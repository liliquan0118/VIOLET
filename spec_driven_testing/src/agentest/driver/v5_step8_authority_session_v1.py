"""AuthorityBoundSession: first real implementation of the decision-authority response
design. See docs/v5_step7_decision_authority_contract_proposal_v0_1.md and
docs/v5_step8_authority_bound_response_proposal_v0_1.md.

Deliberately inherits DialogueSession directly, NOT OfflineDialogueSession /
OfflineConsentSession / DynamicFactSession -- those carry the "classify present/absent,
then extract information_names, then map to a field path" chain this design replaces.
What's reused unmodified from that lineage is base DialogueSession's own
`confirmation_proposal` question construction (render_interpretation_prompt) and
`confirmation_response`'s exact-bundle check -- that machinery already does the one
thing this design also wants there: a mechanical check, not a model's own yes/no
judgment. Quote grounding for that evidence, and for decision_extraction answers, uses
this lineage's own `quote_grounded`/`require_proposal_values_in_quotes` (in
v5_step8_authority_decision_v1.py) rather than the shared v5_step8_dialogue_v1
versions -- see that module for why (real-run findings, and the frozen-acceptance
cost of editing the shared originals instead).

This has not yet been run against a real model. A local mock-callback smoke test
(scripts/run_v5_step8_authority_smoke_v0_1.py --validate-only / a fake client) should
be exercised before spending any real API budget.
"""
from copy import deepcopy
from hashlib import sha256

from .v5_object_references_v1 import seal
from .v5_step7_package_v1 import confirmation_response, PreparationGap
from .v5_step8_dialogue_v1 import DialogueSession, DialogueError, render_interpretation_prompt
from .v5_step8_authority_decision_v1 import (
    decision_authority, make_extraction_question, render_extraction_prompt,
    validate_extraction_answer, decide, compose_reply, audit_grounded,
    require_proposal_values_in_quotes, quote_grounded,
)


class AuthorityBoundSession(DialogueSession):
    def __init__(self, case, public_tools, *, target_limit, interpretation_limit):
        super().__init__(case, public_tools, target_limit=target_limit, interpretation_limit=interpretation_limit)
        self.contract = decision_authority(self.case)
        self.authority_events = []

    def question(self, kind):
        self._require_active()
        if kind == 'confirmation_proposal':
            return super().question(kind)
        if kind == 'decision_extraction':
            if (not self.messages or self.messages[-1]['role'] != 'assistant' or self.pending
                    or self.messages[-1].get('tool_calls')):
                raise DialogueError('interpretation_requires_assistant_text_turn')
            if not isinstance(self.messages[-1].get('content'), str) or not self.messages[-1]['content'].strip():
                raise DialogueError('assistant_text_missing')
            latest = {'message_index': len(self.messages) - 1, 'role': 'assistant', 'content': self.messages[-1]['content']}
            return make_extraction_question(self.contract, latest)
        raise DialogueError('unknown_authority_question_kind')

    def _interpret(self, callback, kind):
        def dispatch(packet):
            if kind == 'decision_extraction':
                prompt, revision = render_extraction_prompt(packet), 'authority_bound_decision_extraction/v0.1'
            elif kind == 'confirmation_proposal':
                prompt, revision = render_interpretation_prompt(packet), 'unchanged_step8_base_confirmation_proposal'
            else:
                raise DialogueError('unknown_authority_interpretation_kind')
            self.interpretations[-1].update(rendered_prompt=prompt, prompt_revision=revision,
                prompt_sha256=sha256(prompt.encode('utf-8')).hexdigest(),
                dispatch_contract='prompt_is_model_text_packet_and_revision_are_local_metadata')
            return callback({'packet': deepcopy(packet), 'prompt': prompt, 'prompt_revision': revision})
        return super()._interpret(dispatch, kind)

    def _stop(self, reason):
        self.stopped = reason
        self.events.append({'kind': reason, 'source_assistant_index': len(self.messages) - 1})
        return None

    def respond(self, interpreter):
        self._require_active()
        try:
            raw = self._interpret(interpreter, 'decision_extraction')
            record = self.interpretations[-1]
            try:
                validated = validate_extraction_answer(record['packet'], raw, self.messages)
            except Exception:
                record['status'] = 'interface_validation_failed'
                raise
            record['status'] = 'interface_valid_semantics_not_certified'

            if validated['end_signal_quote'] is not None and not validated['addressed'] and not validated['unaddressed_request_quotes']:
                return self._stop('agent_end')

            decision = decide(self.contract, validated)

            consent_text = None
            needs_consent = any(d['decision_id'] == 'final_operation_consent' for d in decision['decided'])
            if needs_consent:
                if self.confirmed:
                    return self._stop('repeated_confirmation_not_supported')
                proposal = self._interpret(interpreter, 'confirmation_proposal')
                if not isinstance(proposal, dict) or set(proposal) != {'proposal', 'evidence'}:
                    raise DialogueError('invalid_proposal_answer')
                if proposal['proposal'] is None:
                    return self._stop('confirmation_details_incomplete')
                if not isinstance(proposal['evidence'], list) or not proposal['evidence']:
                    raise DialogueError('proposal_source_evidence_missing')
                for evidence in proposal['evidence']:
                    quote_grounded(evidence['message_index'], evidence['quote'], self.messages)
                # Check value presence against every visible assistant message, not just
                # the interpreter's own self-selected `evidence` subset -- we already
                # have the real, full text the interpreter was scoped to (base
                # DialogueSession.question('confirmation_proposal') hands it every
                # assistant message, not only the one it happens to cite). A real run
                # extracted a fully correct reservation_id but only cited the trailing
                # question sentence as evidence, so the old value-in-evidence-only check
                # rejected a proposal that was genuinely grounded in real text -- not
                # because the value was unverifiable, only because we were checking a
                # narrower window than the one we actually have. This still requires
                # every value to be real, visible Agent text (never invented, never a
                # user-side value) -- it only widens WHERE we're willing to look for it.
                full_assistant_text = [{'message_index': i, 'quote': m['content']}
                                       for i, m in enumerate(self.messages)
                                       if m['role'] == 'assistant' and m.get('content')]
                require_proposal_values_in_quotes(proposal['proposal'], full_assistant_text)
                try:
                    consent_text = confirmation_response(self.case, proposal['proposal'], agent_requested=True)
                except PreparationGap:
                    return self._stop('proposed_operation_does_not_match_prepared_bundle')
                self.confirmed = True

            reply_text = compose_reply(self.contract, decision, consent_text=consent_text)
            if not reply_text.strip():
                if validated['end_signal_quote'] is not None:
                    return self._stop('agent_end')
                unresolved = [d for d in decision['decided'] if d.get('outcome') == 'unresolved_path_hint']
                return self._stop('value_confirmation_path_hint_unresolved' if unresolved else 'no_supported_user_request')

            audit = audit_grounded(self.contract, reply_text)
            if audit['leaked_not_in_known_facts']:
                raise DialogueError('composed_reply_failed_grounding_audit')

            message = {'role': 'user', 'content': reply_text}
            event = {'kind': 'authority_bound_reply', 'message_index': len(self.messages),
                'source_assistant_index': len(self.messages) - 1, 'decided': deepcopy(decision['decided']),
                'unaddressed_request_quotes': list(decision['unaddressed_request_quotes']),
                'audit': audit, 'confirmed_this_turn': consent_text is not None}
            self.events.append(event)
            self.authority_events.append(deepcopy(event))
            self.messages.append(message)
            return deepcopy(message)
        except Exception:
            if self.stopped is None:
                self.stopped = 'interpretation_or_response_validation_error'
            raise

    def finish(self, reason='authority_bound_session_stopped'):
        trace = super().finish(reason)
        trace.pop('trace_fingerprint')
        trace.update(schema_version='agentspectesting.v5-step8-authority-bound-session/v0.1',
            decision_authority_contract=deepcopy(self.contract),
            authority_events=deepcopy(self.authority_events),
            design_reference=['docs/v5_step7_decision_authority_contract_proposal_v0_1.md',
                               'docs/v5_step8_authority_bound_response_proposal_v0_1.md'],
            runtime_scope='first_real_implementation_not_previously_calibrated_on_real_model_output',
            semantic_accuracy_certified=False)
        return seal(trace, 'trace_fingerprint')
