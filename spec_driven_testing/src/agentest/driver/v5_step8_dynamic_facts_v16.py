"""Budgeted, fresh-question fact routing; experimental callback use, not approval.

Only the information branch differs from v0.10. No saved selection, global alias,
payment preference inference, feedback search, retry or online entry point.
"""
from copy import deepcopy
from hashlib import sha256
import json

from agentest.compiler.artifacts import content_sha256
from .v5_object_references_v1 import seal
from .v5_step8_dialogue_v1 import BudgetExhausted, DialogueError
from .v5_step8_session_v10 import VersionedOfflineSession
from .v5_step8_integration_v7 import REQUEST_KINDS
from .v5_step8_fact_paths_v13 import build_catalog, map_exact, mapping_question, project_paths
from .v5_step8_fact_correspondence_v14 import check_answer, parse_answer
from .v5_step8_transport_v11 import CombinedBudget
from .v5_step8_runner_v12 import JournaledCompletions

FACT_POLICY = 'experimental_unique_exact_leaf_or_current_question_model_leaf/v0.16'
PROMPT_REVISION = 'fact_correspondence/v0.13_prompt_v0.16_dispatch'


class StrictJournaledCompletions(JournaledCompletions):
    """Retain raw receipts; reject duplicate keys/nonfinite numbers before return."""
    def interpret(self, request):
        super().interpret(request)
        try:
            answer = parse_answer(self.receipts[-1]['response']['choices'][0]['message']['content'])
        except Exception as exc:
            self.receipts[-1].update(strict_decoding_status='rejected', status='error', error_type=type(exc).__name__)
            self.journal.append('strict_interpretation_unusable', {'error_type': type(exc).__name__})
            raise
        self.receipts[-1]['strict_decoding_status'] = 'accepted_not_semantically_verified'
        self.journal.append('strict_interpretation_decoded_not_semantically_verified', {})
        return answer


class DynamicFactSession(VersionedOfflineSession):
    """Leaf-only experimental selection policy must be explicitly configured.

    A structurally valid model choice is NOT a semantic certificate. Keep the raw
    v0.14 check unchanged; record this separate runtime policy in every handoff.
    All-or-nothing applies to the extracted names, not all requests in the message.
    """
    def __init__(self, *args, fact_policy, total_limit, **kwargs):
        if fact_policy != FACT_POLICY:
            raise ValueError('explicit_experimental_fact_policy_required')
        super().__init__(*args, **kwargs)
        self.budget = CombinedBudget(self.budget.limits['target'],
                                     self.budget.limits['interpretation'], total_limit)
        self.catalog = build_catalog(self.case, self.public_tools)
        self.fact_handoffs = []

    def _correspond(self, callback, packet, extraction, name):
        question = mapping_question(packet, extraction, self.catalog, name)
        context = content_sha256(self.messages)
        dispatch_packet = seal({'kind': 'fact_correspondence', 'question': deepcopy(question)}, 'packet_fingerprint')
        request = {'packet': dispatch_packet, 'prompt': question['prompt'], 'prompt_revision': PROMPT_REVISION}
        record = {'packet': deepcopy(dispatch_packet), 'rendered_prompt': question['prompt'],
                  'prompt_revision': PROMPT_REVISION,
                  'prompt_sha256': sha256(question['prompt'].encode()).hexdigest(),
                  'status': 'attempted'}
        self.interpretations.append(record)
        try:
            answer = self.budget.invoke('interpretation', callback, request)
        except BudgetExhausted:
            record['status'] = 'budget_exhausted_before_call'
            self.stopped = 'interpretation_budget_exhausted'
            raise
        except Exception as exc:
            record.update(status='callback_error', error_type=type(exc).__name__)
            self.stopped = 'interpretation_error'
            raise
        record.update(answer=deepcopy(answer), status='returned_semantics_not_certified')
        try:
            if (context != content_sha256(self.messages)
                    or self.question('information_names') != packet
                    or build_catalog(self.case, self.public_tools) != self.catalog):
                raise DialogueError('correspondence_context_changed')
            checked = check_answer(packet, extraction, self.catalog, question, answer)
        except Exception:
            record['status'] = 'interface_or_source_validation_failed'
            raise
        record.update(status='interface_checked_semantics_not_certified', checked_answer=deepcopy(checked))
        return checked

    def _checked(self, callback, kind):
        if kind != 'information_names': return super()._checked(callback, kind)
        context = content_sha256(self.messages)
        extraction = self._interpret(callback, kind)
        extraction_record = self.interpretations[-1]
        packet = deepcopy(extraction_record['packet'])
        try:
            if context != content_sha256(self.messages) or self.question(kind) != packet:
                raise DialogueError('extraction_context_changed')
            mapping = map_exact(packet, extraction, self.catalog)
            entries = {e['path']: e for e in self.catalog['entries']}
            rows = []
            for original in mapping['rows']:
                row = {**deepcopy(original), 'path': None}
                if row['status'] == 'mapped':
                    path = row['candidate_paths'][0]
                    row['basis'] = 'unique_exact_leaf_name_under_experimental_policy'
                    if entries[path]['kind'] == 'leaf': row['path'] = path
                    else: row['status'] = 'container_scope_unresolved'
                else:
                    checked = self._correspond(callback, packet, extraction, row['information_name'])
                    row['checked_answer'] = checked
                    row['basis'] = 'fresh_model_choice_under_experimental_policy_not_reviewed'
                    if checked['selected_kind'] == 'leaf':
                        row.update(status='mapped', path=checked['answer']['path'])
                    else: row['status'] = 'unresolved' if checked['answer']['path'] is None else 'container_scope_unresolved'
                rows.append(row)
                # One unresolved name blocks the reply. Do not spend on later names.
                if row['status'] != 'mapped': break
            complete = bool(rows) and len(rows) == len(mapping['rows']) and all(r['status'] == 'mapped' for r in rows)
            result = seal({'schema_version': 'agentspectesting.dynamic-fact-handoff/v0.16',
                'source_packet_fingerprint': packet['packet_fingerprint'],
                'source_extraction_fingerprint': content_sha256(extraction),
                'catalog_fingerprint': self.catalog['catalog_fingerprint'], 'rows': rows,
                'unprocessed_name_count': len(mapping['rows'])-len(rows),
                'status': 'mapped' if complete else 'unresolved',
                'paths': list(dict.fromkeys(r['path'] for r in rows)) if complete else [],
                'fact_policy': FACT_POLICY, 'semantic_accuracy_certified': False,
                'scope': 'extracted_names_only_not_message_completeness',
                'online_use_accepted': False, 'confirmation_allowed': False}, 'handoff_fingerprint')
        except Exception:
            extraction_record['status'] = 'fact_handoff_failed'
            raise
        extraction_record['status'] = 'extraction_interface_checked_not_completeness_certified'
        self.fact_handoffs.append(deepcopy(result))
        return result

    def respond(self, interpreter):
        # v0.8 routing/consent retained verbatim except information projection.
        self._require_active()
        try:
            checked = {kind: self._checked(interpreter, kind) for kind in REQUEST_KINDS}
            requests = {kind: result['status'] for kind, result in checked.items()}
            if 'unclear' in requests.values(): return self._stop('unresolved_user_request')
            information = requests['requested_information'] == 'present'
            confirmation = requests['confirmation_request'] == 'present'
            if requests['end_signal'] == 'present':
                return self._stop('inconsistent_end_and_request' if information or confirmation else 'agent_end')
            if information:
                handoff = self._checked(interpreter, 'information_names')
                if handoff['status'] != 'mapped': return self._stop('requested_information_unresolved')
                projected = project_paths(self.case, self.public_tools, self.catalog, handoff['paths'], agent_requested=True)
                entries = {e['path']: e for e in self.catalog['entries']}
                if all(len(entries[p]['tokens']) == 1 for p in handoff['paths']):
                    values = {entries[f['path']]['tokens'][0]: f['value'] for f in projected['facts']}
                else: values = projected
                response = {'role': 'user', 'content': json.dumps(values, ensure_ascii=False, sort_keys=True)}
                self.events.append({'kind': 'provide_requested_fact_paths', 'paths': handoff['paths'],
                    'message_index': len(self.messages), 'source_assistant_index': len(self.messages)-1,
                    'source_handoff_fingerprint': handoff['handoff_fingerprint'],
                    'confirmation_deferred_until_fresh_agent_request': confirmation,
                    'execution_mode': 'callback_simulation_not_online_attestation',
                    'confirmation_emitted': False, 'other_requests_in_message': 'not_claimed_resolved'})
                self.messages.append(response)
                return deepcopy(response)
            if confirmation:
                return self._consent(interpreter, {kind: result['source_packet_fingerprint']
                                                  for kind, result in checked.items()})
            return self._stop('no_supported_user_request')
        except Exception:
            if self.consent_assessments and self.consent_assessments[-1].get('status') == 'identity_pending':
                self.consent_assessments[-1]['status'] = 'identity_failed_without_consent'
            if self.stopped is None: self.stopped = 'interpretation_or_response_validation_error'
            raise

    def finish(self, reason='dynamic_fact_session_stopped'):
        trace = super().finish(reason)
        trace.pop('trace_fingerprint')
        trace.update(schema_version='agentspectesting.dynamic-fact-session/v0.16',
            fact_handoffs=deepcopy(self.fact_handoffs), fact_policy=FACT_POLICY,
            runtime_scope='experimental_callbacks_not_online_attestation',
            fact_correspondence_semantics='requires_separate_review',
            combined_call_limit=self.budget.total_limit,
            confirmation_logic_scope='unchanged_v0.8_consent_with_v0.9_identity_prompt')
        return seal(trace, 'trace_fingerprint')
