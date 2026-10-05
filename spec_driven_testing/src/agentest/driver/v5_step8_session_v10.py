"""Version-bound prompt dispatch over the existing offline consent session."""
from copy import deepcopy
from hashlib import sha256

from .v5_object_references_v1 import seal
from .v5_step8_consent_v8 import OfflineConsentSession, render_question as existing_render
from .v5_step8_identity_question_v9 import render_question as identity_render, PROMPT_REVISION


class VersionedOfflineSession(OfflineConsentSession):
    """Callback receives {packet, prompt, prompt_revision}; only prompt is model text.

    No model client, tool executor or global replacement of historical renderers.
    Routing, answer checks, consent policy and budgets remain inherited unchanged.
    """
    def _interpret(self, callback, kind):
        def dispatch(packet):
            identity = kind == 'original_request_identity'
            prompt = identity_render(packet) if identity else existing_render(packet)
            revision = PROMPT_REVISION if identity else 'unchanged_step8_atomic_extraction_comparison'
            self.interpretations[-1].update(rendered_prompt=prompt, prompt_revision=revision,
                prompt_sha256=sha256(prompt.encode('utf-8')).hexdigest(),
                dispatch_contract='prompt_is_model_text_packet_and_revision_are_local_metadata')
            return callback({'packet': deepcopy(packet), 'prompt': prompt, 'prompt_revision': revision})
        return super()._interpret(dispatch, kind)

    def finish(self, reason='offline_adapter_stopped'):
        trace = super().finish(reason)
        trace.pop('trace_fingerprint')
        trace.pop('identity_question_model_calibrated')
        trace.update(schema_version='agentspectesting.versioned-offline-session/v0.10',
            identity_prompt_revision=PROMPT_REVISION,
            identity_semantics='requires_separate_review_not_certified_by_dispatch',
            runtime_scope='offline_only_no_external_client')
        return seal(trace, 'trace_fingerprint')
