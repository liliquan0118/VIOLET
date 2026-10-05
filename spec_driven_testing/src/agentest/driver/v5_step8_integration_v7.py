"""Offline response routing over audited components; no online confirmation gate."""
from copy import deepcopy
import json

from . import v5_step8_atomic_v4 as atomic
from . import v5_step8_extraction_v5 as extraction
from . import v5_step8_handoff_v6 as handoff
from .v5_object_references_v1 import seal
from .v5_step7_package_v1 import requested_facts
from .v5_step8_dialogue_v1 import DialogueSession, DialogueError


REQUEST_KINDS = ('requested_information', 'confirmation_request', 'end_signal')


def render_question(packet):
    if packet['kind'] in REQUEST_KINDS:
        return atomic.render_question(packet)
    if packet['kind'] == 'information_names':
        return extraction.render_question(packet)
    if packet['kind'] == 'explicit_proposal_change':
        return handoff.render_question(packet)
    raise DialogueError('unknown_integrated_question')


class OfflineDialogueSession(DialogueSession):
    """Exercise routing with supplied callbacks, not an approved online adapter.

    Reuses transport, attempt accounting, tool correlation and fact access. All
    comparison outcomes stop; a no-change claim never becomes user permission.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handoffs = []
        self.projections = []

    def _visible_projection(self):
        visible, indices, omitted = [], [], []
        for index, message in enumerate(self.messages):
            content = message.get('content')
            if message['role'] == 'tool':
                omitted.append({'trace_index': index, 'reason': 'tool_result_not_user_visible'})
                continue
            if content is None or content == '':
                if message['role'] != 'assistant' or not message.get('tool_calls'):
                    raise DialogueError('unsupported_empty_text_history')
                omitted.append({'trace_index': index, 'reason': 'tool_only_assistant_message'})
                continue
            if not isinstance(content, str) or not content.strip():
                raise DialogueError('unsupported_visible_text_history')
            visible.append({'message_index': len(visible), 'role': message['role'], 'content': content})
            indices.append(index)
        packet = handoff.make_comparison(visible)
        projection = {'source_packet_fingerprint': packet['packet_fingerprint'],
                      'visible_to_trace_indices': indices, 'omitted_messages': omitted,
                      'scope': 'user_and_assistant_text_only_not_tool_semantics'}
        return packet, projection

    def question(self, kind):
        self._require_active()
        # Existing guard rejects pending tools, text+tools and missing text.
        super().question('user_request')
        latest = {'message_index': len(self.messages)-1, 'role': 'assistant',
                  'content': self.messages[-1]['content']}
        if kind in REQUEST_KINDS:
            return atomic.make_question(kind, latest)
        if kind == 'information_names':
            return extraction.make_question(kind, latest)
        if kind == 'explicit_proposal_change':
            return self._visible_projection()[0]
        raise DialogueError('unknown_integrated_question')

    def _checked(self, callback, kind):
        answer = self._interpret(callback, kind)
        record = self.interpretations[-1]
        packet = record['packet']
        try:
            if kind in REQUEST_KINDS:
                result = atomic.make_handoff(packet, answer)
            elif kind == 'information_names':
                result = handoff.map_information(packet, answer, sorted(self.view['facts_on_request']))
            else:
                result = handoff.make_handoff(packet, answer)
                projection = self._visible_projection()[1]
                self.projections.append(projection)
        except Exception:
            record['status'] = 'interface_validation_failed'
            raise
        record['status'] = 'interface_valid_semantics_not_certified'
        self.handoffs.append(deepcopy(result))
        return result

    def _stop(self, reason):
        self.stopped = reason
        self.events.append({'kind': reason, 'source_assistant_index': len(self.messages)-1})
        return None

    def respond(self, interpreter):
        self._require_active()
        try:
            requests = {kind: self._checked(interpreter, kind)['status'] for kind in REQUEST_KINDS}
            if 'unclear' in requests.values():
                return self._stop('unresolved_user_request')
            information = requests['requested_information'] == 'present'
            confirmation = requests['confirmation_request'] == 'present'
            if requests['end_signal'] == 'present':
                return self._stop('inconsistent_end_and_request' if information or confirmation else 'agent_end')
            if information:
                mapping = self._checked(interpreter, 'information_names')
                if mapping['status'] != 'mapped':
                    return self._stop('requested_information_unresolved')
                names = mapping['mapped_fields']
                values = requested_facts(self.case, names, agent_requested=True)
                response = {'role': 'user', 'content': json.dumps(values, ensure_ascii=False, sort_keys=True)}
                self.events.append({'kind': 'provide_requested_facts', 'fields': names,
                    'message_index': len(self.messages), 'source_assistant_index': len(self.messages)-1,
                    'source_mapping_fingerprint': mapping['mapping_fingerprint'],
                    'confirmation_deferred_until_fresh_agent_request': confirmation,
                    'execution_mode': 'offline_callback_simulation'})
                self.messages.append(response)
                return deepcopy(response)
            if confirmation:
                comparison = self._checked(interpreter, 'explicit_proposal_change')
                return self._stop({'yes': 'explicit_change_claim_no_confirmation',
                                   'unclear': 'comparison_unresolved_no_confirmation',
                                   'no': 'confirmation_gate_pending_not_authorized'}[comparison['comparison']['answer']])
            return self._stop('no_supported_user_request')
        except Exception:
            if self.stopped is None:
                self.stopped = 'interpretation_or_response_validation_error'
            raise

    def finish(self, reason='offline_adapter_stopped'):
        trace = super().finish(reason)
        trace.pop('trace_fingerprint')
        trace.update(schema_version='agentspectesting.v5-step8-offline-integration/v0.7',
                     handoffs=deepcopy(self.handoffs), visible_text_projections=deepcopy(self.projections),
                     confirmation_allowed=False, online_use_accepted=False,
                     execution_mode='offline_callback_simulation')
        return seal(trace, 'trace_fingerprint')
