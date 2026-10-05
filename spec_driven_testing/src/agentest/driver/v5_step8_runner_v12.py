"""Durable, non-resuming execution wrapper. No credentials or clients at import."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os

from .v5_data_sources_v1 import encode
from .v5_object_references_v1 import seal
from .v5_step8_transport_v11 import CompletionBridges, run_transport


class Journal:
    """Immutable numbered events; fsync file and directory before side effects.

    A reserved attempt is conservatively consumed, not proof the server received it.
    Existing directories are never resumed. Any write failure poisons this writer.
    """
    def __init__(self, directory, header):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        self.failed = False
        self.attempts = dict(target=0, interpretation=0, tools=0)
        self.received = dict(target=0, interpretation=0, tools=0)
        self.append('run_started', header)

    def append(self, event, data):
        if self.failed:
            raise OSError('journal_unusable_no_further_dispatch')
        try:
            item = {'sequence': self.sequence, 'event': event, 'data': deepcopy(data)}
            payload = encode(item)
            path = self.directory / f'{self.sequence:06d}.json'
            with path.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            for directory in (self.directory, self.directory.parent):
                descriptor = os.open(directory, os.O_RDONLY)
                try: os.fsync(descriptor)
                finally: os.close(descriptor)
            self.sequence += 1
            return item['sequence']
        except BaseException:
            self.failed = True
            raise

    def reserve(self, kind, payload):
        counts = {**self.attempts, kind: self.attempts[kind] + 1}
        index = self.append('attempt_reserved_before_dispatch',
                            {'kind': kind, 'request': payload, 'consumed': counts})
        self.attempts = counts
        return index

    def response(self, kind, index, value):
        self.append('response_received', {'kind': kind, 'attempt_sequence': index, 'response': value})
        self.received[kind] += 1


class JournaledCompletions(CompletionBridges):
    """Reuse v0.11 decoding; persist SDK response before parsing/validation."""
    def __init__(self, client, journal, **kwargs):
        super().__init__(client, **kwargs)
        self.journal = journal
        self.original_client = client
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self._dispatch)))

    def _dispatch(self, **payload):
        kind = self.active_kind
        index = self.journal.reserve(kind, payload)
        try:
            response = self.original_client.chat.completions.create(**payload)
            self.journal.response(kind, index, response.model_dump(mode='json'))
            return response
        except BaseException as exc:
            self.journal.append('dispatch_failed_or_interrupted',
                {'kind': kind, 'attempt_sequence': index, 'error_type': type(exc).__name__})
            raise

    def _call(self, kind, payload):
        self.active_kind = kind  # Synchronous transport only; no concurrent calls.
        try:
            result = super()._call(kind, payload)
            self.journal.append('completion_decoded_not_semantically_verified', {'kind': kind})
            return result
        except BaseException as exc:
            self.journal.append('completion_not_usable', {'kind': kind, 'error_type': type(exc).__name__})
            raise


def run_journaled(session, *, bridges, tool_bridge, journal, tool_limit, mode):
    if mode not in ('mock', 'external_sdk_local_tau'):
        raise ValueError('explicit_execution_mode_required')

    def checkpoint():
        # finish() is terminal in the inherited session; never call it mid-run.
        journal.append('session_checkpoint', {
            'messages': session.messages, 'interpretations': session.interpretations,
            'events': session.events, 'pending_tool_calls': session.pending,
            'termination_reason': session.stopped,
            'meaning': 'nonterminal_observation_not_execution_attestation'})

    def callback(function):
        def invoke(payload):
            try: return function(payload)
            finally: checkpoint()
        return invoke

    def tool(call):
        index = journal.reserve('tools', call)
        try:
            response = tool_bridge(call)
            raw = response.model_dump(mode='json', exclude_none=True) if hasattr(response, 'model_dump') else response
            journal.response('tools', index, raw)
            return response
        except BaseException as exc:
            journal.append('dispatch_failed_or_interrupted',
                {'kind': 'tools', 'attempt_sequence': index, 'error_type': type(exc).__name__})
            raise

    def state():
        value = tool_bridge.capture_state()
        journal.append('database_snapshot', value)
        return value

    try:
        raw = run_transport(session, target=callback(bridges.target), interpreter=callback(bridges.interpret),
                            execute_tool=tool, capture_state=state, tool_limit=tool_limit)
        checkpoint()
        result = seal({'schema_version': 'agentspectesting.durable-run/v0.12',
            'raw_transport': raw,
            'execution_evidence': {
                'configured_mode': mode,
                'reserved_attempts': deepcopy(journal.attempts),
                'received_responses': deepcopy(journal.received),
                'external_model_responses_received': mode == 'external_sdk_local_tau' and
                    journal.received['target'] + journal.received['interpretation'] > 0,
                'provider_acceptance_of_failed_attempts': 'unknown',
                'source': 'runner_dispatch_journal_not_inherited_offline_flags',
                'inherited_trace_scope': 'raw_driver_structure_and_decisions_only_not_execution_attestation'},
            'oracle_verdict': 'not_evaluated', 'semantic_review': 'pending',
            'automatic_retries': 0, 'automatic_resume': False}, 'run_fingerprint')
        journal.append('run_finished_review_required', result)
        return result
    except BaseException as exc:
        if not journal.failed:
            journal.append('run_aborted_no_automatic_resume', {'error_type': type(exc).__name__})
        raise
