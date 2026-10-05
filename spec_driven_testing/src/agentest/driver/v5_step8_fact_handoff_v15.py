"""Source-bound reviewed correspondence in an explicitly offline replay session."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

from agentest.compiler.artifacts import content_sha256
from agentest.compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_data_sources_v1 import read_json
from .v5_object_references_v1 import seal
from .v5_step8_fact_paths_v13 import build_catalog, map_exact, project_paths
from .v5_step8_fact_correspondence_v14 import check_answer, parse_answer
from .v5_step8_session_v10 import VersionedOfflineSession
from .v5_step8_integration_v7 import REQUEST_KINDS


def load_reviewed_selection(root, review_path, *, expected_review_fingerprint):
    """Caller pins a reviewed artifact. A hash is provenance, not user identity."""
    root = Path(root).resolve()
    review_path = Path(review_path).resolve()
    if not review_path.is_relative_to(root): raise ValueError('review_outside_workspace')
    review = read_json(review_path)
    verify_fingerprint(review, 'review_fingerprint')
    if review['review_fingerprint'] != expected_review_fingerprint:
        raise ValueError('unexpected_review')
    if review['status'] != 'accepted_for_this_source_question_only':
        raise ValueError('accepted_review_required')
    directory = review_path.parent
    if set(review['raw_file_sha256']) != {f'{i:06d}.json' for i in range(6)}:
        raise ValueError('complete_reviewed_record_set_required')
    for name, digest in review['raw_file_sha256'].items():
        if Path(name).name != name or sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError('reviewed_raw_record_changed')
    header = read_json(directory/'000000.json')
    request = header['data']['request']
    verify_fingerprint(request, 'request_fingerprint')
    if request['request_fingerprint'] != review['source_request_fingerprint']:
        raise ValueError('review_request_mismatch')
    reservation = read_json(directory/'000001.json')
    received = read_json(directory/'000002.json')
    raw_check = read_json(directory/'000004.json')['data']
    run = read_json(directory/'000005.json')['data']
    verify_fingerprint(run, 'run_fingerprint')
    job = request['job']
    if (reservation['data']['kind'] != 'interpretation'
            or received['data']['attempt_sequence'] != reservation['sequence']
            or received['data']['kind'] != 'interpretation'
            or reservation['data']['request']['messages'][1]['content'] != job['question']['prompt']):
        raise ValueError('request_response_correlation_mismatch')
    response = received['data']['response']
    if response['id'] != review['response_id'] or response['choices'][0]['finish_reason'] != 'stop':
        raise ValueError('reviewed_complete_response_required')
    answer = parse_answer(response['choices'][0]['message']['content'])
    checked = check_answer(**job, answer=answer)
    if (checked != raw_check or checked != run['checked_answer'] or answer != review['answer']
            or checked['check_fingerprint'] != review['source_check_fingerprint']
            or run['run_fingerprint'] != review['source_run_fingerprint']
            or job['question']['question_fingerprint'] != review['source_question_fingerprint']):
        raise ValueError('review_answer_chain_mismatch')
    if checked['selected_kind'] != 'leaf' or answer['path'] is None:
        raise ValueError('local_leaf_handoff_only')
    return seal({'schema_version': 'agentspectesting.reviewed-fact-selection/v0.15',
        'review_fingerprint': review['review_fingerprint'],
        'response_file_sha256': review['raw_file_sha256']['000002.json'],
        'response_id': response['id'],
        'source_packet_fingerprint': job['packet']['packet_fingerprint'],
        'source_extraction_fingerprint': content_sha256(job['extraction']),
        'catalog_fingerprint': job['catalog']['catalog_fingerprint'],
        'source_question_fingerprint': job['question']['question_fingerprint'],
        'information_name': job['question']['information_name'], 'path': answer['path'],
        'scope': 'exact_reviewed_question_only_not_a_global_alias',
        'online_disclosure_authorized': False}, 'selection_fingerprint')


def resolve_information(packet, extraction, catalog, selections):
    mapping = map_exact(packet, extraction, catalog)
    entries = {e['path']: e for e in catalog['entries']}
    rows = []
    for original in mapping['rows']:
        row = deepcopy(original)
        if row['status'] == 'mapped':
            path = row['candidate_paths'][0]
            row.update(path=path, basis='unique_exact_leaf_name_local_replay_policy')
            if entries[path]['kind'] != 'leaf': row.update(status='container_scope_unresolved', path=None)
        else:
            candidates = []
            for selection in selections:
                verify_fingerprint(selection, 'selection_fingerprint')
                if (selection['source_packet_fingerprint'] == packet['packet_fingerprint']
                        and selection['source_extraction_fingerprint'] == content_sha256(extraction)
                        and selection['catalog_fingerprint'] == catalog['catalog_fingerprint']
                        and selection['information_name'] == row['information_name']):
                    candidates.append(selection)
            if len(candidates) == 1 and candidates[0]['path'] in entries and entries[candidates[0]['path']]['kind'] == 'leaf':
                chosen = candidates[0]
                row.update(status='mapped', path=chosen['path'], basis='source_bound_reviewed_correspondence',
                           selection_fingerprint=chosen['selection_fingerprint'], review_fingerprint=chosen['review_fingerprint'])
            else:
                row.update(path=None, review_match_count=len(candidates))
        rows.append(row)
    complete = bool(rows) and all(r['status'] == 'mapped' for r in rows)
    return seal({'schema_version': 'agentspectesting.local-fact-handoff/v0.15',
        'source_packet_fingerprint': packet['packet_fingerprint'],
        'source_extraction_fingerprint': content_sha256(extraction),
        'catalog_fingerprint': catalog['catalog_fingerprint'], 'rows': rows,
        'status': 'mapped' if complete else 'unresolved',
        'paths': list(dict.fromkeys(r['path'] for r in rows)) if complete else [],
        'scope': 'extracted_information_only_not_all_requests_in_agent_message',
        'semantic_accuracy_generalized': False, 'confirmation_allowed': False,
        'online_disclosure_authorized': False}, 'handoff_fingerprint')


class OfflineFactReplaySession(VersionedOfflineSession):
    """New local-only session, not a replacement for the frozen online runner.

    Unique exact leaf names use an explicit local replay policy; this is not proof
    of arbitrary request/object-scope semantics. Reviewed aliases require exact
    source matches. Payment preference and action consent are not inferred here.
    """
    def __init__(self, *args, reviewed_selections, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog = build_catalog(self.case, self.public_tools)
        self.reviewed_selections = deepcopy(reviewed_selections)
        self.fact_handoffs = []

    def _checked(self, callback, kind):
        if kind != 'information_names': return super()._checked(callback, kind)
        answer = self._interpret(callback, kind)
        record = self.interpretations[-1]
        try: result = resolve_information(record['packet'], answer, self.catalog, self.reviewed_selections)
        except Exception:
            record['status'] = 'interface_validation_failed'
            raise
        record['status'] = 'source_checked_local_handoff_not_general_semantic_certificate'
        self.fact_handoffs.append(deepcopy(result))
        return result

    def respond(self, interpreter):
        self._require_active()
        try:
            requests = {kind: self._checked(interpreter, kind)['status'] for kind in REQUEST_KINDS}
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
                # Preserve existing top-level response shape; nested replies retain addresses.
                if all(len(entries[p]['tokens']) == 1 for p in handoff['paths']):
                    values = {entries[f['path']]['tokens'][0]: f['value'] for f in projected['facts']}
                else: values = projected
                message = {'role': 'user', 'content': json.dumps(values, ensure_ascii=False, sort_keys=True)}
                self.events.append({'kind': 'provide_requested_fact_paths', 'message_index': len(self.messages),
                    'source_assistant_index': len(self.messages)-1, 'paths': handoff['paths'],
                    'source_handoff_fingerprint': handoff['handoff_fingerprint'],
                    'execution_mode': 'offline_saved_answer_replay',
                    'payment_choice_emitted': False, 'confirmation_emitted': False,
                    'other_requests_in_message': 'not_claimed_resolved'})
                self.messages.append(message)
                return deepcopy(message)
            # This local handoff deliberately does not exercise operation consent.
            return self._stop('confirmation_not_exercised_in_fact_replay' if confirmation else 'no_supported_user_request')
        except Exception:
            if self.stopped is None: self.stopped = 'interpretation_or_response_validation_error'
            raise

    def finish(self, reason='local_fact_replay_stopped'):
        trace = super().finish(reason)
        trace.pop('trace_fingerprint')
        trace.update(schema_version='agentspectesting.local-fact-replay-session/v0.15',
            fact_handoffs=deepcopy(self.fact_handoffs), runtime_scope='offline_saved_answers_no_external_dispatch',
            fact_policy='unique_exact_leaf_names_plus_source_bound_reviewed_correspondence',
            confirmation_logic_scope='not_exercised_no_online_policy_replacement')
        return seal(trace, 'trace_fingerprint')
