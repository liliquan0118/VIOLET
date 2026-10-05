"""Check correspondence answer shape/provenance without authorizing disclosure."""
from copy import deepcopy
import json

from .v5_object_references_v1 import seal
from .v5_step8_fact_paths_v13 import mapping_question


def parse_answer(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise ValueError('duplicate_answer_key')
            value[key] = item
        return value
    def invalid(value): raise ValueError('nonfinite_json_not_allowed')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def check_answer(packet, extraction, catalog, question, answer):
    rebuilt = mapping_question(packet, extraction, catalog, question['information_name'])
    if rebuilt != question:
        raise ValueError('correspondence_question_or_sources_changed')
    if not isinstance(answer, dict) or set(answer) != {'path'}:
        raise ValueError('only_path_field_required')
    path = answer['path']
    entries = {entry['path']: entry for entry in catalog['entries']}
    if path is not None and (not isinstance(path, str) or path not in entries):
        raise ValueError('exact_candidate_path_or_json_null_required')
    return seal({'schema_version': 'agentspectesting.fact-correspondence-check/v0.14',
        'source_question_fingerprint': question['question_fingerprint'],
        'catalog_fingerprint': catalog['catalog_fingerprint'], 'answer': deepcopy(answer),
        'status': 'unresolved' if path is None else 'candidate_selected_not_semantically_verified',
        'selected_kind': None if path is None else entries[path]['kind'],
        'semantic_status': 'not_reviewed', 'request_scope_verified': False,
        'facts_release_authorized': False, 'confirmation_allowed': False,
        'runtime_integration': 'not_enabled'}, 'check_fingerprint')


def run_one(job, bridge, journal):
    """Exactly one interpretation callback, no target, tool or resume path.

    bridge is the existing zero-retry journaled completion bridge. Invalid and
    truncated raw responses stay in its journal. A returned check is not a pass.
    """
    if bridge.journal is not journal or any(journal.attempts.values()):
        raise ValueError('fresh_shared_journal_required')
    question = job['question']
    # Validate sources before reserving any SDK attempt; null is a legal shape.
    check_answer(job['packet'], job['extraction'], job['catalog'], question, {'path': None})
    result = None
    try:
        bridge.interpret({'prompt': question['prompt']})
        # Re-read the preserved content strictly; legacy decoding permits duplicate keys.
        answer = parse_answer(bridge.receipts[-1]['response']['choices'][0]['message']['content'])
        result = check_answer(job['packet'], job['extraction'], job['catalog'], question, answer)
        journal.append('answer_checked_semantic_review_required', result)
    except Exception as exc:
        journal.append('calibration_response_unusable', {'error_type': type(exc).__name__})
    except BaseException as exc:
        if not journal.failed:
            journal.append('calibration_interrupted_no_resume', {'error_type': type(exc).__name__})
        raise
    summary = seal({'schema_version': 'agentspectesting.fact-correspondence-run/v0.14',
        'source_question_fingerprint': question['question_fingerprint'],
        'reserved_attempts': deepcopy(journal.attempts),
        'received_responses': deepcopy(journal.received), 'checked_answer': result,
        'semantic_status': 'not_reviewed', 'facts_release_authorized': False,
        'actual_target_agent_calls': 0, 'actual_business_tool_calls': 0,
        'automatic_retries': 0, 'automatic_resume': False}, 'run_fingerprint')
    journal.append('calibration_finished_review_required', summary)
    return summary
