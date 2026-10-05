"""Select new event-kind review questions from the verified frozen source pool."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import prepare_given_when, verify_fingerprint
from .v5_when_split_review_v1 import build_kind_packets, KIND, validate_split_response

PILOT_SCHEMA = 'agentspectesting.v5-when-kind-expansion-pilot/v0.1'
PREFLIGHT_SCHEMA = 'agentspectesting.v5-when-kind-expansion-preflight/v0.1'


def seal(value, field):
    value[field] = content_sha256(value)
    return value


def select_unreviewed_kind(pool, reviewed, selection):
    verify_fingerprint(pool, 'packet_set_fingerprint')
    verify_fingerprint(reviewed, 'review_fingerprint')
    if reviewed.get('schema_version') != 'agentspectesting.v5-when-kind-reviewed/v0.1' or reviewed['source_packet_set_fingerprint'] != pool['packet_set_fingerprint']:
        raise ValueError('review belongs to another event-kind pool')
    if selection.get('schema_version') != 'agentspectesting.v5-when-kind-expansion-selection/v0.1':
        raise ValueError('expansion selection schema required')
    if selection.get('reuse_reviewed_branches') is not False or selection.get('automatic_followup_calls') is not False:
        raise ValueError('no reviewed sample reuse or automatic followups allowed')
    ids, limit = selection['branch_ids'], selection['max_calls']
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('distinct selected branch IDs required')
    if type(limit) is not int or not 0 < len(ids) <= limit:
        raise ValueError('selection is empty or exceeds call cap')
    candidates, packets_by_id = {}, {}
    for p in pool['packets']:
        verify_fingerprint(p, 'packet_fingerprint')
        if p['task'] != KIND or p['branch_id'] in candidates or p['packet_id'] in packets_by_id:
            raise ValueError('unexpected or duplicate source packet')
        candidates[p['branch_id']] = p
        packets_by_id[p['packet_id']] = p
    reviewed_ids = set()
    for row in reviewed['reviews']:
        p = packets_by_id.get(row['packet_id'])
        if p is None or row['source_packet_fingerprint'] != p['packet_fingerprint'] or row['packet_id'] in reviewed_ids:
            raise ValueError('review source identity mismatch')
        validate_split_response(p, row['response'])
        reviewed_ids.add(row['packet_id'])
    if not set(ids) <= candidates.keys():
        raise ValueError('selected branch not in differing-text pool')
    if any(candidates[key]['packet_id'] in reviewed_ids for key in ids):
        raise ValueError('selected branch already reviewed; do not repeat it')
    packets = [deepcopy(candidates[key]) for key in ids]
    return seal({'schema_version': PILOT_SCHEMA, 'source_packet_set_fingerprint': pool['packet_set_fingerprint'],
                 'source_review_fingerprint': reviewed['review_fingerprint'],
                 'selection_fingerprint': content_sha256(selection), 'packets': packets,
                 'proposed_model': 'deepseek-v4-flash', 'approval_status': 'not_requested',
                 'approved_call_count': 0, 'automatic_retries': 0, 'max_calls_per_packet': 1,
                 'target_agent_calls': 0, 'summary': {'proposed_event_kind_calls': len(packets),
                     'context_calls_ready': 0, 'external_llm_calls': 0}}, 'pilot_fingerprint')


def reconstruct_expansion(input_files):
    """Only local reads; source reconstruction does not grant execution approval."""
    docs = {}
    for key in ('preparation', 'kind_pool', 'kind_review', 'event_kind_template', 'selection',
                'release_manifest', 'release_snapshot'):
        ref = input_files[key]
        data = Path(ref['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['sha256']:
            raise ValueError('expansion input changed: ' + key)
        if key != 'release_snapshot':
            docs[key] = data.decode('utf-8') if key == 'event_kind_template' else json.loads(data)
    manifest = docs['release_manifest']
    verify_fingerprint(manifest, 'release_fingerprint')
    if manifest.get('method_version') != 'v5-source-step-0.1.1':
        raise ValueError('expected frozen source-step v0.1.1')
    if input_files['release_snapshot']['sha256'] != manifest['snapshot']['sha256']:
        raise ValueError('snapshot differs from release manifest')
    with zipfile.ZipFile(input_files['release_snapshot']['path']) as archive:
        paths = manifest['source_step_artifacts']
        intake = json.loads(archive.read('workspace/' + paths['intake']))
        scope = json.loads(archive.read('workspace/' + paths['scope']))
    if intake['intake_fingerprint'] != manifest['source_intake_fingerprint'] or scope['scope_fingerprint'] != manifest['scope_fingerprint']:
        raise ValueError('frozen source identity mismatch')
    expected = prepare_given_when(intake, scope)
    actual = docs['preparation']
    verify_fingerprint(actual, 'preparation_set_fingerprint')
    if actual['source_release']['release_fingerprint'] != manifest['release_fingerprint']:
        raise ValueError('preparation release mismatch')
    strip = lambda x: {k: v for k, v in x.items() if k not in ('source_release', 'preparation_set_fingerprint')}
    if strip(actual) != strip(expected):
        raise ValueError('preparation differs from frozen source reconstruction')
    pool = build_kind_packets(actual, docs['event_kind_template'])
    if pool != docs['kind_pool']:
        raise ValueError('event-kind pool differs from preparation and template')
    return select_unreviewed_kind(pool, docs['kind_review'], docs['selection'])


def validate_expansion_sources(pilot, preflight):
    rebuilt = reconstruct_expansion(preflight['input_files'])
    if pilot != rebuilt:
        raise ValueError('expansion batch differs from exact selected source packets')
    return {'source_reconstruction': 'matched', 'packet_count': len(rebuilt['packets'])}
