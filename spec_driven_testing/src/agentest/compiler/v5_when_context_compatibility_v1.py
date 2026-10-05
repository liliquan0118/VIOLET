"""Contextual suitability of an existing When description, not test execution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .artifacts import content_sha256
from .v5_given_when_preparation_v1 import verify_fingerprint
from .v5_json_mode_preflight_v1 import validate_json_object_messages
from .v5_restriction_batch_v1 import validate_restriction_sources

TASK = "when_context_compatibility"
PILOT_SCHEMA = "agentspectesting.v5-when-context-pilot/v0.1"
PREFLIGHT_SCHEMA = "agentspectesting.v5-when-context-preflight/v0.1"
DECISIONS = {"compatible_under_stated_context", "incompatible", "unclear"}


def seal(value, field):
    value[field] = content_sha256(value)
    return value


def _index(records):
    if not isinstance(records, list):
        raise ValueError("upstream record list required")
    result = {}
    for record in records:
        key = record['gwt']['branch_id']
        if key in result:
            raise ValueError("duplicate upstream branch")
        result[key] = record
    return result


def _render(value, indent=0):
    """Display source material without injecting interpretations of its fields."""
    if isinstance(value, dict):
        return '\n'.join(' ' * indent + str(k) + ':\n' + _render(v, indent + 2)
                         for k, v in value.items())
    if isinstance(value, list):
        return '\n'.join(' ' * indent + '-\n' + _render(v, indent + 2) for v in value) or ' ' * indent + '(empty list)'
    return ' ' * indent + (json.dumps(value) if value is None or isinstance(value, bool) else str(value))


def _sections(branch, user, lookup):
    context = branch['source_context']
    if context['gwt'] != user['gwt'] or context['gwt'] != lookup['gwt']:
        raise ValueError("upstream GWT differs from prepared context")
    for key in ('rule_text', 'origin', 'kind'):
        if context[key] != user[key] or context[key] != lookup[key]:
            raise ValueError("upstream rule context differs")
    if branch['given']['source_text'] != user['gwt']['given'] or branch['when']['source_text'] != user['gwt']['when']:
        raise ValueError("prepared text differs from GWT")
    if branch['when']['suggested_user_trigger_text'] != user['trigger']['trigger']:
        raise ValueError("suggested request source differs")
    for side, record in (('expected', user), ('lookup', lookup)):
        assertion = branch['relation_handoff'][side + '_relation_assertion']
        if assertion['present'] != ('relation' in record) or (assertion['present'] and assertion['value'] != record['relation']):
            raise ValueError("upstream relationship differs from preparation")
    sections = {'Rule': context['rule_text'], 'Given': user['gwt']['given'],
                'Original When': user['gwt']['when'], 'Expected Then': user['gwt']['then'],
                'Suggested request': user['trigger']['trigger']}
    for name, record in (('User-requirements relationship proposal', user),
                         ('Lookup relationship proposal', lookup)):
        sections[name] = ('Proposed relationship between test user and target object: ' + _render(record['relation'])
                          if 'relation' in record else 'Relationship not supplied; no relationship may be inferred from its absence.')
    sections['Context verification'] = (
        'Both relationship entries are upstream proposals, not independent confirmations. '
        'Their correctness and their binding to actual users and objects have not been established by this review. '
        'Any compatible decision relying on them is conditional; inconsistent proposals need clarification.')
    for label, key in (('User-requirements target category proposal', 'root'),):
        sections[label] = _render(user[key]) if key in user else 'Not supplied.'
    sections['Lookup target category proposal'] = _render(lookup['root']) if 'root' in lookup else 'Not supplied.'
    for label, key in (('Additional upstream non-database conditions', 'nonDBconditions'),
                       ('Upstream conversation prerequisites', 'conversation_preconditions'),
                       ('Upstream When qualifiers', 'when_qualifiers')):
        value = user.get(key)
        if value:
            sections[label] = 'Unverified upstream proposal:\n' + _render(value)
    if any(not isinstance(v, str) or not v.strip() for v in sections.values()):
        raise ValueError("nonempty contextual text required")
    return sections


def build_context_pool(parent, gate, preparation, user_records, lookup_records, template):
    for value, field in ((parent, 'pilot_fingerprint'), (gate, 'packet_set_fingerprint'),
                         (preparation, 'preparation_set_fingerprint')):
        verify_fingerprint(value, field)
    if gate.get('schema_version') != 'agentspectesting.v5-when-restriction-packets/v0.1':
        raise ValueError('reviewed event-kind gate required')
    if parent['source_packet_set_fingerprint'] != gate['packet_set_fingerprint']:
        raise ValueError('parent gate mismatch')
    if not isinstance(template, str) or not template.strip():
        raise ValueError('context template required')
    allowed = {p['packet_id']: p for p in gate['packets']}
    branches = {b['branch_id']: b for b in preparation['branches']}
    if len(allowed) != len(gate['packets']) or len(branches) != len(preparation['branches']):
        raise ValueError('duplicate source identity')
    users, lookups = _index(user_records), _index(lookup_records)
    packets, seen = [], set()
    for source in parent['packets']:
        verify_fingerprint(source, 'packet_fingerprint')
        if source != allowed.get(source['packet_id']) or source['task'] != 'when_restriction_preservation':
            raise ValueError('parent is not an admitted event-kind source')
        if source['dependency']['review_fingerprint'] != gate['source_review_fingerprint']:
            raise ValueError('event-kind review mismatch')
        key = source['branch_id']
        if key in seen or key not in branches or key not in users or key not in lookups:
            raise ValueError('duplicate or missing selected branch')
        seen.add(key)
        branch = branches[key]
        verify_fingerprint(branch, 'preparation_fingerprint')
        if source['source_preparation_fingerprint'] != branch['preparation_fingerprint']:
            raise ValueError('source preparation mismatch')
        if branch['structural_issues'] or branch['preparation_status'] != 'prepared_for_semantic_review':
            raise ValueError('branch is not ready for contextual review')
        sections = _sections(branch, users[key], lookups[key])
        if sections['Original When'] != source['model_view']['original_when'] or sections['Suggested request'] != source['model_view']['suggested_description']:
            raise ValueError('selected texts changed')
        messages = [{'role': 'system', 'content': template.strip()},
                    {'role': 'user', 'content': '\n\n'.join(k + ':\n' + v for k, v in sections.items())}]
        validate_json_object_messages(messages)
        provenance = {'branch_id': key, 'parent_pilot_fingerprint': parent['pilot_fingerprint'],
                      'source_packet_fingerprint': source['packet_fingerprint'],
                      'source_preparation_fingerprint': branch['preparation_fingerprint'],
                      'user_record_fingerprint': content_sha256(users[key]),
                      'lookup_record_fingerprint': content_sha256(lookups[key])}
        identity = {'task': TASK, 'messages': messages, 'provenance': provenance}
        packets.append(seal({'packet_id': 'V5-WHEN-CONTEXT::' + content_sha256(identity)[:24],
                             'task': TASK, 'model_view': sections, 'messages': messages,
                             'provenance': provenance, 'approval_status': 'not_requested',
                             'result_status': 'not_run'}, 'packet_fingerprint'))
    if not packets:
        raise ValueError('no selected sources')
    return seal({'schema_version': 'agentspectesting.v5-when-context-pool/v0.1', 'packets': packets,
                 'summary': {'proposed_context_calls': len(packets), 'external_llm_calls': 0},
                 'policy': {'no_required_difference_label': True, 'automatic_contract_promotion': False,
                            'upstream_context_is_not_verified': True}}, 'packet_set_fingerprint')


def validate_context_response(packet, response):
    verify_fingerprint(packet, 'packet_fingerprint')
    if packet['task'] != TASK or not isinstance(response, dict) or set(response) != {'decision', 'reason', 'assumptions', 'evidence'}:
        raise ValueError('context decision, reason, assumptions and evidence required')
    if not isinstance(response['decision'], str) or response['decision'] not in DECISIONS:
        raise ValueError('unknown contextual decision')
    short = lambda v: isinstance(v, str) and bool(v.strip()) and len(v) <= 1600
    if not short(response['reason']):
        raise ValueError('short contextual reason required')
    assumptions = response['assumptions']
    if not isinstance(assumptions, list) or any(not short(v) for v in assumptions) or len(assumptions) != len(set(assumptions)):
        raise ValueError('distinct short assumption strings required')
    if not isinstance(response['evidence'], list) or not response['evidence']:
        raise ValueError('context evidence required for every verdict')
    seen = set()
    for item in response['evidence']:
        if not isinstance(item, dict) or set(item) != {'section', 'quote'}:
            raise ValueError('evidence needs section and quote')
        section, quote = item['section'], item['quote']
        if not isinstance(section, str) or section not in packet['model_view'] or not short(quote) or quote not in packet['model_view'][section]:
            raise ValueError('evidence must quote its supplied section')
        identity = (section, quote)
        if identity in seen:
            raise ValueError('duplicate evidence')
        seen.add(identity)
    return {'packet_id': packet['packet_id'], 'source_packet_fingerprint': packet['packet_fingerprint'],
            'response': deepcopy(response), 'format_status': 'valid', 'semantic_acceptance': 'pending_review',
            'context_verification': 'not_performed', 'runtime_reachability': 'not_assessed',
            'test_readiness': 'not_assessed', 'automatic_contract_promotion': False}


def validate_context_sources(pilot, preflight):
    docs = {}
    for key in ('parent_pilot', 'parent_preflight', 'restriction_gate', 'preparation',
                'user_requirements', 'conditions_matches', 'context_template'):
        ref = preflight['input_files'][key]
        data = Path(ref['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['sha256']:
            raise ValueError('context source changed: ' + key)
        docs[key] = data.decode('utf-8') if key == 'context_template' else json.loads(data)
    parent, check = docs['parent_pilot'], docs['parent_preflight']
    if parent['pilot_fingerprint'] != check['pilot_fingerprint']:
        raise ValueError('context parent identity mismatch')
    validate_restriction_sources(parent, check)
    rebuilt = build_context_pool(parent, docs['restriction_gate'], docs['preparation'],
                                 docs['user_requirements'], docs['conditions_matches'], docs['context_template'])
    if rebuilt['packet_set_fingerprint'] != pilot['source_packet_set_fingerprint'] or rebuilt['packets'] != pilot['packets']:
        raise ValueError('context batch differs from exact source reconstruction')
    return {'source_reconstruction': 'matched', 'packet_count': len(rebuilt['packets'])}
