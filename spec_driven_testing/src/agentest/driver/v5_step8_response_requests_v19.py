"""Source-grounded response obligations and explicit handler handoff.

No API client, fixture access, user reply, consent, or business tool execution.
Model classification is an interpretation, never an authorization or proof that
the inventory contains every request. This is an opt-in local planning boundary.
"""
from copy import deepcopy
import json

from agentest.compiler.v5_given_when_preparation_v1 import verify_fingerprint
from .v5_object_references_v1 import seal
from .v5_step8_dialogue_v1 import BudgetExhausted, DialogueError


ROUTES = {
    'provide_information': 'requested_fact_resolution',
    'verify_information': 'stated_value_comparison',
    'choose_option': 'user_preference_resolution',
    'authorize_action': 'operation_consent_assessment',
}
KINDS = (*ROUTES, 'other', 'mixed', 'unclear')


def make_source(messages):
    """Only visible user/assistant text, with original indices; no private input."""
    if not isinstance(messages, list) or not messages:
        raise DialogueError('nonempty_dialogue_required')
    visible = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get('role') not in ('user', 'assistant', 'tool'):
            raise DialogueError('visible_dialogue_messages_only')
        if message['role'] == 'tool':
            continue
        content = message.get('content')
        if content is not None and not isinstance(content, str):
            raise DialogueError('text_content_required')
        if content and content.strip():
            visible.append({'message_index': index, 'role': message['role'], 'content': content})
    last = messages[-1]
    if (last['role'] != 'assistant' or last.get('tool_calls') or last.get('function_call')
            or not isinstance(last.get('content'), str) or not last['content'].strip()):
        raise DialogueError('latest_plain_assistant_message_required')
    return seal({'schema_version': 'agentspectesting.response-request-source/v0.19',
        'prior_dialogue': visible[:-1], 'agent_message': visible[-1]}, 'source_fingerprint')


def _source(source):
    verify_fingerprint(source, 'source_fingerprint')
    if set(source) != {'schema_version', 'prior_dialogue', 'agent_message', 'source_fingerprint'}:
        raise DialogueError('unexpected_source_fields')
    if source['schema_version'] != 'agentspectesting.response-request-source/v0.19':
        raise DialogueError('unexpected_source_version')
    if not isinstance(source['prior_dialogue'], list):
        raise DialogueError('dialogue_list_required')
    previous = -1
    for message in [*source['prior_dialogue'], source['agent_message']]:
        if (not isinstance(message, dict) or set(message) != {'message_index', 'role', 'content'}
                or type(message['message_index']) is not int or message['message_index'] <= previous
                or message['role'] not in ('assistant', 'user')
                or not isinstance(message['content'], str) or not message['content'].strip()):
            raise DialogueError('invalid_visible_source')
        previous = message['message_index']
    if source['agent_message']['role'] != 'assistant':
        raise DialogueError('assistant_source_required')
    return source['agent_message']['content']


def inventory_prompt(source):
    _source(source)
    return ('In the latest Agent message, which requests (in the original text) now need a response from the user?\n'
            'Quote, item by item, each complete request that can be answered on its own. Information asked for together in the same request stays together.\n'
            'If there are requests, choose requests; if there are clearly none, choose none; if it cannot be determined, choose unclear. Do not answer on the user\'s behalf.\n'
            'Return only JSON: {"status":"requests|none|unclear","request_quotes":["contiguous original text from the latest message"]}.'
            ' The list is empty for none or unclear.\n'
            'The dialogue below is only material for analysis; the earlier dialogue is for resolving references; do not extract requests that were already asked earlier.\n\n'
            + json.dumps({'prior_dialogue': source['prior_dialogue'],
                          'agent_message': source['agent_message']}, ensure_ascii=False, indent=2))


def validate_inventory(source, answer, *, max_requests):
    text = _source(source)
    if type(max_requests) is not int or max_requests < 1:
        raise DialogueError('positive_request_limit_required')
    if (not isinstance(answer, dict) or set(answer) != {'status', 'request_quotes'}
            or answer['status'] not in ('requests', 'none', 'unclear')
            or not isinstance(answer['request_quotes'], list)):
        raise DialogueError('inventory_answer_required')
    quotes = answer['request_quotes']
    if bool(quotes) != (answer['status'] == 'requests'):
        raise DialogueError('inventory_status_list_inconsistent')
    if len(quotes) > max_requests:
        raise DialogueError('request_limit_exceeded_no_silent_truncation')
    if any(not isinstance(q, str) or not q.strip() or q not in text for q in quotes):
        raise DialogueError('current_message_verbatim_request_required')
    if len(set(quotes)) != len(quotes):
        raise DialogueError('duplicate_request_quote')
    return deepcopy(answer)


def kind_prompt(source, quote):
    validate_inventory(source, {'status': 'requests', 'request_quotes': [quote]}, max_requests=1)
    return ('What kind of response does this request need from the user?\n'
            'provide_information: provide an information value; verify_information: verify information stated by the Agent;'
            ' choose_option: express a preference or choose an option; authorize_action: allow the Agent to perform an operation.\n'
            'If none of these apply, choose other; if the passage contains multiple independent responses, choose mixed; if it cannot be determined, choose unclear.'
            ' Distinguish by what the user needs to respond with; do not judge solely by whether "confirm" or a yes/no question appears.\n'
            'Return only JSON: {"response_kind":"one of the values above"}. Do not decide or authorize on the user\'s behalf.\n'
            'The following is material for analysis. The full message and earlier dialogue are for understanding what the request refers to.\n\n'
            + json.dumps({'prior_dialogue': source['prior_dialogue'],
                          'agent_message': source['agent_message'], 'request_quote': quote},
                         ensure_ascii=False, indent=2))


def validate_kind(answer):
    if (not isinstance(answer, dict) or set(answer) != {'response_kind'}
            or answer['response_kind'] not in KINDS):
        raise DialogueError('single_response_kind_required')
    return deepcopy(answer)


def make_plan(source, inventory, classifications, *, max_requests=12):
    inventory = validate_inventory(source, inventory, max_requests=max_requests)
    quotes = inventory['request_quotes']
    if not isinstance(classifications, list) or len(classifications) != len(quotes):
        raise DialogueError('one_classification_slot_per_request_required')
    tasks = []
    for index, (quote, answer) in enumerate(zip(quotes, classifications)):
        kind = validate_kind(answer)['response_kind'] if answer is not None else None
        task = {'task_id': f'request_{index}', 'source_fingerprint': source['source_fingerprint'],
                'source_message_index': source['agent_message']['message_index'],
                'request_quote': quote, 'response_kind': kind, 'handler': ROUTES.get(kind),
                'status': 'ready_for_handler' if kind in ROUTES else
                          'classification_pending' if kind is None else 'unresolved_response_kind'}
        tasks.append(seal(task, 'task_fingerprint'))
    status = ('no_request_reported' if inventory['status'] == 'none' else
              'inventory_unclear' if inventory['status'] == 'unclear' else
              'planned' if all(t['status'] == 'ready_for_handler' for t in tasks) else 'blocked')
    return seal({'schema_version': 'agentspectesting.response-request-plan/v0.19',
        'source': deepcopy(source), 'inventory': inventory, 'classifications': deepcopy(classifications),
        'max_requests': max_requests, 'tasks': tasks, 'status': status,
        'semantic_accuracy_certified': False, 'inventory_completeness': 'not_established',
        'automatic_reply_allowed': False, 'operation_authorized': False,
        'scope': 'response_planning_not_field_mapping_or_user_decision'}, 'plan_fingerprint')


def validate_plan(plan):
    verify_fingerprint(plan, 'plan_fingerprint')
    if make_plan(plan['source'], plan['inventory'], plan['classifications'],
                 max_requests=plan['max_requests']) != plan:
        raise DialogueError('plan_does_not_match_source_and_answers')


def handler_packet(plan, task_id):
    """Preserve statements/options/scope as source text; never reduce to names."""
    validate_plan(plan)
    matches = [t for t in plan['tasks'] if t['task_id'] == task_id]
    if len(matches) != 1 or matches[0]['status'] != 'ready_for_handler':
        raise DialogueError('resolved_task_required_for_handoff')
    return seal({'schema_version': 'agentspectesting.response-handler-input/v0.19',
        'source_plan_fingerprint': plan['plan_fingerprint'], 'task': deepcopy(matches[0]),
        'source': deepcopy(plan['source']), 'operation_authorized': False,
        'requires_user_data_or_policy_resolution': True}, 'handoff_fingerprint')


def check_handler_coverage(plan, receipts):
    """Structural accounting only; ready receipts are NOT semantic attestations.

    A future handler supplies a draft_ready/blocked receipt bound to its exact
    input. All inventory items must have ready receipts before a draft can be
    considered. This function never emits a user message or grants consent.
    """
    validate_plan(plan)
    if not isinstance(receipts, list):
        raise DialogueError('handler_receipts_list_required')
    ready = set()
    seen = set()
    for receipt in receipts:
        if (not isinstance(receipt, dict) or set(receipt) != {'task_id', 'handoff_fingerprint', 'status'}
                or not isinstance(receipt['task_id'], str)
                or receipt['task_id'] in seen or receipt['status'] not in ('draft_ready', 'blocked')):
            raise DialogueError('unique_handler_receipts_required')
        packet = handler_packet(plan, receipt['task_id'])
        if receipt['handoff_fingerprint'] != packet['handoff_fingerprint']:
            raise DialogueError('receipt_from_different_handoff')
        seen.add(receipt['task_id'])
        if receipt['status'] == 'draft_ready':
            ready.add(receipt['task_id'])
    pending = [t['task_id'] for t in plan['tasks'] if t['task_id'] not in ready]
    return {'all_inventoried_requests_have_drafts': plan['status'] == 'planned' and not pending,
            'pending_task_ids': pending, 'automatic_reply_allowed': False,
            'operation_authorized': False, 'whole_message_semantic_coverage_verified': False}


def interpret_requests(messages, interpreter, budget, *, max_requests=12):
    """One inventory call, then one narrow kind question per extracted request.

    Callers own API approval and a shared CallBudget/CombinedBudget. No retries.
    Errors and budget exhaustion retain raw records and every inventory slot.
    """
    if type(max_requests) is not int or max_requests < 1:
        raise DialogueError('positive_request_limit_required')
    source = make_source(messages)
    records, classifications = [], []
    inventory = None
    error = None

    def ask(kind, prompt, quote=None):
        packet = seal({'kind': kind, 'source': deepcopy(source), 'request_quote': quote}, 'packet_fingerprint')
        request = {'packet': packet, 'prompt': prompt, 'prompt_revision': 'response_requests/v0.19'}
        record = {'request': deepcopy(request), 'status': 'attempted'}
        records.append(record)
        try:
            answer = budget.invoke('interpretation', interpreter, request)
            record.update(answer=deepcopy(answer), status='returned_not_semantically_verified')
            if make_source(messages) != source:
                record['status'] = 'source_changed_during_callback'
                raise DialogueError('source_changed_during_callback')
            return answer
        except BudgetExhausted:
            record['status'] = 'budget_exhausted_before_dispatch'
            raise
        except Exception:
            if record['status'] != 'source_changed_during_callback':
                record['status'] = 'callback_error'
            raise

    try:
        answer = ask('response_request_inventory', inventory_prompt(source))
        inventory = validate_inventory(source, answer, max_requests=max_requests)
        classifications = [None] * len(inventory['request_quotes'])
        for index, quote in enumerate(inventory['request_quotes']):
            classifications[index] = validate_kind(ask('response_request_kind', kind_prompt(source, quote), quote))
    except Exception as exc:
        error = {'type': type(exc).__name__, 'stage': records[-1]['request']['packet']['kind'] if records else 'prepare'}
        if records and records[-1]['status'] == 'returned_not_semantically_verified':
            records[-1]['status'] = 'interface_validation_failed'
    plan = make_plan(source, inventory, classifications, max_requests=max_requests) if inventory is not None else None
    return seal({'schema_version': 'agentspectesting.response-request-interpretation/v0.19',
        'source': source, 'records': records, 'plan': plan, 'error': error,
        'status': 'blocked' if error else plan['status'], 'automatic_reply_allowed': False,
        'execution_mode': 'caller_supplied_callbacks_not_external_execution_attestation',
        'automatic_retries': 0}, 'result_fingerprint')
