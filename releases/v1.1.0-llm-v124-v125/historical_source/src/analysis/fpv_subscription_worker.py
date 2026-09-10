"""One SDK turn per subprocess; stdin JSON in, receipt JSON out. No data access.

The controller enforces a process-group timeout. This worker only receives the
prompt/schema, never inventory, cost results, validation labels or credentials.
SDK login remains in its original local store. No key extraction or API fallback.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
import json
import os
import shutil
import sys
import tempfile


def clean_environment():
    for key in ['OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_API_KEY',
                'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_OAUTH_TOKEN',
                'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY']:
        os.environ.pop(key, None)


def run_codex(job, directory):
    from openai_codex import Codex, CodexConfig, ApprovalMode, Sandbox
    config = ('forced_login_method="chatgpt"', 'web_search="disabled"',
              'mcp_servers.codegraph.enabled=false', 'mcp_servers.openaiDeveloperDocs.enabled=false',
              'features.plugins=false', 'features.apps=false', 'features.hooks=false')
    with Codex(CodexConfig(codex_bin=shutil.which('codex'), cwd=directory,
                          env=dict(os.environ), config_overrides=config)) as client:
        thread = client.thread_start(model=job['model'], cwd=directory, ephemeral=True,
            sandbox=Sandbox.read_only, approval_mode=ApprovalMode.deny_all,
            base_instructions=job['policy'], developer_instructions='Do not call tools. Return the requested JSON only.')
        result = thread.run(job['prompt'], output_schema=job['schema'], effort=job['effort'],
            sandbox=Sandbox.read_only, approval_mode=ApprovalMode.deny_all)
    events = [item.model_dump(mode='json') for item in result.items]
    active = [item.get('type') for item in events if item.get('type') not in {'userMessage','agentMessage','reasoning'}]
    return dict(provider='codex_subscription', sdk_version=version('openai-codex'),
        model=job['model'], requested_model=job['model'], resolved_model=None,
        model_id_source='requested_configuration_not_independently_observed',
        effort=job['effort'], final_response=result.final_response,
        raw_events=[e if e.get('type') in {'userMessage','agentMessage','reasoning'} else {'type':e.get('type')} for e in events],
        active_tool_items=active, turns=1, turn_count_source='one_sdk_run_requested',
        usage=result.usage.model_dump(mode='json') if result.usage else None,
        tools_exposed_empty_verified=False, settings=dict(sandbox='read-only', approval='deny-all', overrides=config))


async def run_claude(job, directory):
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, AssistantMessage, SystemMessage
    from claude_agent_sdk import PermissionResultDeny, ToolUseBlock
    async def deny_tool(name, data, context):
        return PermissionResultDeny(message='Tools are disabled for this direct-judgment experiment.')
    options = ClaudeAgentOptions(cli_path=shutil.which('claude'), cwd=directory,
        model=job.get('model'), system_prompt=job['policy'], tools=[], allowed_tools=[],
        mcp_servers={}, strict_mcp_config=True, setting_sources=[], skills=[],
        permission_mode='dontAsk', can_use_tool=deny_tool,
        max_turns=1, effort=job['effort'])
    # JSON is required in the prompt and validated by the controller. We do not
    # use SDK output_format because it may invoke a structured-output tool.
    events, active, response, usage, actual_model, turns = [], [], None, None, None, None
    async for message in query(prompt=job['prompt'], options=options):
        if isinstance(message, SystemMessage):
            if message.subtype == 'init':
                actual_model = message.data.get('model')
            # Whitelist runtime fields; never persist account or config content.
            events.append(dict(type='system', subtype=message.subtype,
                               model=message.data.get('model'), tools=message.data.get('tools')))
        elif isinstance(message, AssistantMessage):
            blocks = []
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    active.append(block.name)
                    blocks.append(dict(type='tool_use', name=block.name))
                elif hasattr(block, 'text'):
                    blocks.append(dict(type='text', text=block.text))
                else:
                    blocks.append(dict(type=type(block).__name__))
            events.append(dict(type='assistant', content=blocks))
        elif isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError('Claude returned error result')
            response, usage, turns = message.result, message.usage, message.num_turns
            events.append(dict(type='result', subtype=message.subtype,
                               num_turns=message.num_turns, is_error=message.is_error))
    return dict(provider='claude_subscription', sdk_version=version('claude-agent-sdk'),
        model=actual_model, requested_model=job.get('model'), resolved_model=actual_model,
        model_id_source='sdk_init_event', effort=job['effort'], final_response=response, raw_events=events,
        active_tool_items=active, turns=turns, turn_count_source='sdk_result', usage=usage, tools_exposed_empty_verified=False,
        settings=dict(tools=[], setting_sources=[], mcp_servers={}, permission='dontAsk'))


def main():
    clean_environment()
    job = json.load(sys.stdin)
    try:
        with tempfile.TemporaryDirectory(prefix='fpv-direct-sdk-') as directory:
            if job['provider'] == 'codex':
                result = run_codex(job, directory)
            elif job['provider'] == 'claude':
                result = asyncio.run(run_claude(job, directory))
            else:
                raise ValueError('Unknown provider')
        if result['active_tool_items']:
            result.update(status='failed', error_kind='active_tools_observed')
        elif not result['final_response']:
            result.update(status='failed', error_kind='empty_response')
        else:
            result['status'] = 'ok'
    except Exception as error:
        result = dict(status='failed', error_kind=type(error).__name__)
    print(json.dumps(result, ensure_ascii=False))
    if result['status'] != 'ok':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
