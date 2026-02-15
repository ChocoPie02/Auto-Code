"""
Copilot SDK Client Factory
============================

Factory for creating GitHub Copilot SDK clients that mirror the interface
of Claude SDK clients. These can be used as drop-in replacements in the
agent session loop.

The CopilotSessionAdapter wraps the Copilot SDK's event-based API into the
same query()/receive_response() interface that run_agent_session() expects.
"""

import asyncio
import logging
import os
from pathlib import Path

from agents.tools_pkg import (
    get_agent_config,
    get_allowed_tools,
    get_default_thinking_level,
    get_required_mcp_servers,
    is_tools_available,
)
from core.copilot_auth import configure_copilot_authentication
from phase_config import get_thinking_budget

logger = logging.getLogger(__name__)


# ============================================================================
# Message type wrappers to match Claude SDK message types
# ============================================================================


class TextBlock:
    """Wrapper matching claude_agent_sdk TextBlock interface."""

    def __init__(self, text: str):
        self.text = text


class ToolUseBlock:
    """Wrapper matching claude_agent_sdk ToolUseBlock interface."""

    def __init__(self, name: str, input: dict | None = None, id: str = ""):
        self.name = name
        self.input = input or {}
        self.id = id


class ToolResultBlock:
    """Wrapper matching claude_agent_sdk ToolResultBlock interface."""

    def __init__(self, content: str = "", is_error: bool = False, tool_use_id: str = ""):
        self.content = content
        self.is_error = is_error
        self.tool_use_id = tool_use_id


class AssistantMessage:
    """Wrapper matching claude_agent_sdk AssistantMessage interface."""

    def __init__(self, content: list):
        self.content = content


class UserMessage:
    """Wrapper matching claude_agent_sdk UserMessage interface."""

    def __init__(self, content: list):
        self.content = content


# ============================================================================
# Copilot Session Adapter
# ============================================================================


class CopilotSessionAdapter:
    """
    Adapts the Copilot SDK client+session to match the Claude SDK client interface.

    This allows run_agent_session() to work with both providers without changes:
    - query(message) → sends message via Copilot session
    - receive_response() → async generator yielding AssistantMessage/UserMessage

    The adapter translates Copilot SDK events into the Claude SDK message types
    that the session loop already knows how to handle.
    """

    def __init__(self, client, session):
        """
        Args:
            client: CopilotClient instance (must already be started)
            session: Copilot session (from client.create_session())
        """
        self._client = client
        self._session = session
        self._messages: asyncio.Queue = asyncio.Queue()
        self._done = False
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Register event handlers that map Copilot events to Claude message types."""
        from copilot import SessionEvent as SessionEventClass
        from copilot.generated.session_events import SessionEventType


        def handle_event(event):
            event_type = event.type

            if event_type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                # Streaming text delta → AssistantMessage with TextBlock
                delta = getattr(event.data, 'delta_content', '') or ''
                if delta:
                    msg = AssistantMessage(content=[TextBlock(delta)])
                    self._messages.put_nowait(msg)

            elif event_type == SessionEventType.ASSISTANT_MESSAGE:
                # Complete assistant message
                content = getattr(event.data, 'content', '') or ''
                if content:
                    msg = AssistantMessage(content=[TextBlock(content)])
                    self._messages.put_nowait(msg)

            elif event_type == SessionEventType.TOOL_EXECUTION_START:
                # Tool invocation
                tool_name = getattr(event.data, 'tool_name', '') or ''
                tool_input = getattr(event.data, 'arguments', None) or {}
                tool_id = getattr(event.data, 'tool_call_id', '') or ''

                if tool_name:
                    block = ToolUseBlock(name=tool_name, input=tool_input, id=tool_id)
                    msg = AssistantMessage(content=[block])
                    self._messages.put_nowait(msg)

            elif event_type == SessionEventType.TOOL_EXECUTION_COMPLETE:
                # Tool result
                result_content = ''
                is_error = False
                tool_use_id = getattr(event.data, 'tool_call_id', '') or ''

                result_obj = getattr(event.data, 'result', None)
                if result_obj:
                    result_content = getattr(result_obj, 'content', '') or ''
                    is_error = getattr(result_obj, 'resultType', 'success') == 'failure'

                block = ToolResultBlock(
                    content=str(result_content),
                    is_error=bool(is_error),
                    tool_use_id=tool_use_id,
                )
                msg = UserMessage(content=[block])
                self._messages.put_nowait(msg)

            elif event_type == SessionEventType.SESSION_IDLE:
                # Session done — signal completion
                self._done = True

            elif event_type == SessionEventType.SESSION_ERROR:
                # Error event
                error_data = getattr(event.data, 'error', None)
                if error_data:
                    if hasattr(error_data, 'message'):
                        error_msg = error_data.message
                    else:
                        error_msg = str(error_data)
                else:
                    error_msg = getattr(event.data, 'message', '') or str(event.data)

                if error_msg:
                    msg = AssistantMessage(content=[TextBlock(f"\n[Copilot Error] {error_msg}\n")])
                    self._messages.put_nowait(msg)
                self._done = True

            elif event_type == SessionEventType.SESSION_SHUTDOWN:
                # Session shutdown
                self._done = True

        self._session.on(handle_event)

    async def query(self, message: str) -> None:
        """
        Send a message to the Copilot session.

        Mirrors ClaudeSDKClient.query() interface.
        """
        self._done = False
        # Clear any stale messages from previous queries
        while not self._messages.empty():
            try:
                self._messages.get_nowait()
            except asyncio.QueueEmpty:
                break

        await self._session.send({"prompt": message})

    async def receive_response(self):
        """
        Async generator yielding AssistantMessage/UserMessage objects.

        Mirrors ClaudeSDKClient.receive_response() interface.
        Messages are queued by event handlers and yielded here.
        """
        while not self._done or not self._messages.empty():
            try:
                msg = await asyncio.wait_for(self._messages.get(), timeout=0.5)
                yield msg
            except asyncio.TimeoutError:
                if self._done:
                    break
                continue

        # Drain any remaining messages
        while not self._messages.empty():
            try:
                yield self._messages.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stop(self):
        """Stop the Copilot client (cleanup)."""
        try:
            await self._client.stop()
        except Exception as e:
            logger.debug(f"Error stopping Copilot client: {e}")


# ============================================================================
# Client Factory Functions
# ============================================================================


def _get_copilot_model(model: str) -> str:
    """
    Map a model shorthand to a Copilot-compatible model ID.

    If the model is a Claude shorthand (opus, sonnet, haiku), map it to a
    reasonable Copilot equivalent. Otherwise pass through as-is.

    Args:
        model: Model shorthand or full ID

    Returns:
        Model ID usable with Copilot SDK
    """
    # Copilot model mappings for Claude shorthands
    COPILOT_MODEL_MAP = {
        "opus": "claude-sonnet-4",
        "opus-1m": "claude-sonnet-4",
        "opus-4.5": "claude-sonnet-4",
        "sonnet": "claude-sonnet-4",
        "haiku": "gpt-4.1-mini",
    }

    # Check if it's a Claude shorthand that needs mapping
    if model in COPILOT_MODEL_MAP:
        mapped = COPILOT_MODEL_MAP[model]
        logger.info(f"Mapped Claude shorthand '{model}' to Copilot model '{mapped}'")
        return mapped

    # Check if it's a full Claude model ID — map to closest Copilot equivalent
    if model.startswith("claude-"):
        if "opus" in model:
            return "claude-sonnet-4"
        elif "haiku" in model:
            return "gpt-4.1-mini"
        elif "sonnet" in model:
            return "claude-sonnet-4"

    # Pass through as-is (user specified a Copilot-native model like gpt-4.1)
    return model


def _build_mcp_servers_config(
    project_dir: Path,
    agent_type: str,
    mcp_config: dict | None = None,
) -> dict:
    """
    Build MCP server configuration in Copilot SDK format.

    Translates the Auto-Code MCP config to Copilot's mcpServers dict format.

    Args:
        project_dir: Project root directory
        agent_type: Agent type (coder, planner, etc.)
        mcp_config: Optional project MCP configuration

    Returns:
        Dict of MCP server configs for Copilot session
    """
    mcp_servers = {}

    # Context7 (code documentation lookup)
    if mcp_config and mcp_config.get("CONTEXT7_ENABLED", "true").lower() == "true":
        mcp_servers["context7"] = {
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp@latest"],
        }

    # Linear (task management)
    linear_api_key = os.environ.get("LINEAR_API_KEY")
    if linear_api_key and mcp_config and mcp_config.get("LINEAR_MCP_ENABLED", "true").lower() == "true":
        mcp_servers["linear"] = {
            "type": "http",
            "url": "https://mcp.linear.app/mcp",
            "headers": {"Authorization": f"Bearer {linear_api_key}"},
        }

    return mcp_servers


async def create_copilot_client(
    project_dir: Path,
    spec_dir: Path,
    model: str,
    agent_type: str = "coder",
    max_thinking_tokens: int | None = None,
    output_format: dict | None = None,
    agents: dict | None = None,
    betas: list[str] | None = None,
    effort_level: str | None = None,
    fast_mode: bool = False,
) -> CopilotSessionAdapter:
    """
    Create a Copilot SDK client configured similarly to create_client() in client.py.

    This function mirrors the interface of the Claude create_client() but uses
    the Copilot SDK internally. Returns a CopilotSessionAdapter that provides
    the same query()/receive_response() interface.

    Args:
        project_dir: Root directory for the project
        spec_dir: Directory containing the spec
        model: Model shorthand or ID
        agent_type: Agent type from AGENT_CONFIGS
        max_thinking_tokens: Token budget for thinking (mapped to Copilot equivalent)
        output_format: Structured output format (if supported by Copilot)
        agents: Subagent definitions (mapped to Copilot custom agents)
        betas: Beta headers (Claude-specific, ignored for Copilot)
        effort_level: Effort level (mapped to Copilot equivalent)
        fast_mode: Fast mode (Claude-specific, ignored for Copilot)

    Returns:
        CopilotSessionAdapter with query()/receive_response() interface
    """
    try:
        from copilot import CopilotClient
    except ImportError:
        raise ImportError(
            "github-copilot-sdk is not installed. "
            "Install it with: pip install github-copilot-sdk"
        )

    # Configure authentication
    configure_copilot_authentication()

    # Resolve model
    copilot_model = _get_copilot_model(model)
    logger.info(f"Creating Copilot client with model: {copilot_model}")

    # Get agent configuration for tools
    config = get_agent_config(agent_type)

    # Build client options
    client_options = {}

    # Pass GitHub token if available
    github_token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if github_token:
        client_options["github_token"] = github_token

    # Create and start client
    client = CopilotClient(client_options if client_options else None)
    await client.start()

    # Build session configuration
    session_config = {
        "model": copilot_model,
        "streaming": True,
    }

    # Set working directory
    if project_dir:
        session_config["cwd"] = str(project_dir.resolve())

    # System prompt from agent config
    system_prompt = config.get("system_prompt")
    if system_prompt:
        session_config["system_message"] = {"content": system_prompt}

    # MCP servers
    from core.client import load_project_mcp_config
    mcp_config = load_project_mcp_config(project_dir)
    mcp_servers = _build_mcp_servers_config(project_dir, agent_type, mcp_config)
    if mcp_servers:
        session_config["mcpServers"] = mcp_servers

    # Custom agents (translate to Copilot format)
    if agents:
        custom_agents = []
        for name, agent_def in agents.items():
            custom_agents.append({
                "name": name,
                "displayName": name,
                "description": agent_def.get("description", ""),
                "prompt": agent_def.get("prompt", ""),
            })
        session_config["customAgents"] = custom_agents

    # Create session
    session = await client.create_session(session_config)

    return CopilotSessionAdapter(client, session)


async def create_copilot_simple_client(
    agent_type: str = "merge_resolver",
    model: str = "gpt-4.1-mini",
    system_prompt: str | None = None,
    cwd: Path | None = None,
    max_turns: int = 1,
    max_thinking_tokens: int | None = None,
    betas: list[str] | None = None,
    effort_level: str | None = None,
    fast_mode: bool = False,
) -> CopilotSessionAdapter:
    """
    Create a minimal Copilot SDK client for single-turn utility operations.

    Mirrors create_simple_client() from core/simple_client.py but uses
    the Copilot SDK.

    Args:
        agent_type: Agent type from AGENT_CONFIGS
        model: Copilot model to use (defaults to gpt-4.1-mini for fast/cheap)
        system_prompt: Optional custom system prompt
        cwd: Working directory for file operations
        max_turns: Maximum conversation turns
        max_thinking_tokens: Override thinking budget
        betas: Beta headers (ignored for Copilot)
        effort_level: Effort level (ignored for simple Copilot clients)
        fast_mode: Fast mode (ignored for Copilot)

    Returns:
        CopilotSessionAdapter for single-turn operations
    """
    try:
        from copilot import CopilotClient
    except ImportError:
        raise ImportError(
            "github-copilot-sdk is not installed. "
            "Install it with: pip install github-copilot-sdk"
        )

    # Configure authentication
    configure_copilot_authentication()

    # Resolve model
    copilot_model = _get_copilot_model(model)

    # Build client options
    client_options = {}
    github_token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if github_token:
        client_options["github_token"] = github_token

    # Create and start client
    client = CopilotClient(client_options if client_options else None)
    await client.start()

    # Build session config
    session_config = {
        "model": copilot_model,
        "streaming": True,
    }

    if cwd:
        session_config["cwd"] = str(cwd.resolve())

    if system_prompt:
        session_config["system_message"] = {"content": system_prompt}

    # Create session
    session = await client.create_session(session_config)

    return CopilotSessionAdapter(client, session)
