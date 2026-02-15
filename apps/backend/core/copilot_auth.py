"""
Copilot Authentication Module
==============================

Handles GitHub Copilot CLI authentication setup and verification.
Copilot SDK authenticates via:
1. GitHub signed-in user (stored CLI credentials from `copilot` login)
2. Environment variables (COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN)
3. BYOK (Bring Your Own Key) — uses third-party API keys directly

This module runs alongside the existing Claude auth system (core/auth.py).
"""

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def check_copilot_cli_installed() -> bool:
    """
    Check if the GitHub Copilot CLI is installed and accessible.

    Returns:
        True if `copilot` command is found in PATH
    """
    copilot_path = shutil.which("copilot")
    if copilot_path:
        logger.info(f"Copilot CLI found at: {copilot_path}")
        return True

    logger.warning(
        "Copilot CLI not found in PATH. "
        "Install it from: https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli"
    )
    return False


def get_copilot_cli_version() -> str | None:
    """
    Get the installed Copilot CLI version.

    Returns:
        Version string, or None if CLI is not installed
    """
    try:
        result = subprocess.run(
            ["copilot", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug(f"Failed to get Copilot CLI version: {e}")
    return None


def get_copilot_auth_token() -> str | None:
    """
    Get the GitHub token for Copilot authentication.

    Checks environment variables in priority order:
    1. COPILOT_GITHUB_TOKEN (recommended for explicit Copilot usage)
    2. GH_TOKEN (GitHub CLI compatible)
    3. GITHUB_TOKEN (GitHub Actions compatible)

    Returns:
        Token string, or None if no token found
    """
    for env_var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            logger.info(f"Using GitHub token from {env_var}")
            return token

    logger.debug("No GitHub token found in environment variables")
    return None


def configure_copilot_authentication() -> bool:
    """
    Configure authentication for the Copilot SDK.

    The Copilot SDK handles authentication automatically:
    - If a GitHub token env var is set, it uses that
    - Otherwise it uses stored credentials from `copilot` CLI login
    - BYOK mode bypasses GitHub auth entirely

    This function primarily validates that auth will work.

    Returns:
        True if authentication is likely to succeed
    """
    # Check if explicit token is available
    token = get_copilot_auth_token()
    if token:
        logger.info("Copilot authentication: using environment variable token")
        return True

    # Check if CLI is installed (for stored credential auth)
    if check_copilot_cli_installed():
        logger.info(
            "Copilot authentication: will use stored CLI credentials"
        )
        return True

    logger.error(
        "Copilot authentication failed: no token and no CLI found. "
        "Either set COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN or "
        "install and authenticate the Copilot CLI."
    )
    return False


def get_copilot_auth_status() -> dict:
    """
    Get the current Copilot authentication status.

    Returns:
        Dict with auth info:
        - authenticated: bool
        - method: str (env_var, cli, none)
        - cli_installed: bool
        - cli_version: str | None
        - token_source: str | None (which env var)
    """
    cli_installed = check_copilot_cli_installed()
    cli_version = get_copilot_cli_version() if cli_installed else None

    # Check for env var token
    token_source = None
    for env_var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(env_var):
            token_source = env_var
            break

    if token_source:
        return {
            "authenticated": True,
            "method": "env_var",
            "cli_installed": cli_installed,
            "cli_version": cli_version,
            "token_source": token_source,
        }
    elif cli_installed:
        return {
            "authenticated": True,
            "method": "cli",
            "cli_installed": True,
            "cli_version": cli_version,
            "token_source": None,
        }
    else:
        return {
            "authenticated": False,
            "method": "none",
            "cli_installed": False,
            "cli_version": None,
            "token_source": None,
        }
