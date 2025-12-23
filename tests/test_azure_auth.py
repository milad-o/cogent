"""Tests for Azure auth resolution helpers.

These tests focus on the internal auth-selection logic and avoid making network
calls or requiring Azure/OpenAI services.
"""

from __future__ import annotations

import pytest

from agenticflow.models.azure import AzureEntraAuth, _resolve_azure_openai_auth


def test_resolve_azure_openai_auth_api_key_explicit() -> None:
    kwargs = _resolve_azure_openai_auth(
        api_key="k",
        azure_ad_token_provider=None,
        entra=None,
        env_api_key_name="AZURE_OPENAI_API_KEY",
    )
    assert kwargs == {"api_key": "k"}


def test_resolve_azure_openai_auth_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-k")
    kwargs = _resolve_azure_openai_auth(
        api_key=None,
        azure_ad_token_provider=None,
        entra=None,
        env_api_key_name="AZURE_OPENAI_API_KEY",
    )
    assert kwargs == {"api_key": "env-k"}


def test_resolve_azure_openai_auth_token_provider_wins() -> None:
    def provider() -> str:
        return "token"

    kwargs = _resolve_azure_openai_auth(
        api_key=None,
        azure_ad_token_provider=provider,
        entra=None,
        env_api_key_name="AZURE_OPENAI_API_KEY",
    )
    assert kwargs["azure_ad_token_provider"] is provider


def test_resolve_azure_openai_auth_rejects_conflict_api_key_and_provider() -> None:
    def provider() -> str:
        return "token"

    with pytest.raises(ValueError, match="either api_key or Entra auth"):
        _resolve_azure_openai_auth(
            api_key="k",
            azure_ad_token_provider=provider,
            entra=None,
            env_api_key_name="AZURE_OPENAI_API_KEY",
        )


def test_resolve_azure_openai_auth_rejects_provider_and_entra_conflict() -> None:
    def provider() -> str:
        return "token"

    with pytest.raises(ValueError, match="Provide only one"):
        _resolve_azure_openai_auth(
            api_key=None,
            azure_ad_token_provider=provider,
            entra=AzureEntraAuth(method="default"),
            env_api_key_name="AZURE_OPENAI_API_KEY",
        )


def test_resolve_azure_openai_auth_entra_client_secret_optional_dependency() -> None:
    # This path requires azure-identity. If not installed, we skip.
    try:
        import azure.identity  # noqa: F401
    except Exception:
        pytest.skip("azure-identity not installed")

    with pytest.raises(ValueError, match="tenant_id, client_id, and client_secret"):
        _resolve_azure_openai_auth(
            api_key=None,
            azure_ad_token_provider=None,
            entra=AzureEntraAuth(method="client_secret"),
            env_api_key_name="AZURE_OPENAI_API_KEY",
        )


def test_resolve_azure_openai_auth_missing_everything_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        _resolve_azure_openai_auth(
            api_key=None,
            azure_ad_token_provider=None,
            entra=None,
            env_api_key_name="AZURE_OPENAI_API_KEY",
        )
