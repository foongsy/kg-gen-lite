import ssl

from kg_gen.kg_gen import ModelConfig, _http_client, _parse_vercel_model


def test_parse_vercel_model():
    assert _parse_vercel_model("vercel:openai/gpt-5.4-mini") == "openai/gpt-5.4-mini"
    assert (
        _parse_vercel_model("vercel_ai_gateway/google/gemini-3.1-flash-lite")
        == "google/gemini-3.1-flash-lite"
    )
    assert _parse_vercel_model("openai/gpt-4o") is None


def test_vercel_prefix_uses_vercel_provider():
    cfg = ModelConfig(model="vercel:openai/gpt-5.4-mini", api_key="test-key")
    pai_model = cfg.build()

    assert pai_model.model_name == "openai/gpt-5.4-mini"
    assert pai_model.system == "vercel"
    assert pai_model.base_url == "https://ai-gateway.vercel.sh/v1/"


def test_vercel_ai_gateway_uses_vercel_provider():
    cfg = ModelConfig(
        model="vercel_ai_gateway/google/gemini-3.1-flash-lite",
        api_key="test-key",
    )
    pai_model = cfg.build()

    assert pai_model.model_name == "google/gemini-3.1-flash-lite"
    assert pai_model.system == "vercel"
    assert pai_model.base_url == "https://ai-gateway.vercel.sh/v1/"


def test_http_client_ssl_verify():
    assert _http_client(True) is None
    client = _http_client(False)
    assert client._transport._pool._ssl_context.verify_mode == ssl.CERT_NONE


def test_ssl_verify_disabled_uses_unverified_http_client():
    cfg = ModelConfig(
        model="vercel_ai_gateway/google/gemini-3.1-flash-lite",
        api_key="test-key",
        ssl_verify=False,
    )
    pai_model = cfg.build()

    http_client = pai_model.client._client
    assert http_client._transport._pool._ssl_context.verify_mode == ssl.CERT_NONE


def test_litellm_provider_for_prefixed_models():
    cfg = ModelConfig(model="openai/gpt-4o", api_key="test-key")
    pai_model = cfg.build()

    assert pai_model.model_name == "openai/gpt-4o"
    assert pai_model.system == "litellm"
