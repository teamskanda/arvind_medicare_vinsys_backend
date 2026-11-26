"""Tests for URL validation — the security boundary for customer-supplied URLs.

DNS is stubbed so these run offline and deterministically.
"""

from __future__ import annotations

import pytest

from helpers import url_validation
from helpers.errors import AppError

# Captured before the autouse fixture below replaces the module attribute, so
# the resolver's own error handling can be tested directly.
REAL_RESOLVE_HOST = url_validation._resolve_host


@pytest.fixture(autouse=True)
def stub_dns(monkeypatch: pytest.MonkeyPatch):
    """Resolve every hostname to a public address unless a test says otherwise."""

    def fake_resolve(host: str, port: int) -> list[str]:
        return ["93.184.216.34"]

    monkeypatch.setattr(url_validation, "_resolve_host", fake_resolve)
    return fake_resolve


# ---------------------------------------------------------------------------
# normalise_url
# ---------------------------------------------------------------------------


def test_adds_https_scheme_when_missing():
    assert url_validation.normalise_url("acme.com") == "https://acme.com/"


def test_preserves_path_and_query_but_drops_fragment():
    result = url_validation.normalise_url("https://acme.com/about?x=1#team")
    assert result == "https://acme.com/about?x=1"


@pytest.mark.parametrize("value", ["", "   ", "https://acme .com", "https://acme\ncom"])
def test_rejects_malformed_input(value: str):
    with pytest.raises(AppError) as exc:
        url_validation.normalise_url(value)
    assert exc.value.status_code == 422


def test_rejects_overlong_url():
    with pytest.raises(AppError):
        url_validation.normalise_url("https://acme.com/" + "a" * 2100)


# ---------------------------------------------------------------------------
# validate_url — scheme, port, hostname
# ---------------------------------------------------------------------------


async def test_accepts_a_normal_public_site():
    validated = await url_validation.validate_url("acme.com")
    assert validated.host == "acme.com"
    assert validated.scheme == "https"
    assert validated.port == 443
    assert validated.resolved_ips == ["93.184.216.34"]


async def test_lowercases_host_and_strips_trailing_dot():
    validated = await url_validation.validate_url("https://ACME.com./")
    assert validated.host == "acme.com"


@pytest.mark.parametrize(
    "url",
    ["ftp://acme.com", "file:///etc/passwd", "gopher://acme.com", "javascript:alert(1)"],
)
async def test_rejects_non_http_schemes(url: str):
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url(url)
    assert exc.value.code in {"invalid_url", "unsafe_url"}


@pytest.mark.parametrize("url", ["http://acme.com:8080", "https://acme.com:22", "https://acme.com:6379"])
async def test_rejects_non_standard_ports(url: str):
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url(url)
    assert exc.value.code == "unsafe_url"


async def test_rejects_embedded_credentials():
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url("https://user:pass@acme.com")
    assert exc.value.code == "unsafe_url"


@pytest.mark.parametrize(
    "host",
    ["localhost", "foo.local", "db.internal", "service.intranet", "box.home.arpa", "a.lan"],
)
async def test_rejects_internal_hostnames(host: str):
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url(f"http://{host}/")
    assert exc.value.code == "unsafe_url"


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "8.8.8.8", "[::1]"]
)
async def test_rejects_ip_literals(host: str):
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url(f"http://{host}/")
    assert exc.value.code == "unsafe_url"


async def test_rejects_single_label_hostname():
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url("http://intranet/")
    assert exc.value.code == "invalid_url"


# ---------------------------------------------------------------------------
# validate_url — DNS answers (the SSRF core)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",        # loopback
        "10.1.2.3",         # RFC1918
        "172.16.5.4",       # RFC1918
        "192.168.0.10",     # RFC1918
        "169.254.169.254",  # cloud metadata endpoint
        "0.0.0.0",          # unspecified
        "100.64.0.1",       # carrier-grade NAT
        "::1",              # IPv6 loopback
        "fd00::1",          # IPv6 unique-local
        "fe80::1",          # IPv6 link-local
    ],
)
async def test_rejects_hosts_resolving_to_internal_addresses(
    monkeypatch: pytest.MonkeyPatch, ip: str
):
    monkeypatch.setattr(url_validation, "_resolve_host", lambda host, port: [ip])
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url("https://evil.example.org/")
    assert exc.value.code == "unsafe_url"


async def test_rejects_when_any_dns_answer_is_internal(monkeypatch: pytest.MonkeyPatch):
    """A host mixing a public and a loopback record must be rejected outright."""
    monkeypatch.setattr(
        url_validation, "_resolve_host", lambda host, port: ["93.184.216.34", "127.0.0.1"]
    )
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url("https://rebind.example.org/")
    assert exc.value.code == "unsafe_url"


async def test_rejects_ipv4_mapped_ipv6_loopback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(url_validation, "_resolve_host", lambda host, port: ["::ffff:127.0.0.1"])
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url("https://evil.example.org/")
    assert exc.value.code == "unsafe_url"


def test_resolver_translates_dns_failure_into_unreachable(monkeypatch: pytest.MonkeyPatch):
    """A non-existent domain must surface as `site_unreachable`, not a 500."""
    import socket

    def boom(*args, **kwargs):
        raise socket.gaierror("nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(AppError) as exc:
        REAL_RESOLVE_HOST("does-not-exist-999.org", 443)
    assert exc.value.code == "site_unreachable"


def test_resolver_rejects_empty_dns_answer(monkeypatch: pytest.MonkeyPatch):
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(AppError) as exc:
        REAL_RESOLVE_HOST("acme.com", 443)
    assert exc.value.code == "site_unreachable"


def test_resolver_deduplicates_addresses(monkeypatch: pytest.MonkeyPatch):
    import socket

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (0, 0, 0, "", ("93.184.216.34", 443)),
            (0, 0, 0, "", ("93.184.216.34", 443)),
            (0, 0, 0, "", ("93.184.216.35", 443)),
        ],
    )
    assert REAL_RESOLVE_HOST("acme.com", 443) == [
        "93.184.216.34",
        "93.184.216.35",
    ]


@pytest.mark.parametrize("tld", ["example", "test", "invalid", "localhost"])
async def test_rejects_reserved_tlds(tld: str):
    """RFC 2606 reserved names are never a real customer website."""
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url(f"https://acme.{tld}/")
    assert exc.value.code == "unsafe_url"


async def test_host_with_port_is_not_mistaken_for_a_scheme():
    """`acme.com:8080` must be read as host:port, then rejected on the port."""
    with pytest.raises(AppError) as exc:
        await url_validation.validate_url("acme.com:8080/pricing")
    assert exc.value.code == "unsafe_url"


async def test_scheme_without_slashes_is_rejected_cleanly():
    """Regression: `javascript:alert(1)` used to raise an unhandled ValueError."""
    for value in ["javascript:alert(1)", "mailto:a@b.com", "data:text/html,<h1>x"]:
        with pytest.raises(AppError) as exc:
            await url_validation.validate_url(value)
        assert exc.value.code == "invalid_url"


# ---------------------------------------------------------------------------
# Pinned-connection target construction
# ---------------------------------------------------------------------------


def test_pinned_target_uses_ip_and_keeps_path():
    validated = url_validation.ValidatedUrl(
        url="https://acme.com/about?x=1",
        scheme="https",
        host="acme.com",
        port=443,
        path="/about",
        resolved_ips=["93.184.216.34"],
    )
    target = url_validation._pinned_target(validated, "93.184.216.34")
    assert target == "https://93.184.216.34:443/about?x=1"


def test_pinned_target_brackets_ipv6():
    validated = url_validation.ValidatedUrl(
        url="https://acme.com/",
        scheme="https",
        host="acme.com",
        port=443,
        path="/",
        resolved_ips=["2606:2800:220:1:248:1893:25c8:1946"],
    )
    target = url_validation._pinned_target(validated, "2606:2800:220:1:248:1893:25c8:1946")
    assert target.startswith("https://[2606:2800:220:1:248:1893:25c8:1946]:443/")


# ---------------------------------------------------------------------------
# Branding extraction
# ---------------------------------------------------------------------------


def test_extracts_title_theme_colour_and_favicon():
    html = """
    <html><head>
      <title>  Acme   Industries | Home </title>
      <meta name="theme-color" content="#7B1E3B">
      <link rel="icon" href="/favicon.ico">
    </head><body>hi</body></html>
    """
    validated = url_validation.ValidatedUrl(
        url="https://acme.com/", scheme="https", host="acme.com", port=443, path="/"
    )
    metadata = url_validation.SiteMetadata()
    url_validation._extract_branding(html, validated, metadata)

    assert metadata.title == "Acme Industries | Home"
    assert metadata.brand_color == "#7b1e3b"
    assert metadata.favicon_url == "https://acme.com/favicon.ico"


def test_ignores_non_hex_theme_colour():
    html = '<head><title>X site</title><meta name="theme-color" content="rebeccapurple"></head>'
    validated = url_validation.ValidatedUrl(
        url="https://acme.com/", scheme="https", host="acme.com", port=443, path="/"
    )
    metadata = url_validation.SiteMetadata()
    url_validation._extract_branding(html, validated, metadata)
    assert metadata.brand_color is None


def test_falls_back_to_og_site_name():
    html = '<head><meta property="og:site_name" content="Acme Co"></head>'
    validated = url_validation.ValidatedUrl(
        url="https://acme.com/", scheme="https", host="acme.com", port=443, path="/"
    )
    metadata = url_validation.SiteMetadata()
    url_validation._extract_branding(html, validated, metadata)
    assert metadata.title == "Acme Co"
