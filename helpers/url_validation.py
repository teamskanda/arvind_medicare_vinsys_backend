"""Validation of customer-submitted website URLs.

The client may hand us any URL during a live demo, so this module treats the
input as hostile. Controls, in order:

 1. Scheme allow-list (http/https only) and default-port allow-list, so the
    URL cannot be used to probe internal services on odd ports.
 2. Hostname sanity: no IP literals, no internal suffixes (`localhost`,
    `.local`, `.internal`, `.home.arpa`).
 3. DNS resolution of *every* A/AAAA record, each of which must be globally
    routable. One private answer rejects the whole host, so a DNS entry
    mixing a public and a 127.0.0.1 record cannot slip through.
 4. The reachability/branding probe connects to a *pinned* resolved IP while
    carrying the original SNI and Host header. The address cannot change
    between the check and the connection, which is what closes the
    DNS-rebinding window.
 5. Redirects are followed manually, re-validating the host at every hop, so
    a public URL cannot bounce us to an internal one.

Note that the crawl itself runs on Firecrawl's infrastructure, not ours — so
our network is never the thing fetching customer URLs at scale. These
controls protect the one request *we* make (the probe) and give the customer
a fast, clear rejection before any Firecrawl credits are spent.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import anyio.to_thread
import httpx

from helpers import errors

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = {80, 443}
BLOCKED_HOST_SUFFIXES = (
    ".local",
    ".localhost",
    ".internal",
    ".intranet",
    ".home.arpa",
    ".lan",
    ".corp",
    ".test",
    ".invalid",
    ".example",
)
BLOCKED_HOSTNAMES = {"localhost", "metadata", "metadata.google.internal"}

USER_AGENT = "SkandaAI-Validator/1.0 (+website knowledge base setup)"
PROBE_TIMEOUT_SECONDS = 10.0
MAX_REDIRECTS = 3
MAX_PROBE_BYTES = 512 * 1024

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_THEME_COLOR_RE = re.compile(
    r"""<meta[^>]+name=["']theme-color["'][^>]+content=["']([^"']+)["']""",
    re.IGNORECASE,
)
_THEME_COLOR_RE_ALT = re.compile(
    r"""<meta[^>]+content=["']([^"']+)["'][^>]+name=["']theme-color["']""",
    re.IGNORECASE,
)
_ICON_RE = re.compile(
    r"""<link[^>]+rel=["'][^"']*icon[^"']*["'][^>]*>""", re.IGNORECASE
)
_HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)
_OG_SITE_NAME_RE = re.compile(
    r"""<meta[^>]+property=["']og:site_name["'][^>]+content=["']([^"']+)["']""",
    re.IGNORECASE,
)
_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@dataclass(slots=True)
class ValidatedUrl:
    url: str
    scheme: str
    host: str
    port: int
    path: str
    resolved_ips: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SiteMetadata:
    """Best-effort extras. Every field may be None; nothing here blocks a crawl."""

    reachable: bool = False
    status_code: int | None = None
    final_url: str | None = None
    title: str | None = None
    brand_color: str | None = None
    favicon_url: str | None = None
    warning: str | None = None


# ---------------------------------------------------------------------------
# Step 1-3: syntax, hostname and DNS validation
# ---------------------------------------------------------------------------


def normalise_url(raw: str) -> str:
    """Make a human-typed URL parseable. `acme.com` -> `https://acme.com/`."""
    candidate = (raw or "").strip()
    if not candidate:
        raise errors.invalid_url("Please enter a website address.")
    if len(candidate) > 2048:
        raise errors.invalid_url("That website address is too long.")
    if "\n" in candidate or "\r" in candidate or " " in candidate:
        raise errors.invalid_url("That website address contains invalid characters.")

    # Resolve the scheme before anything else, so a non-web scheme such as
    # `javascript:` or `file:` is rejected with a clear message instead of
    # being mangled into a bogus https URL.
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):(//)?(.*)$", candidate, re.DOTALL)
    if scheme_match:
        scheme, slashes, remainder = (
            scheme_match.group(1).lower(),
            scheme_match.group(2),
            scheme_match.group(3),
        )
        if slashes:
            if scheme not in ALLOWED_SCHEMES:
                raise errors.invalid_url(
                    "Only http:// and https:// website addresses are supported."
                )
        elif remainder.split("/", 1)[0].isdigit():
            # No "//" but a numeric tail: this is `host:port`, not a scheme.
            candidate = "https://" + candidate
        else:
            raise errors.invalid_url(
                "Only http:// and https:// website addresses are supported."
            )
    else:
        candidate = "https://" + candidate

    try:
        parsed = urlparse(candidate)
        if not parsed.netloc or parsed.hostname is None:
            raise errors.invalid_url("That does not look like a website address.")
        parsed.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        raise errors.invalid_url("That does not look like a website address.")

    # Drop fragments and any credentials; keep path and query.
    return urlunparse(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", "", parsed.query, "")
    )


def _assert_public_ip(raw_ip: str, host: str) -> None:
    try:
        ip = ipaddress.ip_address(raw_ip)
    except ValueError:
        raise errors.unsafe_url(f"Could not interpret the network address for {host}.")

    # `is_global` already covers loopback, private ranges, link-local
    # (including the 169.254.169.254 cloud metadata endpoint), multicast,
    # reserved and unspecified addresses.
    if not ip.is_global:
        raise errors.unsafe_url(
            "That website resolves to a private or internal network address, "
            "so it cannot be crawled."
        )
    if isinstance(ip, ipaddress.IPv6Address) and (ip.ipv4_mapped or ip.sixtofour):
        mapped = ip.ipv4_mapped or ip.sixtofour
        if mapped is not None and not mapped.is_global:
            raise errors.unsafe_url(
                "That website resolves to a private or internal network address, "
                "so it cannot be crawled."
            )


def _resolve_host(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise errors.site_unreachable(
            "We could not find that website. Please check the address and try again."
        )
    except UnicodeError:
        raise errors.invalid_url("That website address is not valid.")

    ips: list[str] = []
    for info in infos:
        address = info[4][0]
        if address not in ips:
            ips.append(address)
    if not ips:
        raise errors.site_unreachable(
            "We could not find that website. Please check the address and try again."
        )
    return ips


async def validate_url(raw: str) -> ValidatedUrl:
    """Full validation. Raises `AppError` with a customer-readable message."""
    normalised = normalise_url(raw)
    parsed = urlparse(normalised)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise errors.invalid_url("Only http:// and https:// website addresses are supported.")

    if parsed.username or parsed.password:
        raise errors.unsafe_url("Website addresses containing login credentials are not accepted.")

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise errors.invalid_url("That does not look like a website address.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise errors.unsafe_url(
            "Only standard web ports (80 and 443) are supported for crawling."
        )

    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise errors.unsafe_url("That address points to an internal host and cannot be crawled.")

    # Reject bare IP literals: a knowledge base is built from a real website,
    # and allowing IPs widens the SSRF surface for no product benefit.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise errors.unsafe_url("Please enter a website domain rather than an IP address.")

    if "." not in host:
        raise errors.invalid_url("Please enter a full website address, for example https://acme.com")

    resolved = await anyio.to_thread.run_sync(_resolve_host, host, port)
    for ip in resolved:
        _assert_public_ip(ip, host)

    return ValidatedUrl(
        url=normalised,
        scheme=parsed.scheme,
        host=host,
        port=port,
        path=parsed.path or "/",
        resolved_ips=resolved,
    )


# ---------------------------------------------------------------------------
# Step 4-5: pinned-IP probe with manual redirect validation
# ---------------------------------------------------------------------------


def _pinned_target(validated: ValidatedUrl, ip: str) -> str:
    literal = f"[{ip}]" if ":" in ip else ip
    netloc = f"{literal}:{validated.port}"
    query = urlparse(validated.url).query
    return urlunparse((validated.scheme, netloc, validated.path, "", query, ""))


async def _pinned_get(
    client: httpx.AsyncClient, validated: ValidatedUrl
) -> httpx.Response:
    """GET the URL over a connection pinned to an already-validated IP."""
    target = _pinned_target(validated, validated.resolved_ips[0])
    return await client.get(
        target,
        headers={
            "Host": validated.host
            if validated.port in (80, 443)
            else f"{validated.host}:{validated.port}",
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en",
        },
        extensions={"sni_hostname": validated.host},
    )


async def probe_site(validated: ValidatedUrl) -> SiteMetadata:
    """Check reachability and pull optional branding.

    Never raises: a failure here is reported as a warning and the crawl is
    still allowed to proceed, because Firecrawl reaches sites our probe
    cannot (JS-heavy pages, bot-protected origins). Firecrawl becomes the
    authority on reachability in that case.
    """
    metadata = SiteMetadata()
    current = validated

    try:
        async with httpx.AsyncClient(
            timeout=PROBE_TIMEOUT_SECONDS,
            follow_redirects=False,
            verify=True,
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                response = await _pinned_get(client, current)
                metadata.status_code = response.status_code
                metadata.final_url = current.url

                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        metadata.warning = "The website returned an incomplete redirect."
                        return metadata
                    # Re-validate the hop: a public URL must not redirect us
                    # onto an internal address.
                    current = await validate_url(urljoin(current.url, location))
                    continue

                if response.status_code >= 400:
                    metadata.warning = (
                        f"The website responded with HTTP {response.status_code}. "
                        "We will still try to crawl it."
                    )
                    return metadata

                metadata.reachable = True
                body = response.content[:MAX_PROBE_BYTES].decode(
                    response.encoding or "utf-8", errors="replace"
                )
                _extract_branding(body, current, metadata)
                return metadata

            metadata.warning = "The website redirected too many times."
            return metadata

    except errors.AppError as exc:
        # A redirect landed somewhere unsafe — that *is* worth surfacing.
        metadata.warning = exc.message
        return metadata
    except (httpx.HTTPError, OSError) as exc:
        # ssl.SSLError and socket errors are OSError subclasses.
        logger.info("Probe failed for %s: %s", validated.host, exc)
        metadata.warning = (
            "We could not load the site directly, but we will still attempt the crawl."
        )
        return metadata
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected probe failure for %s: %s", validated.host, exc)
        metadata.warning = (
            "We could not load the site directly, but we will still attempt the crawl."
        )
        return metadata


def _extract_branding(
    html: str, validated: ValidatedUrl, metadata: SiteMetadata
) -> None:
    """Pull title / theme colour / favicon. Wrapped by the caller's try."""
    title_match = _TITLE_RE.search(html)
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()
        metadata.title = title[:200] or None
    if not metadata.title:
        og_match = _OG_SITE_NAME_RE.search(html)
        if og_match:
            metadata.title = og_match.group(1).strip()[:200] or None

    color_match = _THEME_COLOR_RE.search(html) or _THEME_COLOR_RE_ALT.search(html)
    if color_match:
        color = color_match.group(1).strip()
        if _HEX_COLOR_RE.match(color):
            metadata.brand_color = color.lower()

    icon_tag = _ICON_RE.search(html)
    if icon_tag:
        href_match = _HREF_RE.search(icon_tag.group(0))
        if href_match:
            try:
                candidate = urljoin(validated.url, href_match.group(1).strip())
                if urlparse(candidate).scheme in ALLOWED_SCHEMES:
                    metadata.favicon_url = candidate[:2048]
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


async def check_robots_allowed(validated: ValidatedUrl) -> tuple[bool, str | None]:
    """Check robots.txt for the start URL.

    Returns `(allowed, note)`. We evaluate the `*` group because that is what
    a general-purpose crawler is bound by. A missing or unreadable
    robots.txt means "allowed", matching the standard.

    Firecrawl independently respects robots.txt during the crawl; this check
    exists to fail fast, before credits are spent, with an honest message.
    """
    robots_url = f"{validated.scheme}://{validated.host}:{validated.port}/robots.txt"
    robots_target = ValidatedUrl(
        url=robots_url,
        scheme=validated.scheme,
        host=validated.host,
        port=validated.port,
        path="/robots.txt",
        resolved_ips=validated.resolved_ips,
    )

    try:
        async with httpx.AsyncClient(
            timeout=PROBE_TIMEOUT_SECONDS, follow_redirects=False
        ) as client:
            response = await _pinned_get(client, robots_target)
    except Exception as exc:
        logger.info("robots.txt unreadable for %s: %s", validated.host, exc)
        return True, None

    if response.status_code >= 400:
        return True, None

    try:
        parser = RobotFileParser()
        parser.parse(response.text[:MAX_PROBE_BYTES].splitlines())
        allowed = parser.can_fetch("*", validated.url)
    except Exception as exc:  # pragma: no cover - malformed robots.txt
        logger.info("Could not parse robots.txt for %s: %s", validated.host, exc)
        return True, None

    if allowed:
        return True, None
    return False, (
        "This website's robots.txt asks automated crawlers not to access that page. "
        "We respect that, so it cannot be used as a knowledge base."
    )
