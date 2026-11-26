"""The floating widget loader, and the install snippets customers copy.

Architecture (iframe-first, per the research blueprint and section 7 of the
Phase 3 brief):

    customer's website
      |
      +-- loader script  (this module generates it)
      |     * draws the launcher button
      |     * creates/sizes/positions the iframe
      |     * opens, closes, handles mobile
      |     * contains NO AI logic, NO knowledge base, NO configuration
      |       beyond what is already public
      |
      +-- iframe  ->  {PUBLIC_WIDGET_BASE_URL}/embed/{public_id}?mode=panel
            * this is the actual chatbot
            * talks to our API with a signed visitor session
            * same origin as us, so the customer's page cannot read it

Why the split matters: the loader runs **inside the customer's page**, with
access to their DOM. Anything we put in it is readable by them and by anyone
who can inject script into their site. So it carries only the public_id and
the launcher's appearance. Conversations, retrieval and the model all stay
behind the iframe boundary, where the customer's page cannot reach them —
browsers block cross-origin DOM access, which is the entire security benefit
of this architecture over a script that renders chat directly into the page.

Two things the loader deliberately does not do:

  * It never creates a transparent full-viewport iframe. That pattern is
    common and it swallows every click on the host page. The collapsed state
    here is a small fixed-size launcher, and `pointer-events` is scoped to
    the elements that need it.
  * It never trusts a `postMessage` without checking the origin *and* that
    the message came from our own iframe's window.
"""

from __future__ import annotations

import json
from html import escape

from config import get_settings
from helpers.icons import CLOSE_ICON_PATH, LAUNCHER_ICONS
from models.chatbot import Chatbot
from models.customization import LauncherSettings
from models.deployment import DeploymentSnippet, PlatformInstruction

# Below this viewport width the panel becomes a full-screen sheet. 480px is
# the common phone-portrait boundary; a 400x600 panel on a 360px screen would
# overflow horizontally, which the brief rules out explicitly.
MOBILE_BREAKPOINT = 480


# ---------------------------------------------------------------------------
# Inline SVG icons for the launcher
# ---------------------------------------------------------------------------
# The path data lives in helpers/icons.py, which the customisation studio's
# gallery and its launcher preview also read, so the glyph a customer picks is
# literally the glyph their website draws.

_ICONS: dict[str, str] = {key: icon.path for key, icon in LAUNCHER_ICONS.items()}

_CLOSE_ICON = CLOSE_ICON_PATH


def _icon_markup(launcher: LauncherSettings) -> str:
    """The launcher's open-state glyph.

    A custom `icon_url` wins when present; `deployment.resolve_launcher` has
    already rejected anything that is not an https URL, so this cannot become
    a `javascript:` or `data:` sink.
    """
    if launcher.icon_url:
        return (
            f'<img src="{escape(launcher.icon_url, quote=True)}" alt="" '
            'style="width:60%;height:60%;object-fit:contain;border-radius:6px">'
        )
    path = _ICONS.get(launcher.icon, _ICONS["chat"])
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )


# ---------------------------------------------------------------------------
# The loader script
# ---------------------------------------------------------------------------


def _launcher_radius(launcher: LauncherSettings) -> str:
    """The launcher's corner radius as a CSS length.

    Derived from the shape rather than stored as a number, so "circle" stays
    circular at every size the customer can choose.
    """
    if launcher.shape == "square":
        return "10px"
    if launcher.shape == "rounded":
        return f"{max(8, round(launcher.launcher_size * 0.32))}px"
    return "50%"


def build_loader_config(
    chatbot: Chatbot, launcher: LauncherSettings, *, window_radius: int = 14,
    preview: bool = False,
) -> dict:
    """Generate the JavaScript served at `/w/{public_id}.js`.

    Every value the script embeds goes through `json.dumps`, so a hostile
    stored configuration cannot break out of a string literal and execute in
    the customer's page. The size target is 2-3 KB compressed; readability is
    kept because a customer's developer may well read this file, and an
    unreadable loader is harder to trust than a slightly larger one.

    `window_radius` comes from the chatbot's appearance settings so the
    floating window's corners match the chat UI inside it. It is a number
    here, not a CSS string, because it is clamped before it is interpolated.
    """
    settings = get_settings()

    panel_url = settings.embed_url(chatbot.public_id, mode="panel")
    embed_origin = _origin_of(settings.public_widget_base_url)

    config = {
        "id": chatbot.public_id,
        "url": "about:blank" if preview else panel_url,
        "origin": _origin_of(settings.frontend_url) if preview else embed_origin,
        "name": chatbot.name[:80],
        "accent": launcher.accent_color,
        "size": launcher.launcher_size,
        "radius": _launcher_radius(launcher),
        "side": "left" if launcher.position == "bottom-left" else "right",
        "dx": launcher.offset_x,
        "dy": launcher.offset_y,
        "w": launcher.window_width,
        "h": launcher.window_height,
        # The floating window's own corner radius, kept inside the range the
        # studio offers plus a little, so a stored value can never produce a
        # pill-shaped window.
        "wr": max(0, min(28, window_radius)),
        "autoOpen": launcher.open_on_load,
        "label": launcher.label or "",
        "anim": launcher.open_animation,
        "closeAnim": launcher.close_animation,
        "attn": launcher.attention_animation,
        "icon": _icon_markup(launcher),
        "close": (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" aria-hidden="true">{_CLOSE_ICON}</svg>'
        ),
        "mobile": MOBILE_BREAKPOINT,
        "refresh": None if preview else f"{settings.public_api_base_url}/api/public/bots/{chatbot.public_id}/widget-config",
    }

    return config


def build_loader_script(
    chatbot: Chatbot, launcher: LauncherSettings, *, window_radius: int = 14,
    preview: bool = False,
) -> str:
    config = build_loader_config(chatbot, launcher, window_radius=window_radius, preview=preview)
    return _LOADER_TEMPLATE.replace("__SKANDA_CONFIG__", json.dumps(config))


def build_unavailable_script(public_id: str) -> str:
    """Served when the handle is unknown, draft or unpublished.

    A no-op with one clear console line, rather than a 404. Reasons: the
    customer's own site should not log a failed request because they
    unpublished their chatbot, and an identical response for "unknown" and
    "unpublished" means the endpoint does not confirm which handles exist.
    """
    return (
        "/* SKANDA AI widget */\n"
        "(function(){\n"
        '  if (window.console && console.info) {\n'
        '    console.info("SKANDA AI: this chat assistant is not published, so the '
        'widget was not loaded. Publish it from your SKANDA AI dashboard.");\n'
        "  }\n"
        f"  window.__skandaAIUnavailable = {json.dumps(public_id[:64])};\n"
        "})();\n"
    )


def _origin_of(base_url: str) -> str:
    """Scheme + host + port, which is what `postMessage` compares against."""
    from urllib.parse import urlsplit

    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        return base_url
    return f"{parsed.scheme}://{parsed.netloc}"


# The loader. Written as an IIFE with no globals beyond one namespace object,
# no dependencies, and no build step.
_LOADER_TEMPLATE = r"""/* SKANDA AI chat widget loader. Positions and controls an iframe; holds no AI logic. */
(function () {
  "use strict";
  var C = __SKANDA_CONFIG__;

  // Guard: two copies of the snippet on one page must not draw two launchers.
  // The marker is claimed NOW, not in start(). Two script tags both evaluate
  // before DOMContentLoaded, so a guard that waited for the public handle to
  // be published would let both through and draw two launchers.
  window.__skandaAILoaded = window.__skandaAILoaded || {};
  if (window.__skandaAILoaded[C.id]) { return; }
  window.__skandaAILoaded[C.id] = true;

  var doc = document;
  var open = false;
  var iframe = null;
  var root, launcher, panel;
  var closingTimer = null;

  function isMobile() { return window.innerWidth <= C.mobile; }

  function style(el, css) { for (var k in css) { el.style[k] = css[k]; } }

  // Checked at the moment of use, so changing the OS preference takes effect
  // without a reload. A visitor who asked for less motion gets none.
  function reducedMotion() {
    return !!(window.matchMedia &&
              window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  // Keyframes cannot be expressed inline. One id-guarded <style>, so two
  // chatbots on a page share it, with names that cannot collide.
  function injectStyles() {
    if (doc.getElementById("skanda-ai-keyframes")) { return; }
    var el = doc.createElement("style");
    el.id = "skanda-ai-keyframes";
    el.textContent =
      "@keyframes skandaAiScale{from{opacity:0;transform:scale(.9)}to{opacity:1;transform:scale(1)}}" +
      "@keyframes skandaAiSlide{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}" +
      "@keyframes skandaAiFade{from{opacity:0}to{opacity:1}}" +
      "@keyframes skandaAiPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}" +
      "@keyframes skandaAiBounce{0%,82%,100%{transform:translateY(0)}88%{transform:translateY(-9px)}94%{transform:translateY(-3px)}}" +
      "@keyframes skandaAiGlow{0%,100%{box-shadow:0 6px 20px rgba(0,0,0,.22)}" +
      "50%{box-shadow:0 6px 20px rgba(0,0,0,.22),0 0 0 12px var(--skanda-glow)}}";
    (doc.head || doc.documentElement).appendChild(el);
  }

  function labelNode() {
    var span = doc.createElement("span");
    // textContent, never innerHTML: this is customer text going into their
    // own page, and as markup it would be an injection point there.
    span.textContent = C.label;
    span.style.font = '600 14px/1 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
    span.style.whiteSpace = "nowrap";
    return span;
  }

  // Two looks: closed (chosen icon and shape, optional label) and open (a
  // close cross, always square).
  function paintLauncher(opened) {
    var glow = C.accent;
    if (/^#[0-9a-f]{3}$/i.test(glow)) {
      glow = "#" + glow[1] + glow[1] + glow[2] + glow[2] + glow[3] + glow[3];
    }
    launcher.style.setProperty("--skanda-glow", glow + "26");
    var pill = !opened && !!C.label;
    launcher.innerHTML = opened ? C.close : C.icon;

    var glyph = Math.round(C.size * (opened ? 0.44 : 0.52));
    var svg = launcher.querySelector("svg");
    if (svg) { style(svg, { width: glyph + "px", height: glyph + "px" }); }
    var img = launcher.querySelector("img");
    if (img) {
      style(img, {
        width: Math.round(C.size * 0.58) + "px",
        height: Math.round(C.size * 0.58) + "px",
        objectFit: "contain", borderRadius: "6px"
      });
    }

    if (pill) { launcher.appendChild(labelNode()); }
    style(launcher, {
      width: pill ? "auto" : C.size + "px",
      padding: pill ? "0 " + Math.round(C.size * 0.34) + "px" : "0",
      gap: pill ? "9px" : "0",
      borderRadius: pill ? Math.round(C.size / 2) + "px" : C.radius
    });

    launcher.setAttribute(
      "aria-label", opened ? "Close the chat assistant" : "Open the chat assistant"
    );
    launcher.setAttribute("aria-expanded", opened ? "true" : "false");
  }

  // Runs while closed, to draw the eye. Off unless the customer asked for it.
  function applyAttention() {
    if (!C.attn || C.attn === "none" || reducedMotion()) {
      launcher.style.animation = "";
      return;
    }
    launcher.style.animation =
      C.attn === "bounce" ? "skandaAiBounce 5s ease-in-out 4s infinite" :
      C.attn === "glow"   ? "skandaAiGlow 2.6s ease-in-out 3s infinite" :
                            "skandaAiPulse 2.4s ease-in-out 4s infinite";
  }

  function build() {
    injectStyles();
    root = doc.createElement("div");
    root.className = "skanda-ai-widget";
    root.setAttribute("data-skanda-ai", C.id);
    // Zero-sized and click-through. The launcher and panel inside it are
    // themselves position:fixed, so the container only supplies the stacking
    // context -- it never covers any part of the host page, and only those
    // two elements accept pointer events.
    style(root, {
      position: "fixed", zIndex: "2147483000", bottom: "0", left: "0",
      width: "0", height: "0", pointerEvents: "none"
    });

    panel = doc.createElement("div");
    style(panel, {
      position: "fixed", display: "none", overflow: "hidden", pointerEvents: "auto",
      background: "#ffffff", borderRadius: C.wr + "px",
      boxShadow: "0 12px 40px rgba(0,0,0,.18), 0 2px 8px rgba(0,0,0,.08)",
      border: "1px solid rgba(0,0,0,.08)"
    });

    launcher = doc.createElement("button");
    launcher.type = "button";
    style(launcher, {
      position: "fixed", pointerEvents: "auto", cursor: "pointer",
      height: C.size + "px",
      border: "none", margin: "0",
      background: C.accent, color: "#ffffff",
      display: "flex", alignItems: "center", justifyContent: "center",
      boxShadow: "0 6px 20px rgba(0,0,0,.22)",
      transition: "transform .15s ease, box-shadow .15s ease",
      WebkitTapHighlightColor: "transparent"
    });
    paintLauncher(false);
    applyAttention();
    // Hover and attention both drive `transform`: only one is ever active.
    launcher.addEventListener("mouseenter", function () {
      if (C.attn === "none") { launcher.style.transform = "scale(1.05)"; }
    });
    launcher.addEventListener("mouseleave", function () {
      if (C.attn === "none") { launcher.style.transform = "scale(1)"; }
    });
    launcher.addEventListener("click", function () { open ? hide() : show(); });

    place();
    root.appendChild(panel);
    root.appendChild(launcher);
    (doc.body || doc.documentElement).appendChild(root);
    window.addEventListener("resize", place);
  }

  // Parent-side positioning and sizing. The iframe is given real dimensions
  // rather than being stretched over the viewport.
  function place() {
    var edge = C.side === "left" ? "left" : "right";
    var other = C.side === "left" ? "right" : "left";
    launcher.style[edge] = C.dx + "px";
    launcher.style[other] = "auto";
    launcher.style.bottom = C.dy + "px";

    if (isMobile()) {
      style(panel, {
        left: "0", right: "0", top: "0", bottom: "0",
        width: "auto", height: "auto", borderRadius: "0", border: "none"
      });
    } else {
      var h = Math.max(120, Math.min(C.h, window.innerHeight - C.dy - C.size - 32));
      var w = Math.max(120, Math.min(C.w, window.innerWidth - C.dx * 2));
      panel.style[edge] = C.dx + "px";
      panel.style[other] = "auto";
      style(panel, {
        top: "auto", bottom: (C.dy + C.size + 12) + "px",
        width: w + "px", height: h + "px", borderRadius: C.wr + "px"
      });
    }

    // Crossing the breakpoint while open must not strand the launcher. On a
    // full-screen mobile sheet it would float over the chat; going back to
    // desktop without restoring it would leave no visible close affordance.
    if (open) { launcher.style.display = isMobile() ? "none" : "flex"; }
  }

  function mount() {
    if (iframe) { return; }
    iframe = doc.createElement("iframe");
    iframe.src = C.url;
    iframe.title = C.name + " chat assistant";
    iframe.setAttribute("allow", "clipboard-write");
    // Preserve the iframe origin for API calls, sessions and the message bridge.
    iframe.setAttribute(
      "sandbox",
      "allow-scripts allow-same-origin allow-forms allow-popups"
    );
    style(iframe, { width: "100%", height: "100%", border: "0", display: "block" });
    panel.appendChild(iframe);
  }

  function show() {
    if (closingTimer) { clearTimeout(closingTimer); closingTimer = null; }
    mount();
    open = true;
    panel.style.display = "block";
    place();
    if (C.anim !== "none" && !reducedMotion()) {
      // Grown from the launcher's own corner, not from nowhere.
      panel.style.transformOrigin = (C.side === "left" ? "left" : "right") + " bottom";
      panel.style.animation =
        (C.anim === "slide" ? "skandaAiSlide" :
         C.anim === "fade"  ? "skandaAiFade"  : "skandaAiScale") +
        " .2s cubic-bezier(.2,.8,.2,1)";
    }
    paintLauncher(true);
    launcher.style.animation = "";
    launcher.style.transform = "scale(1)";
    if (isMobile()) { launcher.style.display = "none"; }
    post({ type: "open" });
  }

  function hide() {
    open = false;
    if (closingTimer) { clearTimeout(closingTimer); }
    var animation = C.closeAnim;
    if (animation && animation !== "none" && !reducedMotion()) {
      panel.style.transformOrigin = (C.side === "left" ? "left" : "right") + " bottom";
      panel.style.animation =
        (animation === "slide" ? "skandaAiSlide" :
         animation === "fade" ? "skandaAiFade" : "skandaAiScale") +
        " .18s ease-in reverse both";
      closingTimer = setTimeout(function () {
        if (!open) { panel.style.display = "none"; panel.style.animation = ""; }
        closingTimer = null;
      }, 180);
    } else {
      panel.style.display = "none";
      panel.style.animation = "";
    }
    launcher.style.display = "flex";
    paintLauncher(false);
    applyAttention();
    post({ type: "close" });
  }

  function post(message) {
    if (!iframe || !iframe.contentWindow) { return; }
    message.source = "skanda-ai-parent";
    // Explicit target origin: never "*", which would broadcast to whatever
    // happens to be in the frame.
    iframe.contentWindow.postMessage(message, C.origin);
  }

  // Messages from the chatbot. Three checks before anything is acted on:
  // the origin must be ours, the sender must be our own iframe's window, and
  // the payload must carry our marker.
  window.addEventListener("message", function (event) {
    if (event.origin !== C.origin) { return; }
    if (!iframe || event.source !== iframe.contentWindow) { return; }
    var data = event.data;
    if (!data || data.source !== "skanda-ai-widget") { return; }

    if (data.type === "ready") {
      post({ type: "viewport", mobile: isMobile(), width: window.innerWidth });
    } else if (data.type === "close") {
      hide();
    } else if (data.type === "resize") {
      // Bounded: a compromised frame must not be able to cover the page.
      var h = parseInt(data.height, 10);
      if (!isMobile() && h > 0) {
        panel.style.height = Math.min(Math.max(h, 320), C.h) + "px";
      }
    }
  });

  function start() {
    build();
    if (C.autoOpen) { show(); }
    var controller = {
      id: C.id,
      open: show,
      close: hide,
      toggle: function () { open ? hide() : show(); },
      update: function (next) {
        var wasAuto = C.autoOpen;
        C = next;
        style(launcher, { height: C.size + "px", background: C.accent });
        paintLauncher(open);
        place();
        if (!open) { applyAttention(); }
        if (C.autoOpen !== wasAuto) { C.autoOpen ? show() : hide(); }
      }
    };
    window.__skandaAI = controller;
    // Existing installations receive newly published settings without a new
    // snippet or page reload. Draft settings never enter this endpoint.
    if (C.refresh && window.fetch) {
      var refreshing = false;
      var lastConfig = "";
      var refresh = function () {
        if (refreshing || doc.hidden) { return; }
        refreshing = true;
        fetch(C.refresh, { cache: "no-store", credentials: "omit" }).then(function (res) {
          if (res.status === 404) { root.style.display = "none"; return null; }
          if (!res.ok) { return null; }
          return res.json();
        }).then(function (next) {
          if (!next) { return; }
          root.style.display = "block";
          var serialized = JSON.stringify(next);
          if (serialized !== lastConfig) { controller.update(next); lastConfig = serialized; }
        }).catch(function () {}).then(function () { refreshing = false; });
      };
      setInterval(refresh, 30000);
      window.addEventListener("focus", refresh);
    }
  }

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
"""


# ---------------------------------------------------------------------------
# Install snippets
# ---------------------------------------------------------------------------


def build_snippets(chatbot: Chatbot, launcher: LauncherSettings) -> list[DeploymentSnippet]:
    """The three deployment modes, as ready-to-copy code.

    Generated from `public_id` and the configured base URL — never a
    hardcoded domain and never a hardcoded id, so moving the deployment
    domain is one environment variable.
    """
    settings = get_settings()
    public_id = chatbot.public_id

    inline_url = settings.embed_url(public_id)
    panel_url = settings.embed_url(public_id, mode="panel")
    full_url = settings.embed_url(public_id, mode="full")
    script_url = settings.widget_script_url(public_id)

    widget_code = f'<script src="{script_url}" async></script>'

    iframe_code = (
        "<iframe\n"
        f'  src="{inline_url}"\n'
        '  width="100%"\n'
        f'  height="{launcher.window_height}"\n'
        '  style="border:0;"\n'
        f'  title="{_escape_attribute(chatbot.name)} chat assistant"\n'
        '  loading="lazy">\n'
        "</iframe>"
    )

    return [
        DeploymentSnippet(
            mode="widget",
            label="Floating widget",
            description=(
                "A chat bubble in the corner of every page. Paste this once "
                "before the closing </body> tag."
            ),
            url=panel_url,
            code=widget_code,
            language="html",
            recommended=True,
        ),
        DeploymentSnippet(
            mode="iframe",
            label="Inline iframe",
            description=(
                "Embeds the chatbot inside the page, filling its container. "
                "Use this on a contact or support page, or where the platform "
                "does not allow custom JavaScript."
            ),
            url=inline_url,
            code=iframe_code,
            language="html",
        ),
        DeploymentSnippet(
            mode="link",
            label="Shareable link",
            description=(
                "A full-page chatbot you can send by email or WhatsApp, or "
                "link from social media. No website needed."
            ),
            url=full_url,
        ),
    ]


def _escape_attribute(value: str) -> str:
    """Escape a chatbot name for an HTML attribute in generated snippet text."""
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")[:80]
    )


def platform_instructions() -> list[PlatformInstruction]:
    """Short, per-platform install steps for the Deploy tab.

    `supports_javascript=False` is the honest flag the brief asks for:
    some plans genuinely cannot run a custom script, and the customer needs
    to know that before they try.
    """
    return [
        PlatformInstruction(
            platform="HTML website",
            steps=[
                "Open the HTML file or template that every page shares.",
                "Paste the widget snippet just before the closing </body> tag.",
                "Save and upload the file, then reload your website.",
            ],
        ),
        PlatformInstruction(
            platform="WordPress",
            steps=[
                "In the admin menu, go to Appearance -> Theme File Editor, or "
                "install a plugin such as 'Insert Headers and Footers'.",
                "Paste the widget snippet into the footer section.",
                "Save changes and reload your site.",
            ],
            note=(
                "On WordPress.com, custom scripts need a Business plan or "
                "higher. Use the inline iframe in a Custom HTML block instead."
            ),
        ),
        PlatformInstruction(
            platform="Shopify",
            steps=[
                "Go to Online Store -> Themes -> ... -> Edit code.",
                "Open layout/theme.liquid.",
                "Paste the widget snippet just before </body> and save.",
            ],
        ),
        PlatformInstruction(
            platform="React / Next.js",
            steps=[
                "Render the chatbot in an <iframe> with the inline URL, or",
                "add the loader with next/script: "
                '<Script src="..." strategy="afterInteractive" />.',
                "Avoid injecting the script inside a component that remounts "
                "often; the loader is designed to be added once per page load.",
            ],
        ),
        PlatformInstruction(
            platform="Wix / Squarespace / Webflow (restricted plans)",
            steps=[
                "Add an Embed or Custom HTML element to the page.",
                "Paste the inline iframe snippet into it.",
            ],
            supports_javascript=False,
            note=(
                "These builders block custom scripts on their lower plans. "
                "The inline iframe works on every plan that allows an embed "
                "block, but it cannot float over the page."
            ),
        ),
    ]
