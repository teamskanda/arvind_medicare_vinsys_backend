"""The launcher icon registry. One definition, three consumers.

The same glyph has to appear in three places:

  * the generated loader script, drawn into the customer's own page;
  * the customisation studio's icon gallery;
  * the studio's live launcher preview.

If each built its own copy, the preview would stop matching the thing it is
previewing the first time one of them was edited — which is the specific
failure this phase is meant to avoid. So the path data lives here, in a
module with no dependencies, and travels to the frontend inside the options
catalogue (`helpers/customization.options`).

Each path is drawn on a 24x24 grid as a stroke icon inheriting
`currentColor`, so the launcher's colour controls it and no fill is needed.
Inlined rather than fetched: one fewer request from the customer's website,
and no dependency on an asset host staying up.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LauncherIcon:
    """One selectable launcher glyph."""

    label: str
    path: str


LAUNCHER_ICONS: dict[str, LauncherIcon] = {
    "chat": LauncherIcon(
        "Chat bubble",
        '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9'
        'L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5'
        'a8.48 8.48 0 0 1 8 8z"/>',
    ),
    "robot": LauncherIcon(
        "Robot",
        '<rect x="4" y="8" width="16" height="12" rx="2"/><path d="M12 4v4"/>'
        '<circle cx="9" cy="14" r="1"/><circle cx="15" cy="14" r="1"/>',
    ),
    "sparkle": LauncherIcon(
        "AI sparkle",
        '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/>'
        '<path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    ),
    "support": LauncherIcon(
        "Customer support",
        '<path d="M4 14v-2a8 8 0 0 1 16 0v2"/>'
        '<path d="M4 14h2a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1H4zM20 14h-2a1 1 0 0 0-1 1v3a1 1 0 0 0 1 1h2z"/>'
        '<path d="M12 22h3a2 2 0 0 0 2-2"/>',
    ),
}

CLOSE_ICON_PATH = '<path d="M18 6L6 18"/><path d="M6 6l12 12"/>'
