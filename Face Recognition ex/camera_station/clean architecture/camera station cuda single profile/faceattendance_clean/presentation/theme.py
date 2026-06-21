"""Modern Tk/ttk theme for the desktop station UI."""
from __future__ import annotations

from dataclasses import dataclass
from tkinter import ttk


@dataclass(frozen=True)
class ThemePalette:
    bg: str = "#0b1120"
    surface: str = "#111827"
    surface_2: str = "#162033"
    surface_3: str = "#1f2937"
    border: str = "#2b3a55"
    text: str = "#e5edf6"
    muted: str = "#9caec4"
    subtle: str = "#718096"
    accent: str = "#38bdf8"
    accent_dark: str = "#0ea5e9"
    success: str = "#22c55e"
    warning: str = "#f59e0b"
    danger: str = "#ef4444"
    video_bg: str = "#050816"
    tree_bg: str = "#0f172a"
    tree_alt: str = "#111c31"
    selection: str = "#164e63"


MODERN_PALETTE = ThemePalette()


def apply_modern_theme(root, palette: ThemePalette = MODERN_PALETTE) -> ThemePalette:
    """Apply a cohesive dark theme to a Tk root/toplevel and return the palette."""
    root.configure(bg=palette.bg)
    try:
        root.option_add("*Font", "{Segoe UI} 10")
        root.option_add("*TCombobox*Listbox.background", palette.surface)
        root.option_add("*TCombobox*Listbox.foreground", palette.text)
        root.option_add("*TCombobox*Listbox.selectBackground", palette.selection)
        root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    except Exception:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(".", background=palette.bg, foreground=palette.text, fieldbackground=palette.surface)
    style.configure("TFrame", background=palette.bg)
    style.configure("App.TFrame", background=palette.bg)
    style.configure("Header.TFrame", background=palette.surface, relief="flat")
    style.configure("Toolbar.TFrame", background=palette.surface)
    style.configure("Card.TFrame", background=palette.surface, relief="flat")
    style.configure("RaisedCard.TFrame", background=palette.surface_2, relief="flat")
    style.configure("Metric.TFrame", background=palette.surface_2, relief="flat")
    style.configure("Video.TFrame", background=palette.video_bg, relief="flat")

    style.configure("TLabel", background=palette.bg, foreground=palette.text, font=("Segoe UI", 10))
    style.configure("Card.TLabel", background=palette.surface, foreground=palette.text, font=("Segoe UI", 10))
    style.configure("Video.TLabel", background=palette.video_bg, foreground=palette.muted, font=("Segoe UI", 11))
    style.configure("Muted.TLabel", background=palette.bg, foreground=palette.muted, font=("Segoe UI", 9))
    style.configure("MutedCard.TLabel", background=palette.surface, foreground=palette.muted, font=("Segoe UI", 9))
    style.configure("Title.TLabel", background=palette.bg, foreground="#ffffff", font=("Segoe UI Semibold", 20))
    style.configure("HeaderTitle.TLabel", background=palette.surface, foreground="#ffffff", font=("Segoe UI Semibold", 20))
    style.configure("HeaderSub.TLabel", background=palette.surface, foreground=palette.muted, font=("Segoe UI", 9))
    style.configure("SectionTitle.TLabel", background=palette.surface, foreground="#ffffff", font=("Segoe UI Semibold", 12))
    style.configure("MetricValue.TLabel", background=palette.surface_2, foreground="#ffffff", font=("Segoe UI Semibold", 15))
    style.configure("MetricName.TLabel", background=palette.surface_2, foreground=palette.muted, font=("Segoe UI", 8))
    style.configure("Status.TLabel", background=palette.bg, foreground=palette.success, font=("Segoe UI Semibold", 10))
    style.configure("StatusPill.TLabel", background=palette.surface_2, foreground=palette.success, font=("Segoe UI Semibold", 10), padding=(10, 5))
    style.configure("Danger.TLabel", background=palette.surface, foreground=palette.danger, font=("Segoe UI Semibold", 10))

    for button_style, bg, active in (
        ("TButton", palette.surface_3, palette.border),
        ("Secondary.TButton", palette.surface_3, palette.border),
        ("Accent.TButton", palette.accent_dark, palette.accent),
        ("Primary.TButton", palette.accent_dark, palette.accent),
        ("Danger.TButton", "#7f1d1d", "#991b1b"),
        ("Ghost.TButton", palette.surface, palette.surface_2),
    ):
        style.configure(button_style, font=("Segoe UI Semibold", 9), padding=(10, 7), borderwidth=0, relief="flat")
        style.map(button_style, background=[("active", active), ("pressed", active), ("disabled", palette.surface_2)], foreground=[("disabled", palette.subtle)])
    style.configure("TButton", background=palette.surface_3, foreground=palette.text)
    style.configure("Secondary.TButton", background=palette.surface_3, foreground=palette.text)
    style.configure("Accent.TButton", background=palette.accent_dark, foreground="#ffffff")
    style.configure("Primary.TButton", background=palette.accent_dark, foreground="#ffffff")
    style.configure("Danger.TButton", background="#7f1d1d", foreground="#ffffff")
    style.configure("Ghost.TButton", background=palette.surface, foreground=palette.muted)

    style.configure("TCheckbutton", background=palette.surface, foreground=palette.text, font=("Segoe UI", 10), focuscolor=palette.surface)
    style.map("TCheckbutton", background=[("active", palette.surface)], foreground=[("disabled", palette.subtle)])
    style.configure("TRadiobutton", background=palette.surface, foreground=palette.text, font=("Segoe UI", 10), focuscolor=palette.surface)
    style.map("TRadiobutton", background=[("active", palette.surface)], foreground=[("disabled", palette.subtle)])

    style.configure("TEntry", fieldbackground=palette.surface_2, foreground=palette.text, insertcolor=palette.text, borderwidth=1, relief="flat", padding=(6, 4))
    style.configure("TCombobox", fieldbackground=palette.surface_2, background=palette.surface_2, foreground=palette.text, arrowcolor=palette.text, borderwidth=0, padding=(6, 4))
    style.map("TCombobox", fieldbackground=[("readonly", palette.surface_2)], foreground=[("readonly", palette.text)], background=[("readonly", palette.surface_2)])

    style.configure("Horizontal.TScale", background=palette.surface, troughcolor=palette.surface_3, sliderthickness=15)
    style.configure("TNotebook", background=palette.bg, borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", background=palette.surface, foreground=palette.muted, padding=(18, 9), font=("Segoe UI Semibold", 10), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", palette.surface_2), ("active", palette.surface_3)], foreground=[("selected", "#ffffff"), ("active", "#ffffff")])

    style.configure("Treeview", rowheight=30, font=("Segoe UI", 10), background=palette.tree_bg, fieldbackground=palette.tree_bg, foreground=palette.text, borderwidth=0, relief="flat")
    style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10), background=palette.surface_3, foreground=palette.text, relief="flat", padding=(6, 6))
    style.map("Treeview", background=[("selected", palette.selection)], foreground=[("selected", "#ffffff")])
    style.configure("Vertical.TScrollbar", background=palette.surface_3, troughcolor=palette.surface, bordercolor=palette.surface, arrowcolor=palette.text)
    style.configure("Horizontal.TScrollbar", background=palette.surface_3, troughcolor=palette.surface, bordercolor=palette.surface, arrowcolor=palette.text)

    return palette
