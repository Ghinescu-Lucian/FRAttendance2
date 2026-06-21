"""Reusable Tk widgets for the presentation layer."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional


def section_title(parent, text: str, subtitle: Optional[str] = None):
    frame = ttk.Frame(parent, style="Card.TFrame")
    ttk.Label(frame, text=text, style="SectionTitle.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(frame, text=subtitle, style="MutedCard.TLabel", wraplength=340).pack(anchor="w", pady=(2, 0))
    return frame


def metric_card(parent, title: str, variable: tk.StringVar, width: int = 10):
    card = ttk.Frame(parent, style="Metric.TFrame", padding=(12, 8))
    ttk.Label(card, text=title.upper(), style="MetricName.TLabel").pack(anchor="w")
    ttk.Label(card, textvariable=variable, style="MetricValue.TLabel", width=width).pack(anchor="w")
    return card


def configure_tree_tags(tree, odd_bg: str = "#111c31", even_bg: str = "#0f172a") -> None:
    try:
        tree.tag_configure("odd", background=odd_bg)
        tree.tag_configure("even", background=even_bg)
    except Exception:
        pass


def retag_tree_rows(tree) -> None:
    try:
        for index, item in enumerate(tree.get_children("")):
            tree.item(item, tags=("even" if index % 2 == 0 else "odd",))
    except Exception:
        pass
