"""
AVA Terminal User Interface (TUI)
=================================

A powerful, keyboard-driven terminal interface for AVA using Textual.

Features:
- Full chat interface with markdown support
- Real-time system metrics display
- Command palette (Ctrl+K)
- Keyboard-driven navigation
- Split-pane view
- ASCII thinking animations

Usage:
    python run_tui.py
    # or
    python -m tui
"""

from .app import AVATUI

__all__ = ["AVATUI"]
