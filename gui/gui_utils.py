"""
GUI Utility Functions

Reusable tkinter helpers for window management, styling, and crash reporting.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox

def center_window(win, width=None, height=None):
    """
    Center a tkinter window on screen.

    If width/height are not provided, the window's current dimensions are used.

    Parameters
    ----------
    win : tk.Tk or tk.Toplevel
        Window to center.
    width : int or None
        Override width in pixels.
    height : int or None
        Override height in pixels.
    """
    win.update_idletasks()
    if width is None:
        width  = win.winfo_width()
    if height is None:
        height = win.winfo_height()

    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = (screen_w // 2) - (width // 2)
    y = (screen_h // 2) - (height // 2)

    win.geometry(f"{width}x{height}+{x}+{y}")

def configure_big_treeview_style():
    """
    Configure larger, more readable fonts and spacing for treeview components.
    
    This sets up bigger fonts and taller rows so that the task protocol
    tables are easier to read, especially on high-DPI displays.
    """
    style = ttk.Style()
    
    # Use default theme to ensure rowheight works properly on Windows
    try:
        if style.theme_use() == "vista":
            style.theme_use("default")
    except Exception:
        pass

    # Configure larger treeview styling
    style.configure(
        "Big.Treeview",
        rowheight=36,
        font=("Segoe UI", 13)
    )
    style.configure(
        "Big.Treeview.Heading", 
        font=("Segoe UI", 13, "bold")
    )
    
    # Make notebook tabs bigger too
    style.configure("TNotebook.Tab", font=("Segoe UI", 12, "bold"))

def log_crash(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions to a crash report file."""
    # Try Desktop first, fall back to home directory, then temp directory
    for candidate in [
        Path.home() / "Desktop" / "kinarm_crash_report.txt",
        Path.home() / "kinarm_crash_report.txt",
    ]:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            crash_log_path = candidate
            break
        except OSError:
            continue
    else:
        import tempfile
        crash_log_path = Path(tempfile.gettempdir()) / "kinarm_crash_report.txt"
    
    with open(crash_log_path, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"CRASH REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        f.write(f"\n{'='*80}\n\n")
    
    # Also print to console
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    
    # Show error dialog
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Program Crashed",
            f"The program crashed unexpectedly.\n\n"
            f"A crash report has been saved to:\n{crash_log_path}\n\n"
            f"Please send this file to support."
        )
        root.destroy()
    except Exception:
        pass