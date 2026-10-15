# -*- mode: python ; coding: utf-8 -*-

# PyInstaller spec file for GECKO (Windows, onedir build).

import os

app_name = os.environ.get("GECKO_VERSION", "GECKO")

datas = [
    ('data', 'data'),
    ('gui', 'gui'),
    ('label', 'label'),
    ('utility', 'utility'),
]
binaries = []
hiddenimports = [
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    'pandas',
    'scipy',
    'scipy.signal',
    'scipy.ndimage',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.widgets',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'PyQt5.QtPrintSupport',
]

a = Analysis(
    ['gui/gui_main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Qt5Agg", "QtAgg", "TkAgg", "Agg"]}},
    runtime_hooks=[],
    excludes=['pytest', 'IPython', 'jupyter', 'notebook'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)