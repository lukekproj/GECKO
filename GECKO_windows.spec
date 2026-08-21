# -*- mode: python ; coding: utf-8 -*-

# PyInstaller spec file for GECKO (Windows, onedir build).

import os
from PyInstaller.utils.hooks import collect_all

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

for pkg in ('numpy', 'pandas', 'scipy', 'matplotlib'):
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['gui/gui_main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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