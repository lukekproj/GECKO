# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('data', 'data'), ('gui', 'gui'), ('label', 'label'), ('utility', 'utility')]
binaries = []
hiddenimports = [
    'numpy', 'numpy.core', 'numpy.core._multiarray_umath',
    'pandas', 'scipy', 'scipy.signal', 'scipy.ndimage',
    'matplotlib', 'matplotlib.pyplot',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.widgets',
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
    'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
]

for pkg in ['numpy', 'pandas', 'scipy', 'matplotlib', 'PyQt5']:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

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

app = BUNDLE(
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='GECKO.app',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)