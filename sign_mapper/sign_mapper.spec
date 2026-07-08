# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for WV Sign Mapper.
# Run from the sign_mapper/ directory:
#
#   pip install pyinstaller
#   pyinstaller sign_mapper.spec --clean --noconfirm
#
# Output: dist/WVSignMapper/WVSignMapper.exe  (Windows)
#         dist/WVSignMapper/WVSignMapper       (Mac/Linux)
#
# The dist/WVSignMapper/ folder is self-contained — zip it and distribute.
# Typical size: 1.5–2.5 GB (dominated by PyTorch).

from PyInstaller.utils.hooks import collect_all, collect_submodules

# ultralytics ships YAML model configs and other data files that must travel
# with the bundle — collect_all handles binaries, datas, and hiddenimports.
ult_datas, ult_bins, ult_hidden = collect_all('ultralytics')

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],          # sign_mapper/ — finds gps_parser, geojson_utils, pipeline
    binaries=ult_bins,
    datas=[
        ('static', 'static'),   # Flask static folder (index.html, style.css, …)
        *ult_datas,             # ultralytics YAML configs, default weights, etc.
    ],
    hiddenimports=[
        # Local modules imported lazily inside threads — static analysis misses them
        'pipeline',
        'gps_parser',
        'geojson_utils',
        # ultralytics collected above
        *ult_hidden,
        # sklearn Cython extensions not found by static analysis
        'sklearn.neighbors._partition_nodes',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        # werkzeug internals Flask relies on
        'werkzeug.sansio.utils',
        'werkzeug.sansio.http',
        'werkzeug.routing.rules',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude heavy packages that are present in the venv but unused here
    excludes=['matplotlib', 'pandas', 'notebook', 'IPython', 'pytest',
              'tensorboard', 'tf2onnx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WVSignMapper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # Keep the console window so users see the startup URL and any error output.
    # Switch to console=False + a tray icon if you want a cleaner UX later.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WVSignMapper',
)
