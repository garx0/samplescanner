# -*- mode: python ; coding: utf-8 -*-
"""
Builds the portable program:

    .venv\\Scripts\\pyinstaller.exe --noconfirm SampleScanner.spec

The result is dist\\SampleScanner\\ — a folder with SampleScanner.exe in it and
the python, numpy and scipy it needs inside _internal\\. Unpack that folder
anywhere you own, drop ffmpeg.exe and ffprobe.exe next to the exe, and it
runs: no python, no installation, nothing written outside its own folder.

The folder the build is written into is PyInstaller's own decision, not this
file's: it is whatever --distpath says, and dist\\SampleScanner by default. So a
build can be put somewhere else when the one in dist\\ is in use —

    .venv\\Scripts\\pyinstaller.exe --noconfirm --distpath dist2 --workpath build2 SampleScanner.spec

— and everything below follows it: the readme, VERSION.txt, the empty cache and
results folders and the stash all land in the new folder, and dist\\ is not
touched at all.

One exe, two ways to run it — with no arguments it opens the window, with
arguments it is the program itself:

    SampleScanner.exe
    SampleScanner.exe -q .\\query -s .\\search -v
"""
import os
import shutil

block_cipher = None

# Where this build goes. DISTPATH is what PyInstaller resolved --distpath into
# (its default is <this folder>\\dist), so the rest of this file never has to
# know or care where that is.
_OUT = os.path.join(DISTPATH, "SampleScanner")

# The cache and the results of the frozen program live inside that folder, and
# PyInstaller deletes the whole of it to write the new build into it. On a real
# library the index is gigabytes and hours of decoding, and the exported
# fragments are the whole point of the search, so both are set aside before the
# build and put back when it has finished. The stash is a folder of its own,
# OUTSIDE the one that gets deleted.
_STASH = os.path.join(DISTPATH, "_stash")
if os.path.isdir(_STASH):
    shutil.rmtree(_STASH, ignore_errors=True)
_KEEP = []
for _name in ("cache", "results"):
    _live = os.path.join(_OUT, _name)
    if os.path.isdir(_live):
        _parked = os.path.join(_STASH, _name)
        os.makedirs(_STASH, exist_ok=True)
        shutil.move(_live, _parked)
        _KEEP.append((_parked, _live))

# What travels with the exe: the short readme only. README_all.md is the
# developer documentation and has no business in a user's folder; the window's
# Help button falls back to looking for it next to the exe.
datas = []

# The entry point hands over to samplescanner.main(), and the search itself runs
# in the ssc package — named here as well, so that a refactoring of those lines
# cannot quietly leave a module out of the build.
hiddenimports = ["samplescanner", "ssc"]

# Nothing of these is used, and every one of them is big. Only the obviously
# unrelated ones may go in: scipy probes numpy.testing, numpy.f2py and unittest
# while it imports, so excluding any of those breaks it at run time.
excludes = ["matplotlib", "pandas", "PyQt5", "PyQt6", "PySide2", "PySide6",
            "IPython", "jupyter", "pytest", "pydoc_data", "tkinter.test",
            "numpy.tests", "scipy.tests", "numpy.distutils"]

a = Analysis(
    ["samplescanner_gui.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
    name="SampleScanner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # A CONSOLE program, and that is the point: a windowed one is not waited
    # for by the shell that started it and cannot be stopped with Ctrl+C, so
    # `SampleScanner.exe -q ... -s ...` would return at once and keep running
    # in the background. The window hides the console it was handed (see
    # hide_own_console), so nothing extra is shown for the gui.
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
    upx=False,
    upx_exclude=[],
    name="SampleScanner",
)

# Beside the exe: the short readme under the name users look for, the version
# the build carries — written from ssc/version.py, the single place it is kept,
# never typed here — and the two empty folders the program writes into; an empty
# folder survives a zip as a folder entry, so the archive shows what is where
# before the first run.
_out = _OUT          # the same folder the build was just written into
shutil.copyfile(os.path.join(SPECPATH, "README_dist.md"),
                os.path.join(_out, "README.md"))
with open(os.path.join(SPECPATH, "ssc", "version.py"), encoding="utf-8") as _f:
    _ns = {}
    exec(compile(_f.read(), "version.py", "exec"), _ns)   # no imports in it
with open(os.path.join(_out, "VERSION.txt"), "w", encoding="utf-8") as _f:
    _f.write(_ns["version_line"]() + "\n")
for _empty in ("cache", "results"):
    os.makedirs(os.path.join(_out, _empty), exist_ok=True)

# and the folders that were set aside at the top go back where they belong: a
# rebuild must not cost the user the index of every collection they have, nor
# the fragments a search already found
for _parked, _live in _KEEP:
    if os.path.isdir(_live):
        shutil.rmtree(_live, ignore_errors=True)
    shutil.move(_parked, _live)
if os.path.isdir(_STASH) and not os.listdir(_STASH):
    os.rmdir(_STASH)