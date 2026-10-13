# SampleScanner

Finds which track of your music collection a short audio fragment was taken
from, and at what second and pitch.

## Installation (from source code)

The program reads and writes audio using **ffmpeg and ffprobe**, which are not
bundled with SampleScanner. Install both for your system:

* **Windows** — download a build from <https://ffmpeg.org/download.html> (the
  `gyan.dev` builds are the painless ones: they contain `ffmpeg.exe` and
  `ffprobe.exe` in one `bin` folder). Unpack it and put `ffmpeg.exe` and
  `ffprobe.exe` **next to `samplescanner.py`**, or add that `bin` folder to
  `PATH`.
* **macOS** — `brew install ffmpeg` installs both at once. If you would rather
  not install anything, download a static build from the same page, unpack it,
  and put `ffmpeg` and `ffprobe` next to `samplescanner.py`
  (`chmod +x` them first if macOS does not do it for you).
* **Linux** — `ffmpeg` is in every distribution's repositories:
  `sudo apt install ffmpeg` (Debian/Ubuntu), `sudo dnf install ffmpeg`
  (Fedora), `sudo pacman -S ffmpeg` (Arch). A static build next to
  `samplescanner.py` works too.

A build unpacked next to the program is looked for as `ffmpeg.exe`/`ffprobe.exe` on 
Windows and as `ffmpeg`/`ffprobe` on macOS and Linux. Anything on `PATH` is found 
whatever it is called.

Install **Python 3.9 or newer**, then run `pip install -r requirements.txt`
from this folder.

A virtual environment is the better way to do it, and on recent Debian/Ubuntu
it is the only way that is allowed — those refuse a `pip install` outside one,
so that a package manager can never be broken by a `pip` that did not ask. It
also keeps everything SampleScanner needs inside this folder:

```
python3 -m venv .venv                       # Windows: python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt    # Windows: .venv\Scripts\python.exe ...
.venv/bin/python samplescanner_gui.py       # Windows: .venv\Scripts\python.exe ...
```

On Linux the **window** needs tkinter, which is a separate package in some
distributions: `sudo apt install python3-tk` (Debian/Ubuntu),
`sudo dnf install python3-tkinter` (Fedora). Without it the program itself
still runs from a terminal — only the window is unavailable. On macOS tkinter
comes with the python.org installer and with brew's python.

Note that on macOS and Linux the command is `python3`, not `python`.

After installation and usage, if the author updated the SampleScanner upload 
with a new version, you'll need to move your contents of results and cache 
folders manually in a directory with the new version. You'll be able to reuse 
the cache (aka index, fingerprints) you've generated (unless the author has 
rewritten the way it creates and reads the cache in one of the updates).

## Running it

To run SampleScanner with interface, run `python samplescanner_gui.py`
(`python3 …` on macOS and Linux). Pick 
what to look for, pick where to look, configure pitches to check, press 
"Start the search".

To run SampleScanner in command line, run the following command, specifying 
path to query file/queries folder, sources folder, and pitch range and step:

```
python samplescanner.py -q "<queries>" -s "<sources>" --pitch=<MIN>:<MAX>:<STEP> -v
```

(you can specify `samplescanner_gui.py` instead of `samplescanner.py` here)

`--version` prints the version of the program and stops, if you want to check
which one you have.

Use case: start new search (with one or multiple samples to find in the selected
library) and play the wav files (with the finds worth checking) appearing in the
results/... folder while the search progresses.

**What may be picked**, for both the query and the library:

* one or several audio files (Ctrl and Shift pick several at once);
* a folder — subfolders are read too, and anything that is not audio is skipped;
* a **text file** with a list, one entry per line: folders, audio files and
  `glob` patterns (`F:\music\Artist *\*.flac` on Windows, `~/music/Artist */`
  on macOS and Linux) may be mixed in one list. Blank lines and everything after
  a `#` are ignored. A relative entry counts from the folder of that text file,
  so a list can be moved together with the collection it describes. To pick a
  text file in the window, switch the file type in the dialog from `Audio` to
  `Text`.

## Where it writes

Only into the folder the program is started from: `cache` (the index it builds
once per collection) and `results` (one folder per run, holding a `log_….txt`
with everything the run printed, and the fragments it cut out). With the ready
exe that folder is the one the exe is in; from source code it is the folder you
run the command in, so start it from the unpacked archive — that is where those
two folders belong.

That folder must be **writable**. Unpack the archive somewhere you own — your
home directory is the safe answer — and not into `C:\Program Files`, into
`/Applications`, into a folder of another user, or onto a read-only disk. The
program will let you know before a search starts if it is wrong.
