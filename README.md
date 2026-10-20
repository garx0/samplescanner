# SampleScanner

(Micro)sample searcher for local libraries. Finds which track of your music
collection a short audio fragment was taken from, with timestamp and pitch.

## 1a. Installation (easy way, Windows only)

Download the latest Windows build from the [releases page](https://github.com/garx0/samplescanner/releases).

Download **ffmpeg** from <https://ffmpeg.org/download.html> (the `gyan.dev` builds
recommended for Windows), unpack the archive and put **`ffmpeg.exe` and
`ffprobe.exe` next to `SampleScanner.exe`** (or add its `bin` folder to
`PATH` environment variable). SampleScanner uses ffmpeg to read and write audio.

No other installation needed, the program is portable. Just place the
`SampleScanner` folder anywhere you like and the program will "live" inside it
and write files only inside it.

## 1b. Installation from source code (macOS, Linux, Windows) (this installation method was not tested!)

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

## 2. Running it

To run SampleScanner with interface, run `SampleScanner.exe`. Pick
what to look for, pick where to look, configure pitches to check, press
"Start the search".
(if using source code version, replace `SampleScanner.exe` with
`python samplescanner_gui.py` on Windows / `python3 samplescanner_gui.py`
on macOS/Linux).

To run SampleScanner in command line, run the following command, specifying
path to query file/queries folder (or txt file), sources folder (or txt file),
and pitch range and step:

```
SampleScanner.exe -q "<queries>" -s "<sources>" --pitch=<MIN>:<MAX>:<STEP> -v
```

(if using source code version, replace `SampleScanner.exe` with
`python samplescanner_gui.py` on Windows / `python3 samplescanner_gui.py`
on macOS/Linux).

Use case: start new search (with one or multiple samples to find in the selected
library) and play the wav files (with the finds worth checking) appearing in the
results/... folder while the search progresses.

SampleScanner will take extra time indexing (caching) the files it sees for the
first time, so the next searches in the same folders will be faster (unless you
clear the `cache` folder). The index takes 4.2 KBytes per second of audio
(33.6 kbps), containing aggregated features of audio spectrogram. The index
(cache) is used on scanning stage, then the actual audio files are used on
"verifying" stage.

**What may be picked**, for both the query and the library:

* one or several audio files (Ctrl and Shift pick several at once);
* one or several folders (Ctrl and Shift pick several at once);
* a **text file** with a list of files/folders/glob patterns, one entry per line.
  Blank lines and everything after a `#` are ignored. A relative entry counts
  from the folder of that text file, so a list can be moved together with the
  collection it describes. To pick a text file in the window, switch the file type
  in the dialog from `Audio` to `Text`.

## Where it writes

SampleScanner only writes into the folder the program is started from: into
`cache` (the cache it builds once per collection) and `results` (one folder per
run, holding a `log_….txt` with everything the run printed, and the audio
fragments it cut out). With the ready exe that folder is the one the exe is in;
from source code it is the folder you run the command in, so start it from the
unpacked archive — that is where those two folders belong.

That folder must be **writable**. Don't unpack the archive into
`C:\Program Files`, `/Applications`, a folder of another user, or onto a
read-only disk (you will receive an error before the search starts if the folder
isn't writable).

## How the search works

Two passes: a cheap one over an index of the whole collection, and an exact one
over a handful of places it points at (the actual audio).

**Preparation.** Every file — the collection and the query alike — is decoded to 
**22050 Hz mono**. The query is trimmed of silence and resampled by
the pitch shift.

**The index** (built once per collection, kept on disk next to the program). For
each file: a **log-mel spectrogram in 48 bands**, one frame every 23 ms, plus the
**energy of each frame**. Two small `.npy` files per track — the whole index of
a large collection is a couple of gigabytes (33.6 kbps) instead of the dozens 
of gigabytes the audio itself takes.

**The coarse pass.** The query's log-mel profile, normalised, is correlated
against the profiles of the indexed files; a check on the frame energy throws
away the windows that plainly do not sound like the query. The peaks of that
correlation are the candidates: at most a few places per file, and **24
candidates in total** go on to the next pass. This pass never touches the audio.

**The precise pass.** For those 24 places the original files are re-read around
the candidate position, the full feature bank is computed (log-mel at several
resolutions, one frame every 5.8 ms), and the query is compared with what was
read by **cosine similarity**. That is the `r` in the report.

**What the answer is made of.** Three numbers:

- **`r`** — the cosine similarity, 0 to 1. Real matches sit around 0.45-0.9;
- **`frames`** — the share of the query that is really there. The query is cut
  into frames of 5.8 ms, and a frame counts if the sound in the source at that
  moment looks like that frame (correlation above 0.5). It answers "how much of
  the query is accounted for": `r` can be high while only half of the query
  matches;
- **`margin`** — how much better the winner is than the runner-up: the best
  other moment *in the same file*, and the best moment *in the other files*.
  Repeats of the same fragment and equivalent copies of one recording are not
  counted as competitors, or the margin would always come out zero.

The verdict is decided by those three: **confident** when `r` ≥ 0.55 with
margin ≥ 0.12 and frames ≥ 0.15, or when the margin is decisive (≥ 0.25) and
`r` ≥ 0.45 even if few frames are explained; **probable** from `r` ≥ 0.42;
**weak match** from `r` ≥ 0.32; below that the source is most likely not in the
collection. The fragment is cut out at a lower bar than that — `r` ≥ 0.38 and
margin ≥ 0.08 — because having the audio to listen to is useful even when the
program is not sure.

**Pitch.** The query is tried at every shift in the range, and each shift is a
query of its own: whole semitones first, then the fractions (if step smaller 
than 1 is selected), so that a sample which is not at a whole semitone is still 
found, but only after the likely places have been checked.
