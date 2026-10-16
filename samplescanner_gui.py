#!/usr/bin/env python3
"""
Graphical front end for samplescanner.py — a front door, not a replacement.

One file, two ways to run it:

  SampleScanner.exe                      the window
  SampleScanner.exe -q ... -s ... -v     the program, exactly as in a terminal
  python samplescanner_gui.py             the same split, from a source checkout

The window picks the arguments, builds exactly the command line a user would
type in a terminal and runs it in a separate process, streaming the output as
it arrives. The searching itself is done by samplescanner.py alone, so anything
that can be done from a command line can still be done — the GUI only builds
the command line, which is printed in the run window and can be copied out of
it and used as it is.

Only tkinter is used: it comes with python itself, nothing to install.
"""
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if not getattr(sys, "frozen", False):
    # frozen into an exe, the program lives next to the exe: __file__ then
    # points into the unpack folder, which is not where anything may be written
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def hide_own_console() -> None:
    """
    The exe is a CONSOLE program, on purpose: a windowed one is not waited for
    by the shell that started it, and Ctrl+C does not reach it — the command
    would return at once and the report would keep running in the background,
    unstoppable. A console program fixes both.

    Started from Explorer, though, it is handed a console nobody asked for, and
    that one sits behind the window for as long as the window is open. The
    giveaway is who owns the console window: a console that was created FOR this
    process belongs to this process, and there is nobody to hide it from — while
    a console the parent shell already had belongs to the shell, and that is the
    user's own terminal, to be left alone.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        window = ctypes.windll.kernel32.GetConsoleWindow()
        if not window:
            return
        owner = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
        if owner.value == os.getpid():
            ctypes.windll.user32.ShowWindow(window, 0)        # SW_HIDE
    except Exception:
        pass


# One file, two ways to run it: with arguments it is the program itself — the
# same main() that python samplescanner.py calls — and without them it is the
# window that builds the arguments for it.
if len(sys.argv) > 1:
    from samplescanner import main                             # noqa: E402
    sys.exit(main(sys.argv[1:]))
hide_own_console()

# mirrors AUDIO_EXTENSIONS of ssc.config; a copy rather than an import,
# because loading samplescanner here would pull numpy and scipy into the window
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac",
                    ".opus", ".wma", ".aiff", ".aif", ".ape", ".mp4", ".wv")

PITCH_MIN, PITCH_MAX = -24, 24
PITCH_STEPS = ("1", "0.5", "0.25", "0.125", "0.0625")
PITCH_ORDERS = ("finer", "symm")      # "finer" tries whole semitones first

# How wide a line of the hint under a picker may grow. What may be picked is a
# whole paragraph, so it goes under the title and wraps instead of running off
# the edge of a narrow screen. It has to stay well under the 780 pixels
# _fit_to_screen insists on: the width of the window follows the widest thing in
# it, and a hint that wrapped at 900 would make every window 940 wide.
HINT_WRAP = 600

# The shifts themselves are described by the search itself, not by a second
# copy here: ssc.pitch is pure python (the resampling lives in ssc.audio), so
# this costs nothing and the lines under the sliders cannot drift away from what
# the search will really try.
from ssc.config import DEFAULT_BATCH                                      # noqa: E402
from ssc.format import exe_name, fmt_elapsed
from ssc.launcher import launcher_argv, launcher_dir                 # noqa: E402
from ssc.pitch import (count_pitches, describe_pitch_order,          # noqa: E402
                       describe_pitch_set)
from ssc.status import is_status, plain
from ssc.version import version_line

MONO = "Consolas" if os.name == "nt" else "DejaVu Sans Mono"


# The first word of the line the program prints when it wants an answer from
# the window that started it. Must match CONFIRM_MARKER in samplescanner.py.
CONFIRM_MARKER = "CONFIRM    :"

FFMPEG_HELP = (
    "The program needs ffmpeg AND ffprobe to read audio files, and neither "
    "was found.\n"
    "\n"
    "  1. Download both from https://ffmpeg.org/download.html\n"
    + ("     (on Windows the gyan.dev builds are the painless ones — they\n"
       f"      contain {exe_name('ffmpeg')} and {exe_name('ffprobe')} in one\n"
       "      bin folder)\n"
       if os.name == "nt" else
       "     (on macOS: brew install ffmpeg. On Linux: apt install ffmpeg,\n"
       "      or a static build from the same page)\n")
    + f"  2. Unpack the archive and put BOTH {exe_name('ffmpeg')} AND "
      f"{exe_name('ffprobe')}\n"
      "     NEXT TO THIS PROGRAM — in the same folder as the program file —\n"
      "     or add the bin folder of the archive to PATH.\n"
    "\n"
    "Nothing else has to be installed. When both are in place, press "
    "\"New search\" again.")

# ffmpeg decodes the audio, ffprobe reads what a file declares about itself.
# Without it the search still runs, but the fragments it cuts out can come out
# wrong — so it is worth one question before a long search starts.
FFPROBE_QUESTION = (
    "ffprobe was not found, only ffmpeg.\n"
    "\n"
    "Without ffprobe the found samples may be exported incorrectly.\n"
    "\n"
    f"Put {exe_name('ffprobe')} next to the program file (it comes with "
    "ffmpeg) and press\n"
    "\"New search\" again to search now.\n"
    "\n"
    "Start the search without ffprobe anyway?")

WRITE_HELP = (
    "The program cannot write into the folder it is in:\n\n"
    "{path}\n\n"
    "Its results and its index go there, so it cannot work from where it is "
    "now.\n\nMove the whole program folder somewhere you own — your Documents, "
    "for example — and start it again.")

HELP_TEXT = """\
Start a search by pressing "New search" on the main window, then

- pick "the query" — what to look for: one file or a folder; several items can
  be picked at once with Ctrl and Shift; subfolders are read too, and anything
  that is not audio is skipped. You can also pick a text file with a list of
  files and folders (or glob patterns), one per line — switch the file type in
  the dialog from "Audio" to "Text".
- pick "the sources" — where to look: the same way, files, folders, glob
  patterns or a text file with a list of them, as many as you like.
- set the pitch range with the two handles on the scale, pick the step and pick
  the order. If you leave both handles at 0, the query is searched as it is.
  Otherwise every query is repitched by the values from <MIN> to <MAX>
  semitones, <STEP> apart.

A relative path in a text list counts from the folder that list is in, so a list
of a collection can be moved around together with it.

The line at the bottom of the form is the command that will be run, always up to
date: everything picked is in it as it is, a text list included.

Everything typed in the "extra options" field goes into the command line as it
is, for anything that has no field of its own.

The window that opens streams the full report of the search; it can be
stopped, saved and scrolled. The line VERDICT says how convincing the answer
is: FOUND (confident) — the fragment was found; FOUND (probable) — found, but
something similar was almost as good; WEAK MATCH and NOT FOUND — most likely
not it, worth listening to what it cut out.

Confident and non-confident results for each query can be found as audio
files in `results/<query>__<sources>__<date and time>` folder and its
subfolders (the weak ones in `unsure`). The button "Open results folder" in
the search window shows this folder.

The program needs ffmpeg AND ffprobe to read audio files (they come together,
search up how to download them). If they are not found, this window says so
before anything has to be filled in — put ffmpeg AND ffprobe next to the
program or add their folder to PATH, then press "New search" again. If ffmpeg is
there but ffprobe is not, the window asks whether to search anyway: without
ffprobe the found samples may be exported incorrectly.

First search on a sources folder will take extra time to make an index from it
(aka fingerprints), which will occupy extra space on disk (less than the audio
files themselves, 4 kb per second) in `cache` folder next to the program. Next
searches in the same sources folder will be faster as long as the corresponding
directory on disk exists (otherwise it will be slow again). If you don't need
these folders inside `cache` folder anymore, remove them with "Clear the cache"
on the main window — or manually.

Source audio for searching is converted to 22050 hz with both channels mixed
to mono.

Order of pitches: by default ("finer" order) it circles around 0st, first with
1st step, then with twice smaller step, and so on, up to the chosen step.
The "symm" order just circles around 0 with the chosen step.

You can also run the exe file in command line instead of using the GUI:

  SampleScanner.exe -q "<queries>" -s "<sources>" --pitch=<MIN>:<MAX>:<STEP> -v

The folder this program is in must be writable: `cache` and `results` are
created in it. If it is not, the program says so before a search starts and
asks to be moved somewhere you own.

Nothing is uploaded anywhere: the search runs on this computer only.
"""


def program_dir() -> str:
    """
    The folder the program writes into: the folder of the launcher when the
    program was packed into one — a portable program unpacked anywhere keeps its
    cache and its results next to itself — and the folder of the script otherwise.
    """
    return launcher_dir() or HERE


def is_writable(path: str) -> bool:
    try:
        return os.path.isdir(path) and os.access(path, os.W_OK)
    except OSError:
        return False


def find_ffmpeg():
    """
    The same places samplescanner.py looks for ffmpeg, without importing it (that
    would load numpy and scipy into the window). Returns the path, or None.

    The names looked for beside the program are the ones this system gives them
    — ffmpeg.exe on Windows, plain ffmpeg on macOS and Linux — so a build
    unpacked next to the program is found on all three. Looking only for the
    .exe would have the window claim that ffmpeg is missing while it stands
    right beside it, and there is no worse first impression than that.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    name = exe_name("ffmpeg")
    for root in (program_dir(), HERE):
        for cand in (os.path.join(root, "ffmpeg", "bin", name),
                     os.path.join(root, name)):
            if os.path.isfile(cand):
                return cand
    return None


def find_ffprobe():
    """
    ffprobe next to the ffmpeg that was found, in the same places. Both come
    from the same archive and sit side by side, so looking beside ffmpeg is
    enough — and it is the same rule the program itself uses (ffprobe_bin).
    Returns the path, or None.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        beside = os.path.join(os.path.dirname(ffmpeg), exe_name("ffprobe"))
        if os.path.isfile(beside):
            return beside
    return shutil.which("ffprobe")


def fmt_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} GB"


def open_in_explorer(path: str) -> None:
    """Shows a folder in the file manager of the system."""
    if not os.path.exists(path):
        messagebox.showinfo("Not there",
                            f"{path}\n\nThis folder does not exist.")
        return
    try:
        if os.name == "nt":
            os.startfile(path)                          # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as exc:
        messagebox.showwarning("Cannot open", f"{path}\n\n{exc}")


def child_argv() -> list:
    """
    The command that runs the program in a separate process, the way the window
    has always run it: a search is minutes of solid work, and in its own
    process the window keeps answering, and the "Stop" button really stops it.
    Packed into one launcher the launcher itself is both the window and the
    program, and its arguments decide which of the two this run is.
    """
    return launcher_argv()


def show_argv(argv: list) -> str:
    """
    The command as a person would type it, for the line at the top of the run
    window and the preview under the form.

    Quoting is the quoting of the machine the window runs on: list2cmdline for
    Windows, shlex.join everywhere else. The line is there to be read — and
    copied out and used as it is, which the module docstring promises — and a
    command written for one shell and pasted into the other is a command that
    does not do what it says.
    """
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    import shlex
    return shlex.join(argv)


def elide_path(path: str, limit: int = 56) -> str:
    """A long path shown on one line: the beginning and the end, which is where
    the drive and the program folder are, and no more than `limit` characters."""
    if len(path) <= limit:
        return path
    keep = limit - 3
    return path[:keep // 2] + "..." + path[-(keep - keep // 2):]



# --------------------------------------------------------------------------- #
# The run window: the output of the search as it arrives
# --------------------------------------------------------------------------- #


class RunWindow(tk.Toplevel):
    """
    Shows stdout of the search while it runs. The lines are read in a thread and
    handed over through a queue — tkinter may only be touched from the thread it
    was started in.
    """
    def __init__(self, master, argv, cwd):
        super().__init__(master)
        self.title(f"Search running — {version_line()}")
        self.geometry("1000x640")
        self.minsize(640, 400)
        self.proc = None
        self.queue: queue.Queue = queue.Queue()
        self.results_dir = None
        self.finished = False
        self.silent = 0

        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text="Command:").pack(side=tk.LEFT)
        cmd = ttk.Entry(top)
        cmd.insert(0, show_argv(argv))
        cmd.configure(state="readonly")
        cmd.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(fill=tk.X)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop)
        self.stop_btn.pack(side=tk.LEFT)
        self.open_btn = ttk.Button(bar, text="Open results folder",
                                   command=lambda: open_in_explorer(
                                       self.results_dir or ""),
                                   state="disabled")
        self.open_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Save output…",
                   command=self.save).pack(side=tk.LEFT)
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Follow the output",
                        variable=self.follow).pack(side=tk.LEFT, padx=12)
        ttk.Button(bar, text="Close", command=self.on_close).pack(side=tk.RIGHT)

        wrap = ttk.Frame(self, padding=(8, 0, 8, 8))
        wrap.pack(fill=tk.BOTH, expand=True)
        self.text = tk.Text(wrap, wrap="none", font=(MONO, 9),
                            state="disabled", bg="#101418", fg="#dfe6ec",
                            insertbackground="#dfe6ec")
        vs = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.text.yview)
        hs = ttk.Scrollbar(wrap, orient=tk.HORIZONTAL, command=self.text.xview)
        self.text.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.text.tag_configure("good", foreground="#7ee08a")
        self.text.tag_configure("ok", foreground="#d7e37a")
        self.text.tag_configure("weak", foreground="#ffb454")
        self.text.tag_configure("bad", foreground="#ff7b72")
        self.text.tag_configure("head", foreground="#79c0ff")

        self.status = tk.StringVar(value="starting…")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 3)).pack(fill=tk.X, side=tk.BOTTOM)

        # What the status line shows while the search runs: how long it has been
        # going, and what it is doing now. The second part is NOT guessed here
        # from the text of the report — the program says it, by sending a status
        # line (ssc.status.flash), and all this window does is show whatever
        # arrived last. A line that only ever says "starting…" tells a person
        # nothing for the minutes a big collection takes, which is exactly when
        # they want to know it is alive.
        self._flash = ""
        self._lines = 0
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._start(argv, cwd)
        self.after(60, self._poll)
        self.after(1000, self._tick)

    def _tick(self):
        if self.finished:
            return
        secs = fmt_elapsed(time.perf_counter() - self.t0)
        what = self._flash
        self.status.set(f"running · {secs} · {self._lines} lines"
                        + (f" · {what}" if what else ""))
        self.after(1000, self._tick)

    # ---------- the process ----------

    def _start(self, argv, cwd):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" \
            else 0
        try:
            self.proc = subprocess.Popen(
                argv, cwd=cwd, env=env,
                stdin=subprocess.PIPE,      # the only way back in: see _confirm
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=flags)
        except OSError as exc:
            self._put(f"Error: cannot start the search: {exc}", "bad")
            self._done(1)
            return
        self.t0 = time.perf_counter()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        proc = self.proc
        try:
            for line in proc.stdout:
                self.queue.put(line.rstrip("\n"))
        except (ValueError, OSError):
            pass
        finally:
            try:
                proc.wait()
            except OSError:
                pass
            self.queue.put(None)              # end of the stream

    def _poll(self):
        while True:
            try:
                line = self.queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self._done()
                return
            if is_status(line):
                # what the program is doing NOW: it goes to the status line and
                # not into the report, where it would only be something to
                # scroll past once the moment it describes is over
                self._flash = plain(line)
                continue
            self._put(line, self._tag_for(line))
            self._lines += 1
            if line.startswith(CONFIRM_MARKER):
                self._answer_confirmation(line)
            if line.startswith("RESULTS   :"):
                # the line says "not written" when nothing is to be cut —
                # only a folder that is really there may be opened
                path = line.split(":", 1)[1].strip()
                if os.path.isdir(path):
                    self.results_dir = path
                    self.open_btn.configure(state="normal")
        # the process is gone; if the stream has said nothing for half a second
        # more it never will, so the run is over either way
        if self.proc is not None and self.proc.poll() is not None:
            self.silent += 1
            if self.silent > 5:
                self._done()
                return
        else:
            self.silent = 0
        self.after(120, self._poll)

    def _answer_confirmation(self, line: str):
        """
        The program stopped and asked something — so far only how large the
        index of the collection may be. The question goes into a dialog here,
        and the answer goes back the one way there is to the program: a line on
        its stdin.
        """
        question = line.split(":", 1)[1].strip()
        question = question.split(" [y/N]")[0].strip()
        yes = messagebox.askyesno("A large index", question + "\n\n"
                                  "Building it will take a while and the disk "
                                  "space. Build it anyway?",
                                  parent=self)
        self._write_answer("y" if yes else "n")

    def _write_answer(self, answer: str):
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.write(answer + "\n")
                self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def _tag_for(self, line: str) -> str:
        if line.lstrip().startswith("VERDICT"):
            if "confident" in line:
                return "good"
            if "probable" in line:
                return "ok"
            return "weak"
        if "WEAK MATCH" in line or line.lstrip().startswith("NOT FOUND"):
            return "weak"
        if line.startswith("Error:") or " ERROR " in line:
            return "bad"
        # The name of the query in the REPORT is coloured; the one printed
        # while the search is still running is not — that one is a step of a
        # longer process, and colouring it the same would mark it as an answer.
        if line.startswith(("QUERY  :", "SOURCE  :", "INDEX", "Pitch", "=====",
                            "Summary", "TOTAL")):
            return "head"
        return ""

    def _put(self, line: str, tag: str = ""):
        self.text.configure(state="normal")
        self.text.insert(tk.END, line + "\n", tag or ())
        self.text.configure(state="disabled")
        if self.follow.get():
            self.text.see(tk.END)

    def _done(self, code=None):
        if self.finished:
            return
        self.finished = True
        if code is None and self.proc is not None:
            code = self.proc.poll()
        self.stop_btn.configure(state="disabled")
        secs = time.perf_counter() - getattr(self, "t0", time.perf_counter())
        self._put("")
        self.status.set(f"finished in {secs:.1f} s"
                        + (f", exit code {code}" if code else ""))
        if self.results_dir:
            self._put(f"\nThe fragments are in {self.results_dir}",
                      "head")

    # ---------- buttons ----------

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self._put("\nstopped by the user", "bad")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def save(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Save the output", defaultextension=".txt",
            initialfile="samplescanner_log.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.text.get("1.0", tk.END))
        except OSError as exc:
            messagebox.showwarning("Cannot save", str(exc))

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("The search is still running",
                                       "Stop the search and close?"):
                return
            self.stop()
        self.destroy()


# --------------------------------------------------------------------------- #
# The search window: what to look for, where, and at which pitches
# --------------------------------------------------------------------------- #


class Tip:
    """
    The explanation of a control, shown while the pointer rests on it.

    A window of its own, not text under the control: what these options have to
    say is several lines long, and printed in the form it would take more room
    than the control it belongs to. The pointer is over a small "?" next to it,
    which is why the text is not in the form at all.

    Several tips may be attached to the same control: they share one window and
    only one of them is ever shown.
    """
    _live: Optional["Tip"] = None

    def __init__(self, widgets: list, text: str) -> None:
        self.text = text
        self.win: Optional[tk.Toplevel] = None
        for w in widgets:
            w.bind("<Enter>", self._enter, add="+")
            w.bind("<Leave>", self._leave, add="+")
            w.bind("<ButtonPress>", self._leave, add="+")

    def attach(self, *widgets) -> "Tip":
        for w in widgets:
            w.bind("<Enter>", self._enter, add="+")
            w.bind("<Leave>", self._leave, add="+")
            w.bind("<ButtonPress>", self._leave, add="+")
        return self

    def _enter(self, event) -> None:
        # the pointer crossing from the control onto its own "?" must not close
        # the window it just opened
        if self.win is not None and self.win.winfo_exists():
            return
        Tip._live = self
        x = event.widget.winfo_rootx() + 16
        y = event.widget.winfo_rooty() + event.widget.winfo_height() + 6
        self.win = tk.Toplevel()
        self.win.wm_overrideredirect(True)
        try:
            self.win.attributes("-topmost", True)
        except tk.TclError:
            pass
        ttk.Label(self.win, text=self.text, justify=tk.LEFT, padding=8,
                  wraplength=430, background="#fffbe8", foreground="#222",
                  relief="solid", borderwidth=1).pack()
        self.win.update_idletasks()
        w, h = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        screen_w, screen_h = self.win.winfo_screenwidth(), \
            self.win.winfo_screenheight()
        x = min(x, max(8, screen_w - w - 12))
        # below the control when there is room, above it when there is not
        if y + h > screen_h - 12:
            y = max(8, event.widget.winfo_rooty() - h - 6)
        self.win.geometry(f"+{x}+{y}")

    def _leave(self, _event) -> None:
        if Tip._live is not self:
            return
        if self.win is not None:
            self.win.destroy()
            self.win = None
        Tip._live = None


def tip_mark(parent) -> ttk.Label:
    """
    The small "?" an explanation hangs on. It is created, not placed: a frame
    inside another one is packed and a bare child is gridded, and the caller
    knows which of the two it is dealing with.
    """
    return ttk.Label(parent, text=" ?", foreground="#06c", cursor="hand2",
                     font=(None, 9, "bold"))


class ScrollFrame(ttk.Frame):
    """
    A frame whose content scrolls when the window is made smaller than it.

    The content is put on a canvas, so the window may shrink below the natural
    height of everything inside it instead of cutting the bottom off — the
    buttons that start the search then stay reachable at any size.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient=tk.VERTICAL,
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.bar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner,
                                                  anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self._bind_wheel()

    def _on_inner(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._sync_bar()

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)
        self._sync_bar()

    def _on_scroll_set(self, first, last):
        self.bar.set(first, last)
        self._sync_bar()

    def _sync_bar(self):
        """
        The bar is shown while the content is TALLER THAN THE WINDOW, and that
        is the only thing it looks at.

        It used to ask the scroll position instead — "is the bottom of the
        content on screen?" — which is wrong: Tk reports `last` as exactly 1.0
        whenever the view is scrolled to the end, however much content there
        still is below the top edge. So the bar disappeared the moment the
        wheel reached the bottom, and with the width freed by the hidden bar the
        content could fit after all, and then nothing brought it back.
        """
        region = self.canvas.cget("scrollregion").split()
        try:
            top, bottom = float(region[1]), float(region[3])
        except (IndexError, ValueError):
            return
        needed = bottom - top > self.canvas.winfo_height() + 1
        if needed and not self.bar.winfo_ismapped():
            self.bar.grid()
        elif not needed and self.bar.winfo_ismapped():
            self.bar.grid_remove()

    def _bind_wheel(self, widget=None):
        """The wheel over any part of the content scrolls the content."""
        (widget or self).bind("<MouseWheel>", self._wheel)
        for child in (widget or self).winfo_children():
            self._bind_wheel(child)

    def _wheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = getattr(widget, "master", None)
        if widget is self:
            self.canvas.yview_scroll(-1 * (event.delta // 120), "units")


class PathPicker(ttk.Frame):
    """A list of picked files/folders with the four buttons that change it."""
    def __init__(self, master, title, hint="", on_change=None):
        super().__init__(master, padding=(0, 4, 0, 0))
        self.paths: list = []
        # what to call after the list changed: the window below shows the whole
        # command line, and a preview that still reads "-q -s -v" while a file
        # sits in the list says nothing about what is about to be searched
        self.on_change = on_change
        head = ttk.Frame(self)
        head.pack(fill=tk.X)
        ttk.Label(head, text=title).pack(side=tk.LEFT)
        if hint:
            # What may be picked is a whole paragraph, and a paragraph does not
            # go next to the title: side by side it pushes the edge of the
            # window out to wherever the text ends. Under the title it wraps,
            # and the window stays the size it has always been.
            self.hint = ttk.Label(self, text=hint, foreground="#666",
                                  anchor=tk.W, justify=tk.LEFT,
                                  wraplength=HINT_WRAP)
            self.hint.pack(fill=tk.X, pady=(1, 0))
            self.bind("<Configure>", self._rewrap)
        box = ttk.Frame(self)
        box.pack(fill=tk.BOTH, expand=True, pady=(3, 3))
        self.listbox = tk.Listbox(box, height=3, selectmode=tk.EXTENDED,
                                  font=(MONO, 9), activestyle="none")
        sb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Add files…",
                   command=self.add_files).pack(side=tk.LEFT)
        ttk.Button(btns, text="Add folder…",
                   command=self.add_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Remove",
                   command=self.remove).pack(side=tk.LEFT)
        ttk.Button(btns, text="Clear", command=self.clear).pack(side=tk.LEFT,
                                                                 padx=4)

    def _add(self, items):
        changed = False
        for item in items:
            if item not in self.paths:
                self.paths.append(item)
                self.listbox.insert(tk.END, item)
                changed = True
        if changed and self.on_change:
            self.on_change()

    def _rewrap(self, _event=None):
        """
        Narrows the hint when the picker has become narrower than the line in
        it, so the last words are never cut off on a small screen.

        It only ever narrows. Letting it widen as well would be a loop: the
        width of the window follows the width of its widest part, so a hint that
        asks for more room than it has gets the room and then asks for more.
        """
        width = self.winfo_width()
        if width > 80 and self.hint.cget("wraplength") > width + 8:
            self.hint.configure(wraplength=width)

    def add_files(self):
        types = [("Audio", " ".join(f"*{e}" for e in AUDIO_EXTENSIONS)),
                 ("Text", " ".join(f"*{e}" for e in [".txt"])),
                 ("All files", "*.*")]
        chosen = filedialog.askopenfilenames(
            parent=self.winfo_toplevel(),
            # the title of the dialog is the last place a person looks before
            # giving up on the "Text" filter, and it is the one that is read
            title="Pick audio files — or a .txt file with a list of them",
            filetypes=types)
        if chosen:
            self._add(list(chosen))

    def add_folder(self):
        chosen = filedialog.askdirectory(parent=self.winfo_toplevel(),
                                         title="Pick a folder",
                                         mustexist=True)
        if chosen:
            self._add([chosen])

    def remove(self):
        picked = list(reversed(self.listbox.curselection()))
        for i in picked:
            del self.paths[i]
            self.listbox.delete(i)
        if picked and self.on_change:
            self.on_change()

    def clear(self):
        if not self.paths:
            return
        self.paths.clear()
        self.listbox.delete(0, tk.END)
        if self.on_change:
            self.on_change()

    def __bool__(self):
        return bool(self.paths)


class SearchDialog(tk.Toplevel):
    """
    Collects the arguments of one search and then starts it.

    The content is inside a ScrollFrame and the buttons live outside it, so a
    short screen or a small window costs a scrollbar and never a button that
    cannot be reached.
    """

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("New search")
        self.transient(master)

        scroll = ScrollFrame(self)
        scroll.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))
        self.scroll = scroll
        body = scroll.inner

        self.q_pick = PathPicker(
            body, "1.  What to look for  (the query)",
            "files or folders (or text file with list of files/folders/glob "
            "patterns, one per line); Ctrl and Shift pick several",
            on_change=self._refresh_cmdline)
        self.q_pick.pack(fill=tk.X)
        self.s_pick = PathPicker(
            body, "2.  Where to look  (the sources)",
            "files or folders (or text file with list of files/folders/glob "
            "patterns, one per line); Ctrl and Shift pick several",
            on_change=self._refresh_cmdline)
        self.s_pick.pack(fill=tk.X)

        pitch = ttk.LabelFrame(body, text="3.  Pitch range, in semitones "
                               "(0 = as it is)", padding=6)
        pitch.pack(fill=tk.X, pady=(8, 0))
        # two handles, one scale: the same range and the same ticks under both
        self.sliders: list = []
        self.low = self._slider(pitch, "from", 1)
        self.high = self._slider(pitch, "to", 2)
        ticks = ttk.Frame(pitch)
        ticks.grid(row=3, column=0, columnspan=3, sticky="ew")
        for i in range(13):
            ticks.columnconfigure(i, weight=1)
            ttk.Label(ticks, text=f"{4 * i + PITCH_MIN:+d}",
                      foreground="#666").grid(row=0, column=i)
        row = ttk.Frame(pitch)
        row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Label(row, text="step:").pack(side=tk.LEFT)
        self.step_var = tk.StringVar(value="1")
        self.step = ttk.Combobox(row, textvariable=self.step_var,
                                 values=list(PITCH_STEPS), width=7,
                                 state="readonly")
        self.step.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(row, text="order:").pack(side=tk.LEFT)
        self.order_var = tk.StringVar(value="finer")
        self.order = ttk.Combobox(row, textvariable=self.order_var,
                                  values=list(PITCH_ORDERS), width=7,
                                  state="readonly")
        self.order.pack(side=tk.LEFT, padx=(4, 12))
        # the shifts themselves: the set first, then the pattern they are tried in —
        # the pattern belongs to the step alone, so it never moves with the ends
        self.pitch_label = ttk.Label(pitch, text="", foreground="#0a5",
                                     anchor=tk.W)
        self.pitch_label.grid(row=5, column=0, columnspan=3, sticky="w",
                              pady=(4, 0))
        self.order_label = ttk.Label(pitch, foreground="#0a5", anchor=tk.W)
        self.order_label.grid(row=6, column=0, columnspan=3, sticky="w",
                               pady=(2, 0))
        # the callbacks of the widgets, not variable traces: several traces on
        # variables of one window can end up sharing one name inside tkinter and
        # overwrite each other, which silently kills all but the last one
        self.step.bind("<<ComboboxSelected>>",
                       lambda e: self._update_pitch())
        self.order.bind("<<ComboboxSelected>>",
                        lambda e: self._update_pitch())

        opts = ttk.LabelFrame(body, text="Options", padding=6)
        opts.pack(fill=tk.X, pady=(8, 0))
        self.nonlazy = tk.BooleanVar()
        self.noexport = tk.BooleanVar()
        self.batch = tk.IntVar(value=DEFAULT_BATCH)

        # How the library is walked is two decisions, and what each of them
        # costs is more than fits here: both hang off the small "?" and come on
        # hover - the control itself answers to the pointer as well.
        batch_row = ttk.Frame(opts)
        batch_row.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(batch_row, text="Batch size:").pack(side=tk.LEFT)
        self.batch_spin = ttk.Spinbox(
            batch_row, from_=1, to=99, width=5, textvariable=self.batch,
            command=self._refresh_cmdline)
        self.batch_spin.pack(side=tk.LEFT, padx=(4, 0))
        mark = tip_mark(batch_row)
        mark.pack(side=tk.LEFT, padx=(2, 0))
        Tip([batch_row, self.batch_spin, mark],
            "Process queries in batches. Runs faster in total, but intermediate "
            "results report will be updated less often (you'll see results "
            "for individual queries only after the whole batch is processed).")

        lazy_row = ttk.Frame(opts)
        lazy_row.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.lazymode = ttk.Checkbutton(
            lazy_row,
            text="Non-lazy mode: try every pitch shift, even after a "
                 "confident one", variable=self.nonlazy)
        self.lazymode.pack(side=tk.LEFT)
        mark = tip_mark(lazy_row)
        mark.pack(side=tk.LEFT, padx=(2, 0))
        Tip([self.lazymode, lazy_row, mark],
            "Processes more pitches but isn't actually that much slower "
            "(could even be faster)")
        ttk.Checkbutton(opts, text="do not cut the fragments to files",
                        variable=self.noexport).grid(row=1, column=1,
                                                      sticky="w", pady=(6, 0))
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="extra options:").grid(row=2, column=0, sticky="w",
                                                    pady=(6, 0))
        self.extra = ttk.Entry(opts, font=(MONO, 9))
        self.extra.grid(row=2, column=1, sticky="ew", pady=(6, 0), padx=(6, 0))

        bottom = ttk.Frame(self, padding=(8, 8, 8, 8))
        bottom.pack(fill=tk.X, side=tk.BOTTOM)
        self.bottom = bottom
        self.cmdline = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.cmdline, font=(MONO, 8),
                  foreground="#666", anchor=tk.W).pack(fill=tk.X, pady=(0, 6))
        btns = ttk.Frame(bottom)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Start the search",
                   command=self.start).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel",
                   command=self.destroy).pack(side=tk.RIGHT, padx=6)

        self._update_pitch()
        self.bind("<Return>", lambda e: self.start())
        self._fit_to_screen()

    def _fit_to_screen(self):
        """
        As tall as it wants to be, but never taller than the screen: at a 150 %
        scaling a 1080-pixel screen is 720 points, and a window that does not
        fit cannot be closed with its own buttons. What does not fit is
        scrolled instead — see ScrollFrame.
        """
        self.update_idletasks()
        width = max(self.scroll.inner.winfo_reqwidth(), 780) + 40
        height = (self.scroll.inner.winfo_reqheight()
                  + self.bottom.winfo_reqheight() + 60)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.minsize(min(620, screen_w - 40), min(360, screen_h - 80))
        self.geometry(f"{min(width, screen_w - 40)}"
                      f"x{min(height, screen_h - 80)}")

    def _slider(self, parent, label, row):
        var = tk.IntVar(value=0)
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           padx=(0, 6))
        sc = tk.Scale(parent, from_=PITCH_MIN, to=PITCH_MAX, resolution=1,
                      orient=tk.HORIZONTAL, variable=var, showvalue=True,
                      digits=0, tickinterval=0, length=440,
                      command=lambda value: self._update_pitch())
        sc.grid(row=row, column=1, sticky="ew")
        ttk.Label(parent, text="st").grid(row=row, column=2, sticky="w",
                                          padx=(6, 0))
        self.sliders.append(sc)
        return var

    # ---------- the command line ----------

    def pitch_range(self):
        lo, hi = sorted((self.low.get(), self.high.get()))
        step = float(self.step.get())
        return lo, hi, step

    def _update_pitch(self):
        lo, hi, step = self.pitch_range()
        # the count belongs to this line: it is how many query/shift pairs one
        # run has to answer, and that is what decides the time a run takes - a
        # person moves the handles and has to be able to see what the range now
        # costs
        n = count_pitches(lo, hi, step)
        self.pitch_label.configure(
            text=f"pitches set: {describe_pitch_set(lo, hi, step)}"
                 f" ({n} pitch{'' if n == 1 else 'es'})")
        # a range of 0 to 0 is the query as it is: there is no order to talk
        # about, and a line describing one would only be in the way
        self.order_label.configure(
            text="" if lo == hi == 0 else
            f"pitches order: {describe_pitch_order(step, self.order.get())}")
        self._refresh_cmdline()

    def _refresh_cmdline(self):
        """
        Puts the whole command line below the form, as it would be typed. It is
        the only place the person can see what the picked items turn into — in
        particular that a .txt list is handed over as it is, whole, and not
        opened here.
        """
        self.cmdline.set(show_argv(self.build_argv()))

    def build_argv(self):
        lo, hi, step = self.pitch_range()
        argv = ["-q", *self.q_pick.paths, "-s", *self.s_pick.paths, "-v"]
        argv.append(f"--pitch={lo}" if lo == hi
                    else f"--pitch={lo}:{hi}:{self.step.get()}")
        argv.append(f"--pitchorder={self.order.get()}")
        if self.nonlazy.get():
            argv.append("--nonlazy")
        argv.append(f"--batch={max(1, self.batch.get())}")
        if self.noexport.get():
            argv.append("--no-export")
        # the program must ask before it spends gigabytes on the disk, and the
        # only one who can answer is this window
        argv.append("--confirm-large-index")
        extra = self.extra.get().split()          # no quote handling, by design
        argv.extend(extra)
        return argv

    def start(self):
        if not self.q_pick.paths:
            messagebox.showwarning("Nothing to look for",
                                   "Add at least one query file or folder.",
                                   parent=self)
            return
        if not self.s_pick.paths:
            messagebox.showwarning("Nowhere to look",
                                   "Add at least one search file or folder.",
                                   parent=self)
            return
        missing = [p for p in self.q_pick.paths + self.s_pick.paths
                   if not os.path.exists(p)]
        if missing:
            messagebox.showwarning("Not there",
                                   "These paths do not exist:\n\n"
                                   + "\n".join(missing[:8]), parent=self)
            return
        # the working folder is the folder of the program: there is no field for
        # it any more, and the main window shows where it is
        work = os.path.abspath(self.app.work_dir.get().strip() or HERE)
        try:
            os.makedirs(work, exist_ok=True)
        except OSError as exc:
            messagebox.showwarning("Cannot use that folder", str(exc),
                                   parent=self)
            return
        self.app.work_dir.set(work)
        self.app.open_run_window(self.build_argv(), work)
        self.destroy()


# --------------------------------------------------------------------------- #
# The cache window
# --------------------------------------------------------------------------- #


class CacheWindow(tk.Toplevel):
    """Shows what the cache holds and how much of it is worth deleting."""

    def __init__(self, master, work_dir):
        super().__init__(master)
        self.work_dir = work_dir
        self.title("Cache")
        self.geometry("760x430")
        self.transient(master)

        wrap = ttk.Frame(self, padding=8)
        wrap.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(wrap, font=(MONO, 9), activestyle="none")
        sb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.summary = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.summary, wraplength=720,
                  justify=tk.LEFT, padding=(8, 4)).pack(fill=tk.X)

        bar = ttk.Frame(self, padding=(8, 4, 8, 8))
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side=tk.LEFT)
        ttk.Button(bar, text="Delete the selected",
                   command=self.delete_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Delete all", command=self.delete_all).pack(
            side=tk.LEFT)
        ttk.Button(bar, text="Open the cache folder",
                   command=lambda: open_in_explorer(self.cache_dir())).pack(
            side=tk.RIGHT)
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.refresh()

    def cache_dir(self):
        return os.path.join(self.work_dir.get(), "cache")

    def folders(self):
        """One line per collection: size, file count, when it was last used."""
        base = self.cache_dir()
        if not os.path.isdir(base):
            return []
        out = []
        for name in sorted(os.listdir(base), key=str.lower):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            size = count = 0
            newest = 0.0
            for dirpath, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        st = os.stat(os.path.join(dirpath, f))
                    except OSError:
                        continue
                    size += st.st_size
                    count += 1
                    newest = max(newest, st.st_mtime)
            when = time.strftime("%d.%m.%Y %H:%M",
                                 time.localtime(newest)) if newest else "—"
            out.append((name, size, count, when, path))
        return out

    def refresh(self):
        self.listbox.delete(0, tk.END)
        rows = self.folders()
        for name, size, count, when, _p in rows:
            self.listbox.insert(tk.END, f"{name:<28} {fmt_size(size):>10}   "
                                        f"{count:>5} files   {when}")
        if not rows:
            self.summary.set(f"{self.cache_dir()} does not exist yet — the "
                             f"cache appears after the first search.")
        else:
            self.summary.set(
                f"{len(rows)} collection(s), {fmt_size(sum(r[1] for r in rows))}"
                f" total. Tick the ones to remove with Ctrl or Shift. "
                f"Deleting costs time, not data: the index of a collection is "
                f"built again at its next search.")

    def _delete(self, rows):
        if not rows:
            messagebox.showinfo("Nothing selected",
                                "Tick the collections to delete with Ctrl "
                                "or Shift.", parent=self)
            return
        what = "\n".join(f"  {n}  ({fmt_size(s)})" for n, s, _c, _w, _p in rows)
        total = sum(r[1] for r in rows)
        if not messagebox.askyesno(
                "Delete from the cache?",
                f"{what}\n\n{total and fmt_size(total)} will be freed. The "
                f"next search over them is slower — the index is built again. "
                f"\n\nDelete?", parent=self):
            return
        for _n, _s, _c, _w, path in rows:
            shutil.rmtree(path, ignore_errors=True)
        self.refresh()

    def delete_selected(self):
        rows = self.folders()
        self._delete([rows[i] for i in self.listbox.curselection()
                      if 0 <= i < len(rows)])

    def delete_all(self):
        self._delete(self.folders())


# --------------------------------------------------------------------------- #
# Help and the main window
# --------------------------------------------------------------------------- #


class HelpWindow(tk.Toplevel):
    """
    The short guide, in the same spirit as README_tldr.md: what to press, what
    to fill in, what comes out of it. The full README is one button away.
    """

    def __init__(self, master):
        super().__init__(master)
        self.title("Help")
        self.geometry("860x680")
        self.minsize(560, 400)
        self.transient(master)
        wrap = ttk.Frame(self, padding=8)
        wrap.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(wrap, wrap=tk.WORD, font=(MONO, 9), padx=14, pady=10,
                       spacing1=2, spacing3=2)
        sb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        # the version is the first line of the help as well: somebody comparing
        # two builds asks exactly that, and the window title is not where they
        # will look for an answer they can copy into a message
        help_text = f"{version_line()}\n\n{HELP_TEXT}"
        text.insert("1.0", help_text)
        text.tag_configure("b", font=(MONO, 9, "bold"))
        for i, line in enumerate(help_text.splitlines(), start=1):
            if line.strip().startswith("python samplescanner.py"):
                text.tag_add("b", f"{i}.0", f"{i}.end")
        text.configure(state="disabled")
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill=tk.X)
        # README.md is the only documentation that travels with the program:
        # the developer one (README_all.md) is not shipped, so it is not looked
        # for here either — the button is there only where a file can be opened
        readme = os.path.join(HERE, "README.md")
        if os.path.exists(readme):
            ttk.Button(bar, text="Open the full documentation",
                       command=lambda: open_in_explorer(readme)).pack(
                           side=tk.LEFT)
        ttk.Button(bar, text="Close", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Label(bar, text="The same program runs from a terminal: "
                   "python samplescanner.py --help", foreground="#666").pack(
            side=tk.LEFT, padx=10)


class App(tk.Tk):
    """The first window: what to do."""

    def __init__(self):
        super().__init__()
        self.title(version_line())
        self.geometry("620x380")
        self.resizable(False, False)
        # a portable program keeps everything in the folder it was unpacked
        # into, next to the exe — never in the working directory of whoever
        # started it
        self.work_dir = tk.StringVar(value=program_dir())

        head = ttk.Frame(self, padding=(24, 22, 24, 6))
        head.pack(fill=tk.X)
        ttk.Label(head, text=version_line(),
                  font=("", 16, "bold")).pack(anchor=tk.W)
        ttk.Label(head, text="Find out which track of your collection a "
                  "short audio fragment came from.",
                  foreground="#555").pack(anchor=tk.W, pady=(6, 0))

        menu = ttk.Frame(self, padding=(24, 12, 24, 12))
        menu.pack(fill=tk.BOTH, expand=True)
        for text, cmd, hint in (
                ("New search", self.new_search,
                 "pick the fragment and the collection to search"),
                ("Help", self.show_help,
                 "how to search, where the results go, etc"),
                ("Clear the cache", self.show_cache,
                 "free the disk space the index of the collections takes (choose folders to delete)")):
            b = ttk.Button(menu, text=text, command=cmd)
            b.pack(fill=tk.X, pady=4)
            ttk.Label(menu, text="      " + hint,
                      foreground="#666").pack(anchor=tk.W, pady=(0, 4))

        foot = ttk.Frame(self, padding=(24, 0, 24, 18))
        foot.pack(fill=tk.X)
        ttk.Label(foot, text=elide_path(self.work_dir.get()),
                  foreground="#888", font=(MONO, 8),
                  anchor=tk.W).pack(anchor=tk.W)
        ttk.Label(foot, text="Results and cache live in this folder; "
                  "you can run the program from a terminal too.",
                  foreground="#888", wraplength=560,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if not is_writable(program_dir()):
            # a portable program in a folder it may not write to cannot work;
            # this is the one thing to say before the user types anything
            self.after(200, self._no_write_rights)

    def _no_write_rights(self):
        messagebox.showerror(
            "Cannot write into this folder",
            WRITE_HELP.format(path=program_dir()), parent=self)

    def new_search(self):
        # ffmpeg and ffprobe are checked HERE, before anything is picked:
        # without ffmpeg the search cannot read a single file, and a user who
        # only finds that out after filling the whole form has wasted their
        # time. ffprobe is only needed for the export, so its absence is a
        # question rather than a wall — with the consequence spelled out.
        if not find_ffmpeg():
            messagebox.showerror("ffmpeg and ffprobe are needed", FFMPEG_HELP,
                                 parent=self)
            return
        if not find_ffprobe() and not messagebox.askyesno(
                "ffprobe was not found", FFPROBE_QUESTION, parent=self,
                default=messagebox.NO):
            return
        SearchDialog(self, self)

    def show_help(self):
        HelpWindow(self)

    def show_cache(self):
        CacheWindow(self, self.work_dir)

    def open_run_window(self, argv, work):
        argv = child_argv() + argv
        RunWindow(self, argv, work)

    def on_close(self):
        # a search still running must be stopped the same way as through its
        # own window, with the same question asked first
        for win in [w for w in self.winfo_children()
                    if isinstance(w, RunWindow)]:
            win.on_close()
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
