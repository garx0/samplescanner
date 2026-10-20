"""
Is there a newer version out, and how does it get put in place?

The releases live on GitHub, one archive per version, named after the tag it
belongs to. The question this module answers is asked once per start of the
window, from a thread, and the answer is either a version to offer or nothing:
the program never waits for the network, and a network that is not there is not
something the user should ever be shown.

The putting in place is the part with teeth. A running exe on Windows cannot be
overwritten, so the work is done by a second copy of the program: the one that
is running hands over the archive it has downloaded and leaves, and the second
one waits for it to be gone and then does the writing. That is the ordinary way
of doing it on this system, and the reason the window closes and comes back.

The user's own cache and results are never touched. Not because the archive
happens to carry empty ones - it does, but that is a promise about today's
archive and not about the code - but because the update does not go near those
two folders at all: they are written by the program and belong to the person
using it, and the only thing an update may do about them is leave them be.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import Callable, List, Optional, Sequence, Tuple

from ssc.version import VERSION

# Where the releases live. The repository can be pointed elsewhere with the
# environment variable, which is what a fork or a mirror needs, and what the
# tests use to talk to a public repository instead of a private one.
REPO = os.environ.get("SAMPLESCANNER_UPDATE_REPO", "garx0/samplescanner")

# The folders the program writes into. They belong to the user, and an update
# leaves them exactly as they are.
KEEP_DIRS = ("cache", "results")

# The environment variable that tells the program that has just been put in
# place where the temporary copy of itself is, so that it can remove it.
UPDATE_TEMP_ENV = "SAMPLESCANNER_UPDATE_TEMP"

# The mark inside a temporary copy of the program. Without it the cleanup would
# remove any folder in the temporary directory whose name begins like one of
# ours - and the temporary directory belongs to the system, not to us.
COPY_MARK = ".samplescanner-update-copy"

# The folder inside the archive that holds the program, and the name of the
# archive itself, which carries the version it belongs to.
ARCHIVE_DIR = "SampleScanner"
ARCHIVE_NAME = "SampleScanner-{tag}-windows-portable.zip"

# How long to wait for the network, in seconds. The window does not wait at all
# - it carries on being a window - but the thread that asks must not hang about
# for ever either, or it would keep a connection open for the whole run.
CHECK_TIMEOUT = 8.0
DOWNLOAD_TIMEOUT = 120.0


# --------------------------------------------------------------------------- #
# The numbers
# --------------------------------------------------------------------------- #

def parse_version(text: str) -> Optional[Tuple[int, ...]]:
    """The number in a tag or a version string: 'v0.2.7' -> (0, 2, 7)."""
    m = re.search(r"(\d+(?:\.\d+)*)", text or "")
    if not m:
        return None
    try:
        return tuple(int(part) for part in m.group(1).split("."))
    except ValueError:
        return None


def is_newer(tag: str, current: str = VERSION) -> bool:
    """
    Whether the tag is a version above the one running.

    '0.3' and '0.3.0' are the same version, so the shorter one is padded with
    zeros before the comparison - otherwise a release called v0.3 would look
    older than the v0.2.7 it follows.
    """
    a, b = parse_version(tag), parse_version(current)
    if a is None or b is None:
        return False
    length = max(len(a), len(b))
    a = a + (0,) * (length - len(a))
    b = b + (0,) * (length - len(b))
    return a > b


# --------------------------------------------------------------------------- #
# The releases
# --------------------------------------------------------------------------- #

def api_url(repo: str = REPO) -> str:
    return f"https://api.github.com/repos/{repo}/releases/latest"


def releases_url(repo: str = REPO) -> str:
    return f"https://github.com/{repo}/releases"


def asset_name(tag: str) -> str:
    """The name of the archive of that version."""
    return ARCHIVE_NAME.format(tag=tag)


def download_url(tag: str, repo: str = REPO) -> str:
    """Where the archive of that version is."""
    return (f"https://github.com/{repo}/releases/download/{tag}/"
            f"{asset_name(tag)}")


def latest_tag(repo: str = REPO, timeout: float = CHECK_TIMEOUT) -> Optional[str]:
    """
    The tag of the newest release, or None when it cannot be told.

    The releases of a repository are asked for through its API, which answers
    with the newest one first and needs no key for a public repository. Nothing
    here raises: a network that is not there is an ordinary answer, and the
    caller treats it as "nothing to offer".

    The API has two ways to ask, and they do not answer the same question. The
    `latest` one gives the newest release that is NOT marked as a prerelease -
    which is the right answer for most repositories and the wrong one for a
    repository whose releases are all prereleases: there is no such release, the
    API answers 404, and the program would sit quietly for ever while a newer
    version waits on the page. So when the plain answer is not there, the whole
    list is asked for and the newest of them is taken.
    """
    request = urllib.request.Request(api_url(repo), headers={
        "User-Agent": f"SampleScanner/{VERSION}",
        "Accept": "application/vnd.github+json",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            body = answer.read().decode("utf-8", "replace")
        tag = str(json.loads(body).get("tag_name") or "").strip()
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return None
        tag = ""
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    if tag:
        return tag
    return _newest_prerelease(repo, timeout)


def _newest_prerelease(repo: str, timeout: float) -> Optional[str]:
    """
    The newest release there is, prereleases included.

    The list comes back newest first, so the first one that has a tag is the
    newest; the tags are still compared by number, because a repository whose
    releases are made by hand does not always put them in order.
    """
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases", headers={
            "User-Agent": f"SampleScanner/{VERSION}",
            "Accept": "application/vnd.github+json",
        })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            body = answer.read().decode("utf-8", "replace")
        releases = json.loads(body)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    best: Optional[str] = None
    best_version: Optional[tuple] = None
    for release in releases:
        tag = str(release.get("tag_name") or "").strip()
        if not tag:
            continue
        version = parse_version(tag)
        if version is None:
            continue
        if best_version is None or version > best_version:
            best, best_version = tag, version
    return best


def download(url: str, dest: str, timeout: float = DOWNLOAD_TIMEOUT,
             progress: Optional[Callable[[int, int], None]] = None) -> str:
    """
    Fetches a file to `dest`, saying how far it has got.

    The size is not always known - a server may answer without it - and then
    the progress is counted without a total, which the caller shows as a bar
    that cannot say how much is left.
    """
    request = urllib.request.Request(url, headers={
        "User-Agent": f"SampleScanner/{VERSION}",
    })
    done = 0
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        total = int(answer.headers.get("Content-Length") or 0)
        with open(dest, "wb") as out:
            while True:
                chunk = answer.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    return dest


# --------------------------------------------------------------------------- #
# The archive
# --------------------------------------------------------------------------- #

def extract(zip_path: str, target: str,
            keep: Sequence[str] = KEEP_DIRS) -> List[str]:
    """
    The contents of the archive into the folder, and what was written.

    Only the part of the archive under the program's own folder is taken, and
    the two folders in `keep` are left alone whatever the archive holds: they
    are the user's, not the program's.

    A path that climbs out of the folder is refused rather than written: an
    archive is a stranger's writing until it has been looked at.
    """
    written: List[str] = []
    prefix = ARCHIVE_DIR + "/"
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            if not rel:
                continue
            parts = rel.split("/")
            if rel.startswith("/") or ".." in parts:
                raise ValueError(f"the archive holds a path outside itself: {name}")
            if parts[0] in keep:
                continue
            dest = os.path.join(target, *parts)
            if info.is_dir():
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with archive.open(info) as source, open(dest, "wb") as out:
                shutil.copyfileobj(source, out)
            written.append(rel)
    return written


# --------------------------------------------------------------------------- #
# Putting it in place
# --------------------------------------------------------------------------- #

def _wait_until_replaceable(path: str, timeout: float = 120.0) -> bool:
    """
    Wait until the file can be written over.

    Not needed for the exe any more - it is renamed out of the way first, which
    frees its name - but a file that is being written by something else still
    has to be waited for, and this is how.
    """
    end = time.time() + timeout
    while True:
        try:
            with open(path, "r+b"):
                return True
        except OSError:
            if time.time() >= end:
                return False
            time.sleep(0.3)


def _copy_folder(base: str, pid: int) -> str:
    """
    A free folder in the temporary directory for the copy of the program.

    The name carries the number of the process that is making the copy, so that
    two updates never share one. If a folder of that name is there already and
    it is not ours - it has no mark in it - then it belongs to somebody else
    and is left alone, and a number is added to the name until a free one is
    found.
    """
    name = f"sssc-update-{pid}"
    path = os.path.join(base, name)
    counter = 0
    while os.path.exists(path):
        if os.path.isfile(os.path.join(path, COPY_MARK)):
            shutil.rmtree(path, ignore_errors=True)   # ours, left over
            break
        counter += 1
        name = f"sssc-update-{pid}-{counter}"
        path = os.path.join(base, name)
    return path


def copy_for_update(target: str) -> str:
    """
    A copy of the program, to run the update from.

    The program that does the writing cannot be the program in the folder being
    written to: it has the files of that folder open - the exe it is running
    from, the libraries it has loaded - and a file a process has open cannot be
    written by it, nor even renamed by it. Only another process can.

    So the second copy of the program is made in a temporary folder, and it
    writes the new program into the folder the person is using. The user's own
    cache and results are not copied: they are large, and they are not ours.

    The folder is named after the process that made it, so that two updates
    never share one, and it is removed when the update is over - by the program
    that was just put in place, which is the only one that can.
    """
    # The copy is made in the temporary folder of the system and NOT in the
    # folder of the program: a folder cannot be copied into itself.
    base = tempfile.gettempdir()
    temp = _copy_folder(base, os.getpid())
    os.makedirs(temp, exist_ok=True)
    # A mark inside the copy, so that only a copy of THIS program is ever
    # removed by the cleanup: a folder in the temporary directory that happens
    # to have a similar name is somebody else's and is left alone.
    with open(os.path.join(temp, COPY_MARK), "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    for name in os.listdir(target):
        if name in KEEP_DIRS:
            continue
        source = os.path.join(target, name)
        dest = os.path.join(temp, name)
        try:
            if os.path.isdir(source):
                shutil.copytree(source, dest)
            else:
                shutil.copy2(source, dest)
        except OSError as exc:
            raise RuntimeError(f"the program could not be copied: {exc}")
    return temp


def _is_our_copy(path: str) -> bool:
    """
    Whether a folder in the temporary directory is a copy of this program.

    The copy carries a mark inside it, put there when it was made. A copy made
    by an OLDER version of the program carries no mark - it did not know about
    marks - and it still has to be recognised, or the update from that version
    would leave a hundred and seventy megabytes in the temporary folder for
    ever. A folder that holds the program itself is taken for such a copy.

    A folder that is neither is somebody else's and is left alone: the
    temporary directory belongs to the system, not to us.
    """
    if os.path.isfile(os.path.join(path, COPY_MARK)):
        return True
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(name.lower().endswith(".exe") for name in names)


def clean_update_folders(base: Optional[str] = None) -> int:
    """
    Removes the temporary copies of the program that updates leave behind.

    The copy is made by the program that is about to be replaced and is removed
    by the program that replaced it, which is the only one that can: the copy
    holds the exe the updater is running from, and a file that is being run
    cannot be deleted.

    A copy whose update is going on RIGHT NOW is left alone: the folder is
    named after the process that made it, and that process is still alive while
    the update runs. Removing it then would delete the copy out from under the
    update that is filling it - which is exactly what happened before this was
    looked at.

    A folder that is not a copy of this program is left alone as well: the
    temporary directory of the system is shared with everything else on it.
    """
    base = base or tempfile.gettempdir()
    removed = 0
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    for name in names:
        if not name.startswith("sssc-update-"):
            continue
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        pid = name[len("sssc-update-"):].split("-")[0]
        if pid.isdigit() and _process_alive(int(pid)):
            continue                    # an update is going on right now
        if not _is_our_copy(path):
            continue                    # not ours: somebody else's folder
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError:
            continue                    # held: an update is going on
    return removed


def _process_alive(pid: int) -> bool:
    """Whether a process with that number is still running."""
    if pid <= 0 or os.name != "nt":
        return False
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259         # STILL_ACTIVE
    except Exception:
        return False


def clean_update_archives(base: Optional[str] = None) -> int:
    """
    Removes the archives of updates that did not finish.

    A download that was interrupted - the network went, the program was closed
    - leaves its half-read archive behind, and an archive of a release is sixty
    megabytes. The normal ending is that the copy of the program that does the
    writing removes it as soon as it has unpacked it, but that copy only gets
    there if the download was whole.

    Only files that are clearly ours are touched: the name says what they are,
    and the number of the process that fetched them says whether that process
    is still at work on them.

    An archive that cannot be removed is one that somebody is holding open, and
    a download into it is going on right now; it is left alone and looked at
    again at the next start.
    """
    base = base or tempfile.gettempdir()
    removed = 0
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    for name in names:
        if not name.startswith("SampleScanner-") or not name.endswith(".zip"):
            continue
        if "-update" not in name:
            continue
        stem = name[:-len(".zip")]
        pid_text = stem.rsplit("-", 1)[-1]
        path = os.path.join(base, name)
        if pid_text.isdigit() and _process_alive(int(pid_text)):
            continue                # a download is going on right now
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue                # held: a download is going on
    return removed


def _same_file(one: str, two: str) -> bool:
    """Whether the two files are the same, so that the one on disk can be left."""
    try:
        if os.path.getsize(one) != os.path.getsize(two):
            return False
        with open(one, "rb") as a, open(two, "rb") as b:
            while True:
                x, y = a.read(1 << 20), b.read(1 << 20)
                if x != y:
                    return False
                if not x:
                    return True
    except OSError:
        return False


def _put_file(source: str, dest: str) -> bool:
    """
    Writes the new file over the old one.

    Nothing is holding these files: the program that is running was started
    from a copy of itself in a temporary folder, and the program it replaced
    has left. So they are written over in place, which is simpler than moving
    them aside first and tidying up afterwards.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(source, dest)
    return True


def apply_update(zip_path: str, target: str, exe_path: str,
                 restart: bool = True) -> str:
    """
    Put the archive in place over the program that is running.

    Everything is unpacked into a folder of its own first, so that a download
    that turns out to be broken leaves the program as it was rather than half
    replaced. A file that is already the right one is left alone: most of the
    hundred and something files of a build are the same libraries in every
    version, and a file that does not have to be written does not have to be
    moved aside either.

    Files that were there before and are not in the new archive are left alone.
    The program does not know what a person has put next to the exe - ffmpeg
    is the one that matters - and a file it cannot recognise is not its to
    delete.

    The version that was put in place is read out of the archive and returned:
    the number the program is running is not the number that was just written,
    and saying the wrong one would be the last thing it says.
    """
    staging = os.path.join(target, ".update-staging")
    shutil.rmtree(staging, ignore_errors=True)
    version = "unknown"
    try:
        written = extract(zip_path, staging)
        if not written:
            raise RuntimeError("the archive holds nothing of the program")
        version = _version_of(staging)
        for rel in written:
            source = os.path.join(staging, *rel.split("/"))
            dest = os.path.join(target, *rel.split("/"))
            if os.path.isfile(dest) and _same_file(source, dest):
                continue                     # already the right file
            _put_file(source, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    # the folders the program writes into are the user's: they are made if they
    # are not there, and are never emptied
    for name in KEEP_DIRS:
        os.makedirs(os.path.join(target, name), exist_ok=True)
    if restart:
        # the temporary copy of the program is told where it is through the
        # environment, so that the program that is started now can remove it
        # once it is running - the copy holds the exe this one is running from,
        # and nothing else can delete that
        _start(exe_path, target, os.environ.get(UPDATE_TEMP_ENV) or None)
    return version


def _version_of(folder: str) -> str:
    """The version the archive put in place, as its own VERSION.txt says it."""
    try:
        with open(os.path.join(folder, "VERSION.txt"), encoding="utf-8-sig") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def _start(exe: str, target: str, temp: Optional[str] = None) -> None:
    """
    Starts the program again, with a window of its own.

    It is told where the temporary copy of itself is, so that it can remove it
    once it is running: the copy holds the exe the updater is running from, and
    the updater cannot delete a file it is running from.
    """
    flags = 0
    if os.name == "nt":
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.DETACHED_PROCESS)
    env = dict(os.environ)
    if temp:
        env[UPDATE_TEMP_ENV] = temp
    subprocess.Popen([exe], cwd=target, env=env, creationflags=flags,
                     close_fds=True)


def spawn_updater(zip_path: str, target: str, exe_path: str,
                  wait_pid: int = 0) -> None:
    """
    Starts the second copy of the program that does the writing.

    It is a copy of the program, made in a temporary folder, started with the
    one argument that says it is the one that writes, and with no window of its
    own: the person using the program is about to see the window close and come
    back, and a black console blinking in between is noise.

    The copy is told where it is through the environment, so that the program it
    puts in place can remove it afterwards - the copy holds the exe the updater
    is running from, and nothing else can delete that.
    """
    from ssc.launcher import UPDATE_FLAG
    temp = copy_for_update(target)
    updater_exe = os.path.join(temp, os.path.basename(exe_path))
    command = [updater_exe, UPDATE_FLAG, zip_path, target, exe_path,
               str(wait_pid)]
    env = dict(os.environ)
    env[UPDATE_TEMP_ENV] = temp
    flags = 0
    if os.name == "nt":
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW)
    subprocess.Popen(command, cwd=temp, env=env, creationflags=flags,
                     close_fds=True)


def apply_update_main(argv: Sequence[str]) -> int:
    """
    The entry point of the second copy: it puts the archive in place and starts
    the program again.

    It answers with a number, as every entry point of this program does, and
    says what it did on the console - there is no window to say it in.
    """
    if len(argv) < 3:
        print("usage: --apply-update <archive> <folder> <exe> [pid]")
        return 2
    zip_path, target, exe_path = argv[0], argv[1], argv[2]
    if len(argv) > 3 and argv[3].isdigit():
        # the process that started this one: it may still be leaving, and the
        # program it runs is about to be replaced
        _wait_for_process(int(argv[3]))
    try:
        version = apply_update(zip_path, target, exe_path)
    except Exception as exc:                         # noqa: BLE001 - reported
        print(f"update failed: {exc}")
        return 1
    finally:
        # The archive has been read and unpacked, and it is sixty megabytes of
        # somebody's disk. It was a temporary file and it goes back to being one:
        # a portable program that leaves a copy of itself in the temporary
        # folder of the system for ever is not portable.
        try:
            os.remove(zip_path)
        except OSError:
            pass
    print(f"updated to {version}")
    return 0


def _wait_for_process(pid: int, timeout: float = 120.0) -> None:
    """
    Waits until the process with that number is gone.

    The program that starts the update is still running while the update waits
    for it: it has the files of the program folder open, and they cannot be
    written over while it holds them. So the wait is not a courtesy, it is the
    whole point - without it the update writes over an exe that is running.

    A process that has ended has an exit code of its own, and one that is still
    running has 259, which is what STILL_ACTIVE means. The two are easy to mix
    up, and mixing them up means writing over a running program.
    """
    if pid <= 0 or os.name != "nt":
        return
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        end = time.time() + timeout
        while time.time() < end:
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return                     # it is already gone
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle,
                                                       ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)
            if code.value != 259:         # STILL_ACTIVE is 259
                return                     # it has ended
            time.sleep(0.2)
    except Exception:
        return
