"""Every number the program is tuned by, in one place."""

from __future__ import annotations

import numpy as np
from typing import Tuple

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 22050          # internal sample rate
HOP = 128                    # 5.8 ms between frames
FMIN = 30.0
FMAX = 11025.0               # = SAMPLE_RATE / 2
CHUNK_FRAMES = 4096          # block size when computing the STFT

# Feature bank: (n_fft, n_mels, take the first time difference, weight)
BANK: Tuple[Tuple[int, int, bool, float], ...] = (
    (1024, 48, False, 1.0),
    (1024, 48, True, 1.0),
    (2048, 48, False, 1.0),
    (2048, 48, True, 1.0),
    (2048, 64, False, 1.0),
    (2048, 64, True, 1.0),
)

# Thresholds for discarding degenerate (nearly constant) bands
MIN_STD_RATIO = 0.05         # min(std) / max(std) of the two sides
MIN_STD_FLOOR = 0.01         # window std relative to the band std of the file
FRAME_SUPPORT_THRESHOLD = 0.5  # frame r above which a frame counts as "matched"

# Cosine similarity is unstable in quiet parts (tiny window variance on a
# near-zero signal yields spuriously high values), so windows whose energy is
# below this fraction of the file median are not considered at all.
ENERGY_FLOOR = 0.05

# --- decision thresholds ---
# "confident": the ordinary rule — needs both good similarity and a clear margin.
CONF_SIM = 0.55
CONF_MARGIN = 0.12
CONF_SUPPORT = 0.15
# Decisive margin: if the found position beats everything else (including in
# other files) by a lot, that alone is enough even with medium similarity and
# low frame support — typical for a heavily processed sample.
DECISIVE_MARGIN = 0.25
DECISIVE_SIM = 0.45
# Equivalent sources: the same recording often sits in a collection several
# times (different mixes, edits, releases). Then several different files match
# with a high and nearly equal r, and the "margin over the other files" is
# almost zero even though the answer is obvious. Such candidates are treated as
# one group: they do not compete with each other, they are all reported and all
# exported.
# The group is taken as the best moment of every file, ranked by r, that stays
# within EQUIV_TOL of the best and is itself credible (EQUIV_MIN_SIM). The rule
# only fires if the group is separated from the rest by EQUIV_GAP — otherwise
# the candidates are genuinely indistinguishable, not duplicates of one source,
# and the honest small margin is the right answer.
EQUIV_TOL = 0.12
EQUIV_GAP = 0.10
EQUIV_MIN_SIM = 0.55
MAX_EQUIVALENTS = 12         # not worth listing 30 identical rips
# Fragment export: a lower threshold than "confident" (see export_decision).
EXPORT_MIN_SIM = 0.38
EXPORT_MIN_MARGIN = 0.08

LOOP_Z_FRACTION = 0.95      # fraction of the max z-profile above which we look for loops
LOOP_SEPARATION = 0.30      # s, closer than this the moments count as one
LOOP_TOL = 0.02             # s, tolerance when fitting the period
MAX_REPORTED_OCCURRENCES = 8
MIN_REPETITION_COUNT = 3    # this many equal-quality moments means a repeat

CANDIDATES_PER_FILE = 5      # peaks taken from the combined profile of each file
CANDIDATES_PER_MEMBER = 3    # peaks from each individual feature (for recall)
LOCAL_EXCLUSION = 0.75       # s, offsets within this are considered one place
LOCAL_REFINE = 2             # +- this many frames searched at the final stage
PEAK_EXCLUSION = int(round(LOCAL_EXCLUSION * SAMPLE_RATE / HOP))  # same, in frames

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac",
                    ".opus", ".wma", ".aiff", ".aif", ".ape", ".mp4", ".wv"}

EPS = 1e-12


# Coarse (cheap) on-disk index
# --------------------------------------------------------------------------- #
# The full feature bank (3 resolutions, 5.8 ms hop) costs ~700 KB per second
# of audio, i.e. ~165 GB for a 12 GB mp3 collection — that does not fit in
# memory. So the wide search runs off a "coarse" index: one resolution,
# 23 ms hop, float16. Chosen on the test data: with n_fft=1024, hop=512 and
# 48 bands the true position reliably lands at the top of the profile, while
# the index is 166x smaller. Only one file is loaded into RAM at a time.
# Accuracy is restored by a second, precise pass: for the shortlisted
# candidates the position is re-read from the original file (a window of a
# few seconds) and the full bank is computed — hence all reported numbers and
# thresholds stay exactly the same.

COARSE_NFFT = 1024
COARSE_HOP = 512           # ≈23 ms — enough to locate the position
COARSE_BANDS = 48
COARSE_DTYPE = np.float16
# The coarse similarity scales the query by its OWN std, and the std of a
# single frame is 0 — with one coarse frame no band is informative and every
# source comes back as -inf, i.e. no candidates at all. A query shorter than
# this many coarse frames is padded (see coarse_views).
COARSE_MIN_FRAMES = 3
FINE_PAD = 1.0             # s, margin when re-reading a candidate window
# The shortest window the precise pass can work with: the widest feature of the
# bank is 2048 samples long, so anything under four of them has no frame to
# correlate. It is a minimum for the WINDOW that is re-read from a source, never
# for a query — a query is left exactly as long as it is, because padding it
# costs more than it buys (see fine_verify for the case this is about).
FINE_MIN_SAMPLES = 4 * 2048
FINE_CANDIDATES = 24       # how many candidates go to the precise pass
COARSE_PER_MEMBER = 6      # peaks taken from each coarse feature
FINE_MAX_TOTAL = 128       # ceiling on the shortlist; every file gets one peak
                           # until the budget is used up, which is what keeps a
                           # true match from being out-zoned by other files
# The precise pass reads one window per candidate, and starting ffmpeg for one
# costs several times what decoding that window does (measured: 57 ms of the
# 77 ms a candidate takes). So the windows of a whole batch go through one ffmpeg
# each, every window keeping its own input and its own -ss - see
# load_audio_many, which is where the rule that makes the result identical to
# separate reads is explained and where the fallback lives.
BATCH_WINDOWS = 32         # windows per ffmpeg; more is barely quicker and the
                           # command line grows with every one of them
BATCH_MAX_COMMAND = 8000   # characters of command line per ffmpeg, a collection
                           # path being repeated once per input
# 2 is the shared cache tree: cache/<X>_/<Y>_/ with a manifest per <Y> that
# records which library folder it stands for. Version 1 indexes are NOT read -
# see the migration in ssc/cache.py's design notes - so a user with a v1 cache
# rebuilds it or converts it.
INDEX_VERSION = 2

REF_TAIL_PAD = 512   # zero frames appended past the end (≈3 s at HOP=128)

# What one second of audio costs in the on-disk index: one coarse frame per
# COARSE_HOP samples, COARSE_BANDS mel values as float16 plus one float32
# frame energy. Used to tell the user how big the index of a collection will
# get BEFORE it is written; measured against real indexes the estimate comes
# out about 5 % high, which is the safe direction.
INDEX_BYTES_PER_SECOND = int((COARSE_BANDS * 2 + 4) * SAMPLE_RATE / COARSE_HOP)
# Per file: the .npy headers, the manifest entry and the slack of the file
# system. Small, but it is what keeps the estimate from going under.
INDEX_BYTES_PER_FILE = 8 * 1024
# Above this the index is worth a question before it is built. In the window
# the user is asked; from a terminal it is only announced, because there is
# nobody there to answer.
INDEX_SIZE_WARN = 2 * 1024 ** 3


# One results subfolder per run, so nothing of different runs ever mixes up:
#     results/<query folder>__<search folder>__<YYYYmmdd-HHMMSS>/
# It holds the fragments of every query of that run, the log of the run
# (log_<query folder>__<search folder>__<YYYYmmdd-HHMMSS>.txt) and the ones
# that did not reach the export threshold in its unsure/ subfolder.
DEFAULT_RESULTS_PREFIX = "results"
DEFAULT_UNSURE_SUBDIR = "unsure"
# How many queries one pass over the collection answers. Making a file ready to
# be compared with something costs the same whatever is compared with it, so
# the more queries share the pass the less of that work there is per query.
# Four is a compromise: the search gets quicker with every step, but the report
# of a query is printed only when the last query of its batch is done, and that
# wait grows with the batch.
DEFAULT_BATCH = 4
# The sweep goes through the collection in rounds: one walk gives every query
# still in play its next shift, and the queries that come back confident leave.
# A round that stops less than a 1/ROUND_STOP_SHARE of them is not worth its
# own walk, because the next one will not stop many either - from there the
# rest of the shifts is cheaper to answer in a single pass. Raise it to split
# into fewer rounds, lower it to give up sooner.
ROUND_STOP_SHARE = 4
# Where the fragment goes when it was searched at a pitch shift: the copy in
# the run folder is shifted back and sounds like the query, the copy that keeps
# the source's own pitch goes here.
ORIG_PITCH_SUBDIR = "orig_pitch"
# Where a confident match is written the moment it is found, before the rest of
# the sweep has run: a sweep of a large collection over many shifts is hours
# long, and a result that is only written at the end is a result that stopping
# halfway throws away. The layout below it is the one of a normal export - the
# fragment corrected to the query in here, the source's own pitch in
# orig_pitch/ - so the folder can be listened to exactly like the run folder.
INTERMEDIATE_SUBDIR = "intermediate"
LOG_PREFIX = "log"
RESULTS_STAMP = "%Y%m%d-%H%M%S"
# Below this similarity there is nothing worth listening to at all (false
# candidates sit at 0.2-0.36), so nothing is exported into the unsure folder.
UNSURE_MIN_SIM = 0.15
# Names of the exported fragments. Win32 without the \\?\ prefix allows 260
# characters for the WHOLE path, and the results folder already spends ~85 of
# them, so the fragment names have to be budgeted: the source name is shortened
# to fit (see fragment_stem), never the timestamp.
MAX_PATH_CHARS = 240
PATH_TAIL_RESERVE = 12       # ".flac" plus the "_2" of a repeated name
QUERY_STEM_MAX = 60
SOURCE_STEM_MAX = 90
MIN_STEM = 12                # below this a name stops being readable
# Margin at the edges when cutting: compensates for the inaccuracy of seeking
# inside an encoded file (an mp3 decoder may start slightly early or late).
EXTRACT_PAD = 0.5
# The export starts EARLIER than the position that was found, and ends later by
# the same amount. The search picks its candidates on a 23 ms grid and the
# precise pass only narrows that down, so a position can still be a good part of
# a step late; worse, the profile of a sharp transient peaks at its loudest
# part, not at its onset. Measured over the test queries: 25 of 26 exports land
# exactly on the onset, and the sharpest transient was 41 ms late — more than
# one coarse step. The pre-roll is sized on that worst case and put on both
# sides, so neither the beginning nor the end of the sample can be lost.
# Extra audio around it is harmless context, whereas losing the sample itself
# would be a real cut. Nothing about the index or the search changes.
EXPORT_PREROLL = 1.75 * COARSE_HOP / SAMPLE_RATE   # ≈41 ms

