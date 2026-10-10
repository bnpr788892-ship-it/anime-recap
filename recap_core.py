"""
recap_core.py - the testable logic of the anime recap automation.

Nothing in this file imports Google, Gemini, Whisper or edge-tts at module
level, so the unit tests can run with mocks and never touch a real service.
main.py wires these pieces to the real services.
"""
import csv
import datetime
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

# ======================================================================
# Statuses and columns
# ======================================================================
PENDING = "PENDING"
PROCESSING = "PROCESSING"
TRANSCRIBING = "TRANSCRIBING"
GENERATING_SCRIPT = "GENERATING_SCRIPT"
GENERATING_VOICE = "GENERATING_VOICE"
EDITING = "EDITING"
UPLOADING = "UPLOADING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
RETRY_PENDING = "RETRY_PENDING"

ACTIVE_STATUSES = {PROCESSING, TRANSCRIBING, GENERATING_SCRIPT,
                   GENERATING_VOICE, EDITING, UPLOADING}

OUT_KEYS = ("long", "short1", "short2")

KIND_SEASON, KIND_EPISODE, KIND_MEGA = "SEASON", "EPISODE", "MEGA"

COLUMNS = [
    "unit_key", "source_id", "kind", "filename", "season", "episode",
    "expected_episodes", "title", "date_received", "status", "season_status",
    "started_at", "completed_at", "segment_start", "segment_end",
    "source_duration", "narration_duration", "output_duration", "duration_diff",
    "sync_status", "subtitle_status", "scene_match", "warnings",
    "transcription_status", "script_status", "voice_status",
    "long_status", "shorts_status", "youtube_status",
    "yt_long_id", "yt_short1_id", "yt_short2_id",
    "yt_long_verify", "yt_short1_verify", "yt_short2_verify",
    "drive_srt_link", "drive_long_link", "drive_short1_link", "drive_short2_link",
    "short_plan", "segments", "summary", "thumbnail", "meta",
    "yt_long_url", "yt_short1_url", "yt_short2_url",
    "retry_count", "failed_stage", "error", "last_updated",
]

# Header used by the first Google Sheet version (main.py v2) - migrated on load.
OLD_HEADER_MAP = {
    "Source file ID": "source_id",
    "Source name": "filename",
    "Status": "status",
    "Long video YouTube ID": "yt_long_id",
    "Short 1 YouTube ID": "yt_short1_id",
    "Short 2 YouTube ID": "yt_short2_id",
    "Date (UTC)": "last_updated",
}


def _yt_col(k):
    return f"yt_{k}_id"


def _drive_col(k):
    return f"drive_{k}_link"


def _verify_col(k):
    return f"yt_{k}_verify"


# ======================================================================
# Time helpers
# ======================================================================
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def fmt_ts(dt):
    return dt.strftime(TS_FORMAT) + " UTC"


def parse_ts(text):
    """Parse a timestamp, tolerating the formats Google Sheets may give back."""
    if not text:
        return None
    t = str(text).strip().replace(" UTC", "").replace("Z", "")
    for f in (TS_FORMAT, "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S",
              "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.datetime.strptime(t, f)
        except ValueError:
            continue
    return None


# ======================================================================
# Errors, classification, retries with exponential backoff
# ======================================================================
class PermanentError(Exception):
    """Will not succeed if retried."""


class TemporaryError(Exception):
    """Worth retrying."""


class QuotaExceeded(PermanentError):
    """An API daily quota is used up. Stop the whole run."""


class MissingCredentials(PermanentError):
    pass


class MediaError(Exception):
    """ffmpeg / ffprobe / output validation problem."""


class StageError(Exception):
    def __init__(self, stage, original, column=None):
        super().__init__(f"{stage}: {original}")
        self.stage = stage
        self.original = original
        self.column = column


QUOTA_WORDS = ("quotaexceeded", "dailylimitexceeded", "uploadlimitexceeded")


def _status_code(exc):
    resp = getattr(exc, "resp", None)
    if resp is not None and getattr(resp, "status", None):
        return int(resp.status)
    r = getattr(exc, "response", None)
    if r is not None and getattr(r, "status_code", None):
        return int(r.status_code)
    return None


def _error_text(exc):
    text = str(exc)
    content = getattr(exc, "content", None)
    if isinstance(content, (bytes, bytearray)):
        text += " " + content.decode("utf-8", "ignore")
    elif isinstance(content, str):
        text += " " + content
    return text.lower()


def classify_error(exc):
    """Returns 'quota', 'permanent', 'temporary' or 'unknown'."""
    if isinstance(exc, QuotaExceeded):
        return "quota"
    if isinstance(exc, PermanentError):
        return "permanent"
    if isinstance(exc, TemporaryError):
        return "temporary"
    text = _error_text(exc)
    if any(w in text.replace(" ", "") for w in QUOTA_WORDS):
        return "quota"
    code = _status_code(exc)
    if code is not None:
        if code in (408, 429) or code >= 500:
            return "temporary"
        if code in (400, 401, 403, 404, 405, 409, 410):
            return "permanent"
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "temporary"
    return "unknown"


def with_retries(fn, attempts=4, base_delay=2.0, max_delay=60.0,
                 sleep=time.sleep, label="", log=print):
    """Run fn(); retry only temporary failures with exponential backoff."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            kind = classify_error(e)
            if kind == "quota" and not isinstance(e, QuotaExceeded):
                raise QuotaExceeded(scrub(str(e))) from e
            if kind != "temporary" or i == attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (i - 1)))
            delay += random.uniform(0, base_delay / 4)
            log(f"[retry] {label} attempt {i}/{attempts} failed "
                f"({type(e).__name__}); waiting {delay:.1f}s")
            sleep(delay)


# ======================================================================
# Secret scrubbing
# ======================================================================
SECRET_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_CLIENT_SECRET",
                    "GOOGLE_REFRESH_TOKEN", "GOOGLE_CLIENT_ID",
                    "TELEGRAM_BOT_TOKEN")


def scrub(text):
    text = str(text)
    for name in SECRET_ENV_NAMES:
        val = os.environ.get(name)
        if val and len(val) >= 6:
            text = text.replace(val, "***")
    text = re.sub(r"(?i)(key|token|secret|access_token|refresh_token)=([^&\s'\"]+)",
                  r"\1=***", text)
    return text


# ======================================================================
# Configuration
# ======================================================================
def _flag(value, default):
    if value is None or value == "":
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Config:
    long_dir: str = ""
    shorts_dir: str = ""
    max_videos: int = 2
    max_retries: int = 3
    stale_hours: float = 7.0
    burn_subs: bool = True
    clip_len: float = 50.0
    privacy: str = "private"
    attempts: int = 4
    base_delay: float = 2.0
    seg_len: float = 8.0
    sub_font: str = "Noto Sans Devanagari"
    sub_size_long: int = 18      # ASS units, scaled to video height (288 base)
    sub_size_short: int = 11
    sub_margin_long: int = 25
    sub_margin_short: int = 60
    wrap_landscape: int = 42
    wrap_portrait: int = 22
    # --- v4 settings ---
    wps: float = 2.2                 # Hindi words per second
    episode_seconds: float = 3600.0  # target episode length (1 hour)
    min_last_episode: float = 600.0  # a shorter leftover is folded into the previous episode
    snap_tolerance: float = 120.0    # boundary adjustment tolerance (seconds)
    max_episodes: int = 24           # safety limit per source
    season_number: int = 1
    mega_enabled: bool = True
    mega_max_seconds: float = 41400.0   # YouTube limit is 12h; keep a margin
    duration_tol: float = 0.0        # 0 = automatic (3 frames, at least 0.15s)
    src_audio_volume: float = 0.15   # original sound under the narration (0 = mute)
    scene_window: float = 30.0       # fallback scene window (seconds)
    batch_scenes: int = 20
    shorts_per_episode: int = 2
    short_min: float = 20.0
    short_max: float = 58.0
    uploads_per_run: int = 6         # YouTube API quota guard (about 1,600 units each)
    run_budget_minutes: float = 270.0
    require_confirmed: bool = False  # True = mega only when YouTube confirmed every episode
    sleep: Callable = field(default=time.sleep, repr=False)
    clock: Callable = field(default=utcnow, repr=False)

    def now(self):
        return fmt_ts(self.clock())

    @classmethod
    def from_env(cls, env=None):
        e = env if env is not None else os.environ

        def num(name, default, cast=float):
            try:
                return cast(e.get(name) or default)
            except ValueError:
                return default

        return cls(
            long_dir=e.get("LONG_FOLDER_ID", ""),
            shorts_dir=e.get("SHORTS_FOLDER_ID", ""),
            max_videos=num("MAX_VIDEOS", 2, int),
            max_retries=num("MAX_RETRIES", 3, int),
            burn_subs=_flag(e.get("BURN_SUBTITLES"), True),
            clip_len=num("SHORT_CLIP_SECONDS", 50.0),
            privacy=e.get("YT_PRIVACY") or "private",
            seg_len=num("SEGMENT_SECONDS", 8.0),
            sub_font=e.get("SUB_FONT") or "Noto Sans Devanagari",
            sub_size_long=num("SUB_SIZE_LONG", 18, int),
            sub_size_short=num("SUB_SIZE_SHORT", 11, int),
            sub_margin_long=num("SUB_MARGIN_LONG", 25, int),
            sub_margin_short=num("SUB_MARGIN_SHORT", 60, int),
            wps=num("HINDI_WORDS_PER_SECOND", 2.2),
            episode_seconds=num("EPISODE_MINUTES", 60.0) * 60.0,
            min_last_episode=num("MIN_LAST_EPISODE_MINUTES", 10.0) * 60.0,
            snap_tolerance=num("BOUNDARY_TOLERANCE_SECONDS", 120.0),
            max_episodes=num("MAX_EPISODES_PER_SOURCE", 24, int),
            season_number=num("SEASON_NUMBER", 1, int),
            mega_enabled=_flag(e.get("MEGA_ENABLED"), True),
            mega_max_seconds=num("MEGA_MAX_SECONDS", 41400.0),
            duration_tol=num("DURATION_TOLERANCE_SECONDS", 0.0),
            src_audio_volume=num("SOURCE_AUDIO_VOLUME", 0.15),
            scene_window=num("SCENE_WINDOW_SECONDS", 30.0),
            batch_scenes=num("NARRATION_BATCH_SCENES", 20, int),
            shorts_per_episode=min(2, max(0, num("SHORTS_PER_EPISODE", 2, int))),
            short_min=num("SHORT_MIN_SECONDS", 20.0),
            short_max=num("SHORT_MAX_SECONDS", 58.0),
            uploads_per_run=num("MAX_YT_UPLOADS_PER_RUN", 6, int),
            run_budget_minutes=num("RUN_TIME_BUDGET_MINUTES", 270.0),
            require_confirmed=_flag(e.get("REQUIRE_CONFIRMED_UPLOADS"), False),
        )


REQUIRED_ENV = ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
                "GOOGLE_REFRESH_TOKEN", "GEMINI_API_KEY",
                "INCOMING_FOLDER_ID", "LONG_FOLDER_ID",
                "SHORTS_FOLDER_ID", "DONE_FOLDER_ID"]


def sheet_id_from_env(env=None):
    e = env if env is not None else os.environ
    raw = (e.get("GOOGLE_SHEET_ID") or e.get("SHEET_ID") or "").strip()
    m = re.search(r"/d/([A-Za-z0-9_-]+)", raw)       # a pasted full link also works
    return m.group(1) if m else raw


def missing_env(env=None):
    e = env if env is not None else os.environ
    missing = [n for n in REQUIRED_ENV if not (e.get(n) or "").strip()]
    if not sheet_id_from_env(e).strip():
        missing.append("GOOGLE_SHEET_ID (or SHEET_ID)")
    return missing


# ======================================================================
# Persistent state (Google Sheet through a small backend interface)
# ======================================================================
def drive_link(file_id):
    return f"https://drive.google.com/file/d/{file_id}/view"


def id_from_link(link):
    m = re.search(r"/d/([^/?#]+)", link or "")
    return m.group(1) if m else (link or "")


class StateStore:
    """
    One row per unit of work, keyed by unit_key:
      <drive file id>          the season (source video) row
      <drive file id>:E01      an episode row
      <drive file id>:MEGA     the full-season mega video row
    backend needs: read_csv() -> str | None, write_csv(text).
    """

    def __init__(self, backend, sleep=time.sleep, base_delay=2.0, log=print):
        self.backend = backend
        self.rows = {}
        self._sleep = sleep
        self._base_delay = base_delay
        self._log = log

    # ---- loading / saving ----
    def load(self):
        text = self.backend.read_csv() or ""
        self.rows = {}
        parsed = [r for r in csv.reader(io.StringIO(text))
                  if any(c.strip() for c in r)]
        if not parsed:
            return
        header = [h.strip() for h in parsed[0]]
        changed = False
        if "source_id" not in header:
            if header and header[0] in OLD_HEADER_MAP:
                header = [OLD_HEADER_MAP.get(h, h) for h in header]
                changed = True
            else:
                return  # unknown sheet layout: treat as empty
        for cells in parsed[1:]:
            rec = dict.fromkeys(COLUMNS, "")
            for h, v in zip(header, cells):
                if h in rec:
                    rec[h] = v.strip()
            if not rec["source_id"] and not rec["unit_key"]:
                continue
            if not rec["unit_key"]:                      # a row from an older version
                changed = True
                for r in self._migrate_legacy(rec):
                    self._put(r)
                continue
            self._put(rec)
        if changed:
            self.save()

    def _put(self, rec):
        key = rec["unit_key"]
        old = self.rows.get(key)
        if old is None or old["status"] != COMPLETED:
            self.rows[key] = rec          # one row per unit

    @staticmethod
    def _migrate_legacy(rec):
        """Rows written by older versions (one row per video, keyed by Drive id)."""
        st = rec["status"]
        sid = rec["source_id"]
        if st == "DONE":
            st = rec["status"] = COMPLETED
        if st == COMPLETED:
            rec.update(unit_key=sid, kind=KIND_SEASON, expected_episodes="1",
                       season_status="COMPLETE", season="1")
            return [rec]
        outputs = any(rec.get(c) for c in ("yt_long_id", "yt_short1_id", "yt_short2_id",
                                           "drive_long_link", "drive_short1_link",
                                           "drive_short2_link"))
        if not outputs:
            return []                    # nothing was uploaded: start fresh
        rec.update(unit_key=f"{sid}:E01", kind=KIND_EPISODE, episode="1", season="1",
                   status=RETRY_PENDING, retry_count="0", failed_stage="", error="")
        return [rec]

    def to_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(COLUMNS)
        for rec in self.rows.values():
            w.writerow([rec.get(c, "") for c in COLUMNS])
        return buf.getvalue()

    def save(self):
        text = self.to_csv()
        with_retries(lambda: self.backend.write_csv(text), attempts=4,
                     base_delay=self._base_delay, sleep=self._sleep,
                     label="sheet write", log=self._log)

    # ---- row access ----
    def get(self, key):
        return self.rows.get(key)

    def upsert(self, key, **fields):
        rec = self.rows.get(key)
        if rec is None:
            rec = dict.fromkeys(COLUMNS, "")
            rec["unit_key"] = key
            rec["source_id"] = key.split(":")[0]
            self.rows[key] = rec
        for k, v in fields.items():
            if k not in COLUMNS:
                raise KeyError(f"unknown column {k}")
            rec[k] = "" if v is None else str(v)
        rec["last_updated"] = fmt_ts(utcnow())
        self.save()
        return rec

    def claim(self, key, filename, now_text, **extra):
        """Mark a unit as PROCESSING. Counts a dead earlier run as a try."""
        rec = self.rows.get(key) or dict.fromkeys(COLUMNS, "")
        retries = int(rec.get("retry_count") or 0)
        if rec.get("status") in ACTIVE_STATUSES:
            retries += 1  # an earlier run started this and never finished
        return self.upsert(
            key, filename=filename, status=PROCESSING,
            started_at=rec.get("started_at") or now_text,
            retry_count=retries, failed_stage="", error="", completed_at="", **extra)

    def children(self, source_id, kind):
        return sorted((r for r in self.rows.values()
                       if r["source_id"] == source_id and r["kind"] == kind),
                      key=lambda r: int(r.get("episode") or 0))


def is_eligible(rec, cfg, now=None):
    """Should this source be processed in this run?"""
    now = now or cfg.clock()
    if rec is None:
        return True
    st = rec.get("status") or PENDING
    if st in (COMPLETED, FAILED):
        return False
    if int(rec.get("retry_count") or 0) >= cfg.max_retries:
        return False
    if st in ACTIVE_STATUSES:
        last = parse_ts(rec.get("last_updated"))
        if last is None:
            return True  # unreadable time: treat the old run as dead
        return (now - last) > datetime.timedelta(hours=cfg.stale_hours)
    return st in (PENDING, RETRY_PENDING)


# ======================================================================
# Subtitles (Hindi, synchronized to the narration)
# ======================================================================
def format_srt_time(sec):
    sec = max(0.0, float(sec))
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_time(text):
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", text.strip())
    if not m:
        raise ValueError(f"bad srt time: {text}")
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def wrap_text(text, max_line=22):
    """At most two balanced lines, so subtitles stay readable on a phone."""
    text = " ".join(str(text).split())
    words = text.split()
    if len(text) <= max_line or len(words) < 2:
        return text
    i = min(range(1, len(words)),
            key=lambda k: max(len(" ".join(words[:k])), len(" ".join(words[k:]))))
    return " ".join(words[:i]) + "\n" + " ".join(words[i:])


def cues_to_srt(cues, max_line=42):
    out = []
    for n, c in enumerate(cues, start=1):
        out.append(f"{n}\n{format_srt_time(c['start'])} --> "
                   f"{format_srt_time(c['end'])}\n{wrap_text(c['text'], max_line)}\n")
    return "\n".join(out)


def write_srt(cues, path, max_line=42):
    with open(path, "w", encoding="utf-8") as f:
        f.write(cues_to_srt(cues, max_line))
    return path


def parse_srt(text):
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r", "").strip())
    for b in blocks:
        lines = [l for l in b.split("\n")]
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        a, z = lines[idx].split("-->")
        body = " ".join(" ".join(lines[idx + 1:]).split())
        if body:
            cues.append({"start": parse_srt_time(a), "end": parse_srt_time(z),
                         "text": body})
    return cues


def slice_cues(cues, start, end):
    """Cues inside [start, end], shifted so the clip starts at 0."""
    out = []
    for c in cues:
        if c["end"] <= start or c["start"] >= end:
            continue
        s = max(c["start"], start) - start
        e = min(c["end"], end) - start
        if e - s >= 0.2:
            out.append({"start": s, "end": e, "text": c["text"]})
    return out


def build_cues_from_boundaries(boundaries, total_duration, max_words=7, max_chars=40):
    """boundaries: [{'offset': sec, 'duration': sec, 'text': str}] from edge-tts."""
    words = []
    for b in boundaries:
        toks = str(b.get("text", "")).split()
        if not toks:
            continue
        off = float(b["offset"])
        dur = max(float(b.get("duration", 0)), 0.01)
        weights = [max(len(t), 1) for t in toks]
        tot = sum(weights)
        t0 = off
        for t, w in zip(toks, weights):
            d = dur * w / tot
            words.append((t0, t0 + d, t))
            t0 += d
    cues, cur = [], []

    def flush():
        if cur:
            cues.append({"start": cur[0][0], "end": cur[-1][1],
                         "text": " ".join(x[2] for x in cur)})
            cur.clear()

    for w in words:
        cur.append(w)
        chars = len(" ".join(x[2] for x in cur))
        if len(cur) >= max_words or chars >= max_chars or w[2][-1:] in "।.?!":
            flush()
    flush()
    return _tidy_cues(cues, total_duration)


def build_cues_proportional(text, total_duration, max_words=7):
    """Fallback when no timing events exist: spread text by length over the audio."""
    chunks = []
    for s in [s for s in re.split(r"(?<=[।.?!])\s+", str(text).strip()) if s]:
        words = s.split()
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))
    if not chunks or total_duration <= 0:
        return []
    weights = [max(len(c), 1) for c in chunks]
    tot = sum(weights)
    cues, t = [], 0.0
    for c, w in zip(chunks, weights):
        d = total_duration * w / tot
        cues.append({"start": t, "end": t + d, "text": c})
        t += d
    return _tidy_cues(cues, total_duration)


def _tidy_cues(cues, total_duration):
    cues = sorted(cues, key=lambda c: c["start"])
    for i, c in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else total_duration
        c["end"] = max(c["end"], c["start"] + 0.3)       # not too short to read
        c["end"] = min(c["end"] + 0.15, nxt) if nxt > c["start"] else c["end"]
        if total_duration:
            c["end"] = min(c["end"], total_duration)
    return [c for c in cues if c["end"] > c["start"]]


def make_cues(boundaries, total_duration, script_text, max_words=7):
    cues = []
    if boundaries:
        cues = build_cues_from_boundaries(boundaries, total_duration, max_words)
    if not cues:
        cues = build_cues_proportional(script_text, total_duration, max_words)
    return cues


# ======================================================================
# Media helpers (ffmpeg / ffprobe)
# ======================================================================
LONG_VF = ("scale=1280:720:force_original_aspect_ratio=decrease,"
           "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=30")


def _run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, check=True, capture_output=True,
                              text=True, cwd=cwd, stdin=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"{cmd[0]} failed (exit {e.returncode}): "
                         f"{(e.stderr or '')[-400:]}") from e
    except FileNotFoundError as e:
        raise MediaError(f"{cmd[0]} is not installed") from e


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height",
         "-of", "json", path], capture_output=True, text=True,
        stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise MediaError(f"ffprobe cannot read {os.path.basename(path)}: "
                         f"{(r.stderr or '')[-200:]}")
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    v = [s for s in streams if s.get("codec_type") == "video"]
    a = [s for s in streams if s.get("codec_type") == "audio"]
    try:
        dur = float(data.get("format", {}).get("duration") or 0)
    except ValueError:
        dur = 0.0
    return {"duration": dur, "has_video": bool(v), "has_audio": bool(a),
            "width": int(v[0].get("width", 0)) if v else 0,
            "height": int(v[0].get("height", 0)) if v else 0}


def validate_output(path, kind="long", expected_duration=None, tolerance=0.1, abs_tol=None):
    """Raise MediaError unless the file is a playable video with audio."""
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        raise MediaError(f"{os.path.basename(path)} is missing or empty")
    info = probe(path)
    if not info["has_video"]:
        raise MediaError(f"{os.path.basename(path)} has no video stream")
    if not info["has_audio"]:
        raise MediaError(f"{os.path.basename(path)} has no audio stream")
    if info["duration"] < 1.0:
        raise MediaError(f"{os.path.basename(path)} is shorter than 1 second")
    if expected_duration:
        allowed = abs_tol if abs_tol is not None else max(2.0, expected_duration * tolerance)
        if abs(info["duration"] - expected_duration) > allowed:
            raise MediaError(
                f"{os.path.basename(path)} is {info['duration']:.1f}s, "
                f"expected about {expected_duration:.1f}s")
    if kind == "short" and info["height"] <= info["width"]:
        raise MediaError(f"{os.path.basename(path)} is not vertical")
    return info


def vertical_filter(width, height):
    """9:16 filter for both landscape and portrait sources."""
    if height and width / height > 9 / 16 * 1.02:
        return "crop=ih*9/16:ih,scale=1080:1920,setsar=1"
    return ("scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1")


def subtitles_filter(srt_name, font, size, margin_v, alignment=2):
    style = (f"FontName={font},FontSize={size},PrimaryColour=&H00FFFFFF&,"
             f"OutlineColour=&H00000000&,BorderStyle=1,Outline=2,Shadow=0,"
             f"Alignment={alignment},MarginV={margin_v}")
    return f"subtitles={srt_name}:force_style='{style}'"


def render_short(long_video, start, length, out, srt=None, workdir=".",
                 font="Noto Sans Devanagari", font_size=11, margin_v=60):
    info = probe(long_video)
    vf = vertical_filter(info["width"], info["height"])
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    out = os.path.abspath(out)
    cwd = None
    if srt:
        shutil.copyfile(srt, os.path.join(workdir, "short_subs.srt"))
        vf += "," + subtitles_filter("short_subs.srt", font, font_size, margin_v)
        cwd = workdir
    _run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", os.path.abspath(long_video),
          "-t", f"{length:.3f}", "-vf", vf,
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
          "-movflags", "+faststart", out], cwd=cwd)
    return out



# ---- v4 media helpers: exact-duration episodes ----
def stream_info(path):
    """Per-stream start/duration, to check audio and video line up."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,start_time,duration:format=duration", "-of", "json", path],
        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise MediaError(f"ffprobe cannot read {os.path.basename(path)}")
    data = json.loads(r.stdout or "{}")
    out = {"format_duration": float(data.get("format", {}).get("duration") or 0)}
    for s in data.get("streams", []):
        t = s.get("codec_type")
        if t in ("video", "audio") and t not in out:
            def f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            out[t] = {"start": f(s.get("start_time")) or 0.0, "duration": f(s.get("duration"))}
    return out


def check_av_sync(path, expected, tol, start_tol=0.15):
    """Returns (ok, details). Audio and video must both span the expected length."""
    info = stream_info(path)
    problems = []
    v, a = info.get("video"), info.get("audio")
    if not v or not a:
        return False, "missing audio or video stream"
    vd = v["duration"] or info["format_duration"]
    ad = a["duration"] or info["format_duration"]
    if abs(v["start"]) > start_tol or abs(a["start"]) > start_tol:
        problems.append(f"streams start late (video {v['start']:.2f}s, audio {a['start']:.2f}s)")
    if abs(vd - expected) > tol:
        problems.append(f"video is {vd:.2f}s, expected {expected:.2f}s")
    if abs(ad - expected) > max(tol, 0.25):
        problems.append(f"audio is {ad:.2f}s, expected {expected:.2f}s")
    return (not problems), ("; ".join(problems) or f"video {vd:.2f}s audio {ad:.2f}s")


def silence_gaps(path, min_gap=8.0, noise="-45dB"):
    """Long stretches of near-silence in the audio (reported as warnings)."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path, "-af", f"silencedetect=noise={noise}:d={min_gap}",
         "-f", "null", "-"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    gaps, start = [], None
    for line in (r.stderr or "").splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            start = float(m.group(1))
        m = re.search(r"silence_end: ([\d.]+)", line)
        if m and start is not None:
            gaps.append((start, float(m.group(1))))
            start = None
    return gaps


def has_audio(path):
    return probe(path)["has_audio"]


def extract_audio(src, start, dur, out):
    """16 kHz mono wav of one segment, for speech recognition."""
    _run(["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
          "-vn", "-ac", "1", "-ar", "16000", out])
    return out


def make_proxy(src, start, dur, out):
    """Small silent copy of one segment for video analysis (keeps uploads small)."""
    _run(["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
          "-an", "-vf", "scale=-2:360,fps=2", "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "32", "-pix_fmt", "yuv420p", out])
    return out


def to_wav(src, out, rate=24000):
    """Decode any audio into the 24 kHz mono 16-bit wav the narration builder expects."""
    _run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ac", "1", "-ar", str(rate),
          "-c:a", "pcm_s16le", out])
    return out


def render_episode(src, start, dur, narration_wav, out, srt=None, workdir=".",
                   font="Noto Sans Devanagari", font_size=18, margin_v=25,
                   src_audio_volume=0.15, source_has_audio=None):
    """
    The picture is the original footage from `start` for exactly `dur` seconds,
    in order: nothing is looped, repeated, frozen or stretched. The narration
    wav is exactly `dur` long (see assemble_narration_wav).
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    out = os.path.abspath(out)
    if source_has_audio is None:
        source_has_audio = has_audio(src)
    vf = LONG_VF
    cwd = None
    if srt:
        shutil.copyfile(srt, os.path.join(workdir, "subs.srt"))
        vf += "," + subtitles_filter("subs.srt", font, font_size, margin_v)
        cwd = workdir
    fade = f"afade=t=out:st={max(0.0, dur - 0.8):.3f}:d=0.8"
    loud = "loudnorm=I=-16:TP=-1.5:LRA=11"
    if source_has_audio and src_audio_volume > 0:
        achain = (f"[0:a]volume={src_audio_volume}[s];[1:a][s]amix=inputs=2:duration=first:"
                  f"dropout_transition=0,volume=2,{loud},{fade}[a]")
    else:
        achain = f"[1:a]{loud},{fade}[a]"
    cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", os.path.abspath(src),
           "-i", os.path.abspath(narration_wav),
           "-filter_complex", f"[0:v]{vf}[v];{achain}",
           "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", out]
    _run(cmd, cwd=cwd)
    return out


def concat_videos(paths, out, workdir="."):
    """Join already-matching episode files without re-encoding."""
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    lst = os.path.join(workdir, "mega_concat.txt")
    with open(lst, "w") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    _run(["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", lst,
          "-c", "copy", "-movflags", "+faststart", os.path.abspath(out)])
    return out


def find_cut_point(src, t, tol=120.0, threshold=0.3):
    """
    Nearest picture cut (scene change) to time t within +-tol seconds, so a long
    source is split between scenes instead of mid-shot. Returns t if none found.
    """
    a = max(0.0, t - tol)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-ss", f"{a:.3f}", "-t", f"{2 * tol:.3f}", "-i", src,
         "-an", "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    cuts = [a + float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr or "")]
    return min(cuts, key=lambda c: abs(c - t)) if cuts else t


def add_cta_cue(cues, clip_dur, text, last_seconds=3.5):
    """On-screen call to action in the last seconds of a Short (inside its exact duration)."""
    start = max(0.0, clip_dur - last_seconds)
    kept = []
    for c in cues:
        if c["start"] >= start:
            continue
        kept.append(dict(c, end=min(c["end"], start)))
    kept.append({"start": start, "end": clip_dur, "text": text})
    return [c for c in kept if c["end"] - c["start"] > 0.05]


# ======================================================================
# Shorts planning
# ======================================================================
def parse_ai_picks(text, n_cues):
    """Parse Gemini's JSON answer; ignore anything invalid."""
    if not text:
        return []
    t = re.sub(r"```(?:json)?", "", text)
    a, z = t.find("["), t.rfind("]")
    if a < 0 or z <= a:
        return []
    try:
        items = json.loads(t[a:z + 1])
    except ValueError:
        return []
    picks = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            idx = int(it.get("start_cue"))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < n_cues:
            continue
        picks.append({"start_cue": idx,
                      "title": str(it.get("title") or "").strip(),
                      "description": str(it.get("description") or "").strip()})
    return picks


def _short_title(title, fallback):
    title = (title or "").strip() or fallback
    if "#shorts" not in title.lower():
        title = title[:88].rstrip() + " #Shorts"
    return title[:100]


def plan_shorts(cues, total_duration, picks=None, count=2, clip_len=50.0,
                name="video"):
    """Two non-overlapping windows. AI picks first, fixed positions as fallback."""
    clip = min(clip_len, max(10.0, total_duration / (count + 1)))
    plans = []

    def free(start, length):
        if start < 0 or start + length > total_duration + 0.01:
            return False
        return all(start >= p["start"] + p["len"] + 1 or
                   start + length + 1 <= p["start"] for p in plans)

    def add(start, title, desc):
        length = min(clip, total_duration - start)
        if length < min(clip, 15):
            start = max(0.0, total_duration - clip)
            length = min(clip, total_duration - start)
        if length <= 0 or not free(start, length):
            return False
        n = len(plans) + 1
        plans.append({"start": round(start, 2), "len": round(length, 2),
                      "title": _short_title(title, f"{name} Part {n}"),
                      "desc": (desc or f"{name} Hindi recap") + "\n\n#Shorts #Anime #HindiRecap"})
        return True

    for p in picks or []:
        if len(plans) >= count:
            break
        idx = p["start_cue"]
        if 0 <= idx < len(cues):
            add(cues[idx]["start"], p.get("title"), p.get("description"))
    for frac in (0.25, 0.6, 0.8, 0.4, 0.1):
        if len(plans) >= count:
            break
        pos = total_duration * frac
        starts = [c["start"] for c in cues if c["start"] <= pos]
        add(starts[-1] if starts else pos, None, None)
    return plans[:count]


# ======================================================================
# YouTube verification
# ======================================================================
def verify_youtube_upload(youtube, video_id):
    """
    CONFIRMED  - YouTube lists the video as uploaded/processed
    REJECTED   - YouTube reports failed/rejected/deleted
    NOT_FOUND  - API answered but did not list it
    UNVERIFIED - could not check (for example the login has no read permission)
    """
    try:
        resp = youtube.videos().list(part="status", id=video_id).execute()
    except Exception:  # noqa: BLE001
        return "UNVERIFIED"
    items = resp.get("items") or []
    if not items:
        return "NOT_FOUND"
    status = (items[0].get("status") or {}).get("uploadStatus", "")
    if status in ("uploaded", "processed"):
        return "CONFIRMED"
    if status in ("failed", "rejected", "deleted"):
        return "REJECTED"
    return "UNVERIFIED"


def _yt_summary(rec):
    ids = [rec.get(_yt_col(k)) for k in OUT_KEYS]
    if not any(ids):
        return ""
    if not all(ids):
        return "PARTIAL"
    ver = [rec.get(_verify_col(k)) for k in OUT_KEYS]
    return "CONFIRMED" if all(v == "CONFIRMED" for v in ver) else "ID_RETURNED"


# ======================================================================
# Names
# ======================================================================
def safe_name(stem, sid=""):
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:40] or "video"
    return f"{base}_{re.sub(r'[^A-Za-z0-9]', '', sid)[:6]}" if sid else base


# ======================================================================
# Summary and notifications
# ======================================================================
def write_step_summary(text, path=None):
    path = path or os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write("```\n" + text + "\n```\n")
    return True


def send_telegram(text, token=None, chat_id=None, opener=urllib.request.urlopen):
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": scrub(text)[:3900]}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    try:
        opener(req, timeout=20)
        return True
    except Exception:  # noqa: BLE001
        return False  # a notification problem must never break the run


def exit_code_for(final):
    return 0 if final == "SUCCESS" else (3 if final == "QUOTA_EXCEEDED" else 1)
