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

COLUMNS = [
    "source_id", "filename", "date_received", "status",
    "started_at", "completed_at",
    "transcription_status", "script_status", "voice_status",
    "long_status", "shorts_status", "youtube_status",
    "yt_long_id", "yt_short1_id", "yt_short2_id",
    "yt_long_verify", "yt_short1_verify", "yt_short2_verify",
    "drive_srt_link", "drive_long_link", "drive_short1_link", "drive_short2_link",
    "short_plan", "retry_count", "failed_stage", "error", "last_updated",
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
        )


REQUIRED_ENV = ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
                "GOOGLE_REFRESH_TOKEN", "GEMINI_API_KEY",
                "INCOMING_FOLDER_ID", "LONG_FOLDER_ID",
                "SHORTS_FOLDER_ID", "DONE_FOLDER_ID"]


def sheet_id_from_env(env=None):
    e = env if env is not None else os.environ
    return e.get("GOOGLE_SHEET_ID") or e.get("SHEET_ID") or ""


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
    One row per source video, keyed by the Google Drive file ID.
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
        migrated = False
        if "source_id" not in header:
            if header and header[0] in OLD_HEADER_MAP:
                header = [OLD_HEADER_MAP.get(h, h) for h in header]
                migrated = True
            else:
                return  # unknown sheet layout: treat as empty
        for cells in parsed[1:]:
            rec = dict.fromkeys(COLUMNS, "")
            for h, v in zip(header, cells):
                if h in rec:
                    rec[h] = v.strip()
            if not rec["source_id"]:
                continue
            if migrated:
                self._migrate_row(rec)
            existing = self.rows.get(rec["source_id"])
            if existing is None or existing["status"] != COMPLETED:
                self.rows[rec["source_id"]] = rec  # one row per source
        if migrated:
            self.save()

    @staticmethod
    def _migrate_row(rec):
        st = rec["status"]
        if st == "DONE":
            rec["status"] = COMPLETED
        elif st.startswith("FAILED"):
            rec["error"] = st[:150]
            rec["status"] = RETRY_PENDING
        elif not st:
            rec["status"] = PENDING

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
    def get(self, sid):
        return self.rows.get(sid)

    def upsert(self, sid, **fields):
        rec = self.rows.get(sid)
        if rec is None:
            rec = dict.fromkeys(COLUMNS, "")
            rec["source_id"] = sid
            self.rows[sid] = rec
        for k, v in fields.items():
            if k not in COLUMNS:
                raise KeyError(f"unknown column {k}")
            rec[k] = "" if v is None else str(v)
        rec["last_updated"] = fmt_ts(utcnow())
        self.save()
        return rec

    def claim(self, sid, filename, now_text):
        """Mark a source as PROCESSING. Counts a dead earlier run as a try."""
        rec = self.rows.get(sid) or dict.fromkeys(COLUMNS, "")
        retries = int(rec.get("retry_count") or 0)
        if rec.get("status") in ACTIVE_STATUSES:
            retries += 1  # an earlier run started this and never finished
        return self.upsert(
            sid, filename=filename, status=PROCESSING,
            started_at=rec.get("started_at") or now_text,
            retry_count=retries, failed_stage="", error="", completed_at="")


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


def validate_output(path, kind="long", expected_duration=None, tolerance=0.1):
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
        allowed = max(2.0, expected_duration * tolerance)
        if abs(info["duration"] - expected_duration) > allowed:
            raise MediaError(
                f"{os.path.basename(path)} is {info['duration']:.1f}s, "
                f"expected about {expected_duration:.1f}s")
    if kind == "short" and info["height"] <= info["width"]:
        raise MediaError(f"{os.path.basename(path)} is not vertical")
    return info


def plan_segments(src_dur, target_dur, seg_len=8.0):
    """
    Which pieces of the source to use so the footage covers the whole story
    in order (instead of looping the first seconds). Returns [(start, length)].
    """
    if src_dur <= 0 or target_dur <= 0:
        return []
    if target_dur <= src_dur:
        n = max(1, math.ceil(target_dur / seg_len))
        seg = target_dur / n
        return [(i * (src_dur / n), seg) for i in range(n)]
    # narration is longer than the footage: walk through the source and repeat
    segs, total, pos, guard = [], 0.0, 0.0, 0
    while total < target_dur - 0.01 and guard < 100000:
        guard += 1
        length = min(seg_len, src_dur - pos)
        if length < 0.5 and pos > 0:
            pos = 0.0
            length = min(seg_len, src_dur)
        # never ask for a sliver shorter than 0.5s; the final -t trims any overshoot
        length = min(length, max(target_dur - total, 0.5), src_dur - pos)
        if length <= 0:
            break
        segs.append((pos, length))
        total += length
        pos += length
        if pos >= src_dur - 0.01:
            pos = 0.0
    return segs


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


def make_long_video_legacy(src, voice, out):
    """The original editor (loops the footage). Kept as the fallback."""
    dur = probe(voice)["duration"]
    _run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", src, "-i", voice,
          "-map", "0:v", "-map", "1:a", "-t", str(dur),
          "-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "veryfast",
          "-c:a", "aac", out])
    return out


def make_long_video_advanced(src, voice, out, srt=None, workdir=".",
                             seg_len=8.0, font="Noto Sans Devanagari",
                             font_size=18, margin_v=25):
    """
    Footage is taken from across the whole source in order, narration is
    loudness-normalized with short fades, subtitles are optional.
    (This does not understand the story: it is not semantic scene matching.)
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    out = os.path.abspath(out)
    target = probe(voice)["duration"]
    sdur = probe(src)["duration"]
    segs = plan_segments(sdur, target, seg_len)
    if not segs:
        raise MediaError("could not plan any footage segments")
    clips = []
    try:
        for i, (st, ln) in enumerate(segs):
            clip = os.path.join(workdir, f"seg_{i:04d}.mp4")
            _run(["ffmpeg", "-y", "-ss", f"{st:.3f}", "-i", src,
                  "-t", f"{ln:.3f}", "-vf", LONG_VF, "-an",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                  "-pix_fmt", "yuv420p", clip])
            clips.append(clip)
        lst = os.path.join(workdir, "concat.txt")
        with open(lst, "w") as f:
            for c in clips:
                f.write(f"file '{c}'\n")
        af = (f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d=0.5,"
              f"afade=t=out:st={max(0.0, target - 1.0):.3f}:d=1.0")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
               "-i", voice, "-map", "0:v", "-map", "1:a", "-t", f"{target:.3f}"]
        cwd = None
        if srt:
            shutil.copyfile(srt, os.path.join(workdir, "subs.srt"))
            cmd += ["-vf", subtitles_filter("subs.srt", font, font_size, margin_v),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p"]
            cwd = workdir
        else:
            cmd += ["-c:v", "copy"]
        cmd += ["-af", af, "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", out]
        _run(cmd, cwd=cwd)
    finally:
        for c in clips:
            try:
                os.remove(c)
            except OSError:
                pass
    return out


def render_long_with_fallback(editors, src, voice, srt, out, expected_duration,
                              validate=validate_output, log=print):
    """
    Try: advanced + subtitles -> advanced without subtitles -> original loop editor.
    editors = {'advanced': f(src, voice, out, srt), 'legacy': f(src, voice, out)}
    Returns the name of the editor that produced a valid file.
    """
    attempts = []
    if srt:
        attempts.append(("advanced+subtitles",
                         lambda: editors["advanced"](src, voice, out, srt)))
    attempts.append(("advanced", lambda: editors["advanced"](src, voice, out, None)))
    attempts.append(("legacy", lambda: editors["legacy"](src, voice, out)))
    last = None
    for name, fn in attempts:
        try:
            fn()
            validate(out, "long", expected_duration)
            return name
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"[editor] {name} failed: {scrub(e)[:200]}")
            try:
                os.remove(out)
            except OSError:
                pass
    raise MediaError(f"all video editors failed: {scrub(last)[:200]}")


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
# Pipeline for one source video (resumable)
# ======================================================================
def safe_name(stem, sid=""):
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:40] or "video"
    return f"{base}_{re.sub(r'[^A-Za-z0-9]', '', sid)[:6]}" if sid else base


def process_source(deps, store, source, cfg, workdir, log=print):
    """
    deps provides: download_source, transcribe, write_script, synthesize,
    duration, render_long, render_short, validate, pick_shorts, upload_drive,
    download_drive, upload_youtube, verify_youtube, move_to_done.
    Every finished output is written to the sheet immediately, so a later
    failure never causes it to be created or uploaded again.
    """
    sid, name = source["id"], source["name"]
    stem = os.path.splitext(name)[0]
    safe = safe_name(stem, sid)
    os.makedirs(workdir, exist_ok=True)
    stats = {"long_generated": 0, "shorts_generated": 0,
             "yt_uploaded": 0, "yt_confirmed": 0}
    ctx = {}
    store.claim(sid, name, cfg.now())

    def row():
        return store.get(sid)

    def has(col):
        return bool((row().get(col) or "").strip())

    def stage(status, fn, col=None, retry=True):
        store.upsert(sid, status=status)
        try:
            if retry:
                return with_retries(fn, attempts=cfg.attempts,
                                    base_delay=cfg.base_delay, sleep=cfg.sleep,
                                    label=status, log=log)
            return fn()
        except Exception as e:  # noqa: BLE001
            raise StageError(status, e, col) from e

    def p(suffix):
        return os.path.join(workdir, f"{safe}{suffix}")

    # ---- generation (only when something still has to be rendered) ----
    def get_source():
        if "src" not in ctx:
            path = p("_source.mp4")
            stage(TRANSCRIBING, lambda: deps.download_source(sid, path),
                  "transcription_status")
            ctx["src"] = path
        return ctx["src"]

    def ensure_generation():
        if "cues" in ctx:
            return
        src = get_source()
        zh = stage(TRANSCRIBING, lambda: deps.transcribe(src), "transcription_status")
        store.upsert(sid, transcription_status="DONE")
        script = stage(GENERATING_SCRIPT, lambda: deps.write_script(zh), "script_status")
        store.upsert(sid, script_status="DONE")
        voice = p("_voice.mp3")
        bounds = stage(GENERATING_VOICE, lambda: deps.synthesize(script, voice),
                       "voice_status")
        store.upsert(sid, voice_status="DONE")
        vdur = deps.duration(voice)
        ctx.update(voice=voice, voice_duration=vdur,
                   cues=make_cues(bounds, vdur, script))

    def get_cues():
        if "cues" not in ctx:
            if has("drive_srt_link"):
                srt_local = p("_resume.srt")
                stage(UPLOADING, lambda: deps.download_drive(
                    id_from_link(row()["drive_srt_link"]), srt_local), "shorts_status")
                with open(srt_local, encoding="utf-8") as f:
                    ctx["cues"] = parse_srt(f.read())
            else:
                ensure_generation()
        return ctx["cues"]

    # ---- long video ----
    def build_long():
        ensure_generation()
        src = get_source()
        srt_path = p("_hindi.srt")
        write_srt(ctx["cues"], srt_path, cfg.wrap_landscape)
        out = p("_hindi_recap.mp4")
        stage(EDITING, lambda: deps.render_long(
            src, ctx["voice"], srt_path if cfg.burn_subs else None, out),
            "long_status", retry=False)
        stage(EDITING, lambda: deps.validate(out, "long", ctx["voice_duration"]),
              "long_status", retry=False)
        # subtitle file first, then the video: a recorded long video always
        # has its subtitle file recorded too (needed to resume Shorts).
        srt_id = stage(UPLOADING, lambda: deps.upload_drive(srt_path, cfg.long_dir),
                       "long_status")
        store.upsert(sid, drive_srt_link=drive_link(srt_id))
        long_id = stage(UPLOADING, lambda: deps.upload_drive(out, cfg.long_dir),
                        "long_status")
        store.upsert(sid, drive_long_link=drive_link(long_id), long_status="DONE")
        ctx["long"] = out
        stats["long_generated"] = 1

    def long_local():
        if "long" not in ctx:
            path = p("_hindi_recap.mp4")
            stage(UPLOADING, lambda: deps.download_drive(
                id_from_link(row()["drive_long_link"]), path), "youtube_status")
            ctx["long"] = path
        return ctx["long"]

    def upload_yt(key, path, title, desc):
        vid = stage(UPLOADING, lambda: deps.upload_youtube(path, title, desc),
                    "youtube_status")
        try:
            store.upsert(sid, **{_yt_col(key): vid})
        except Exception:
            log(f"!!! {key} WAS UPLOADED to YouTube (id={vid}) but saving it to "
                f"the sheet failed. Add it by hand to avoid a duplicate upload.")
            raise
        stats["yt_uploaded"] += 1
        try:
            verdict = deps.verify_youtube(vid)
        except Exception:  # noqa: BLE001
            verdict = "UNVERIFIED"
        if verdict == "CONFIRMED":
            stats["yt_confirmed"] += 1
        store.upsert(sid, **{_verify_col(key): verdict})
        store.upsert(sid, youtube_status=_yt_summary(row()))

    if not has("drive_long_link"):
        build_long()
    if not has(_yt_col("long")):
        upload_yt("long", long_local(), f"{stem} Hindi Recap",
                  "Hindi recap of the story.")

    # ---- shorts ----
    def get_plan():
        raw = (row().get("short_plan") or "").strip()
        if raw:
            try:
                plan = json.loads(raw)
                if isinstance(plan, list) and len(plan) >= 2:
                    return plan
            except ValueError:
                pass
        cues = get_cues()
        total = deps.duration(long_local())
        try:
            picks = parse_ai_picks(deps.pick_shorts(cues), len(cues)) \
                if hasattr(deps, "pick_shorts") else []
        except Exception as e:  # noqa: BLE001
            log(f"[shorts] AI clip choice failed, using fixed positions: {scrub(e)[:150]}")
            picks = []
        plan = plan_shorts(cues, total, picks, clip_len=cfg.clip_len, name=stem)
        store.upsert(sid, short_plan=json.dumps(plan, ensure_ascii=False))
        return plan

    plan = None
    for i, key in enumerate(("short1", "short2")):
        if has(_yt_col(key)):
            continue
        path = p(f"_{key}.mp4")
        if has(_drive_col(key)):
            stage(UPLOADING, lambda: deps.download_drive(
                id_from_link(row()[_drive_col(key)]), path), "shorts_status")
        else:
            plan = plan or get_plan()
            w = plan[i]
            cues = get_cues()
            sl = slice_cues(cues, w["start"], w["start"] + w["len"])
            srt_path = p(f"_{key}.srt")
            write_srt(sl, srt_path, cfg.wrap_portrait)
            lv = long_local()
            stage(EDITING, lambda: deps.render_short(
                lv, w, srt_path if (cfg.burn_subs and sl) else None, path),
                "shorts_status", retry=False)
            stage(EDITING, lambda: deps.validate(path, "short", w["len"]),
                  "shorts_status", retry=False)
            sid_drive = stage(UPLOADING, lambda: deps.upload_drive(path, cfg.shorts_dir),
                              "shorts_status")
            store.upsert(sid, **{_drive_col(key): drive_link(sid_drive)})
            stats["shorts_generated"] += 1
        stored = (row().get("short_plan") or "").strip()
        try:
            meta = json.loads(stored)[i] if stored else {}
        except (ValueError, IndexError):
            meta = {}
        upload_yt(key, path, meta.get("title") or f"{stem} Part {i + 1} #Shorts",
                  meta.get("desc") or f"{stem} Hindi recap #Shorts")
    if has("drive_short1_link") and has("drive_short2_link"):
        store.upsert(sid, shorts_status="DONE")

    # ---- finish: only when every output exists and is recorded ----
    r = row()
    missing = [k for k in OUT_KEYS if not (r.get(_yt_col(k)) and r.get(_drive_col(k)))]
    if missing:
        raise StageError("FINALIZE", RuntimeError(f"outputs missing: {missing}"))
    store.upsert(sid, status=COMPLETED, completed_at=cfg.now(), error="",
                 failed_stage="", youtube_status=_yt_summary(row()))
    try:
        with_retries(lambda: deps.move_to_done(sid), attempts=cfg.attempts,
                     base_delay=cfg.base_delay, sleep=cfg.sleep, label="move to Done",
                     log=log)
    except Exception as e:  # noqa: BLE001
        log(f"[warn] marked COMPLETED but could not move to Done yet "
            f"(will retry next run): {scrub(e)[:150]}")
    return {"status": COMPLETED, **stats}


# ======================================================================
# Whole run: discover, skip duplicates, process, summarize
# ======================================================================
def final_status(summary):
    if summary["quota_exceeded"]:
        return "QUOTA_EXCEEDED"
    if summary["failed"] and summary["succeeded"]:
        return "PARTIAL"
    if summary["failed"]:
        return "FAILED"
    return "SUCCESS"


def run_all(deps, store, cfg, workdir="work", log=print):
    store.load()
    s = {"discovered": 0, "new": 0, "duplicates_skipped": 0, "succeeded": 0,
         "failed": [], "long_generated": 0, "shorts_generated": 0,
         "yt_uploaded": 0, "yt_confirmed": 0, "awaiting_retry": 0,
         "gave_up": 0, "quota_exceeded": False, "final_status": ""}
    found = deps.list_incoming()
    s["discovered"] = len(found)
    for f in found:
        rec = store.get(f["id"])
        if rec is None:
            store.upsert(f["id"], filename=f["name"], date_received=cfg.now(),
                         status=PENDING)
            s["new"] += 1
        elif rec["status"] == COMPLETED:
            s["duplicates_skipped"] += 1
            log(f"[duplicate] already completed, not processing again: {f['name']}")
            try:
                deps.move_to_done(f["id"])
            except Exception as e:  # noqa: BLE001
                log(f"[warn] could not move duplicate to Done: {scrub(e)[:120]}")
    now = cfg.clock()
    eligible = [f for f in found if is_eligible(store.get(f["id"]), cfg, now)]
    for f in eligible[:cfg.max_videos]:
        sub = os.path.join(workdir, safe_name(os.path.splitext(f["name"])[0], f["id"]))
        try:
            res = process_source(deps, store, f, cfg, sub, log)
            s["succeeded"] += 1
            for k in ("long_generated", "shorts_generated", "yt_uploaded", "yt_confirmed"):
                s[k] += res[k]
        except Exception as e:  # noqa: BLE001
            stage_name, col, orig = "UNKNOWN", None, e
            if isinstance(e, StageError):
                stage_name, col, orig = e.stage, e.column, e.original
            kind = classify_error(orig)
            msg = scrub(f"{type(orig).__name__}: {orig}")[:300]
            rec = store.get(f["id"]) or {}
            retries = int(rec.get("retry_count") or 0)
            if kind == "quota":
                status = RETRY_PENDING          # not the video's fault: no retry used
            elif kind == "permanent":
                status = FAILED
            else:
                retries += 1
                status = FAILED if retries >= cfg.max_retries else RETRY_PENDING
            fields = dict(status=status, failed_stage=stage_name, error=msg,
                          retry_count=retries)
            if col:
                fields[col] = "FAILED"
            store.upsert(f["id"], **fields)
            s["failed"].append({"name": f["name"], "stage": stage_name,
                                "error": msg, "status": status})
            log(f"[FAILED] {f['name']} at {stage_name}: {msg}")
            if kind == "quota":
                s["quota_exceeded"] = True
                break
        finally:
            shutil.rmtree(sub, ignore_errors=True)
    s["awaiting_retry"] = sum(1 for r in store.rows.values() if r["status"] == RETRY_PENDING)
    s["gave_up"] = sum(1 for r in store.rows.values() if r["status"] == FAILED)
    s["final_status"] = final_status(s)
    return s


# ======================================================================
# Summary and notifications
# ======================================================================
def build_summary_text(s):
    lines = [
        "Anime Recap daily summary",
        f"Final status: {s['final_status']}",
        f"Sources found in Incoming: {s['discovered']} (new: {s['new']}, "
        f"already completed and skipped: {s['duplicates_skipped']})",
        f"Videos processed successfully: {s['succeeded']}",
        f"Videos failed this run: {len(s['failed'])}",
        f"Long videos generated: {s['long_generated']}",
        f"Shorts generated: {s['shorts_generated']}",
        f"YouTube uploads (ID returned): {s['yt_uploaded']}",
        f"YouTube uploads confirmed by YouTube: {s['yt_confirmed']}",
        f"Jobs awaiting retry: {s['awaiting_retry']}",
        f"Jobs that gave up (FAILED): {s['gave_up']}",
    ]
    if s["quota_exceeded"]:
        lines.append("API quota was exceeded. The run stopped early; "
                     "remaining jobs wait for the next run.")
    for f in s["failed"]:
        lines.append(f"- {f['name']}: failed at {f['stage']} -> {f['status']} "
                     f"({f['error'][:120]})")
    if s["yt_uploaded"] > s["yt_confirmed"]:
        lines.append("Note: some uploads returned an ID but could not be "
                     "independently confirmed (the YouTube login may lack read permission).")
    return "\n".join(lines)


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
