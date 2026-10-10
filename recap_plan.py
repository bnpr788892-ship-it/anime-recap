"""
recap_plan.py - pure planning helpers (no network, no ffmpeg).

Everything here is deterministic and unit-tested: duration and word budgets,
scene-map validation, episode splitting, metadata cleaning, and the code that
builds the narration track so it can never be longer than the video.
"""
import math
import re
import wave

# ---------------- constants (all overridable through Config) -------------
DEFAULT_WPS = 2.2          # natural Hindi storytelling, words per second
NARRATION_FILL = 0.85      # use at most 85% of a scene window for speech
PAD = 0.25                 # lead-in / tail inside each scene window (seconds)
MIN_WINDOW = 4.0           # scenes shorter than this get no narration
MIN_WORDS = 3
WAV_RATE = 24000


# ======================================================================
# Text and duration budgets
# ======================================================================
def count_words(text):
    return len(str(text or "").split())


def estimate_seconds(text, wps=DEFAULT_WPS):
    return count_words(text) / wps


def word_budget(window_seconds, wps=DEFAULT_WPS, fill=NARRATION_FILL):
    """How many Hindi words fit naturally in a scene window (0 = stay silent)."""
    if window_seconds < MIN_WINDOW:
        return 0
    n = int((window_seconds - 2 * PAD) * wps * fill)
    return n if n >= MIN_WORDS else 0


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[।.?!])\s+", str(text or "").strip()) if s.strip()]


def trim_to_budget(text, budget):
    """Drop whole sentences from the end until the text fits. May return ''."""
    sents = split_sentences(text)
    while sents and count_words(" ".join(sents)) > budget:
        sents.pop()
    return " ".join(sents)


def drop_last_sentence(text):
    sents = split_sentences(text)
    return " ".join(sents[:-1])


GENERIC_OPENERS = ("आज आपका स्वागत है", "तो दोस्तों", "दोस्तों", "नमस्कार", "हेलो")


def strip_generic_opener(text):
    t = str(text or "").strip()
    for g in GENERIC_OPENERS:
        if t.startswith(g):
            t = t[len(g):].lstrip(" ,।!-—")
    return t.strip()


# ======================================================================
# Duration checks
# ======================================================================
def default_tolerance(fps=30.0):
    """Allowed difference between wanted and measured duration (seconds)."""
    return max(3.0 / fps, 0.15)


def duration_check(measured, expected, tol):
    diff = measured - expected
    return abs(diff) <= tol, round(diff, 3)


# ======================================================================
# Scene maps
# ======================================================================
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def validate_scene_map(scenes, seg_dur, min_scene=2.0):
    """
    Make a scene list safe to use: sorted, inside [0, seg_dur], no overlaps,
    no gaps. Returns (scenes, warnings). Never invents story content.
    """
    warnings = []
    clean = []
    for s in scenes or []:
        a, b = _num((s or {}).get("start")), _num((s or {}).get("end"))
        if a is None or b is None:
            warnings.append("scene without usable times ignored")
            continue
        a, b = max(0.0, a), min(seg_dur, b)
        if b - a <= 0:
            warnings.append(f"scene {a:.0f}-{b:.0f}s outside the video ignored")
            continue
        d = dict(s)
        d["start"], d["end"] = a, b
        clean.append(d)
    clean.sort(key=lambda x: x["start"])
    fixed = []
    for s in clean:
        if fixed and s["start"] < fixed[-1]["end"] - 1e-6:
            warnings.append(f"scene overlap at {s['start']:.0f}s trimmed")
            s["start"] = fixed[-1]["end"]
            if s["end"] - s["start"] <= 0:
                continue
        fixed.append(s)
    out = []
    pos = 0.0
    for s in fixed:
        if s["start"] - pos > 1e-6:
            if s["start"] - pos <= 10.0 and out:
                out[-1]["end"] = s["start"]            # small gap: extend previous scene
            else:
                out.append({"start": pos, "end": s["start"], "summary": "", "filler": True})
                warnings.append(f"gap {pos:.0f}-{s['start']:.0f}s covered by an unlabeled scene")
        out.append(s)
        pos = s["end"]
    if seg_dur - pos > 1e-6:
        if out and seg_dur - pos <= 10.0:
            out[-1]["end"] = seg_dur
        else:
            out.append({"start": pos, "end": seg_dur, "summary": "", "filler": True})
            warnings.append(f"end {pos:.0f}-{seg_dur:.0f}s covered by an unlabeled scene")
    # merge very short scenes into the previous one
    merged = []
    for s in out:
        if merged and s["end"] - s["start"] < min_scene:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)
    for i, s in enumerate(merged):
        s["index"] = i
    return merged, warnings


def windows_from_transcript(seg_dur, transcript, win=30.0):
    """
    Fallback scene map when video analysis is unavailable: equal time windows
    carrying the spoken lines of that window. The match is time-based only and
    is reported as unverified.
    """
    if seg_dur <= 0:
        return []
    n = max(1, int(seg_dur // win))
    step = seg_dur / n
    scenes = []
    for i in range(n):
        a, b = i * step, (i + 1) * step if i < n - 1 else seg_dur
        lines = [t["text"] for t in transcript or []
                 if t["end"] > a and t["start"] < b and t.get("text")]
        scenes.append({"index": i, "start": a, "end": b, "summary": "",
                       "transcript": " ".join(lines)[:1500], "confidence": "unverified"})
    return scenes


# ======================================================================
# Episodes and seasons
# ======================================================================
def plan_episodes(total, episode_len=3600.0, min_last=600.0, snap_points=None, snap_tol=120.0,
                  snap_fn=None):
    """
    Split [0, total] into contiguous episodes of about episode_len. A leftover
    shorter than min_last is folded into the previous episode, so nothing
    tiny is ever produced. Boundaries snap to nearby scene boundaries.
    The episodes always cover the source with no gap and no overlap.
    """
    if total <= 0:
        return []
    bounds, k = [0.0], 1
    while True:
        t = k * episode_len
        remaining = total - t
        if remaining <= 0.001 or remaining < min_last:
            break
        b = t
        cand = None
        if snap_fn:
            cand = snap_fn(t)
        elif snap_points:
            cand = min(snap_points, key=lambda p: abs(p - t))
        if cand is not None and abs(cand - t) <= snap_tol and cand > bounds[-1] + 60 \
                and total - cand >= min_last:
            b = cand
        bounds.append(b)
        k += 1
    bounds.append(total)
    return [{"n": i + 1, "start": round(bounds[i], 3), "end": round(bounds[i + 1], 3)}
            for i in range(len(bounds) - 1)]


def check_episode_coverage(episodes, total, tol=0.002):
    """Problems list: empty means no gaps, overlaps or missing parts."""
    problems = []
    if not episodes:
        return ["no episodes"]
    if abs(episodes[0]["start"]) > tol:
        problems.append("first episode does not start at 0")
    for a, b in zip(episodes, episodes[1:]):
        if abs(a["end"] - b["start"]) > tol:
            problems.append(f"gap/overlap between episode {a['n']} and {b['n']}")
    if abs(episodes[-1]["end"] - total) > tol:
        problems.append("last episode does not end at the source end")
    return problems


def season_id(source_id, season_no=1):
    return f"S{season_no}-{re.sub(r'[^A-Za-z0-9]', '', source_id)[:10]}"


def episode_key(source_id, n):
    return f"{source_id}:E{n:02d}"


def mega_key(source_id):
    return f"{source_id}:MEGA"


# ======================================================================
# YouTube ids and metadata
# ======================================================================
def is_valid_youtube_id(value):
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", str(value or "")))


def youtube_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


BANNED_WORDS = ("gemini", "claude", "chatgpt", "openai", "edge-tts", "edge tts",
                "as an ai", "ai generated", "language model")
PLACEHOLDER_TITLE = re.compile(r"^\s*(untitled|video|title|placeholder|test|recap)\s*\d*\s*$", re.I)


def _clean(s):
    return " ".join(str(s or "").replace("<", " ").replace(">", " ").split())


def _has_banned(s):
    low = str(s).lower()
    return any(w in low for w in BANNED_WORDS)


def _cut(s, n):
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return (cut or s[:n]).rstrip(" ,.-—")


def sanitize_metadata(meta, fallback_title, max_hashtags=4):
    """Enforce YouTube limits and remove placeholders / generator names."""
    meta = meta if isinstance(meta, dict) else {}
    title = _clean(meta.get("title"))
    if not title or PLACEHOLDER_TITLE.match(title) or _has_banned(title):
        title = _clean(fallback_title)
    title = _cut(title, 100)
    lines = [l for l in str(meta.get("description") or "").replace("\r", "").split("\n")
             if not _has_banned(l)]
    desc = "\n".join(_clean(l) if l.strip() else "" for l in lines).strip()
    tags, seen, total = [], set(), 0
    for t in meta.get("tags") or []:
        t = _clean(t).lstrip("#")
        if not t or len(t) > 30 or _has_banned(t) or t.lower() in seen:
            continue
        if total + len(t) + 1 > 450 or len(tags) >= 15:
            break
        seen.add(t.lower())
        tags.append(t)
        total += len(t) + 1
    hashtags = []
    for h in meta.get("hashtags") or []:
        h = "#" + re.sub(r"\s+", "", _clean(h).lstrip("#"))
        if len(h) > 1 and not _has_banned(h) and h.lower() not in [x.lower() for x in hashtags]:
            hashtags.append(h)
    hashtags = hashtags[:max_hashtags]
    if hashtags:
        desc = (desc + "\n\n" + " ".join(hashtags)).strip()
    return {"title": title, "description": _cut(desc, 4800), "tags": tags,
            "hashtags": hashtags,
            "thumbnail_text": _cut(_clean(meta.get("thumbnail_text")), 40),
            "thumbnail_concept": _cut(_clean(meta.get("thumbnail_concept")), 300)}


# ======================================================================
# Narration track (can never be longer than the video)
# ======================================================================
def offset_cues(cues, offset, clip_end):
    out = []
    for c in cues:
        s, e = c["start"] + offset, min(c["end"] + offset, clip_end)
        if e - s > 0.05:
            out.append({"start": s, "end": e, "text": c["text"]})
    return out


def validate_subtitles(cues, total, clips=None, eps=0.05):
    """Returns a list of problems (empty = fine)."""
    problems = []
    last_end = 0.0
    for i, c in enumerate(cues):
        if c["start"] < -eps:
            problems.append(f"cue {i} starts before 0")
        if c["end"] <= c["start"]:
            problems.append(f"cue {i} has no length")
        if c["end"] > total + eps:
            problems.append(f"cue {i} runs past the video end")
        if c["start"] < last_end - eps:
            problems.append(f"cue {i} overlaps the previous cue")
        last_end = max(last_end, c["end"])
        if clips and not any(c["start"] >= k["start"] - eps and
                             c["end"] <= k["start"] + k["dur"] + eps for k in clips):
            problems.append(f"cue {i} is not inside any spoken clip")
    return problems


def assemble_narration_wav(clips, total_seconds, out_path, rate=WAV_RATE):
    """
    clips: [{'start': sec, 'dur': sec, 'wav': path}] (24 kHz mono 16-bit wavs).
    Writes ONE wav that is exactly total_seconds long: silence in the gaps,
    each clip at its scene time. Raises if anything would overlap or overrun,
    so the narration can never push the video past its source duration.
    """
    clips = sorted(clips, key=lambda c: c["start"])
    total_frames = int(round(total_seconds * rate))
    pos = 0
    with wave.open(out_path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)

        def silence(n):
            while n > 0:
                k = min(n, rate)
                out.writeframes(b"\x00\x00" * k)
                n -= k
        for c in clips:
            with wave.open(c["wav"], "rb") as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, rate):
                    raise ValueError("narration clip must be 24 kHz mono 16-bit wav")
                frames = w.readframes(w.getnframes())
            start_f = int(round(c["start"] * rate))
            n = len(frames) // 2
            if start_f < pos:
                raise ValueError("narration clips overlap")
            if start_f + n > total_frames:
                raise ValueError("narration would run past the end of the video")
            silence(start_f - pos)
            out.writeframes(frames)
            pos = start_f + n
        silence(total_frames - pos)
    return out_path


# ======================================================================
# Shorts planning (story-aware, snaps to subtitle boundaries)
# ======================================================================
def plan_shorts_v2(cues, total, picks=None, count=2, min_len=20.0, max_len=58.0, name="video"):
    plans = []

    def free(a, b):
        return all(a >= p["start"] + p["len"] + 1 or b + 1 <= p["start"] for p in plans)

    def window(start):
        start = max(0.0, min(start, max(0.0, total - min_len)))
        ends = [c["end"] for c in cues if start + min_len <= c["end"] <= start + max_len]
        end = max(ends) if ends else min(total, start + max_len)
        if end - start < min_len * 0.75:
            return None
        return start, end

    def add(start, title, desc):
        w = window(start)
        if not w or not free(*w):
            return False
        n = len(plans) + 1
        plans.append({"start": round(w[0], 2), "len": round(w[1] - w[0], 2),
                      "title": title or f"{name} Part {n}", "desc": desc or "",
                      "cta_start": round(max(0.0, (w[1] - w[0]) - 3.5), 2)})
        return True

    for p in picks or []:
        if len(plans) >= count:
            break
        i = p.get("start_cue")
        if isinstance(i, int) and 0 <= i < len(cues):
            add(cues[i]["start"], p.get("title"), p.get("description"))
    for frac in (0.25, 0.6, 0.8, 0.4, 0.1):
        if len(plans) >= count:
            break
        pos = total * frac
        starts = [c["start"] for c in cues if c["start"] <= pos]
        add(starts[-1] if starts else pos, None, None)
    return plans[:count]


def wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ======================================================================
# Gemini helpers that need no network
# ======================================================================
def parse_timestamp(v):
    """'MM:SS', 'HH:MM:SS', '12.5' or a number -> seconds (None if unusable)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().replace(",", ".").split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    t = 0.0
    for n in nums:
        t = t * 60 + n
    return t


_VERSION = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash$")
_AVOID = ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "preview", "exp", "vision")


def pick_gemini_model(names):
    """
    Choose a text+video capable Flash model from the list this API key can use.
    Prefer the newest stable 'gemini-X-flash', then 'gemini-flash-latest',
    then any other flash model. Returns None if nothing suitable exists.
    """
    names = [n.split("/")[-1] for n in names]
    stable = sorted(((float(m.group(1)), n) for n in names for m in [_VERSION.match(n)] if m),
                    reverse=True)
    if stable:
        return stable[0][1]
    if "gemini-flash-latest" in names:
        return "gemini-flash-latest"
    other = [n for n in names if "flash" in n and not any(a in n for a in _AVOID)]
    if other:
        return sorted(other)[-1]
    anyflash = [n for n in names if "flash" in n and not any(a in n for a in ("image", "tts", "live", "audio", "embedding"))]
    return sorted(anyflash)[-1] if anyflash else None


def parse_json_loose(text):
    """Parse a JSON answer that may be wrapped in code fences or chatter."""
    import json
    t = re.sub(r"```(?:json)?", "", str(text or "")).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        a, z = t.find(opener), t.rfind(closer)
        if a >= 0 and z > a:
            try:
                return json.loads(t[a:z + 1])
            except ValueError:
                continue
    raise ValueError("no JSON found in the model answer")


def parse_scene_answer(data):
    """Gemini scene list -> scenes with seconds. Unusable items are skipped."""
    items = data.get("scenes") if isinstance(data, dict) else data
    scenes = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        a, b = parse_timestamp(it.get("start")), parse_timestamp(it.get("end"))
        if a is None or b is None or b <= a:
            continue
        scenes.append({"start": a, "end": b, "summary": str(it.get("summary") or "")[:600],
                       "characters": it.get("characters") or [],
                       "confidence": str(it.get("confidence") or "medium").lower()})
    return scenes
