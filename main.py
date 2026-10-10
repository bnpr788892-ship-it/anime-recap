import asyncio
import io
import os
import shutil
import sys
import time

import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

import recap_core as core
import recap_pipeline as pipe
import recap_plan as plan

# ---------- Settings (GitHub Secrets / workflow env) ----------
VOICE = os.environ.get("HINDI_VOICE") or "hi-IN-MadhurNeural"   # unchanged voice
PRIVACY = os.environ.get("YT_PRIVACY") or "private"             # unchanged: private by default
GEMINI = "https://generativelanguage.googleapis.com"

_whisper = None
_model_cache = None


# ======================================================================
# Gemini (the key is always sent in a header, never in a URL)
# ======================================================================
def _gheaders(extra=None):
    h = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    h.update(extra or {})
    return h


def resolve_model():
    """GEMINI_MODEL if set, otherwise the newest Flash model this key can actually use."""
    global _model_cache
    explicit = os.environ.get("GEMINI_MODEL", "").strip()
    if explicit:
        return explicit
    if _model_cache:
        return _model_cache
    r = requests.get(f"{GEMINI}/v1beta/models?pageSize=200", headers=_gheaders(), timeout=60)
    r.raise_for_status()
    names = [m["name"] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    pick = plan.pick_gemini_model(names)
    if not pick:
        raise core.PermanentError("this Gemini key has no Flash model available; "
                                  "set the GEMINI_MODEL setting to a model name from Google AI Studio")
    print("[gemini] using model:", pick)
    _model_cache = pick
    return pick


def gemini_call(parts, json_mode=False, timeout=900, low_res=False):
    global _model_cache
    body = {"contents": [{"parts": parts}]}
    cfg = {}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    if low_res:
        cfg["mediaResolution"] = "MEDIA_RESOLUTION_LOW"
    if cfg:
        body["generationConfig"] = cfg
    url = f"{GEMINI}/v1beta/models/{resolve_model()}:generateContent"
    hdr = _gheaders({"Content-Type": "application/json"})
    r = requests.post(url, headers=hdr, json=body, timeout=timeout)
    if r.status_code == 400 and low_res:       # this model may not accept the resolution option
        body["generationConfig"].pop("mediaResolution", None)
        r = requests.post(url, headers=hdr, json=body, timeout=timeout)
    if r.status_code == 404:
        _model_cache = None
    r.raise_for_status()
    cands = r.json().get("candidates") or []
    text = "".join(p.get("text", "") for c in cands[:1]
                   for p in (c.get("content") or {}).get("parts", []))
    if not text.strip():
        raise core.TemporaryError("Gemini returned an empty answer")
    return text


def gemini(prompt, json_mode=False):
    return gemini_call([{"text": prompt}], json_mode)


def gemini_json(prompt):
    return plan.parse_json_loose(gemini(prompt, json_mode=True))


def upload_to_gemini(path):
    size = os.path.getsize(path)
    start = requests.post(
        f"{GEMINI}/upload/v1beta/files",
        headers=_gheaders({"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                           "X-Goog-Upload-Header-Content-Length": str(size),
                           "X-Goog-Upload-Header-Content-Type": "video/mp4",
                           "Content-Type": "application/json"}),
        json={"file": {"display_name": "recap-episode"}}, timeout=120)
    start.raise_for_status()
    url = start.headers["X-Goog-Upload-URL"]
    with open(path, "rb") as f:
        up = requests.post(url, headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                         "X-Goog-Upload-Command": "upload, finalize"},
                           data=f, timeout=3600)
    up.raise_for_status()
    info = up.json()["file"]
    for _ in range(120):                        # wait until Google has processed the video
        st = requests.get(f"{GEMINI}/v1beta/{info['name']}", headers=_gheaders(), timeout=60)
        st.raise_for_status()
        state = st.json().get("state")
        if state == "ACTIVE":
            return info["name"], info["uri"]
        if state == "FAILED":
            raise core.PermanentError("Gemini could not process the video file")
        time.sleep(10)
    raise core.TemporaryError("Gemini video processing took too long")


def delete_gemini_file(name):
    try:
        requests.delete(f"{GEMINI}/v1beta/{name}", headers=_gheaders(), timeout=60)
    except Exception:  # noqa: BLE001
        pass


# ======================================================================
# Speech-to-text and text-to-speech
# ======================================================================
def transcribe_timed(wav_path):
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel("small", compute_type="int8")
    segments, _ = _whisper.transcribe(wav_path, language="zh")
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


async def _synthesize(text, path):
    """Hindi voice + timing events (word or sentence boundaries, depending on edge-tts version)."""
    import edge_tts
    bounds = []
    comm = edge_tts.Communicate(text, VOICE)
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                bounds.append({"offset": chunk["offset"] / 1e7,
                               "duration": chunk["duration"] / 1e7,
                               "text": chunk["text"]})
    return bounds


def make_voice(text, path):
    return asyncio.run(_synthesize(text, path))


# ======================================================================
# YouTube upload (privacy unchanged)
# ======================================================================
def upload_youtube(youtube, path, title, desc="", tags=None):
    body = {
        "snippet": {"title": title[:100], "description": desc, "categoryId": "1",
                    "tags": tags or [], "defaultLanguage": "hi", "defaultAudioLanguage": "hi"},
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(path, chunksize=64 * 1024 * 1024, resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk(num_retries=5)
    print("Uploaded to YouTube:", resp["id"])
    return resp["id"]


# ======================================================================
# Google Sheet as persistent state (uses the existing Drive permission)
# ======================================================================
class DriveSheetBackend:
    def __init__(self, drive, sheet_id):
        self.drive, self.sheet_id = drive, sheet_id

    def read_csv(self):
        data = self.drive.files().export_media(fileId=self.sheet_id, mimeType="text/csv").execute()
        return data.decode("utf-8") if isinstance(data, bytes) else data

    def write_csv(self, text):
        media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype="text/csv", resumable=False)
        self.drive.files().update(fileId=self.sheet_id, media_body=media).execute()


# ======================================================================
# Real services behind the pipeline
# ======================================================================
STYLE = (
    "You are a skilled Indian male storyteller narrating an anime story in natural, conversational "
    "Hindi (Devanagari). Confident and expressive: suspense in mysteries, intensity in action, a warmer "
    "calmer tone in emotional scenes. No news-reader formality, no machine-translated phrasing, no "
    "filler, no repeated introductions, no generic openings such as 'आज आपका स्वागत है' or "
    "'तो दोस्तों', and do not repeat character names more than needed. Use ONLY events that appear in "
    "the scene notes or dialogue given; never invent facts, dialogue or events.")


class RealDeps:
    def __init__(self, drive, youtube, cfg):
        self.drive, self.youtube, self.cfg = drive, youtube, cfg
        self.incoming = os.environ["INCOMING_FOLDER_ID"]
        self.done = os.environ["DONE_FOLDER_ID"]
        self.workdir = "editwork"

    # ---- Drive
    def list_incoming(self):
        res = self.drive.files().list(
            q=f"'{self.incoming}' in parents and mimeType contains 'video/' and trashed=false",
            fields="files(id,name)", orderBy="name", pageSize=100).execute()
        return res.get("files", [])

    def _download(self, file_id, path):
        req = self.drive.files().get_media(fileId=file_id)
        with io.FileIO(path, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=64 * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk(num_retries=5)

    def download_source(self, file_id, path):
        self._download(file_id, path)

    def download_drive(self, file_id, path):
        self._download(file_id, path)

    def upload_drive(self, path, folder_id):
        media = MediaFileUpload(path, resumable=True, chunksize=64 * 1024 * 1024)
        f = self.drive.files().create(
            body={"name": os.path.basename(path), "parents": [folder_id]},
            media_body=media, fields="id").execute(num_retries=5)
        return f["id"]

    def drive_size(self, file_id):
        f = self.drive.files().get(fileId=file_id, fields="size").execute()
        return int(f.get("size") or 0)

    def disk_free(self):
        return shutil.disk_usage(".").free

    def move_to_done(self, file_id):
        f = self.drive.files().get(fileId=file_id, fields="parents").execute()
        self.drive.files().update(
            fileId=file_id, addParents=self.done,
            removeParents=",".join(f.get("parents", []))).execute()

    # ---- measuring and cutting
    def probe(self, path):
        return core.probe(path)

    def find_cut(self, src, t, tol):
        return core.find_cut_point(src, t, tol)

    def cut_audio(self, src, start, dur, out):
        return core.extract_audio(src, start, dur, out)

    def transcribe_timed(self, wav):
        return transcribe_timed(wav)

    # ---- Gemini: pictures
    def analyze_scenes(self, src, start, dur, transcript):
        os.makedirs(self.workdir, exist_ok=True)
        proxy = os.path.join(self.workdir, "proxy.mp4")
        core.make_proxy(src, start, dur, proxy)
        name = None
        try:
            name, uri = upload_to_gemini(proxy)
            lines = "\n".join(f"[{t['start']:.0f}s] {t['text']}" for t in transcript[:400])
            prompt = (
                f"This video is {dur:.0f} seconds long. Identify its meaningful scenes in chronological "
                "order. For each scene give: start and end as MM:SS (or HH:MM:SS), a one or two sentence "
                "factual summary of what visibly happens (English), the characters visible, and your "
                "confidence (high, medium or low). Scenes must not overlap and must cover the whole video; "
                "use short scenes during rapid cuts. Describe only what you can actually see or hear; if "
                'unsure, say so and use confidence "low". Answer ONLY with JSON: '
                '{"scenes":[{"start":"00:00","end":"00:20","summary":"...","characters":["..."],'
                '"confidence":"high"}]}.\nSpoken lines (Chinese) with times:\n' + lines)
            text = gemini_call([{"file_data": {"mime_type": "video/mp4", "file_uri": uri}},
                                {"text": prompt}], json_mode=True, low_res=True, timeout=1800)
            scenes = plan.parse_scene_answer(plan.parse_json_loose(text))
            return scenes or None
        finally:
            if name:
                delete_gemini_file(name)
            if os.path.exists(proxy):
                os.remove(proxy)

    # ---- Gemini: words
    def write_narration(self, scenes, info):
        scene_lines = []
        for s in scenes:
            scene_lines.append(
                f"- scene {s['index']} ({s['start']:.0f}s-{s['end']:.0f}s), at most {s['budget']} words. "
                f"Visible: {s.get('summary') or '(not described)'}. Dialogue (Chinese): {s.get('transcript') or '-'}")
        limit = "Use at most 60% of each word limit." if info.get("shorter") else \
            "Never exceed a scene's word limit; fewer words are fine."
        prompt = (
            f"{STYLE}\n\nStory: {info['title']}. Part {info['episode']} of {info['episodes']}. "
            f"What was just narrated (continue smoothly, do not repeat it): {info.get('prior') or '(this is the start)'}\n"
            "Write the narration for each scene below, in order, in simple spoken Hindi. Keep the most important "
            "events and character actions, drop repetition and unnecessary description. "
            f"{limit} Do not add an ending or conclusion unless the scene is the very last.\n"
            + "\n".join(scene_lines) +
            '\nAnswer ONLY with JSON: {"scenes":[{"i":<scene number>,"text":"<Hindi narration>"}]}')
        data = gemini_json(prompt)
        items = data.get("scenes") if isinstance(data, dict) else data
        return {int(x["i"]): str(x["text"]) for x in items or [] if "i" in x and x.get("text")}

    def pick_shorts(self, cues):
        lines, size = [], 0
        for i, c in enumerate(cues):
            line = f"{i}|{c['start']:.0f}s|{c['text']}"
            size += len(line)
            if size > 45000:
                break
            lines.append(line)
        prompt = (
            "Below is the Hindi narration of an anime recap, one line per subtitle: index|start|text. "
            "Choose 2 different moments that make strong YouTube Shorts: each starts at a dramatic, curious "
            "or cliffhanger line, tells a coherent mini-story, and ends at a natural suspense point. They must "
            "be at least 60 seconds apart. Do not promise anything the story does not deliver. Answer ONLY with "
            'a JSON array of 2 objects: {"start_cue": <index>, "title": "<Hindi hook title under 80 '
            'characters>", "description": "<one or two Hindi sentences>"}.\n\n' + "\n".join(lines))
        return gemini(prompt, json_mode=True)

    def _meta(self, task, data):
        prompt = (
            f"{task}\nRules: accurate to the story only; natural readable Hindi (English words allowed); no "
            "placeholder titles; no keyword stuffing; no fake credits or official affiliations; no unrelated "
            "trending tags; never mention AI tools or generator names. Answer ONLY with JSON: "
            '{"title":"...","description":"...","tags":["..."],"hashtags":["#..."],'
            '"thumbnail_text":"<under 30 characters>","thumbnail_concept":"...","cta":"<one short Hindi line>"}.\n'
            f"Material:\n{data}")
        return gemini_json(prompt)

    def episode_metadata(self, info):
        return self._meta(
            f"Write YouTube metadata for part {info['episode']} of {info['episodes']} (season {info['season']}) "
            f"of an anime story recap. Title: a unique, descriptive story title" +
            (f" followed by 'Season {info['season']} Episode {info['episode']}'" if info["episodes"] > 1 else "") +
            f"; working name of the story: {info['story']}. Description: a useful 3-5 line summary without "
            "spoilers beyond the story so far, plus a short disclosure that this is a Hindi narration/recap. "
            "Up to 12 search tags and 3 hashtags.",
            f"Scene notes: {info['summaries']}\nNarration: {info['narration']}")

    def short_metadata(self, info):
        return self._meta(
            "Write metadata for a vertical YouTube Short cut from a longer recap video. Title: a curiosity-"
            "provoking hook that is true to the clip (under 80 characters). Description: 1-2 sentences, then say "
            f"the full recap is available. Full video: {info['full_url']}. 'cta' is a short on-screen Hindi "
            "line (under 60 characters) that adapts to this clip's story and invites viewers to watch the full video "
            "in the related video / description.",
            f"Full video title: {info['title']}\nClip narration: {info['excerpt']}")

    def mega_metadata(self, info):
        eps = "\n".join(f"{e['n']}. {e['title']}: {e['summary']}" for e in info["episodes"])
        return self._meta(
            f"Write metadata for the FULL SEASON {info['season']} compilation of {info['count']} episodes "
            f"({info['hours']} hours) of the anime story recap '{info['story']}'. The title must look like "
            f"'FULL SEASON {info['season']} - All {info['count']} Episodes Complete | <story title>'. Description: "
            "overview plus a chapter-style list of the episodes.", eps)

    # ---- narration audio
    def synthesize(self, text, path):
        return make_voice(text, path)

    def audio_duration(self, path):
        return core.probe(path)["duration"]

    def to_wav(self, src, out):
        return core.to_wav(src, out)

    # ---- video
    def render_episode(self, src, start, dur, narration_wav, out, srt):
        c = self.cfg
        return core.render_episode(
            src, start, dur, narration_wav, out, srt, workdir=os.path.join(self.workdir, "ep"),
            font=c.sub_font, font_size=c.sub_size_long, margin_v=c.sub_margin_long,
            src_audio_volume=c.src_audio_volume)

    def render_short(self, long_video, window, srt, out):
        c = self.cfg
        return core.render_short(
            long_video, window["start"], window["len"], out, srt,
            workdir=os.path.join(self.workdir, "short"), font=c.sub_font,
            font_size=c.sub_size_short, margin_v=c.sub_margin_short)

    def validate_video(self, path, kind, expected, tol):
        return core.validate_output(path, kind, expected, abs_tol=tol)

    def av_check(self, path, expected, tol):
        return core.check_av_sync(path, expected, tol)

    def silence_gaps(self, path):
        return core.silence_gaps(path)

    def concat_videos(self, paths, out):
        return core.concat_videos(paths, out, workdir=os.path.join(self.workdir, "mega"))

    # ---- YouTube
    def upload_youtube(self, path, title, desc, tags):
        return upload_youtube(self.youtube, path, title, desc, tags)

    def verify_youtube(self, video_id):
        return core.verify_youtube_upload(self.youtube, video_id)


def build_services():
    creds = Credentials(
        None,
        refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/youtube.upload",
        ],
    )
    return (build("drive", "v3", credentials=creds),
            build("youtube", "v3", credentials=creds))


def finish(text, final, notify):
    print(text)
    core.write_step_summary(text)
    if notify:
        core.send_telegram(text)
    print(f"FINAL STATUS: {final}")
    sys.exit(core.exit_code_for(final))


def main():
    missing = core.missing_env()
    if missing:
        text = ("Anime Recap cannot start. Missing credentials/settings: " + ", ".join(missing)
                + "\nAdd them in GitHub: Settings > Secrets and variables > Actions.")
        print(text)
        core.write_step_summary(text)
        core.send_telegram(text)
        print("FINAL STATUS: MISCONFIGURED")
        sys.exit(2)

    try:
        cfg = core.Config.from_env()
        drive, youtube = build_services()
        store = core.StateStore(DriveSheetBackend(drive, core.sheet_id_from_env()),
                                sleep=cfg.sleep, base_delay=cfg.base_delay)
        deps = RealDeps(drive, youtube, cfg)
        summary = pipe.run_all(deps, store, cfg)
    except Exception as e:  # noqa: BLE001
        text = ("Anime Recap crashed before finishing: "
                + core.scrub(type(e).__name__ + ": " + str(e))[:500])
        finish(text, "FAILED", notify=True)
        return

    text = pipe.build_summary_text(summary)
    final = summary["final_status"]
    worked = summary["discovered"] > 0
    finish(text, final, notify=(final != "SUCCESS") or worked)


if __name__ == "__main__":
    main()
