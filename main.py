import asyncio
import csv
import io
import os
import subprocess
import sys

import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

import recap_core as core

# ---------- Settings (come from GitHub Secrets / workflow env) ----------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = os.environ.get("HINDI_VOICE", "hi-IN-MadhurNeural")
PRIVACY = os.environ.get("YT_PRIVACY", "private")  # unchanged default: private

_whisper = None


# ======================================================================
# Original building blocks (kept; the pipeline in recap_core calls them)
# ======================================================================
def run(cmd):
    subprocess.run(cmd, check=True)


def duration(path):
    return core.probe(path)["duration"]


def transcribe(path):
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel("small", compute_type="int8")
    segments, _ = _whisper.transcribe(path, language="zh")
    return " ".join(s.text for s in segments)


def gemini(prompt, timeout=300):
    # The key goes in a header, not in the URL, so it can never end up in an error message.
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    r = requests.post(
        url, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"],
                      "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=timeout)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def hindi_script(chinese_text):
    prompt = (
        "Below is a Chinese transcript of an anime episode. Write an engaging "
        "Hindi recap script (storytelling style, simple Hindi, no headings, "
        "no stage directions, just the narration text).\n\n" + chinese_text[:60000]
    )
    return gemini(prompt)


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


def make_long_video(src, voice, out):
    """Original editor (loops the footage). Used as the fallback."""
    return core.make_long_video_legacy(src, voice, out)


def make_shorts(long_video, prefix):
    """Original fixed-position Shorts. Kept for reference; the pipeline now plans clips itself."""
    total = duration(long_video)
    paths = []
    for i, frac in enumerate([0.25, 0.6], start=1):
        out = f"{prefix}_short{i}.mp4"
        run(["ffmpeg", "-y", "-ss", str(total * frac), "-i", long_video, "-t", "50",
             "-vf", "crop=ih*9/16:ih,scale=1080:1920",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", out])
        paths.append(out)
    return paths


def upload_youtube(youtube, path, title, desc=""):
    body = {
        "snippet": {"title": title[:100], "description": desc, "categoryId": "1"},
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(path, chunksize=-1, resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    print("Uploaded to YouTube:", resp["id"])
    return resp["id"]


# ======================================================================
# Google Sheet as persistent state (uses the existing Drive permission)
# ======================================================================
class DriveSheetBackend:
    def __init__(self, drive, sheet_id):
        self.drive, self.sheet_id = drive, sheet_id

    def read_csv(self):
        data = self.drive.files().export_media(
            fileId=self.sheet_id, mimeType="text/csv").execute()
        return data.decode("utf-8") if isinstance(data, bytes) else data

    def write_csv(self, text):
        media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")),
                                  mimetype="text/csv", resumable=False)
        self.drive.files().update(fileId=self.sheet_id, media_body=media).execute()


# ======================================================================
# Real services behind the pipeline
# ======================================================================
class RealDeps:
    def __init__(self, drive, youtube, cfg):
        self.drive, self.youtube, self.cfg = drive, youtube, cfg
        self.incoming = os.environ["INCOMING_FOLDER_ID"]
        self.done = os.environ["DONE_FOLDER_ID"]
        self.workdir = "editwork"

    # Drive
    def list_incoming(self):
        res = self.drive.files().list(
            q=f"'{self.incoming}' in parents and mimeType contains 'video/' and trashed=false",
            fields="files(id,name)", orderBy="name", pageSize=100).execute()
        return res.get("files", [])

    def _download(self, file_id, path):
        req = self.drive.files().get_media(fileId=file_id)
        with io.FileIO(path, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = dl.next_chunk()

    def download_source(self, file_id, path):
        self._download(file_id, path)

    def download_drive(self, file_id, path):
        self._download(file_id, path)

    def upload_drive(self, path, folder_id):
        media = MediaFileUpload(path, resumable=True)
        f = self.drive.files().create(
            body={"name": os.path.basename(path), "parents": [folder_id]},
            media_body=media, fields="id").execute()
        return f["id"]

    def move_to_done(self, file_id):
        f = self.drive.files().get(fileId=file_id, fields="parents").execute()
        self.drive.files().update(
            fileId=file_id, addParents=self.done,
            removeParents=",".join(f.get("parents", []))).execute()

    # AI / audio
    def transcribe(self, path):
        return transcribe(path)

    def write_script(self, zh_text):
        return hindi_script(zh_text)

    def synthesize(self, text, path):
        return make_voice(text, path)

    def duration(self, path):
        return duration(path)

    def pick_shorts(self, cues):
        lines, size = [], 0
        for i, c in enumerate(cues):
            line = f"{i}|{c['start']:.0f}s|{c['text']}"
            size += len(line)
            if size > 45000:
                break
            lines.append(line)
        prompt = (
            "Below is the Hindi narration of an anime recap, one line per subtitle: "
            "index|start|text. Choose 2 different moments that would make strong "
            "YouTube Shorts: each should start at a dramatic, curious or cliffhanger "
            "line and have at least 50 seconds of story after it. The two must be at "
            "least 60 seconds apart. Answer ONLY with a JSON array of 2 objects: "
            '{"start_cue": <index>, "title": "<Hindi hook title under 80 characters>", '
            '"description": "<one or two Hindi sentences>"}.\n\n' + "\n".join(lines))
        return gemini(prompt)

    # Video
    def render_long(self, src, voice, srt, out):
        cfg = self.cfg
        editors = {
            "advanced": lambda s, v, o, subs: core.make_long_video_advanced(
                s, v, o, subs, workdir=os.path.join(self.workdir, "long"),
                seg_len=cfg.seg_len, font=cfg.sub_font,
                font_size=cfg.sub_size_long, margin_v=cfg.sub_margin_long),
            "legacy": lambda s, v, o: make_long_video(s, v, o),
        }
        used = core.render_long_with_fallback(
            editors, src, voice, srt, out, core.probe(voice)["duration"])
        print("[editor] long video made with:", used)
        return out

    def render_short(self, long_video, window, srt, out):
        cfg = self.cfg
        return core.render_short(
            long_video, window["start"], window["len"], out, srt,
            workdir=os.path.join(self.workdir, "short"), font=cfg.sub_font,
            font_size=cfg.sub_size_short, margin_v=cfg.sub_margin_short)

    def validate(self, path, kind, expected_duration=None):
        return core.validate_output(path, kind, expected_duration)

    # YouTube
    def upload_youtube(self, path, title, desc):
        return upload_youtube(self.youtube, path, title, desc)

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
        text = ("Anime Recap cannot start. Missing credentials/settings: "
                + ", ".join(missing)
                + "\nAdd them in GitHub: Settings > Secrets and variables > Actions.")
        print(text)
        core.write_step_summary(text)
        core.send_telegram(text)
        print("FINAL STATUS: MISCONFIGURED")
        sys.exit(2)

    try:
        cfg = core.Config.from_env()
        drive, youtube = build_services()
        store = core.StateStore(
            DriveSheetBackend(drive, core.sheet_id_from_env()),
            sleep=cfg.sleep, base_delay=cfg.base_delay)
        deps = RealDeps(drive, youtube, cfg)
        summary = core.run_all(deps, store, cfg)
    except Exception as e:  # noqa: BLE001
        text = f"Anime Recap crashed before finishing: {core.scrub(type(e).__name__ + ': ' + str(e))[:500]}"
        finish(text, "FAILED", notify=True)
        return

    text = core.build_summary_text(summary)
    final = summary["final_status"]
    # Telegram (optional): every failure/quota problem, plus the daily summary when work happened.
    worked = summary["discovered"] > 0
    finish(text, final, notify=(final != "SUCCESS") or worked)


if __name__ == "__main__":
    main()
