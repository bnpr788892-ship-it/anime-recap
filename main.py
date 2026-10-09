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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
VOICE = os.environ.get("HINDI_VOICE", "hi-IN-MadhurNeural")
PRIVACY = os.environ.get("YT_PRIVACY", "private")

_whisper = None


# ======================================================================
# Original building blocks
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
    model = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    response = requests.post(
        url,
        headers={
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=timeout,
    )

    if response.status_code == 404:
        raise RuntimeError(
            f"Gemini model '{model}' was not found (HTTP 404). "
            "Check that GEMINI_MODEL names a model available to your API key."
        )

    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(
            f"Gemini returned no response candidates for model '{model}'."
        )

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        raise RuntimeError(
            f"Gemini returned an empty response for model '{model}'."
        )

    return text


def hindi_script(chinese_text):
    prompt = (
        "Below is a Chinese transcript of an anime episode. Write an engaging "
        "Hindi recap script in a storytelling style, using simple Hindi. "
        "Do not include headings or stage directions. Return only the "
        "narration text.\n\n"
        + chinese_text[:60000]
    )
    return gemini(prompt)


async def _synthesize(text, path):
    """Generate Hindi speech and collect available timing events."""
    import edge_tts

    bounds = []
    comm = edge_tts.Communicate(text, VOICE)

    with open(path, "wb") as audio_file:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                bounds.append(
                    {
                        "offset": chunk["offset"] / 1e7,
                        "duration": chunk["duration"] / 1e7,
                        "text": chunk["text"],
                    }
                )

    return bounds


def make_voice(text, path):
    return asyncio.run(_synthesize(text, path))


def make_long_video(src, voice, out):
    """Use the existing legacy video editor as a fallback."""
    return core.make_long_video_legacy(src, voice, out)


def make_shorts(long_video, prefix):
    """Create two fixed-position Shorts as a fallback."""
    total = duration(long_video)
    paths = []

    for i, frac in enumerate([0.25, 0.6], start=1):
        out = f"{prefix}_short{i}.mp4"

        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(total * frac),
                "-i",
                long_video,
                "-t",
                "50",
                "-vf",
                "crop=ih*9/16:ih,scale=1080:1920",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                out,
            ]
        )

        paths.append(out)

    return paths


def upload_youtube(youtube, path, title, desc=""):
    body = {
        "snippet": {
            "title": title[:100],
            "description": desc,
            "categoryId": "1",
        },
        "status": {
            "privacyStatus": PRIVACY,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        _, response = request.next_chunk()

    print("Uploaded to YouTube:", response["id"])
    return response["id"]


# ======================================================================
# Google Sheet state backend through Google Drive
# ======================================================================
class DriveSheetBackend:
    def __init__(self, drive, sheet_id):
        self.drive = drive
        self.sheet_id = sheet_id

    def read_csv(self):
        data = (
            self.drive.files()
            .export_media(fileId=self.sheet_id, mimeType="text/csv")
            .execute()
        )
        return data.decode("utf-8") if isinstance(data, bytes) else data

    def write_csv(self, text):
        media = MediaIoBaseUpload(
            io.BytesIO(text.encode("utf-8")),
            mimetype="text/csv",
            resumable=False,
        )

        self.drive.files().update(
            fileId=self.sheet_id,
            media_body=media,
        ).execute()


# ======================================================================
# Real services behind the existing recap_core pipeline
# ======================================================================
class RealDeps:
    def __init__(self, drive, youtube, cfg):
        self.drive = drive
        self.youtube = youtube
        self.cfg = cfg

        self.incoming = os.environ["INCOMING_FOLDER_ID"]
        self.done = os.environ["DONE_FOLDER_ID"]
        self.workdir = "editwork"

    def list_incoming(self):
        result = (
            self.drive.files()
            .list(
                q=(
                    f"'{self.incoming}' in parents "
                    "and mimeType contains 'video/' and trashed=false"
                ),
                fields="files(id,name)",
                orderBy="name",
                pageSize=100,
            )
            .execute()
        )

        return result.get("files", [])

    def _download(self, file_id, path):
        request = self.drive.files().get_media(fileId=file_id)

        with io.FileIO(path, "wb") as file_handle:
            downloader = MediaIoBaseDownload(file_handle, request)
            done = False

            while not done:
                _, done = downloader.next_chunk()

    def download_source(self, file_id, path):
        self._download(file_id, path)

    def download_drive(self, file_id, path):
        self._download(file_id, path)

    def upload_drive(self, path, folder_id):
        media = MediaFileUpload(path, resumable=True)

        result = (
            self.drive.files()
            .create(
                body={
                    "name": os.path.basename(path),
                    "parents": [folder_id],
                },
                media_body=media,
                fields="id",
            )
            .execute()
        )

        return result["id"]

    def move_to_done(self, file_id):
        file_data = (
            self.drive.files()
            .get(fileId=file_id, fields="parents")
            .execute()
        )

        self.drive.files().update(
            fileId=file_id,
            addParents=self.done,
            removeParents=",".join(file_data.get("parents", [])),
        ).execute()

    # AI and audio
    def transcribe(self, path):
        return transcribe(path)

    def write_script(self, zh_text):
        return hindi_script(zh_text)

    def synthesize(self, text, path):
        return make_voice(text, path)

    def duration(self, path):
        return duration(path)

    def pick_shorts(self, cues):
        lines = []
        size = 0

        for index, cue in enumerate(cues):
            line = f"{index}|{cue['start']:.0f}s|{cue['text']}"
            size += len(line)

            if size > 45000:
                break

            lines.append(line)

        prompt = (
            "Below is the Hindi narration of an anime recap, one line per "
            "subtitle in the format index|start|text. Choose two different "
            "moments suitable for YouTube Shorts. Each should begin with a "
            "dramatic, curious, or cliffhanger line and have at least 50 "
            "seconds of story after it. The two moments must be at least "
            "60 seconds apart. Answer ONLY with a JSON array containing "
            'two objects: {"start_cue": <index>, '
            '"title": "<Hindi hook title under 80 characters>", '
            '"description": "<one or two Hindi sentences>"}.\n\n'
            + "\n".join(lines)
        )

        return gemini(prompt)

    # Video rendering
    def render_long(self, src, voice, srt, out):
        cfg = self.cfg

        editors = {
            "advanced": lambda source, audio, output, subtitles: (
                core.make_long_video_advanced(
                    source,
                    audio,
                    output,
                    subtitles,
                    workdir=os.path.join(self.workdir, "long"),
                    seg_len=cfg.seg_len,
                    font=cfg.sub_font,
                    font_size=cfg.sub_size_long,
                    margin_v=cfg.sub_margin_long,
                )
            ),
            "legacy": lambda source, audio, output: (
                make_long_video(source, audio, output)
            ),
        }

        used = core.render_long_with_fallback(
            editors,
            src,
            voice,
            srt,
            out,
            core.probe(voice)["duration"],
        )

        print("[editor] long video made with:", used)
        return out

    def render_short(self, long_video, window, srt, out):
        cfg = self.cfg

        return core.render_short(
            long_video,
            window["start"],
            window["len"],
            out,
            srt,
            workdir=os.path.join(self.workdir, "short"),
            font=cfg.sub_font,
            font_size=cfg.sub_size_short,
            margin_v=cfg.sub_margin_short,
        )

    def validate(self, path, kind, expected_duration=None):
        return core.validate_output(path, kind, expected_duration)

    # YouTube
    def upload_youtube(self, path, title, desc):
        return upload_youtube(self.youtube, path, title, desc)

    def verify_youtube(self, video_id):
        return core.verify_youtube_upload(self.youtube, video_id)


def build_services():
    credentials = Credentials(
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

    drive = build("drive", "v3", credentials=credentials)
    youtube = build("youtube", "v3", credentials=credentials)

    return drive, youtube


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
        text = (
            "Anime Recap cannot start. Missing credentials/settings: "
            + ", ".join(missing)
            + "\nAdd them in GitHub: Settings > Secrets and variables > Actions."
        )

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
            sleep=cfg.sleep,
            base_delay=cfg.base_delay,
        )

        deps = RealDeps(drive, youtube, cfg)
        summary = core.run_all(deps, store, cfg)

    except Exception as error:  # noqa: BLE001
        message = (
            "Anime Recap crashed before finishing: "
            + core.scrub(
                type(error).__name__ + ": " + str(error)
            )[:500]
        )

        finish(message, "FAILED", notify=True)
        return

    text = core.build_summary_text(summary)
    final = summary["final_status"]
    worked = summary["discovered"] > 0

    finish(
        text,
        final,
        notify=(final != "SUCCESS") or worked,
    )


if __name__ == "__main__":
    main()
