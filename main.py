import os, io, asyncio, subprocess, requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import edge_tts
from faster_whisper import WhisperModel

# ---------- Settings (come from GitHub Secrets) ----------
INCOMING = os.environ["INCOMING_FOLDER_ID"]   # Drive folder: Incoming Videos
LONG_DIR = os.environ["LONG_FOLDER_ID"]       # Drive folder: Long Videos
SHORTS_DIR = os.environ["SHORTS_FOLDER_ID"]   # Drive folder: Shorts
DONE_DIR = os.environ["DONE_FOLDER_ID"]       # Drive folder: Done (processed videos go here)
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = os.environ.get("HINDI_VOICE", "hi-IN-MadhurNeural")
PRIVACY = os.environ.get("YT_PRIVACY", "private")  # change to "public" when you trust it
MAX_VIDEOS = 2  # 2 long videos per day

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
drive = build("drive", "v3", credentials=creds)
youtube = build("youtube", "v3", credentials=creds)


def run(cmd):
    subprocess.run(cmd, check=True)


def duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path])
    return float(out)


def list_new_videos():
    res = drive.files().list(
        q=f"'{INCOMING}' in parents and mimeType contains 'video/' and trashed=false",
        fields="files(id,name)", orderBy="name", pageSize=MAX_VIDEOS).execute()
    return res.get("files", [])


def download(file_id, path):
    req = drive.files().get_media(fileId=file_id)
    with io.FileIO(path, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()


def transcribe(path):
    model = WhisperModel("small", compute_type="int8")
    segments, _ = model.transcribe(path, language="zh")
    return " ".join(s.text for s in segments)


def hindi_script(chinese_text):
    prompt = (
        "Below is a Chinese transcript of an anime episode. Write an engaging "
        "Hindi recap script (storytelling style, simple Hindi, no headings, "
        "no stage directions, just the narration text).\n\n" + chinese_text[:60000]
    )
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}")
    r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=300)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


async def make_voice(text, path):
    await edge_tts.Communicate(text, VOICE).save(path)


def make_long_video(src, voice, out):
    dur = duration(voice)
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", src, "-i", voice,
         "-map", "0:v", "-map", "1:a", "-t", str(dur),
         "-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "veryfast",
         "-c:a", "aac", out])


def make_shorts(long_video, prefix):
    total = duration(long_video)
    paths = []
    for i, frac in enumerate([0.25, 0.6], start=1):
        out = f"{prefix}_short{i}.mp4"
        run(["ffmpeg", "-y", "-ss", str(total * frac), "-i", long_video, "-t", "50",
             "-vf", "crop=ih*9/16:ih,scale=1080:1920",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", out])
        paths.append(out)
    return paths


def upload_youtube(path, title, desc=""):
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


def upload_drive(path, folder_id):
    media = MediaFileUpload(path, resumable=True)
    drive.files().create(
        body={"name": os.path.basename(path), "parents": [folder_id]},
        media_body=media, fields="id").execute()


def move_to_done(file_id):
    f = drive.files().get(fileId=file_id, fields="parents").execute()
    drive.files().update(
        fileId=file_id, addParents=DONE_DIR,
        removeParents=",".join(f["parents"])).execute()


def main():
    videos = list_new_videos()
    if not videos:
        print("No new videos in Incoming folder.")
        return
    for v in videos:
        name = os.path.splitext(v["name"])[0]
        print("Processing:", name)
        src = "source.mp4"
        download(v["id"], src)

        text = transcribe(src)
        script = hindi_script(text)
        asyncio.run(make_voice(script, "voice.mp3"))

        long_out = f"{name}_hindi_recap.mp4"
        make_long_video(src, "voice.mp3", long_out)
        shorts = make_shorts(long_out, name)

        upload_youtube(long_out, f"{name} Hindi Recap")
        for i, s in enumerate(shorts, start=1):
            upload_youtube(s, f"{name} Part {i} #Shorts")

        upload_drive(long_out, LONG_DIR)
        for s in shorts:
            upload_drive(s, SHORTS_DIR)
        move_to_done(v["id"])
        print("Finished:", name)


if __name__ == "__main__":
    main()
