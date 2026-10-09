"""
Run with:  python -m unittest discover -s tests -v

External services (Drive, YouTube, Gemini, Whisper, edge-tts, Telegram) are
replaced by fakes: these tests never upload anything.
The media tests use the real ffmpeg on tiny generated videos.
"""
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recap_core as core  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class MemoryBackend:
    def __init__(self, text=""):
        self.text = text
        self.writes = 0
        self.fail_writes = 0

    def read_csv(self):
        return self.text

    def write_csv(self, text):
        if self.fail_writes:
            self.fail_writes -= 1
            raise core.TemporaryError("sheet temporarily unavailable")
        self.text = text
        self.writes += 1


class HttpErr(Exception):
    def __init__(self, status, content=b""):
        super().__init__(f"HTTP {status}")
        self.resp = type("R", (), {"status": status})()
        self.content = content


def make_cfg(**kw):
    base = dict(long_dir="LONGDIR", shorts_dir="SHORTDIR", attempts=3,
                base_delay=0.0, sleep=lambda s: None, max_retries=3)
    base.update(kw)
    return core.Config(**base)


class FakeDeps:
    """Records every call; stores 'uploaded' files so they can be downloaded again."""

    def __init__(self, incoming=None):
        self.incoming = incoming if incoming is not None else [{"id": "SRC1", "name": "ep1.mp4"}]
        self.calls = []
        self.drive_files = {}
        self.counter = 0
        self.fail = {}          # method name -> list of exceptions to raise (consumed)
        self.verify_result = "CONFIRMED"
        self.moved = []
        self.status_when_moved = []
        self.store = None

    def _maybe_fail(self, name):
        q = self.fail.get(name)
        if q:
            exc = q[0] if isinstance(q, list) and len(q) == 1 and q[0] is not None and getattr(q[0], "_forever", False) else q.pop(0)
            if exc is not None:
                raise exc

    def count(self, name):
        return sum(1 for c in self.calls if c == name)

    def list_incoming(self):
        return list(self.incoming)

    def download_source(self, fid, path):
        self.calls.append("download_source")
        open(path, "wb").write(b"src")

    def download_drive(self, fid, path):
        self.calls.append("download_drive")
        open(path, "wb").write(self.drive_files[fid])

    def transcribe(self, path):
        self.calls.append("transcribe")
        self._maybe_fail("transcribe")
        return "chinese text"

    def write_script(self, zh):
        self.calls.append("write_script")
        return "यह पहला वाक्य है। यह दूसरा वाक्य है और थोड़ा लंबा है। " * 12

    def synthesize(self, text, path):
        self.calls.append("synthesize")
        open(path, "wb").write(b"voice")
        return []   # no timing events -> proportional subtitles

    def duration(self, path):
        return 200.0

    def pick_shorts(self, cues):
        self.calls.append("pick_shorts")
        return '[{"start_cue": 5, "title": "Hook one", "description": "d1"},' \
               ' {"start_cue": 40, "title": "Hook two", "description": "d2"}]'

    def render_long(self, src, voice, srt, out):
        self.calls.append("render_long")
        open(out, "wb").write(b"longvideo")

    def render_short(self, lv, window, srt, out):
        self.calls.append("render_short")
        open(out, "wb").write(b"shortvideo")

    def validate(self, path, kind, expected=None):
        self.calls.append("validate")

    def upload_drive(self, path, folder):
        self.calls.append("upload_drive")
        self._maybe_fail("upload_drive")
        self.counter += 1
        fid = f"DRV{self.counter}"
        self.drive_files[fid] = open(path, "rb").read()
        return fid

    def upload_youtube(self, path, title, desc):
        self.calls.append("upload_youtube")
        self._maybe_fail("upload_youtube")
        self.counter += 1
        return f"YT{self.counter}"

    def verify_youtube(self, vid):
        self.calls.append("verify_youtube")
        return self.verify_result

    def move_to_done(self, fid):
        self.calls.append("move_to_done")
        self.moved.append(fid)
        if self.store is not None:
            self.status_when_moved.append(self.store.get(fid)["status"])


def run_once(deps, backend, cfg=None, tmp=None):
    cfg = cfg or make_cfg()
    store = core.StateStore(backend, sleep=lambda s: None, base_delay=0.0, log=lambda *a: None)
    deps.store = store
    summary = core.run_all(deps, store, cfg, workdir=tmp, log=lambda *a: None)
    return summary, store


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)


# ----------------------------------------------------------------------
# Error handling and retries
# ----------------------------------------------------------------------
class TestRetries(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(core.classify_error(HttpErr(503)), "temporary")
        self.assertEqual(core.classify_error(HttpErr(429)), "temporary")
        self.assertEqual(core.classify_error(HttpErr(401)), "permanent")
        self.assertEqual(core.classify_error(HttpErr(403, b'{"reason":"quotaExceeded"}')), "quota")
        self.assertEqual(core.classify_error(ConnectionError("x")), "temporary")
        self.assertEqual(core.classify_error(ValueError("x")), "unknown")

    def test_temporary_errors_retry_with_exponential_backoff(self):
        delays, calls = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 4:
                raise HttpErr(503)
            return "ok"
        out = core.with_retries(flaky, attempts=5, base_delay=2.0,
                                sleep=delays.append, log=lambda *a: None)
        self.assertEqual(out, "ok")
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(delays), 3)
        self.assertTrue(2.0 <= delays[0] < 2.6)
        self.assertTrue(4.0 <= delays[1] < 4.6)
        self.assertTrue(8.0 <= delays[2] < 8.6)

    def test_retry_limit_is_respected(self):
        calls = []

        def always():
            calls.append(1)
            raise HttpErr(500)
        with self.assertRaises(HttpErr):
            core.with_retries(always, attempts=3, sleep=lambda s: None, log=lambda *a: None)
        self.assertEqual(len(calls), 3)

    def test_permanent_error_is_not_retried(self):
        calls = []

        def bad():
            calls.append(1)
            raise HttpErr(400)
        with self.assertRaises(HttpErr):
            core.with_retries(bad, attempts=5, sleep=lambda s: None, log=lambda *a: None)
        self.assertEqual(len(calls), 1)

    def test_quota_error_becomes_quota_exceeded(self):
        with self.assertRaises(core.QuotaExceeded):
            core.with_retries(lambda: (_ for _ in ()).throw(HttpErr(403, b"quotaExceeded")),
                              attempts=3, sleep=lambda s: None, log=lambda *a: None)

    def test_scrub_hides_keys(self):
        os.environ["GEMINI_API_KEY"] = "SUPERSECRETKEY123"
        try:
            text = core.scrub("error for url ...?key=abc123def&x=1 and SUPERSECRETKEY123")
        finally:
            del os.environ["GEMINI_API_KEY"]
        self.assertNotIn("SUPERSECRETKEY123", text)
        self.assertNotIn("abc123def", text)


# ----------------------------------------------------------------------
# Sheet rows / state
# ----------------------------------------------------------------------
class TestState(unittest.TestCase):
    def test_upsert_never_creates_duplicate_rows(self):
        b = MemoryBackend()
        s = core.StateStore(b, sleep=lambda x: None, base_delay=0)
        s.upsert("A", filename="a.mp4", status=core.PENDING)
        s.upsert("A", status=core.PROCESSING)
        s.upsert("B", filename="b.mp4")
        self.assertEqual(len(s.rows), 2)
        s2 = core.StateStore(b)
        s2.load()
        self.assertEqual(len(s2.rows), 2)
        self.assertEqual(s2.get("A")["status"], core.PROCESSING)

    def test_duplicate_rows_in_sheet_are_merged_preferring_completed(self):
        s = core.StateStore(MemoryBackend())
        header = ",".join(core.COLUMNS)

        def line(status):
            cells = dict.fromkeys(core.COLUMNS, "")
            cells.update(source_id="A", status=status)
            return ",".join(cells[c] for c in core.COLUMNS)
        s.backend.text = "\n".join([header, line(core.COMPLETED), line(core.PENDING)])
        s.load()
        self.assertEqual(len(s.rows), 1)
        self.assertEqual(s.get("A")["status"], core.COMPLETED)

    def test_old_sheet_format_is_migrated(self):
        old = ("Source file ID,Source name,Status,Long video YouTube ID,"
               "Short 1 YouTube ID,Short 2 YouTube ID,Date (UTC)\n"
               "A,a.mp4,DONE,Y1,Y2,Y3,2026-10-08 10:00\n"
               "B,b.mp4,FAILED: boom,,,,2026-10-08 11:00\n")
        s = core.StateStore(MemoryBackend(old), sleep=lambda x: None, base_delay=0)
        s.load()
        self.assertEqual(s.get("A")["status"], core.COMPLETED)
        self.assertEqual(s.get("A")["yt_long_id"], "Y1")
        self.assertEqual(s.get("B")["status"], core.RETRY_PENDING)
        self.assertIn("source_id", s.backend.text.splitlines()[0])

    def test_sheet_write_is_retried(self):
        b = MemoryBackend()
        b.fail_writes = 2
        s = core.StateStore(b, sleep=lambda x: None, base_delay=0, log=lambda *a: None)
        s.upsert("A", status=core.PENDING)
        self.assertEqual(b.writes, 1)

    def test_eligibility(self):
        cfg = make_cfg()
        now = datetime.datetime(2026, 10, 9, 12, 0, 0)
        fresh = core.fmt_ts(now - datetime.timedelta(minutes=30))
        old = core.fmt_ts(now - datetime.timedelta(hours=9))
        e = lambda **k: core.is_eligible(k, cfg, now)  # noqa: E731
        self.assertTrue(e(status=core.PENDING))
        self.assertTrue(e(status=core.RETRY_PENDING, retry_count="1"))
        self.assertFalse(e(status=core.RETRY_PENDING, retry_count="3"))
        self.assertFalse(e(status=core.COMPLETED))
        self.assertFalse(e(status=core.FAILED))
        self.assertFalse(e(status=core.EDITING, last_updated=fresh))   # another run is working
        self.assertTrue(e(status=core.EDITING, last_updated=old))      # a dead run
        self.assertTrue(e(status=core.EDITING, last_updated=""))

    def test_timestamp_formats_from_google_sheets(self):
        self.assertIsNotNone(core.parse_ts("2026-10-09 12:00:00 UTC"))
        self.assertIsNotNone(core.parse_ts("10/09/2026 12:00:00"))
        self.assertIsNone(core.parse_ts("garbage"))


# ----------------------------------------------------------------------
# Pipeline: duplicates, state, failures, resume
# ----------------------------------------------------------------------
class TestPipeline(TempDirTest):
    def test_happy_path_records_everything_and_moves_last(self):
        deps, backend = FakeDeps(), MemoryBackend()
        summary, store = run_once(deps, backend, tmp=self.tmp)
        row = store.get("SRC1")
        self.assertEqual(row["status"], core.COMPLETED)
        for k in ("yt_long_id", "yt_short1_id", "yt_short2_id", "drive_long_link",
                  "drive_short1_link", "drive_short2_link", "drive_srt_link", "started_at",
                  "completed_at"):
            self.assertTrue(row[k], k)
        self.assertEqual(row["transcription_status"], "DONE")
        self.assertEqual(row["youtube_status"], "CONFIRMED")
        self.assertEqual(deps.count("upload_youtube"), 3)
        self.assertEqual(deps.moved, ["SRC1"])
        # moved to Done only after COMPLETED was recorded
        self.assertEqual(deps.status_when_moved, [core.COMPLETED])
        self.assertEqual(summary["final_status"], "SUCCESS")
        self.assertEqual((summary["long_generated"], summary["shorts_generated"],
                          summary["yt_confirmed"]), (1, 2, 3))

    def test_ai_picks_and_titles_are_stored(self):
        deps, backend = FakeDeps(), MemoryBackend()
        _, store = run_once(deps, backend, tmp=self.tmp)
        self.assertIn("Hook one #Shorts", store.get("SRC1")["short_plan"])

    def test_completed_source_is_not_processed_again(self):
        deps, backend = FakeDeps(), MemoryBackend()
        run_once(deps, backend, tmp=self.tmp)
        deps2 = FakeDeps()                      # same file still sitting in Incoming
        summary, _ = run_once(deps2, backend, tmp=self.tmp)
        self.assertEqual(summary["duplicates_skipped"], 1)
        for name in ("transcribe", "render_long", "upload_youtube", "upload_drive"):
            self.assertEqual(deps2.count(name), 0, name)
        self.assertEqual(deps2.moved, ["SRC1"])  # cleaned up into Done

    def test_same_name_different_id_is_a_different_source(self):
        deps = FakeDeps(incoming=[{"id": "A", "name": "ep.mp4"}, {"id": "B", "name": "ep.mp4"}])
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(store.get("A")["status"], core.COMPLETED)
        self.assertEqual(store.get("B")["status"], core.COMPLETED)

    def test_max_videos_per_run(self):
        deps = FakeDeps(incoming=[{"id": f"S{i}", "name": f"e{i}.mp4"} for i in range(5)])
        summary, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(summary["succeeded"], 2)
        self.assertEqual(store.get("S4")["status"], core.PENDING)

    def test_failure_records_stage_and_keeps_source(self):
        deps = FakeDeps()
        deps.fail["transcribe"] = [ValueError("whisper exploded")]
        summary, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("SRC1")
        self.assertEqual(row["status"], core.RETRY_PENDING)
        self.assertEqual(row["failed_stage"], core.TRANSCRIBING)
        self.assertEqual(row["transcription_status"], "FAILED")
        self.assertEqual(row["retry_count"], "1")
        self.assertIn("whisper exploded", row["error"])
        self.assertEqual(deps.moved, [])                      # original preserved
        self.assertNotEqual(row["status"], core.COMPLETED)
        self.assertEqual(summary["final_status"], "FAILED")
        self.assertEqual(summary["awaiting_retry"], 1)

    def test_gives_up_after_retry_limit(self):
        backend = MemoryBackend()
        for attempt in range(1, 5):
            deps = FakeDeps()
            deps.fail["transcribe"] = [ValueError("still broken")]
            _, store = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(store.get("SRC1")["status"], core.FAILED)
        self.assertEqual(store.get("SRC1")["retry_count"], "3")
        self.assertEqual(deps.count("transcribe"), 0)        # 4th run did not even try

    def test_permanent_error_fails_immediately(self):
        deps = FakeDeps()
        deps.fail["transcribe"] = [HttpErr(401)]
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(store.get("SRC1")["status"], core.FAILED)

    def test_temporary_error_is_retried_inside_the_run(self):
        deps = FakeDeps()
        deps.fail["transcribe"] = [HttpErr(503), HttpErr(503)]
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(store.get("SRC1")["status"], core.COMPLETED)
        self.assertEqual(deps.count("transcribe"), 3)

    def test_resume_after_youtube_failure_does_not_redo_or_reupload(self):
        backend = MemoryBackend()
        deps1 = FakeDeps()
        # long uploads fine; the first Short's YouTube upload fails permanently-ish
        deps1.fail["upload_youtube"] = [None, ValueError("youtube hiccup")]
        s1, store1 = run_once(deps1, backend, tmp=self.tmp)
        row = store1.get("SRC1")
        self.assertEqual(row["status"], core.RETRY_PENDING)
        self.assertTrue(row["yt_long_id"])
        self.assertFalse(row["yt_short1_id"])
        self.assertTrue(row["drive_short1_link"])            # rendered + saved already
        long_id = row["yt_long_id"]

        deps2 = FakeDeps()
        deps2.drive_files = dict(deps1.drive_files)          # same Drive contents
        s2, store2 = run_once(deps2, backend, tmp=self.tmp)
        row2 = store2.get("SRC1")
        self.assertEqual(row2["status"], core.COMPLETED)
        self.assertEqual(row2["yt_long_id"], long_id)        # untouched
        self.assertEqual(deps2.count("upload_youtube"), 2)   # only the two Shorts
        for name in ("transcribe", "write_script", "synthesize", "render_long"):
            self.assertEqual(deps2.count(name), 0, name)     # nothing regenerated
        self.assertEqual(deps2.count("render_short"), 1)     # only Short 2 (Short 1 came from Drive)

    def test_quota_stops_the_run_and_is_reported(self):
        deps = FakeDeps(incoming=[{"id": "A", "name": "a.mp4"}, {"id": "B", "name": "b.mp4"}])
        deps.fail["upload_youtube"] = [HttpErr(403, b'{"error":{"errors":[{"reason":"quotaExceeded"}]}}')]
        summary, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertTrue(summary["quota_exceeded"])
        self.assertEqual(summary["final_status"], "QUOTA_EXCEEDED")
        self.assertEqual(store.get("A")["status"], core.RETRY_PENDING)
        self.assertEqual(store.get("A")["retry_count"], "0")   # quota is not the video's fault
        self.assertEqual(store.get("B")["status"], core.PENDING)  # never started
        self.assertEqual(core.exit_code_for(summary["final_status"]), 3)

    def test_fresh_processing_row_from_another_run_is_left_alone(self):
        backend = MemoryBackend()
        s = core.StateStore(backend, sleep=lambda x: None, base_delay=0)
        s.upsert("SRC1", filename="ep1.mp4", status=core.EDITING)   # timestamp = now
        deps = FakeDeps()
        summary, _ = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(summary["succeeded"], 0)
        self.assertEqual(deps.count("transcribe"), 0)

    def test_summary_text_does_not_overclaim(self):
        deps = FakeDeps()
        deps.verify_result = "UNVERIFIED"
        summary, _ = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(summary["yt_uploaded"], 3)
        self.assertEqual(summary["yt_confirmed"], 0)
        text = core.build_summary_text(summary)
        self.assertIn("confirmed by YouTube: 0", text)
        self.assertIn("could not be independently confirmed", text)


# ----------------------------------------------------------------------
# Verification, subtitles, planning, notifications
# ----------------------------------------------------------------------
class FakeYoutube:
    def __init__(self, items=None, error=None):
        self._items, self._error = items, error

    def videos(self):
        return self

    def list(self, **kw):
        return self

    def execute(self):
        if self._error:
            raise self._error
        return {"items": self._items}


class TestSmallPieces(unittest.TestCase):
    def test_youtube_verification_results(self):
        v = core.verify_youtube_upload
        self.assertEqual(v(FakeYoutube([{"status": {"uploadStatus": "processed"}}]), "x"), "CONFIRMED")
        self.assertEqual(v(FakeYoutube([{"status": {"uploadStatus": "rejected"}}]), "x"), "REJECTED")
        self.assertEqual(v(FakeYoutube([]), "x"), "NOT_FOUND")
        self.assertEqual(v(FakeYoutube(error=HttpErr(403)), "x"), "UNVERIFIED")

    def test_cues_from_word_boundaries(self):
        b = [{"offset": i * 0.5, "duration": 0.4, "text": f"w{i}"} for i in range(20)]
        cues = core.build_cues_from_boundaries(b, 10.0, max_words=7)
        self.assertEqual(len(cues), 3)
        self.assertAlmostEqual(cues[0]["start"], 0.0)
        for a, c in zip(cues, cues[1:]):
            self.assertLessEqual(a["end"], c["start"] + 1e-6)
        self.assertLessEqual(cues[-1]["end"], 10.0)

    def test_proportional_fallback_covers_the_audio(self):
        cues = core.build_cues_proportional("एक दो तीन। चार पाँच छह सात आठ नौ दस ग्यारह।", 20.0, 5)
        self.assertTrue(cues)
        self.assertLessEqual(cues[-1]["end"], 20.0)
        self.assertGreater(cues[-1]["end"], 15.0)

    def test_srt_roundtrip_and_format(self):
        cues = [{"start": 0.0, "end": 1.5, "text": "नमस्ते दोस्तों"},
                {"start": 2.0, "end": 3.25, "text": "यह दूसरी लाइन है"}]
        srt = core.cues_to_srt(cues)
        self.assertIn("00:00:00,000 --> 00:00:01,500", srt)
        back = core.parse_srt(srt)
        self.assertEqual([c["text"] for c in back], [c["text"] for c in cues])
        self.assertAlmostEqual(back[1]["end"], 3.25)

    def test_wrap_gives_at_most_two_short_lines(self):
        t = core.wrap_text("यह एक काफी लंबा वाक्य है जो फोन पर पढ़ने लायक होना चाहिए", 22)
        self.assertLessEqual(len(t.split("\n")), 2)
        self.assertEqual(core.wrap_text("छोटा", 22), "छोटा")

    def test_slice_cues_shifts_to_clip_start(self):
        cues = [{"start": 0, "end": 4, "text": "a"}, {"start": 10, "end": 14, "text": "b"},
                {"start": 30, "end": 34, "text": "c"}]
        out = core.slice_cues(cues, 9, 20)
        self.assertEqual([c["text"] for c in out], ["b"])
        self.assertAlmostEqual(out[0]["start"], 1.0)

    def test_plan_segments_cover_source_in_order(self):
        segs = core.plan_segments(600, 100, 8)
        self.assertEqual(len(segs), 13)
        self.assertAlmostEqual(sum(l for _, l in segs), 100, places=3)
        starts = [s for s, _ in segs]
        self.assertEqual(starts, sorted(starts))
        self.assertLess(starts[-1] + segs[-1][1], 600.001)
        self.assertGreater(starts[-1], 500)             # reaches the end of the story

    def test_plan_segments_loops_when_narration_is_longer(self):
        segs = core.plan_segments(20, 70, 8)
        total = sum(l for _, l in segs)
        self.assertGreaterEqual(total, 69.99)
        self.assertLess(total, 70.6)
        for s, l in segs:
            self.assertLessEqual(s + l, 20.001)

    def test_plan_segments_terminates_when_narration_is_barely_longer(self):
        # regression: used to loop forever when the voice was a few hundredths longer
        for src, target in ((6.0, 6.03), (6.0, 6.0), (10.0, 10.004), (0.3, 5.0)):
            segs = core.plan_segments(src, target, 3)
            self.assertTrue(segs)
            self.assertGreaterEqual(sum(l for _, l in segs), target - 0.02)

    def test_vertical_filter_for_landscape_and_portrait(self):
        self.assertIn("crop=ih*9/16:ih", core.vertical_filter(1280, 720))
        self.assertNotIn("crop=", core.vertical_filter(720, 1280))

    def test_parse_ai_picks_ignores_garbage(self):
        txt = '```json\n[{"start_cue": 3, "title": "T"}, {"start_cue": 999}, {"x": 1}, "no"]\n```'
        picks = core.parse_ai_picks(txt, 10)
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0]["start_cue"], 3)
        self.assertEqual(core.parse_ai_picks("not json", 10), [])

    def test_plan_shorts_non_overlapping_and_fallback(self):
        cues = [{"start": i * 5.0, "end": i * 5.0 + 4, "text": "x"} for i in range(60)]
        plan = core.plan_shorts(cues, 300.0, [{"start_cue": 2, "title": "A", "description": ""},
                                              {"start_cue": 3, "title": "B", "description": ""}])
        self.assertEqual(len(plan), 2)           # second pick overlapped -> fallback position used
        a, b = sorted(plan, key=lambda p: p["start"])
        self.assertLessEqual(a["start"] + a["len"], b["start"])
        self.assertTrue(all(p["title"].endswith("#Shorts") for p in plan))
        plain = core.plan_shorts(cues, 300.0, None)
        self.assertEqual(len(plain), 2)

    def test_telegram_is_optional_and_never_raises(self):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        self.assertFalse(core.send_telegram("hi"))
        sent = []
        ok = core.send_telegram("hello", "TOKEN123", "42", opener=lambda req, timeout=0: sent.append(req))
        self.assertTrue(ok)
        self.assertEqual(len(sent), 1)

        def boom(req, timeout=0):
            raise OSError("down")
        self.assertFalse(core.send_telegram("hello", "TOKEN123", "42", opener=boom))

    def test_missing_credentials_are_listed_by_name_only(self):
        env = {"GOOGLE_CLIENT_ID": "x"}
        missing = core.missing_env(env)
        self.assertIn("GEMINI_API_KEY", missing)
        self.assertIn("GOOGLE_SHEET_ID (or SHEET_ID)", missing)
        full = {n: "v" for n in core.REQUIRED_ENV}
        full["SHEET_ID"] = "s"
        self.assertEqual(core.missing_env(full), [])

    def test_step_summary_file(self):
        with tempfile.NamedTemporaryFile("r+", suffix=".md") as f:
            self.assertTrue(core.write_step_summary("hello", f.name))
            self.assertIn("hello", open(f.name).read())

    def test_editor_fallback_order(self):
        order = []

        def adv(src, voice, out, srt):
            order.append("advanced+subs" if srt else "advanced")
            raise core.MediaError("nope")

        def legacy(src, voice, out):
            order.append("legacy")
            open(out, "wb").write(b"x")
        d = tempfile.mkdtemp()
        out = os.path.join(d, "o.mp4")
        used = core.render_long_with_fallback(
            {"advanced": adv, "legacy": legacy}, "s", "v", "subs.srt", out, 10.0,
            validate=lambda *a, **k: None, log=lambda *a: None)
        self.assertEqual(used, "legacy")
        self.assertEqual(order, ["advanced+subs", "advanced", "legacy"])
        shutil.rmtree(d)


# ----------------------------------------------------------------------
# Real ffmpeg checks on tiny generated videos
# ----------------------------------------------------------------------
@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestMediaWithRealFfmpeg(TempDirTest):
    def make_av(self, name, seconds, size="640x360"):
        path = os.path.join(self.tmp, name)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25",
                        "-f", "lavfi", "-i", "sine=frequency=440", "-t", str(seconds),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
                       check=True, capture_output=True)
        return path

    def make_voice(self, seconds):
        path = os.path.join(self.tmp, "voice.mp3")
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300",
                        "-t", str(seconds), "-c:a", "libmp3lame", path],
                       check=True, capture_output=True)
        return path

    def test_validation_accepts_good_and_rejects_bad_files(self):
        good = self.make_av("good.mp4", 4)
        info = core.validate_output(good, "long", 4)
        self.assertTrue(info["has_audio"] and info["has_video"])
        with self.assertRaises(core.MediaError):
            core.validate_output(good, "long", 40)            # wrong length
        empty = os.path.join(self.tmp, "empty.mp4")
        open(empty, "wb").close()
        with self.assertRaises(core.MediaError):
            core.validate_output(empty)
        junk = os.path.join(self.tmp, "junk.mp4")
        open(junk, "wb").write(b"not a video" * 200)
        with self.assertRaises(core.MediaError):
            core.validate_output(junk)
        with self.assertRaises(core.MediaError):
            core.validate_output(good, "short")               # landscape is not a Short

    def test_advanced_long_video_without_subtitles(self):
        src = self.make_av("src.mp4", 30)
        voice = self.make_voice(12)
        out = os.path.join(self.tmp, "long.mp4")
        core.make_long_video_advanced(src, voice, out, None,
                                      workdir=os.path.join(self.tmp, "w"), seg_len=4)
        info = core.validate_output(out, "long", 12)
        self.assertEqual((info["width"], info["height"]), (1280, 720))

    def test_advanced_long_video_with_subtitles_and_short(self):
        src = self.make_av("src.mp4", 30)
        voice = self.make_voice(14)
        cues = core.build_cues_proportional("one two three four five six. seven eight nine ten.", 14.0, 4)
        srt = core.write_srt(cues, os.path.join(self.tmp, "a.srt"))
        out = os.path.join(self.tmp, "long.mp4")
        core.make_long_video_advanced(src, voice, out, srt, workdir=os.path.join(self.tmp, "w"),
                                      seg_len=5, font="DejaVu Sans")
        core.validate_output(out, "long", 14)
        short = os.path.join(self.tmp, "short.mp4")
        sl = core.slice_cues(cues, 2, 12)
        srt2 = core.write_srt(sl, os.path.join(self.tmp, "b.srt"), 22)
        core.render_short(out, 2, 10, short, srt2, workdir=os.path.join(self.tmp, "w2"),
                          font="DejaVu Sans")
        info = core.validate_output(short, "short", 10)
        self.assertEqual((info["width"], info["height"]), (1080, 1920))

    def test_legacy_editor_still_works(self):
        src = self.make_av("src.mp4", 5)
        voice = self.make_voice(9)
        out = os.path.join(self.tmp, "legacy.mp4")
        core.make_long_video_legacy(src, voice, out)
        core.validate_output(out, "long", 9)

    def test_portrait_source_is_handled(self):
        src = self.make_av("portrait.mp4", 12, size="360x640")
        voice = self.make_voice(6)
        out = os.path.join(self.tmp, "long.mp4")
        core.make_long_video_advanced(src, voice, out, None,
                                      workdir=os.path.join(self.tmp, "w"), seg_len=3)
        info = core.validate_output(out, "long", 6)
        self.assertEqual((info["width"], info["height"]), (1280, 720))


if __name__ == "__main__":
    unittest.main()
