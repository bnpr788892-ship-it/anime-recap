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
import recap_plan as plan  # noqa: E402

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


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)


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
        self.assertIsNone(s.get("B"))        # failed with nothing uploaded: starts fresh
        self.assertIn("unit_key", s.backend.text.splitlines()[0])

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




    def test_vertical_filter_for_landscape_and_portrait(self):
        self.assertIn("crop=ih*9/16:ih", core.vertical_filter(1280, 720))
        self.assertNotIn("crop=", core.vertical_filter(720, 1280))

    def test_parse_ai_picks_ignores_garbage(self):
        txt = '```json\n[{"start_cue": 3, "title": "T"}, {"start_cue": 999}, {"x": 1}, "no"]\n```'
        picks = core.parse_ai_picks(txt, 10)
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0]["start_cue"], 3)
        self.assertEqual(core.parse_ai_picks("not json", 10), [])


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


# ----------------------------------------------------------------------
# Real ffmpeg checks on tiny generated videos
# ----------------------------------------------------------------------
@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestMediaWithRealFfmpeg(TempDirTest):
    def make_av(self, name, seconds, size="640x360", audio=True):
        path = os.path.join(self.tmp, name)
        cmd = ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25"]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440"]
        cmd += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        cmd += ["-c:a", "aac"] if audio else ["-an"]
        subprocess.run(cmd + [path], check=True, capture_output=True)
        return path

    def narration(self, total, clips):
        """clips: [(start, seconds)] of a quiet tone, assembled into one wav of exactly `total` s."""
        parts = []
        for i, (st, dur) in enumerate(clips):
            wav = os.path.join(self.tmp, f"c{i}.wav")
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", "sine=frequency=300",
                            "-t", str(dur), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", wav],
                           check=True, capture_output=True)
            parts.append({"start": st, "dur": dur, "wav": wav})
        out = os.path.join(self.tmp, "narration.wav")
        plan.assemble_narration_wav(parts, total, out)
        return out

    def test_validation_accepts_good_and_rejects_bad_files(self):
        good = self.make_av("good.mp4", 4)
        info = core.validate_output(good, "long", 4)
        self.assertTrue(info["has_audio"] and info["has_video"])
        with self.assertRaises(core.MediaError):
            core.validate_output(good, "long", 40)
        with self.assertRaises(core.MediaError):
            core.validate_output(good, "long", 4.5, abs_tol=0.15)   # strict tolerance
        empty = os.path.join(self.tmp, "empty.mp4")
        open(empty, "wb").close()
        with self.assertRaises(core.MediaError):
            core.validate_output(empty)
        junk = os.path.join(self.tmp, "junk.mp4")
        open(junk, "wb").write(b"not a video" * 200)
        with self.assertRaises(core.MediaError):
            core.validate_output(junk)
        with self.assertRaises(core.MediaError):
            core.validate_output(good, "short")
        silent = self.make_av("silent.mp4", 4, audio=False)
        with self.assertRaises(core.MediaError):
            core.validate_output(silent)

    def check_episode(self, src_seconds, start, dur, **kw):
        src = self.make_av(f"src{src_seconds}.mp4", src_seconds, **kw)
        nar = self.narration(dur, [(0.5, min(3.0, dur - 1.0))] if dur > 2 else [])
        out = os.path.join(self.tmp, f"ep_{src_seconds}_{dur}.mp4")
        core.render_episode(src, start, dur, nar, out, None, workdir=os.path.join(self.tmp, "w"))
        tol = plan.default_tolerance(30.0)
        info = core.validate_output(out, "long", dur, abs_tol=tol)
        ok, detail = core.check_av_sync(out, dur, tol)
        self.assertTrue(ok, detail)
        return info, out

    def test_episode_keeps_the_exact_source_duration_short_normal_and_longer(self):
        for src_s, start, dur in ((10, 0, 10), (20, 0, 20), (20, 6, 8)):
            info, _ = self.check_episode(src_s, start, dur)
            self.assertAlmostEqual(info["duration"], dur, delta=plan.default_tolerance(30.0))
            self.assertEqual((info["width"], info["height"]), (1280, 720))

    def test_a_very_short_clip_works(self):
        info, _ = self.check_episode(3, 0, 3)
        self.assertAlmostEqual(info["duration"], 3.0, delta=0.2)

    def test_episode_with_a_silent_source_and_with_a_portrait_source(self):
        self.check_episode(12, 0, 12, audio=False)
        info, _ = self.check_episode(12, 0, 12, size="360x640")
        self.assertEqual((info["width"], info["height"]), (1280, 720))

    def test_subtitles_are_burned_in_and_the_length_is_unchanged(self):
        src = self.make_av("s.mp4", 12)
        nar = self.narration(12, [(1.0, 4.0)])
        cues = [{"start": 1.0, "end": 5.0, "text": "Subtitle check"}]
        srt = core.write_srt(cues, os.path.join(self.tmp, "a.srt"))
        out = os.path.join(self.tmp, "sub.mp4")
        core.render_episode(src, 0, 12, nar, out, srt, workdir=os.path.join(self.tmp, "w"),
                            font="DejaVu Sans")
        core.validate_output(out, "long", 12, abs_tol=plan.default_tolerance(30.0))

    def test_short_is_vertical_and_exactly_its_window_long(self):
        _, ep = self.check_episode(16, 0, 16)
        short = os.path.join(self.tmp, "short.mp4")
        cues = core.add_cta_cue([{"start": 0.0, "end": 4.0, "text": "one two"}], 10.0, "CTA text")
        srt = core.write_srt(cues, os.path.join(self.tmp, "sh.srt"), 22)
        core.render_short(ep, 2.0, 10.0, short, srt, workdir=os.path.join(self.tmp, "w2"),
                          font="DejaVu Sans")
        info = core.validate_output(short, "short", 10.0, abs_tol=plan.default_tolerance(30.0))
        self.assertEqual((info["width"], info["height"]), (1080, 1920))

    def test_joining_episodes_gives_the_summed_duration(self):
        a = self.check_episode(6, 0, 6)[1]
        b = self.check_episode(9, 0, 9)[1]
        out = os.path.join(self.tmp, "mega.mp4")
        core.concat_videos([a, b], out, workdir=os.path.join(self.tmp, "w3"))
        info = core.validate_output(out, "long", 15.0, abs_tol=0.5)
        ok, detail = core.check_av_sync(out, 15.0, 0.5)
        self.assertTrue(ok, detail)

    def test_silence_detection_and_cut_finder(self):
        src = self.make_av("src.mp4", 20)
        nar = self.narration(20, [(0.5, 2.0)])
        out = os.path.join(self.tmp, "ep.mp4")
        core.render_episode(src, 0, 20, nar, out, None, workdir=os.path.join(self.tmp, "w"),
                            src_audio_volume=0)
        gaps = core.silence_gaps(out, min_gap=5.0)
        self.assertTrue(gaps and gaps[0][1] - gaps[0][0] >= 5.0)
        # a picture with a hard cut at 10 s
        a = os.path.join(self.tmp, "a.mp4")
        b = os.path.join(self.tmp, "b.mp4")
        for p, color in ((a, "red"), (b, "blue")):
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=25",
                            "-t", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", p],
                           check=True, capture_output=True)
        cut = os.path.join(self.tmp, "cut.mp4")
        core.concat_videos([a, b], cut, workdir=os.path.join(self.tmp, "w5"))
        found = core.find_cut_point(cut, 12.0, tol=8.0)
        self.assertAlmostEqual(found, 10.0, delta=0.3)
        self.assertEqual(core.find_cut_point(a, 5.0, tol=3.0), 5.0)    # no cut: unchanged

    def test_av_sync_check_catches_a_length_mismatch(self):
        good = self.make_av("g.mp4", 6)
        ok, detail = core.check_av_sync(good, 9.0, 0.15)
        self.assertFalse(ok)
        self.assertIn("expected", detail)


if __name__ == "__main__":
    unittest.main()
