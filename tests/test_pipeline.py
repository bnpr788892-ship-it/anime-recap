"""
Pipeline tests. Every external service (Drive, YouTube, Gemini, Whisper,
edge-tts, ffmpeg) is replaced by FakeDeps, so nothing real is called or uploaded.
"""
import os
import shutil
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recap_core as core  # noqa: E402
import recap_pipeline as pipe  # noqa: E402
import recap_plan as plan  # noqa: E402


class MemoryBackend:
    def __init__(self):
        self.text = ""
        self.fail_writes = 0

    def read_csv(self):
        return self.text

    def write_csv(self, text):
        if self.fail_writes:
            self.fail_writes -= 1
            raise core.TemporaryError("sheet busy")
        self.text = text


class Crash(BaseException):
    """Simulates the runner dying (not an ordinary exception)."""


def cfg_small(**kw):
    base = dict(long_dir="LONG", shorts_dir="SHORTS", attempts=2, base_delay=0.0,
                sleep=lambda s: None, max_retries=3, episode_seconds=60.0,
                min_last_episode=20.0, shorts_per_episode=2, scene_window=20.0,
                uploads_per_run=50, short_min=8.0, short_max=15.0)
    base.update(kw)
    return core.Config(**base)


class FakeDeps:
    def __init__(self, sources=None):
        # sources: {id: (name, seconds)}
        self.sources = sources or {"SRC1": ("The Door.mp4", 60.0)}
        self.dur = {}               # path -> seconds
        self.drive = {}             # drive id -> (seconds, bytes)
        self.calls = []
        self.fail = {}              # method -> [exceptions or None], consumed in order
        self.moved = []
        self.yt_titles = []
        self.yt_desc = []
        self.yt_ids = []
        self.n = 0
        self.render_extra = 0.0     # make rendered video longer/shorter than asked
        self.verdict = "CONFIRMED"
        self.verified = []
        self.long_text = False
        self.vision = True
        self.cut_shift = 0.0
        self.store = None
        self.status_when_moved = []
        self.bad_id = False
        self.mega_total = None

    # ---- helpers
    def count(self, name):
        return self.calls.count(name)

    def _maybe_fail(self, name):
        q = self.fail.get(name)
        if q:
            e = q.pop(0)
            if e is not None:
                raise e

    def _new_id(self):
        self.n += 1
        return f"DRV{self.n:04d}"

    # ---- Drive
    def list_incoming(self):
        return [{"id": k, "name": v[0]} for k, v in self.sources.items()
                if k not in self.moved]

    def download_source(self, fid, path):
        self.calls.append("download_source")
        open(path, "wb").write(b"src")
        self.dur[path] = self.sources[fid][1]

    def download_drive(self, fid, path):
        self.calls.append("download_drive")
        d, b = self.drive[fid]
        open(path, "wb").write(b)
        self.dur[path] = d

    def upload_drive(self, path, folder):
        self.calls.append("upload_drive")
        self._maybe_fail("upload_drive")
        fid = self._new_id()
        self.drive[fid] = (self.dur.get(path, 0), open(path, "rb").read())
        return fid

    def drive_size(self, fid):
        return 1_000_000

    def disk_free(self):
        return 10 ** 12

    def move_to_done(self, fid):
        self.calls.append("move_to_done")
        self.moved.append(fid)
        if self.store is not None:
            self.status_when_moved.append(self.store.get(fid)["status"])

    # ---- YouTube
    def upload_youtube(self, path, title, desc, tags):
        self.calls.append("upload_youtube")
        self._maybe_fail("upload_youtube")
        self.n += 1
        vid = "https://drive.google.com/drive/folders/abc" if self.bad_id else f"YTvid{self.n:06d}"
        self.yt_titles.append(title)
        self.yt_desc.append(desc)
        self.yt_ids.append(vid)
        return vid

    def verify_youtube(self, vid):
        self.calls.append("verify_youtube")
        self.verified.append(vid)
        return self.verdict

    # ---- media
    def probe(self, path):
        return {"duration": self.dur[path], "has_audio": True, "has_video": True}

    def find_cut(self, src, t, tol):
        return t + self.cut_shift

    def cut_audio(self, src, start, dur, out):
        self.calls.append("cut_audio")
        open(out, "wb").write(b"a")

    def transcribe_timed(self, path):
        self.calls.append("transcribe")
        self._maybe_fail("transcribe")
        return [{"start": 0.0, "end": 4.0, "text": "你好"}, {"start": 5.0, "end": 9.0, "text": "再见"}]

    def analyze_scenes(self, src, start, dur, transcript):
        self.calls.append("analyze_scenes")
        if not self.vision:
            raise RuntimeError("no video analysis")
        n = max(1, int(dur // 20))
        step = dur / n
        return [{"start": i * step, "end": (i + 1) * step, "summary": f"scene {i}",
                 "confidence": "high"} for i in range(n)]

    def write_narration(self, scenes, info):
        self.calls.append("write_narration")
        self._maybe_fail("write_narration")
        out = {}
        for s in scenes:
            words = int(s["budget"] * (3.0 if self.long_text and not info.get("shorter") else 0.8))
            sents = max(1, words // 5)
            out[s["index"]] = " ".join(" ".join(["शब्द"] * 5) + "।" for _ in range(sents))
        return out

    def synthesize(self, text, path):
        self.calls.append("synthesize")
        open(path, "wb").write(b"mp3")
        self.dur[path] = plan.count_words(text) / 2.2
        return []

    def audio_duration(self, path):
        return self.dur[path]

    def to_wav(self, mp3, wav):
        n = int(self.dur[mp3] * plan.WAV_RATE)
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(plan.WAV_RATE)
            w.writeframes(b"\x00\x00" * n)
        self.dur[wav] = self.dur[mp3]

    def render_episode(self, src, start, D, nar, out, srt):
        self.calls.append("render_episode")
        self._maybe_fail("render_episode")
        open(out, "wb").write(b"video-%d" % self.n)
        self.dur[out] = D + self.render_extra
        self.last_nar = nar

    def render_short(self, lv, w, srt, out):
        self.calls.append("render_short")
        open(out, "wb").write(b"short")
        self.dur[out] = w["len"]

    def validate_video(self, path, kind, expected, tol):
        if abs(self.dur[path] - expected) > tol:
            raise core.MediaError(f"{os.path.basename(path)} is {self.dur[path]:.2f}s, expected {expected:.2f}s")

    def av_check(self, path, expected, tol):
        return True, "ok"

    def silence_gaps(self, path):
        return []

    def concat_videos(self, paths, out):
        self.calls.append("concat")
        open(out, "wb").write(b"mega")
        self.dur[out] = sum(self.dur[p] for p in paths) if self.mega_total is None else self.mega_total

    # ---- Gemini-style text
    def episode_metadata(self, info):
        return {"title": f"{info['story']} S{info['season']}E{info['episode']} - कहानी",
                "description": "कहानी का सार।", "tags": ["anime", "हिंदी रीकैप"],
                "hashtags": ["#Anime"], "thumbnail_text": "देखो"}

    def short_metadata(self, info):
        return {"title": f"रहस्य {info['part']}", "description": "छोटा सार।", "tags": ["shorts"],
                "hashtags": ["#Anime"], "cta": "असली खुलासा बाकी है!"}

    def mega_metadata(self, info):
        return {"title": f"FULL SEASON {info['season']} - All {info['count']} Episodes Complete | {info['story']}",
                "description": "पूरा सीज़न।", "tags": ["anime"], "hashtags": ["#FullSeason"]}

    def pick_shorts(self, cues):
        self.calls.append("pick_shorts")
        return '[{"start_cue": 1, "title": "रहस्य खुला", "description": "d"}]'


def run_once(deps, backend, cfg=None, tmp=None, monotonic=lambda: 0.0):
    cfg = cfg or cfg_small()
    store = core.StateStore(backend, sleep=lambda s: None, base_delay=0.0, log=lambda *a: None)
    deps.store = store
    s = pipe.run_all(deps, store, cfg, workdir=tmp, log=lambda *a: None, monotonic=monotonic)
    return s, store


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)


class TestDurationRule(T):
    def test_short_source_keeps_its_exact_duration(self):
        deps = FakeDeps({"S": ("clip.mp4", 10.0)})
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(row["status"], core.COMPLETED)
        self.assertEqual(float(row["output_duration"]), 10.0)
        self.assertEqual(float(row["duration_diff"]), 0.0)
        self.assertEqual(row["sync_status"], "OK")
        self.assertEqual(store.get("S")["status"], core.COMPLETED)
        self.assertEqual(deps.count("concat"), 0)                  # one episode: no mega
        self.assertEqual(s["final_status"], "SUCCESS")

    def test_normal_source_is_one_episode_with_same_length(self):
        deps = FakeDeps({"S": ("ep.mp4", 39.0)})
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(float(store.get("S:E01")["output_duration"]), 39.0)

    def test_very_short_source_gets_no_forced_narration(self):
        deps = FakeDeps({"S": ("tiny.mp4", 3.0)})
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(row["status"], core.COMPLETED)
        self.assertEqual(float(row["output_duration"]), 3.0)
        self.assertEqual(deps.count("synthesize"), 0)
        self.assertIn("no narration", row["warnings"])
        self.assertEqual(row["subtitle_status"], "NONE (no narration)")

    def test_narration_longer_than_budget_is_shortened_not_the_video_lengthened(self):
        deps = FakeDeps({"S": ("ep.mp4", 60.0)})
        deps.long_text = True
        _, store = run_once(deps, MemoryBackend(), cfg_small(episode_seconds=600.0), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(row["status"], core.COMPLETED)
        self.assertEqual(float(row["output_duration"]), 60.0)
        self.assertLessEqual(float(row["narration_duration"]), 60.0 * 0.9)

    def test_video_of_wrong_length_is_never_marked_successful(self):
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        deps.render_extra = 3.0                                   # export came out 3 s too long
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertNotEqual(row["status"], core.COMPLETED)
        self.assertEqual(float(row["duration_diff"]), 3.0)         # the difference is reported
        self.assertEqual(row["failed_stage"], core.EDITING)
        self.assertEqual(deps.count("upload_youtube"), 0)
        self.assertEqual(deps.moved, [])
        self.assertNotEqual(s["final_status"], "SUCCESS")

    def test_narration_track_is_exactly_the_episode_length(self):
        deps = FakeDeps({"S": ("ep.mp4", 45.0)})
        run_once(deps, MemoryBackend(), cfg_small(episode_seconds=600.0), tmp=self.tmp)
        # the wav is removed with the work folder; rebuild the check on a copy
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        clip = os.path.join(d, "c.wav")
        with wave.open(clip, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(plan.WAV_RATE)
            w.writeframes(b"\x00\x00" * (5 * plan.WAV_RATE))
        out = os.path.join(d, "n.wav")
        plan.assemble_narration_wav([{"start": 40.0, "dur": 5.0, "wav": clip}], 45.0, out)
        self.assertAlmostEqual(plan.wav_seconds(out), 45.0, places=3)
        with self.assertRaises(ValueError):                         # would overrun the video
            plan.assemble_narration_wav([{"start": 42.0, "dur": 5.0, "wav": clip}], 45.0, out)


class TestEpisodesAndMega(T):
    def test_two_episodes_then_one_mega_video(self):
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        e1, e2 = store.get("S:E01"), store.get("S:E02")
        self.assertEqual((float(e1["segment_start"]), float(e1["segment_end"])), (0.0, 60.0))
        self.assertEqual((float(e2["segment_start"]), float(e2["segment_end"])), (60.0, 130.0))
        self.assertEqual(float(e1["output_duration"]) + float(e2["output_duration"]), 130.0)
        self.assertEqual(store.get("S")["expected_episodes"], "2")
        mega = store.get("S:MEGA")
        self.assertEqual(mega["status"], core.COMPLETED)
        self.assertAlmostEqual(float(mega["output_duration"]), 130.0, places=2)
        self.assertEqual(deps.count("concat"), 1)
        self.assertTrue(plan.is_valid_youtube_id(mega["yt_long_id"]))
        self.assertEqual(store.get("S")["season_status"], "COMPLETE")
        # the mega video is the LAST upload and happens once
        self.assertIn("FULL SEASON", deps.yt_titles[-1])
        self.assertEqual(sum("FULL SEASON" in t for t in deps.yt_titles), 1)
        self.assertEqual(deps.status_when_moved, [core.COMPLETED])

    def test_episode_titles_are_unique_and_numbered(self):
        deps = FakeDeps({"S": ("big.mp4", 190.0)})
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        titles = [store.get(f"S:E0{i}")["title"] for i in (1, 2, 3)]
        self.assertEqual(len(set(titles)), 3)
        self.assertEqual([store.get(f"S:E0{i}")["episode"] for i in (1, 2, 3)], ["1", "2", "3"])
        self.assertEqual(store.get("S:MEGA")["status"], core.COMPLETED)

    def test_failed_episode_blocks_the_mega_video(self):
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        deps.fail["render_episode"] = [None, ValueError("render crashed")]
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(store.get("S:E01")["status"], core.COMPLETED)
        self.assertEqual(store.get("S:E02")["status"], core.RETRY_PENDING)
        self.assertIsNone(store.get("S:MEGA"))
        self.assertEqual(deps.count("concat"), 0)
        self.assertNotEqual(store.get("S")["status"], core.COMPLETED)
        self.assertEqual(deps.moved, [])

    def test_retry_finishes_the_episode_without_duplicate_uploads_then_makes_mega(self):
        backend = MemoryBackend()
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        deps.fail["render_episode"] = [None, ValueError("render crashed")]
        run_once(deps, backend, tmp=self.tmp)
        uploads_after_first = deps.count("upload_youtube")
        long1 = core.StateStore(backend)
        long1.load()
        id1 = long1.get("S:E01")["yt_long_id"]
        s, store = run_once(deps, backend, tmp=self.tmp)           # same fake world, run again
        self.assertEqual(store.get("S:E01")["yt_long_id"], id1)    # untouched
        self.assertEqual(store.get("S:E02")["status"], core.COMPLETED)
        self.assertEqual(store.get("S:MEGA")["status"], core.COMPLETED)
        # first run: ep1 long + 2 shorts = 3. second run: ep2 long + 2 shorts + mega = 4
        self.assertEqual(uploads_after_first, 3)
        self.assertEqual(deps.count("upload_youtube"), 7)
        self.assertEqual(deps.count("concat"), 1)

    def test_mega_failure_is_retried_without_reuploading_episodes(self):
        backend = MemoryBackend()
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        # uploads: ep1 (3) + ep2 (3) succeed, the 7th (mega) fails
        deps.fail["upload_youtube"] = [None] * 6 + [ValueError("youtube hiccup")]
        s, store = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(store.get("S:E01")["status"], core.COMPLETED)
        self.assertEqual(store.get("S:E02")["status"], core.COMPLETED)
        self.assertEqual(store.get("S:MEGA")["status"], core.RETRY_PENDING)
        self.assertEqual(deps.moved, [])
        s2, store2 = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(store2.get("S:MEGA")["status"], core.COMPLETED)
        self.assertEqual(deps.count("upload_youtube"), 7 + 1)      # only the mega again
        self.assertEqual(deps.count("concat"), 1)                  # not joined twice
        self.assertEqual(deps.moved, ["S"])

    def test_mega_blocked_by_platform_limit_keeps_episodes_and_explains(self):
        backend = MemoryBackend()
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        s, store = run_once(deps, backend, cfg_small(mega_max_seconds=100.0), tmp=self.tmp)
        self.assertEqual(store.get("S:E01")["status"], core.COMPLETED)
        self.assertEqual(store.get("S:E02")["status"], core.COMPLETED)
        self.assertTrue(store.get("S")["season_status"].startswith("MEGA_BLOCKED"))
        self.assertNotEqual(store.get("S")["status"], core.COMPLETED)   # never falsely complete
        self.assertEqual(deps.moved, [])
        self.assertTrue(s["blocked"])
        s2, store2 = run_once(deps, backend, cfg_small(mega_max_seconds=100.0, mega_enabled=False),
                              tmp=self.tmp)
        self.assertEqual(store2.get("S")["status"], core.COMPLETED)
        self.assertEqual(deps.count("upload_youtube"), 6)           # no extra episode uploads

    def test_mega_disabled(self):
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        _, store = run_once(deps, MemoryBackend(), cfg_small(mega_enabled=False), tmp=self.tmp)
        self.assertIsNone(store.get("S:MEGA"))
        self.assertEqual(store.get("S")["status"], core.COMPLETED)

    def test_boundary_snaps_to_a_nearby_cut_but_still_covers_everything(self):
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        deps.cut_shift = 5.0
        _, store = run_once(deps, MemoryBackend(), cfg_small(snap_tolerance=20.0), tmp=self.tmp)
        e1, e2 = store.get("S:E01"), store.get("S:E02")
        self.assertEqual(float(e1["segment_end"]), 65.0)
        self.assertEqual(float(e1["segment_end"]), float(e2["segment_start"]))
        self.assertEqual(float(e2["segment_end"]), 130.0)

    def test_split_plan_has_no_gaps_or_overlaps_for_hours_long_sources(self):
        for hours, expected in ((0.5, 1), (1, 1), (2, 2), (3, 3), (10, 10)):
            total = hours * 3600.0
            eps = plan.plan_episodes(total, 3600.0, 600.0)
            self.assertEqual(len(eps), expected, hours)
            self.assertEqual(plan.check_episode_coverage(eps, total), [])
            self.assertAlmostEqual(sum(e["end"] - e["start"] for e in eps), total, places=2)


class TestUploads(T):
    def test_wrong_id_is_never_verified_or_recorded(self):
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        deps.bad_id = True
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(deps.verified, [])                          # a Drive URL never reaches YouTube
        self.assertEqual(row["yt_long_id"], "")
        self.assertEqual(row["status"], core.FAILED)
        self.assertIn("valid YouTube video id", row["error"])
        self.assertNotEqual(row["status"], core.COMPLETED)

    def test_rejected_upload_stops_the_shorts_and_is_not_successful(self):
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        deps.verdict = "REJECTED"
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(row["status"], core.FAILED)
        self.assertEqual(deps.count("upload_youtube"), 1)            # Shorts were not published
        self.assertEqual(row["yt_long_verify"], "REJECTED")

    def test_shorts_link_to_the_verified_long_video(self):
        deps = FakeDeps({"S": ("ep.mp4", 60.0)})
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        long_id = row["yt_long_id"]
        self.assertTrue(plan.is_valid_youtube_id(long_id))
        self.assertEqual(row["yt_long_url"], plan.youtube_url(long_id))
        shorts = [d for t, d in zip(deps.yt_titles, deps.yt_desc) if "#shorts" in t.lower()]
        self.assertEqual(len(shorts), 2)
        for d in shorts:
            self.assertIn(plan.youtube_url(long_id), d)
        self.assertTrue(all(plan.is_valid_youtube_id(row[f"yt_short{i}_id"]) for i in (1, 2)))

    def test_short_windows_follow_the_duration_rule(self):
        deps = FakeDeps({"S": ("ep.mp4", 60.0)})
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        import json
        windows = json.loads(store.get("S:E01")["short_plan"])
        self.assertEqual(len(windows), 2)
        for w in windows:
            self.assertGreaterEqual(w["len"], 8.0)
            self.assertLessEqual(w["len"], 15.0)
            self.assertLessEqual(w["start"] + w["len"], 60.0 + 0.01)

    def test_unconfirmed_uploads_are_reported_honestly(self):
        deps = FakeDeps({"S": ("ep.mp4", 60.0)})
        deps.verdict = "UNVERIFIED"
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(s["yt_confirmed"], 0)
        self.assertEqual(s["yt_uploaded"], 3)
        self.assertEqual(store.get("S:E01")["youtube_status"], "ID_RETURNED")
        self.assertIn("could not confirm", pipe.build_summary_text(s))

    def test_upload_guard_stops_the_run_and_resumes_without_duplicates(self):
        backend = MemoryBackend()
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        s, store = run_once(deps, backend, cfg_small(uploads_per_run=2), tmp=self.tmp)
        self.assertIn("upload guard", s["stopped"])
        self.assertEqual(deps.count("upload_youtube"), 2)
        self.assertNotEqual(store.get("S")["status"], core.COMPLETED)
        for _ in range(6):                                           # "next days"
            run_once(deps, backend, cfg_small(uploads_per_run=2), tmp=self.tmp)
        _, final = run_once(deps, backend, cfg_small(uploads_per_run=2), tmp=self.tmp)
        self.assertEqual(final.get("S")["status"], core.COMPLETED)
        self.assertEqual(deps.count("upload_youtube"), 7)            # 3 + 3 + mega, each exactly once
        self.assertEqual(len(set(deps.yt_ids)), 7)

    def test_quota_error_stops_the_run_and_is_not_counted_as_a_retry(self):
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        err = type("E", (Exception,), {})("HTTP 403 quotaExceeded")
        deps.fail["upload_youtube"] = [err]
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertTrue(s["quota_exceeded"])
        self.assertEqual(s["final_status"], "QUOTA_EXCEEDED")
        self.assertEqual(store.get("S:E01")["status"], core.RETRY_PENDING)
        self.assertEqual(store.get("S:E01")["retry_count"], "0")


class TestStateAndRecovery(T):
    def test_completed_source_is_never_processed_again(self):
        backend = MemoryBackend()
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        run_once(deps, backend, tmp=self.tmp)
        deps.moved.clear()                                           # pretend the move failed
        before = list(deps.calls)
        s, _ = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(s["duplicates_skipped"], 1)
        for name in ("transcribe", "render_episode", "upload_youtube", "download_source"):
            self.assertEqual(deps.calls.count(name), before.count(name), name)
        self.assertEqual(deps.moved, ["S"])

    def test_same_name_different_id_are_different_sources(self):
        deps = FakeDeps({"A": ("same.mp4", 20.0), "B": ("same.mp4", 20.0)})
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(store.get("A")["status"], core.COMPLETED)
        self.assertEqual(store.get("B")["status"], core.COMPLETED)

    def test_failure_records_stage_error_and_keeps_the_source(self):
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        deps.fail["transcribe"] = [ValueError("whisper exploded")]
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(row["status"], core.RETRY_PENDING)
        self.assertEqual(row["failed_stage"], core.TRANSCRIBING)
        self.assertEqual(row["transcription_status"], "FAILED")
        self.assertEqual(row["retry_count"], "1")
        self.assertIn("whisper exploded", row["error"])
        self.assertEqual(deps.moved, [])
        self.assertEqual(s["final_status"], "FAILED")

    def test_gives_up_after_the_retry_limit(self):
        backend = MemoryBackend()
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        for _ in range(4):
            deps.fail["transcribe"] = [ValueError("still broken")] * 5
            _, store = run_once(deps, backend, tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(row["status"], core.FAILED)
        self.assertEqual(row["retry_count"], "3")
        self.assertEqual(deps.moved, [])

    def test_temporary_errors_are_retried_inside_the_run(self):
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        deps.fail["write_narration"] = [core.TemporaryError("503")]
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        self.assertEqual(store.get("S:E01")["status"], core.COMPLETED)

    def test_recovery_after_the_runner_dies_mid_episode(self):
        backend = MemoryBackend()
        deps = FakeDeps({"S": ("big.mp4", 130.0)})
        deps.fail["upload_youtube"] = [None, Crash()]                # dies on the 2nd upload (a Short)
        with self.assertRaises(Crash):
            run_once(deps, backend, tmp=self.tmp)
        uploads = deps.count("upload_youtube")
        # the sheet still knows what is finished; the old run is stale
        s, store = run_once(deps, backend, cfg_small(stale_hours=-1), tmp=self.tmp)
        self.assertEqual(store.get("S")["status"], core.COMPLETED)
        self.assertEqual(len(deps.yt_ids), 7)                        # 7 real uploads in total
        self.assertEqual(len(set(deps.yt_ids)), 7)                   # never a duplicate
        self.assertEqual(deps.count("upload_youtube"), 8)            # +1 attempt that died mid-way
        self.assertGreaterEqual(uploads, 1)

    def test_fresh_work_from_another_run_is_left_alone(self):
        backend = MemoryBackend()
        st = core.StateStore(backend, sleep=lambda s: None, base_delay=0.0)
        st.upsert("S", kind=core.KIND_SEASON, filename="ep.mp4", status=core.EDITING)
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        s, _ = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(deps.count("transcribe"), 0)

    def test_time_budget_stops_before_starting_new_work(self):
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        ticks = iter([0.0] + [10 ** 6] * 50)
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp, monotonic=lambda: next(ticks))
        self.assertIn("time budget", s["stopped"])
        self.assertEqual(deps.count("transcribe"), 0)
        self.assertNotEqual(store.get("S")["status"], core.COMPLETED)

    def test_video_analysis_failure_falls_back_and_is_flagged(self):
        deps = FakeDeps({"S": ("ep.mp4", 40.0)})
        deps.vision = False
        _, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        self.assertEqual(row["status"], core.COMPLETED)
        self.assertEqual(row["scene_match"], "TRANSCRIPT_WINDOWS_UNVERIFIED")
        self.assertIn("not verified", row["warnings"])

    def test_sheet_records_the_full_report(self):
        deps = FakeDeps({"S": ("ep.mp4", 40.0)})
        s, store = run_once(deps, MemoryBackend(), tmp=self.tmp)
        row = store.get("S:E01")
        for col in ("season", "episode", "expected_episodes", "segment_start", "segment_end",
                    "output_duration", "narration_duration", "sync_status", "subtitle_status",
                    "scene_match", "yt_long_id", "yt_long_url", "drive_long_link", "completed_at"):
            self.assertTrue(row[col] != "", col)
        self.assertEqual(row["scene_match"], "GEMINI_VIDEO")
        text = pipe.build_summary_text(s)
        self.assertIn("Duration report", text)

    def test_sheet_write_failures_are_retried(self):
        backend = MemoryBackend()
        backend.fail_writes = 2
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        _, store = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(store.get("S:E01")["status"], core.COMPLETED)

    def test_old_failed_row_is_reset_so_the_video_is_tried_again(self):
        backend = MemoryBackend()
        cols = core.COLUMNS
        old = dict.fromkeys(cols, "")
        old.update(source_id="S", filename="ep.mp4", status="FAILED", failed_stage="GENERATING_SCRIPT")
        import csv, io
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([c for c in cols if c != "unit_key"])
        w.writerow([old[c] for c in cols if c != "unit_key"])
        backend.text = buf.getvalue()
        deps = FakeDeps({"S": ("ep.mp4", 30.0)})
        _, store = run_once(deps, backend, tmp=self.tmp)
        self.assertEqual(store.get("S")["status"], core.COMPLETED)


if __name__ == "__main__":
    unittest.main()
