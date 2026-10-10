"""
End-to-end run of the whole pipeline on tiny REAL video files with the real
FFmpeg code. Only the network services (Drive, YouTube, Gemini, Whisper, TTS)
are fake; the TTS voice is replaced by a quiet tone of the right length.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import recap_core as core  # noqa: E402
import recap_pipeline as pipe  # noqa: E402
import recap_plan as plan  # noqa: E402
from test_pipeline import FakeDeps, MemoryBackend, cfg_small  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class RealMediaDeps(FakeDeps):
    def __init__(self, sources, src_file, workdir):
        super().__init__(sources)
        self.src_file, self.workdir = src_file, workdir

    def download_source(self, fid, path):
        self.calls.append("download_source")
        shutil.copyfile(self.src_file, path)

    def probe(self, path):
        return core.probe(path)

    def cut_audio(self, src, start, dur, out):
        return core.extract_audio(src, start, dur, out)

    def synthesize(self, text, path):
        self.calls.append("synthesize")
        secs = max(0.5, plan.count_words(text) / 2.2)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", "sine=frequency=250",
                        "-t", f"{secs:.3f}", "-c:a", "libmp3lame", path], check=True, capture_output=True)
        return []

    def audio_duration(self, path):
        return core.probe(path)["duration"]

    def to_wav(self, mp3, wav):
        core.to_wav(mp3, wav)

    def render_episode(self, src, start, D, nar, out, srt):
        self.calls.append("render_episode")
        self.srt_used = srt
        core.render_episode(src, start, D, nar, out, srt, workdir=os.path.join(self.workdir, "re"),
                            font="DejaVu Sans")

    def render_short(self, lv, w, srt, out):
        self.calls.append("render_short")
        core.render_short(lv, w["start"], w["len"], out, srt, workdir=os.path.join(self.workdir, "rs"),
                          font="DejaVu Sans")

    def validate_video(self, path, kind, expected, tol):
        core.validate_output(path, kind, expected, abs_tol=tol)

    def av_check(self, path, expected, tol):
        return core.check_av_sync(path, expected, tol)

    def silence_gaps(self, path):
        return core.silence_gaps(path, min_gap=8.0)

    def concat_videos(self, paths, out):
        self.calls.append("concat")
        core.concat_videos(paths, out, workdir=os.path.join(self.workdir, "cc"))

    def find_cut(self, src, t, tol):
        return core.find_cut_point(src, t, tol)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class TestWholePipelineOnRealFiles(unittest.TestCase):
    def test_two_episodes_and_a_mega_video_keep_their_exact_durations(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        src = os.path.join(tmp, "source.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25",
                        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "40", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", src], check=True, capture_output=True)
        deps = RealMediaDeps({"S": ("story.mp4", 40.0)}, src, tmp)
        cfg = cfg_small(episode_seconds=25.0, min_last_episode=8.0, shorts_per_episode=1,
                        short_min=6.0, short_max=10.0, scene_window=10.0, snap_tolerance=3.0)
        store = core.StateStore(MemoryBackend(), sleep=lambda s: None, base_delay=0.0, log=lambda *a: None)
        deps.store = store
        s = pipe.run_all(deps, store, cfg, workdir=os.path.join(tmp, "work"), log=lambda *a: None)
        tol = plan.default_tolerance(30.0)
        self.assertEqual(s["final_status"], "SUCCESS", s["failed"])
        e1, e2, mega = store.get("S:E01"), store.get("S:E02"), store.get("S:MEGA")
        self.assertEqual(e1["status"], core.COMPLETED)
        self.assertEqual(e2["status"], core.COMPLETED)
        total = 0.0
        for row in (e1, e2):
            want = float(row["segment_end"]) - float(row["segment_start"])
            got = float(row["output_duration"])
            self.assertLessEqual(abs(got - want), tol, (want, got))          # the exact-duration rule
            self.assertEqual(row["sync_status"], "OK")
            total += got
        self.assertAlmostEqual(float(e1["segment_end"]), float(e2["segment_start"]), places=3)
        self.assertAlmostEqual(float(e2["segment_end"]), 40.0, places=2)    # whole source covered
        self.assertEqual(mega["status"], core.COMPLETED)
        self.assertLessEqual(abs(float(mega["output_duration"]) - total), 0.6)
        self.assertEqual(store.get("S")["status"], core.COMPLETED)
        # the files the pipeline uploaded are real, playable and the Shorts are vertical
        checked_short = False
        for fid, (dur, blob) in deps.drive.items():
            p = os.path.join(tmp, f"{fid}.bin")
            open(p, "wb").write(blob)
            try:
                info = core.probe(p)
            except core.MediaError:
                continue                                                    # the .srt files
            if info["height"] == 1920:
                checked_short = True
                self.assertEqual(info["width"], 1080)
                self.assertLessEqual(info["duration"], 10.1)
        self.assertTrue(checked_short)
        text = pipe.build_summary_text(s)
        self.assertIn("Duration report", text)


if __name__ == "__main__":
    unittest.main()
