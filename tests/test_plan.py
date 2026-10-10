import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recap_core as core  # noqa: E402
import recap_plan as plan  # noqa: E402


class TestBudgets(unittest.TestCase):
    def test_word_budget_scales_with_the_time_available(self):
        self.assertEqual(plan.word_budget(3.0), 0)                 # too short to speak naturally
        self.assertEqual(plan.word_budget(2.0), 0)
        small, mid, big = plan.word_budget(10), plan.word_budget(30), plan.word_budget(600)
        self.assertTrue(0 < small < mid < big)
        self.assertLess(big / 600.0, plan.DEFAULT_WPS)             # never faster than natural speech
        self.assertGreater(plan.word_budget(10, wps=3.0), small)    # configurable speaking rate

    def test_long_videos_are_not_truncated_by_a_short_video_limit(self):
        hour = plan.word_budget(3600)
        self.assertGreater(hour, 5000)

    def test_trim_drops_whole_sentences_only(self):
        text = "एक दो तीन चार। पाँच छह सात आठ। नौ दस ग्यारह बारह।"
        self.assertEqual(plan.trim_to_budget(text, 8), "एक दो तीन चार। पाँच छह सात आठ।")
        self.assertEqual(plan.trim_to_budget(text, 3), "")          # nothing fits: stay silent
        self.assertEqual(plan.trim_to_budget(text, 100), text)
        self.assertEqual(plan.drop_last_sentence(text), "एक दो तीन चार। पाँच छह सात आठ।")

    def test_generic_openers_are_removed(self):
        self.assertEqual(plan.strip_generic_opener("तो दोस्तों, कहानी शुरू होती है।"), "कहानी शुरू होती है।")
        self.assertEqual(plan.strip_generic_opener("आज आपका स्वागत है। एक लड़का था।"), "एक लड़का था।")

    def test_duration_tolerance_depends_on_frame_rate(self):
        self.assertAlmostEqual(plan.default_tolerance(30), 0.15)
        self.assertAlmostEqual(plan.default_tolerance(10), 0.3)
        self.assertEqual(plan.duration_check(10.1, 10.0, 0.15), (True, 0.1))
        self.assertFalse(plan.duration_check(10.4, 10.0, 0.15)[0])


class TestSceneMaps(unittest.TestCase):
    def test_gaps_overlaps_and_out_of_range_scenes_are_repaired(self):
        raw = [{"start": 0, "end": 20, "summary": "a"}, {"start": 18, "end": 40, "summary": "b"},
               {"start": 60, "end": 80, "summary": "c"}, {"start": "x", "end": 5},
               {"start": 200, "end": 300}]
        scenes, warns = plan.validate_scene_map(raw, 100.0)
        self.assertEqual(scenes[0]["start"], 0.0)
        self.assertEqual(scenes[-1]["end"], 100.0)
        for a, b in zip(scenes, scenes[1:]):
            self.assertAlmostEqual(a["end"], b["start"])            # no gaps, no overlaps
        self.assertTrue(any("overlap" in w for w in warns))
        self.assertTrue(any("ignored" in w for w in warns))
        self.assertEqual([s["index"] for s in scenes], list(range(len(scenes))))

    def test_empty_scene_list_returns_nothing_invented(self):
        scenes, _ = plan.validate_scene_map([], 50.0)
        self.assertEqual([s["start"] for s in scenes], [0.0])       # one unlabeled filler, no story
        self.assertTrue(scenes[0].get("filler"))

    def test_transcript_windows_cover_the_whole_video(self):
        tr = [{"start": 3, "end": 8, "text": "你好"}, {"start": 70, "end": 75, "text": "再见"}]
        win = plan.windows_from_transcript(100.0, tr, 30.0)
        self.assertEqual(win[0]["start"], 0.0)
        self.assertEqual(win[-1]["end"], 100.0)
        self.assertEqual(win[0]["confidence"], "unverified")
        self.assertIn("你好", win[0]["transcript"])
        self.assertIn("再见", win[2]["transcript"])


class TestEpisodes(unittest.TestCase):
    def test_leftover_shorter_than_the_minimum_is_folded_into_the_last_episode(self):
        eps = plan.plan_episodes(2 * 3600 + 300, 3600, 600)
        self.assertEqual(len(eps), 2)
        self.assertEqual(eps[-1]["end"], 7500)

    def test_short_source_stays_one_episode(self):
        self.assertEqual(len(plan.plan_episodes(1800, 3600, 600)), 1)
        self.assertEqual(len(plan.plan_episodes(39, 3600, 600)), 1)

    def test_snapping_moves_boundaries_but_keeps_full_coverage(self):
        eps = plan.plan_episodes(7300, 3600, 600, snap_fn=lambda t: t + 40)
        self.assertEqual(eps[0]["end"], 3640)
        self.assertEqual(plan.check_episode_coverage(eps, 7300), [])
        far = plan.plan_episodes(7300, 3600, 600, snap_fn=lambda t: t + 500, snap_tol=120)
        self.assertEqual(far[0]["end"], 3600)                        # too far: not used

    def test_coverage_checker_finds_gaps_and_overlaps(self):
        bad = [{"n": 1, "start": 0, "end": 50}, {"n": 2, "start": 60, "end": 100}]
        self.assertTrue(plan.check_episode_coverage(bad, 100))
        over = [{"n": 1, "start": 0, "end": 60}, {"n": 2, "start": 50, "end": 100}]
        self.assertTrue(plan.check_episode_coverage(over, 100))

    def test_keys_are_stable(self):
        self.assertEqual(plan.episode_key("abc", 3), "abc:E03")
        self.assertEqual(plan.mega_key("abc"), "abc:MEGA")
        self.assertEqual(plan.season_id("1jq0OY2ZOq-JEs", 2), "S2-1jq0OY2ZOq")


class TestMetadata(unittest.TestCase):
    def test_placeholders_and_generator_names_are_removed(self):
        meta = plan.sanitize_metadata(
            {"title": "Untitled", "description": "Made with Gemini\nअसली कहानी",
             "tags": ["anime", "Gemini", "anime", "x" * 40], "hashtags": ["Anime", "#Recap", "bad tag"]},
            "Door Episode 2")
        self.assertEqual(meta["title"], "Door Episode 2")
        self.assertNotIn("Gemini", meta["description"])
        self.assertEqual(meta["tags"], ["anime"])
        self.assertTrue(all(h.startswith("#") and " " not in h for h in meta["hashtags"]))
        self.assertIn("#Anime", meta["description"])

    def test_limits_are_enforced(self):
        meta = plan.sanitize_metadata({"title": "क" * 300, "description": "ख" * 9000,
                                       "tags": [f"tag{i}" for i in range(100)]}, "t")
        self.assertLessEqual(len(meta["title"]), 100)
        self.assertLessEqual(len(meta["description"]), 4800)
        self.assertLessEqual(len(meta["tags"]), 15)
        self.assertNotIn("<", plan.sanitize_metadata({"title": "a <b> c"}, "t")["title"])

    def test_youtube_id_validation(self):
        self.assertTrue(plan.is_valid_youtube_id("dQw4w9WgXcQ"))
        for bad in ("", None, "https://drive.google.com/drive/folders/1ES_CDeMwzgK3QZku1Hr1TozOKZoAN_3B",
                    "1ES_CDeMwzgK3QZku1Hr1TozOKZoAN_3B", "/tmp/video.mp4", "short"):
            self.assertFalse(plan.is_valid_youtube_id(bad), bad)
        self.assertEqual(plan.youtube_url("dQw4w9WgXcQ"), "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_a_pasted_sheet_link_is_accepted(self):
        link = "https://docs.google.com/spreadsheets/d/125_Czn8GDfSQko_2-l_eirRCovizHFXSvgNky82TITE/edit?usp=drivesdk"
        self.assertEqual(core.sheet_id_from_env({"SHEET_ID": link}), "125_Czn8GDfSQko_2-l_eirRCovizHFXSvgNky82TITE")
        self.assertEqual(core.sheet_id_from_env({"SHEET_ID": " abc123 \n"}), "abc123")


class TestSubtitlesAndShorts(unittest.TestCase):
    def test_subtitle_validation_catches_early_late_and_overlapping_cues(self):
        clips = [{"start": 10.0, "dur": 5.0}]
        good = [{"start": 10.2, "end": 12.0, "text": "a"}, {"start": 12.0, "end": 14.5, "text": "b"}]
        self.assertEqual(plan.validate_subtitles(good, 60.0, clips), [])
        early = [{"start": 8.0, "end": 9.5, "text": "a"}]
        self.assertTrue(plan.validate_subtitles(early, 60.0, clips))        # before the speech
        late = [{"start": 58.0, "end": 61.0, "text": "a"}]
        self.assertTrue(plan.validate_subtitles(late, 60.0))                # past the video end
        overlap = [{"start": 1, "end": 5, "text": "a"}, {"start": 4, "end": 6, "text": "b"}]
        self.assertTrue(plan.validate_subtitles(overlap, 60.0))

    def test_offset_cues_never_pass_the_clip_end(self):
        out = plan.offset_cues([{"start": 0, "end": 3, "text": "a"}, {"start": 4, "end": 6, "text": "b"}],
                               offset=10.0, clip_end=14.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["end"], 13.0)

    def test_shorts_are_planned_on_story_lines_without_overlap(self):
        cues = [{"start": i * 4.0, "end": i * 4.0 + 3.5, "text": f"c{i}"} for i in range(60)]
        picks = [{"start_cue": 3, "title": "A", "description": "x"},
                 {"start_cue": 4, "title": "B", "description": "y"}]
        w = plan.plan_shorts_v2(cues, 240.0, picks, min_len=20, max_len=58)
        self.assertEqual(len(w), 2)
        a, b = sorted(w, key=lambda x: x["start"])
        self.assertLessEqual(a["start"] + a["len"], b["start"])
        for x in w:
            self.assertTrue(20 * 0.75 <= x["len"] <= 58)
            self.assertAlmostEqual(x["start"] % 4.0, 0.0, places=1)          # starts on a story line
            self.assertTrue(any(abs(x["start"] + x["len"] - c["end"]) < 0.02 for c in cues))
        self.assertEqual(w[0]["title"], "A")

    def test_tiny_episode_gets_fewer_or_no_shorts_instead_of_fake_ones(self):
        cues = [{"start": 0.0, "end": 2.5, "text": "x"}]
        self.assertEqual(plan.plan_shorts_v2(cues, 3.0, None, min_len=20, max_len=58), [])

    def test_cta_is_inside_the_short_duration(self):
        cues = core.add_cta_cue([{"start": 0.0, "end": 9.0, "text": "a"}], 10.0, "CTA")
        self.assertEqual(cues[-1]["text"], "CTA")
        self.assertEqual(cues[-1]["end"], 10.0)
        self.assertLessEqual(cues[0]["end"], cues[-1]["start"])


if __name__ == "__main__":
    unittest.main()


class TestGeminiHelpers(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual(plan.parse_timestamp("01:30"), 90.0)
        self.assertEqual(plan.parse_timestamp("1:02:03"), 3723.0)
        self.assertEqual(plan.parse_timestamp(12.5), 12.5)
        self.assertEqual(plan.parse_timestamp("12,5"), 12.5)
        self.assertIsNone(plan.parse_timestamp("soon"))
        self.assertIsNone(plan.parse_timestamp(None))

    def test_model_choice_prefers_newest_stable_flash(self):
        names = ["models/gemini-2.5-flash", "models/gemini-3.5-flash", "models/gemini-3.5-flash-lite",
                 "models/gemini-3.6-flash-preview", "models/gemini-2.5-pro", "models/gemini-3.5-flash-image"]
        self.assertEqual(plan.pick_gemini_model(names), "gemini-3.5-flash")
        self.assertEqual(plan.pick_gemini_model(["gemini-flash-latest", "gemini-2.5-pro"]), "gemini-flash-latest")
        self.assertEqual(plan.pick_gemini_model(["gemini-4-flash-preview", "gemini-2.5-pro"]), "gemini-4-flash-preview")
        self.assertIsNone(plan.pick_gemini_model(["gemini-2.5-pro", "embedding-001"]))

    def test_loose_json_and_scene_answers(self):
        data = plan.parse_json_loose('Here you go:\n```json\n{"scenes": [{"start": "00:05", "end": "00:20", '
                                     '"summary": "a"}, {"start": "bad", "end": 3}]}\n```')
        scenes = plan.parse_scene_answer(data)
        self.assertEqual(len(scenes), 1)
        self.assertEqual((scenes[0]["start"], scenes[0]["end"]), (5.0, 20.0))
        self.assertEqual(plan.parse_scene_answer([{"start": 1, "end": 4}])[0]["confidence"], "medium")
        with self.assertRaises(ValueError):
            plan.parse_json_loose("no json here")
