"""
recap_pipeline.py - season / episode / mega-video orchestration.

Rules this file enforces:
  * every episode is exactly as long as its slice of the source (never looped,
    stretched or padded); a violation fails the job instead of "succeeding"
  * every finished output is written to the sheet at once, so a later failure
    never creates or uploads it a second time
  * the mega video is made only after EVERY episode is completed and recorded
"""
import json
import os
import shutil
import time

import recap_core as core
import recap_plan as plan
from recap_core import (COMPLETED, EDITING, FAILED, GENERATING_SCRIPT, GENERATING_VOICE,
                        KIND_EPISODE, KIND_MEGA, KIND_SEASON, PENDING, PROCESSING,
                        RETRY_PENDING, TRANSCRIBING, UPLOADING, MediaError,
                        PermanentError, QuotaExceeded, StageError, scrub)

DEFAULT_CTA = "असली खुलासा अभी बाकी है! पूरी कहानी: विवरण में दिया वीडियो देखें"


class RunStop(Exception):
    """Stop the whole run cleanly (time budget or upload guard). Nothing failed."""


class MegaBlocked(PermanentError):
    """The mega video cannot be made for a known, explainable reason."""


# ======================================================================
# Run state and small helpers
# ======================================================================
class Run:
    def __init__(self, deps, store, cfg, workroot="work", log=print, monotonic=time.monotonic):
        self.deps, self.store, self.cfg, self.log = deps, store, cfg, log
        self.workroot = workroot
        self._mono = monotonic
        self.t0 = monotonic()
        self.uploads = 0
        self.s = {"discovered": 0, "new": 0, "duplicates_skipped": 0,
                  "seasons_completed": 0, "episodes_completed": 0, "mega_completed": 0,
                  "failed": [], "blocked": [], "warnings": [],
                  "long_generated": 0, "shorts_generated": 0,
                  "yt_uploaded": 0, "yt_confirmed": 0,
                  "awaiting_retry": 0, "gave_up": 0,
                  "quota_exceeded": False, "stopped": "", "final_status": "",
                  "duration_report": []}

    def over_budget(self):
        return (self._mono() - self.t0) / 60.0 > self.cfg.run_budget_minutes

    def reserve_upload(self):
        if self.uploads >= self.cfg.uploads_per_run:
            raise RunStop("upload guard: YouTube API quota allows about "
                          f"{self.cfg.uploads_per_run} uploads per day")
        self.uploads += 1


class Unit:
    """Convenience wrapper around one sheet row."""

    def __init__(self, run, key):
        self.run, self.key = run, key

    def row(self):
        return self.run.store.get(self.key) or {}

    def has(self, col):
        return bool((self.row().get(col) or "").strip())

    def set(self, **f):
        return self.run.store.upsert(self.key, **f)

    def stage(self, status, fn, col=None, retry=True):
        self.set(status=status)
        try:
            if retry:
                return core.with_retries(fn, attempts=self.run.cfg.attempts,
                                         base_delay=self.run.cfg.base_delay,
                                         sleep=self.run.cfg.sleep, label=status,
                                         log=self.run.log)
            return fn()
        except RunStop:
            raise
        except Exception as e:  # noqa: BLE001
            raise StageError(status, e, col) from e


def record_failure(store, key, exc, cfg, log=print):
    stage_name, col, orig = "UNKNOWN", None, exc
    if isinstance(exc, StageError):
        stage_name, col, orig = exc.stage, exc.column, exc.original
    kind = core.classify_error(orig)
    msg = scrub(f"{type(orig).__name__}: {orig}")[:300]
    rec = store.get(key) or {}
    retries = int(rec.get("retry_count") or 0)
    if kind == "quota":
        status = RETRY_PENDING                    # not the video's fault: no retry used
    elif kind == "permanent":
        status = FAILED
    else:
        retries += 1
        status = FAILED if retries >= cfg.max_retries else RETRY_PENDING
    fields = dict(status=status, failed_stage=stage_name, error=msg, retry_count=retries)
    if col:
        fields[col] = "FAILED"
    store.upsert(key, **fields)
    log(f"[FAILED] {key} at {stage_name}: {msg}")
    return kind, status, stage_name, msg


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def attach_transcript(scenes, transcript):
    for s in scenes:
        if s.get("transcript"):
            continue
        lines = [t["text"] for t in transcript or []
                 if t["end"] > s["start"] and t["start"] < s["end"] and t.get("text")]
        s["transcript"] = " ".join(lines)[:1500]
    return scenes


def tolerance(cfg):
    return cfg.duration_tol or plan.default_tolerance(30.0)


def safe_name(stem, sid=""):
    return core.safe_name(stem, sid)


# ======================================================================
# One episode
# ======================================================================
def process_episode(run, src, ep, season_rec, workdir):
    """
    src: {'id','name','path','duration','has_audio'}; ep: {'n','start','end'}.
    Resumable: each finished output is recorded before the next step starts.
    """
    deps, store, cfg, log = run.deps, run.store, run.cfg, run.log
    key = plan.episode_key(src["id"], ep["n"])
    u = Unit(run, key)
    stem = os.path.splitext(src["name"])[0]
    safe = safe_name(stem, src["id"]) + f"_E{ep['n']:02d}"
    total_eps = int(season_rec.get("expected_episodes") or 1)
    D = round(ep["end"] - ep["start"], 3)
    tol = tolerance(cfg)
    os.makedirs(workdir, exist_ok=True)
    stats = {"long_generated": 0, "shorts_generated": 0, "yt_uploaded": 0, "yt_confirmed": 0}
    ctx = {}
    store.claim(key, src["name"], cfg.now(), kind=KIND_EPISODE,
                season=str(cfg.season_number), episode=str(ep["n"]),
                expected_episodes=str(total_eps), segment_start=ep["start"],
                segment_end=ep["end"], source_duration=D)

    def p(suffix):
        return os.path.join(workdir, safe + suffix)

    def get_source():
        if not src.get("path"):
            run_ensure_source(run, src)
        return src["path"]

    # ---------------- build: transcript, scenes, narration, render ----------------
    def build_episode():
        path = u.stage(TRANSCRIBING, get_source, "transcription_status")
        wav16 = p("_16k.wav")
        u.stage(TRANSCRIBING, lambda: deps.cut_audio(path, ep["start"], D, wav16),
                "transcription_status")
        transcript = u.stage(TRANSCRIBING, lambda: deps.transcribe_timed(wav16),
                             "transcription_status")
        u.set(transcription_status="DONE")

        warns, raw = [], None
        try:
            raw = deps.analyze_scenes(path, ep["start"], D, transcript)
        except Exception as e:  # noqa: BLE001
            warns.append(f"video analysis unavailable: {scrub(e)[:120]}")
        scenes, w = plan.validate_scene_map(raw, D) if raw else ([], [])
        warns += w
        if scenes:
            method = "GEMINI_VIDEO"
            low = sum(1 for s in scenes if str(s.get("confidence", "")).lower() == "low")
            if low:
                warns.append(f"{low} scene(s) matched with low confidence - please review")
        else:
            scenes = plan.windows_from_transcript(D, transcript, cfg.scene_window)
            method = "TRANSCRIPT_WINDOWS_UNVERIFIED"
            warns.append("scene match is time-based only (not verified against the pictures)")
        attach_transcript(scenes, transcript)
        for s in scenes:
            s["budget"] = plan.word_budget(s["end"] - s["start"], cfg.wps)
        todo = [s for s in scenes if s["budget"] > 0]

        info = {"title": stem, "episode": ep["n"], "episodes": total_eps,
                "prior": "", "shorter": False, "wps": cfg.wps}
        texts = {}
        for batch in chunks(todo, cfg.batch_scenes):
            got = u.stage(GENERATING_SCRIPT, lambda: deps.write_narration(batch, dict(info)),
                          "script_status")
            texts.update({int(k): v for k, v in (got or {}).items()})
            last = plan.split_sentences(texts.get(batch[-1]["index"], ""))
            info["prior"] = " ".join(last[-2:])
        over = [s for s in todo if plan.count_words(texts.get(s["index"], "")) > s["budget"] * 1.15]
        if over:
            got = u.stage(GENERATING_SCRIPT,
                          lambda: deps.write_narration(over, dict(info, shorter=True)),
                          "script_status")
            for k, v in (got or {}).items():
                if plan.count_words(v) < plan.count_words(texts.get(int(k), "x " * 999)):
                    texts[int(k)] = v
        u.set(script_status="DONE")

        clips, cues, spoken = [], [], 0.0
        for s in todo:
            window = s["end"] - s["start"] - 2 * plan.PAD
            text = plan.trim_to_budget(plan.strip_generic_opener(texts.get(s["index"], "")),
                                       s["budget"])
            mp3 = p(f"_n{s['index']:04d}.mp3")
            bounds, clip_dur = [], 0.0
            while text:
                bounds = u.stage(GENERATING_VOICE, lambda: deps.synthesize(text, mp3),
                                 "voice_status")
                clip_dur = deps.audio_duration(mp3)
                if clip_dur <= window:
                    break
                text = plan.drop_last_sentence(text)     # shorten the script, never speed up
            if not text:
                warns.append(f"scene {s['index']}: no narration fits {window:.0f}s - left silent")
                continue
            wav = p(f"_n{s['index']:04d}.wav")
            deps.to_wav(mp3, wav)
            clip_dur = plan.wav_seconds(wav)
            start = s["start"] + plan.PAD
            clips.append({"start": start, "dur": clip_dur, "wav": wav})
            cues += plan.offset_cues(core.make_cues(bounds, clip_dur, text), start, start + clip_dur)
            spoken += clip_dur
            s["narration"] = text
            if os.path.exists(mp3):
                os.remove(mp3)
        u.set(voice_status="DONE")
        if not clips:
            warns.append("no narration was produced for this episode (very short or silent scenes)")

        nar = p("_narration.wav")
        plan.assemble_narration_wav(clips, D, nar)        # exactly D seconds, by construction
        problems = plan.validate_subtitles(cues, D, clips)
        if problems:
            raise MediaError("subtitle timing problem: " + "; ".join(problems[:3]))
        srt = p("_hindi.srt")
        core.write_srt(cues, srt, cfg.wrap_landscape)
        sub_status = "OK" if cues else "NONE (no narration)"

        out = p("_hindi_recap.mp4")
        use_srt = srt if (cfg.burn_subs and cues) else None
        u.stage(EDITING, lambda: deps.render_episode(path, ep["start"], D, nar, out, use_srt),
                "long_status", retry=False)
        measured = deps.probe(out)["duration"]
        u.set(output_duration=round(measured, 3), duration_diff=round(measured - D, 3),
              narration_duration=round(spoken, 2), scene_match=method)   # recorded even if it fails
        u.stage(EDITING, lambda: deps.validate_video(out, "long", D, tol), "long_status",
                retry=False)
        ok, detail = deps.av_check(out, D, tol)
        u.set(sync_status="OK" if ok else f"FAILED: {detail}", subtitle_status=sub_status)
        if not ok:
            raise StageError(EDITING, MediaError(f"audio/video sync check failed: {detail}"),
                             "long_status")
        for a, b in deps.silence_gaps(out):
            warns.append(f"silent gap {a:.0f}-{b:.0f}s")
        run.s["duration_report"].append(
            {"unit": key, "source": D, "final": round(measured, 3),
             "diff": round(measured - D, 3), "narration": round(spoken, 2),
             "sync": "OK", "subtitles": sub_status, "scene_match": method})

        summary_text = " ".join(s.get("narration", "") for s in scenes)
        try:
            meta = deps.episode_metadata({
                "story": stem, "season": cfg.season_number, "episode": ep["n"],
                "episodes": total_eps, "summaries": " ".join(s.get("summary", "") for s in scenes)[:4000],
                "narration": summary_text[:5000]})
        except Exception as e:  # noqa: BLE001
            warns.append(f"metadata generator failed, plain title used: {scrub(e)[:100]}")
            meta = {}
        fallback = (f"{stem} - Season {cfg.season_number} Episode {ep['n']}"
                    if total_eps > 1 else f"{stem} Hindi Recap")
        meta = plan.sanitize_metadata(meta, fallback)
        u.set(title=meta["title"], thumbnail=meta["thumbnail_text"], summary=summary_text[:400],
              meta=json.dumps(meta, ensure_ascii=False), warnings=" | ".join(warns)[:900])

        srt_id = u.stage(UPLOADING, lambda: deps.upload_drive(srt, cfg.long_dir), "long_status")
        u.set(drive_srt_link=core.drive_link(srt_id))
        long_id = u.stage(UPLOADING, lambda: deps.upload_drive(out, cfg.long_dir), "long_status")
        u.set(drive_long_link=core.drive_link(long_id), long_status="DONE")
        ctx["long"], ctx["cues"], ctx["scenes"] = out, cues, scenes
        stats["long_generated"] = 1

    def long_local():
        if "long" not in ctx:
            path = p("_hindi_recap.mp4")
            u.stage(UPLOADING, lambda: deps.download_drive(
                core.id_from_link(u.row()["drive_long_link"]), path), "youtube_status")
            ctx["long"] = path
        return ctx["long"]

    def get_cues():
        if "cues" not in ctx:
            local = p("_resume.srt")
            u.stage(UPLOADING, lambda: deps.download_drive(
                core.id_from_link(u.row()["drive_srt_link"]), local), "shorts_status")
            with open(local, encoding="utf-8") as f:
                ctx["cues"] = core.parse_srt(f.read())
        return ctx["cues"]

    def upload_youtube(prefix, path, meta):
        run.reserve_upload()
        vid = u.stage(UPLOADING, lambda: deps.upload_youtube(
            path, meta["title"], meta["description"], meta.get("tags", [])), "youtube_status")
        if not plan.is_valid_youtube_id(vid):
            raise StageError(UPLOADING, PermanentError(
                "the upload did not return a valid YouTube video id"), "youtube_status")
        try:
            u.set(**{f"yt_{prefix}_id": vid, f"yt_{prefix}_url": plan.youtube_url(vid)})
        except Exception:
            log(f"!!! {prefix} WAS UPLOADED to YouTube (id={vid}) but the sheet write failed. "
                f"Add it by hand to avoid a duplicate upload.")
            raise
        stats["yt_uploaded"] += 1
        try:
            verdict = deps.verify_youtube(vid) if plan.is_valid_youtube_id(vid) else "UNVERIFIED"
        except Exception:  # noqa: BLE001
            verdict = "UNVERIFIED"
        if verdict == "CONFIRMED":
            stats["yt_confirmed"] += 1
        u.set(**{f"yt_{prefix}_verify": verdict})
        u.set(youtube_status=core._yt_summary(u.row()))
        if verdict == "REJECTED":
            raise StageError(UPLOADING, PermanentError("YouTube reports this upload as rejected"),
                             "youtube_status")
        return vid

    # ---------------- run the steps ----------------
    if not u.has("drive_long_link"):
        build_episode()
    meta = json.loads(u.row().get("meta") or "{}") or plan.sanitize_metadata({}, u.row().get("title") or stem)
    if not u.has("yt_long_id"):
        upload_youtube("long", long_local(), meta)
    long_url = plan.youtube_url(u.row()["yt_long_id"])

    # ---------------- Shorts ----------------
    keys = [("short1", 0), ("short2", 1)][:cfg.shorts_per_episode]

    def get_plan():
        raw = (u.row().get("short_plan") or "").strip()
        if raw:
            try:
                stored = json.loads(raw)
                if isinstance(stored, list) and stored:
                    return stored
            except ValueError:
                pass
        cues = get_cues()
        total = deps.probe(long_local())["duration"]
        try:
            picks = core.parse_ai_picks(deps.pick_shorts(cues), len(cues))
        except Exception as e:  # noqa: BLE001
            log(f"[shorts] AI clip choice failed, using fixed positions: {scrub(e)[:120]}")
            picks = []
        windows = plan.plan_shorts_v2(cues, total, picks, count=cfg.shorts_per_episode,
                                      min_len=cfg.short_min, max_len=cfg.short_max,
                                      name=u.row().get("title") or stem)
        u.set(short_plan=json.dumps(windows, ensure_ascii=False))
        return windows

    windows = None
    for prefix, i in keys:
        if u.has(f"yt_{prefix}_id"):
            continue
        path = p(f"_{prefix}.mp4")
        if u.has(f"drive_{prefix}_link"):
            u.stage(UPLOADING, lambda: deps.download_drive(
                core.id_from_link(u.row()[f"drive_{prefix}_link"]), path), "shorts_status")
        else:
            windows = windows or get_plan()
            if i >= len(windows):
                log(f"[shorts] not enough story for Short {i + 1}; skipped")
                continue
            w = windows[i]
            cues = get_cues()
            sl = core.slice_cues(cues, w["start"], w["start"] + w["len"])
            try:
                meta_s = deps.short_metadata({
                    "title": u.row().get("title") or stem, "episode": ep["n"], "part": i + 1,
                    "excerpt": " ".join(c["text"] for c in sl)[:1500], "full_url": long_url,
                    "hook": w.get("title", "")})
            except Exception as e:  # noqa: BLE001
                log(f"[shorts] metadata generator failed: {scrub(e)[:100]}")
                meta_s = {}
            cta = " ".join(str(meta_s.get("cta") or "").split())
            if not cta or len(cta) > 70:
                cta = DEFAULT_CTA
            ms = plan.sanitize_metadata(
                {**meta_s, "title": meta_s.get("title") or w.get("title"),
                 "description": f"{meta_s.get('description') or w.get('desc') or ''}"
                                f"\n\nपूरा वीडियो: {long_url}",
                 "hashtags": (meta_s.get("hashtags") or []) + ["#Shorts"]},
                f"{u.row().get('title') or stem} Part {i + 1}")
            if "#shorts" not in ms["title"].lower():
                ms["title"] = (ms["title"][:90] + " #Shorts").strip()
            u.set(short_plan=json.dumps(
                [dict(x, **({"title": ms["title"]} if k == i else {})) for k, x in enumerate(windows)],
                ensure_ascii=False))
            srt_s = p(f"_{prefix}.srt")
            core.write_srt(core.add_cta_cue(sl, w["len"], cta), srt_s, cfg.wrap_portrait)
            lv = long_local()
            u.stage(EDITING, lambda: deps.render_short(lv, w, srt_s if cfg.burn_subs else None, path),
                    "shorts_status", retry=False)
            u.stage(EDITING, lambda: deps.validate_video(path, "short", w["len"], tol),
                    "shorts_status", retry=False)
            drv = u.stage(UPLOADING, lambda: deps.upload_drive(path, cfg.shorts_dir), "shorts_status")
            u.set(**{f"drive_{prefix}_link": core.drive_link(drv)})
            u.set(meta=json.dumps(meta, ensure_ascii=False))
            ctx.setdefault("short_meta", {})[prefix] = ms
            stats["shorts_generated"] += 1
        ms = ctx.get("short_meta", {}).get(prefix)
        if ms is None:                                  # resumed: rebuild from stored plan
            stored = json.loads(u.row().get("short_plan") or "[]")
            t = stored[i]["title"] if i < len(stored) and stored[i].get("title") else \
                f"{u.row().get('title') or stem} Part {i + 1} #Shorts"
            ms = plan.sanitize_metadata(
                {"title": t, "description": f"पूरा वीडियो: {long_url}", "hashtags": ["#Shorts"]}, t)
        upload_youtube(prefix, path, ms)
    if all(u.has(f"drive_{k}_link") for k, _ in keys):
        u.set(shorts_status="DONE")

    # ---------------- finish ----------------
    planned = len(windows) if windows is not None else len(keys)   # a tiny source may fit fewer Shorts
    need = ["long"] + [k for k, i in keys if i < planned]
    missing = [k for k in need if not (u.has(f"yt_{k}_id") and u.has(f"drive_{k}_link"))]
    if missing:
        raise StageError("FINALIZE", RuntimeError(f"outputs missing: {missing}"))
    u.set(status=COMPLETED, completed_at=cfg.now(), error="", failed_stage="",
          youtube_status=core._yt_summary(u.row()))
    return stats


# ======================================================================
# Source file, season and mega video
# ======================================================================
def run_ensure_source(run, src):
    """Download the source once per run and measure it with ffprobe."""
    if src.get("path"):
        return src["path"]
    os.makedirs(run.workroot, exist_ok=True)
    path = os.path.join(run.workroot, safe_name(os.path.splitext(src["name"])[0], src["id"]) + "_source.mp4")
    core.with_retries(lambda: run.deps.download_source(src["id"], path),
                      attempts=run.cfg.attempts, base_delay=run.cfg.base_delay,
                      sleep=run.cfg.sleep, label="download source", log=run.log)
    info = run.deps.probe(path)
    if info["duration"] <= 0 or not info.get("has_video", True):
        raise MediaError("the source video cannot be read (duration is zero)")
    src.update(path=path, duration=info["duration"], has_audio=info.get("has_audio", False))
    return path


def ok_for_mega(row, cfg):
    if row.get("status") != COMPLETED or not plan.is_valid_youtube_id(row.get("yt_long_id")):
        return False
    v = row.get("yt_long_verify", "")
    if v in ("REJECTED",):
        return False
    return v == "CONFIRMED" if cfg.require_confirmed else True


def process_mega(run, src, episodes, workdir):
    deps, store, cfg, log = run.deps, run.store, run.cfg, run.log
    key = plan.mega_key(src["id"])
    u = Unit(run, key)
    stem = os.path.splitext(src["name"])[0]
    bad = [r["episode"] for r in episodes if not ok_for_mega(r, cfg)]
    if bad:
        raise MegaBlocked(f"episodes not completed and verified: {bad}")
    total = sum(float(r["output_duration"]) for r in episodes)
    if total > cfg.mega_max_seconds:
        raise MegaBlocked(
            f"the full season is {total / 3600:.1f} h; YouTube allows at most 12 h per video "
            f"(limit set to {cfg.mega_max_seconds / 3600:.1f} h). Episodes are safe; "
            f"set MEGA_ENABLED=0 to finish without the mega video")
    sizes = [deps.drive_size(core.id_from_link(r["drive_long_link"])) for r in episodes]
    if deps.disk_free() < 2.3 * sum(sizes):
        raise MegaBlocked(
            f"not enough disk space on the runner to join {sum(sizes) / 1e9:.1f} GB of episodes; "
            f"episodes are safe, set MEGA_ENABLED=0 or make the mega video elsewhere")
    store.claim(key, src["name"], cfg.now(), kind=KIND_MEGA, season=str(cfg.season_number),
                expected_episodes=str(len(episodes)), source_duration=round(total, 3))
    os.makedirs(workdir, exist_ok=True)
    stats = {"yt_uploaded": 0, "yt_confirmed": 0}
    if not u.has("drive_long_link"):
        files, cues_all, offset = [], [], 0.0
        for r in episodes:
            path = os.path.join(workdir, f"ep{int(r['episode']):02d}.mp4")
            u.stage(UPLOADING, lambda: deps.download_drive(core.id_from_link(r["drive_long_link"]), path),
                    "long_status")
            real = deps.probe(path)["duration"]
            if abs(real - float(r["output_duration"])) > 0.5:
                raise MediaError(f"episode {r['episode']} file is {real:.1f}s, sheet says "
                                 f"{float(r['output_duration']):.1f}s - not joining a damaged episode")
            files.append(path)
            if r.get("drive_srt_link"):
                sp = os.path.join(workdir, f"ep{int(r['episode']):02d}.srt")
                u.stage(UPLOADING, lambda: deps.download_drive(core.id_from_link(r["drive_srt_link"]), sp),
                        "long_status")
                with open(sp, encoding="utf-8") as f:
                    for c in core.parse_srt(f.read()):
                        cues_all.append({"start": c["start"] + offset, "end": c["end"] + offset,
                                         "text": c["text"]})
            offset += real
        out = os.path.join(workdir, safe_name(stem, src["id"]) + "_FULL_SEASON.mp4")
        u.stage(EDITING, lambda: deps.concat_videos(files, out), "long_status", retry=False)
        mtol = 0.3 + 0.1 * len(files)
        u.stage(EDITING, lambda: deps.validate_video(out, "long", total, mtol), "long_status", retry=False)
        ok, detail = deps.av_check(out, total, mtol)
        measured = deps.probe(out)["duration"]
        u.set(output_duration=round(measured, 3), duration_diff=round(measured - total, 3),
              sync_status="OK" if ok else f"FAILED: {detail}", narration_duration="",
              subtitle_status="OK" if cues_all else "NONE")
        if not ok:
            raise StageError(EDITING, MediaError(f"mega video sync check failed: {detail}"), "long_status")
        problems = plan.validate_subtitles(cues_all, total + 0.5)
        if problems:
            raise StageError(EDITING, MediaError("mega subtitle problem: " + "; ".join(problems[:3])),
                             "long_status")
        srt = os.path.join(workdir, "full_season.srt")
        core.write_srt(cues_all, srt, cfg.wrap_landscape)
        srt_id = u.stage(UPLOADING, lambda: deps.upload_drive(srt, cfg.long_dir), "long_status")
        u.set(drive_srt_link=core.drive_link(srt_id))
        mid = u.stage(UPLOADING, lambda: deps.upload_drive(out, cfg.long_dir), "long_status")
        u.set(drive_long_link=core.drive_link(mid), long_status="DONE")
        run.s["duration_report"].append({"unit": key, "source": round(total, 3),
                                         "final": round(measured, 3), "diff": round(measured - total, 3),
                                         "narration": "-", "sync": "OK", "subtitles": "OK",
                                         "scene_match": "-"})
        ctx_path = out
    else:
        ctx_path = os.path.join(workdir, safe_name(stem, src["id"]) + "_FULL_SEASON.mp4")
    if not u.has("yt_long_id"):
        if not os.path.exists(ctx_path):
            u.stage(UPLOADING, lambda: deps.download_drive(core.id_from_link(u.row()["drive_long_link"]), ctx_path),
                    "youtube_status")
        try:
            meta = deps.mega_metadata({
                "story": stem, "season": cfg.season_number, "count": len(episodes),
                "hours": round(total / 3600, 2),
                "episodes": [{"n": r["episode"], "title": r["title"], "summary": r["summary"]}
                             for r in episodes]})
        except Exception as e:  # noqa: BLE001
            log(f"[mega] metadata generator failed: {scrub(e)[:100]}")
            meta = {}
        meta = plan.sanitize_metadata(
            meta, f"FULL SEASON {cfg.season_number} - All {len(episodes)} Episodes Complete | {stem}")
        run.reserve_upload()
        vid = u.stage(UPLOADING, lambda: deps.upload_youtube(
            ctx_path, meta["title"], meta["description"], meta.get("tags", [])), "youtube_status")
        if not plan.is_valid_youtube_id(vid):
            raise StageError(UPLOADING, PermanentError("the upload did not return a valid YouTube video id"),
                             "youtube_status")
        u.set(yt_long_id=vid, yt_long_url=plan.youtube_url(vid), title=meta["title"])
        stats["yt_uploaded"] += 1
        try:
            verdict = deps.verify_youtube(vid)
        except Exception:  # noqa: BLE001
            verdict = "UNVERIFIED"
        if verdict == "CONFIRMED":
            stats["yt_confirmed"] += 1
        u.set(yt_long_verify=verdict, youtube_status="CONFIRMED" if verdict == "CONFIRMED" else "ID_RETURNED")
        if verdict == "REJECTED":
            raise StageError(UPLOADING, PermanentError("YouTube reports the mega upload as rejected"),
                             "youtube_status")
    u.set(status=COMPLETED, completed_at=cfg.now(), error="", failed_stage="")
    return stats


def process_season(run, f):
    """One source video: plan episodes, process each, then the mega video."""
    deps, store, cfg, log = run.deps, run.store, run.cfg, run.log
    sid = f["id"]
    src = {"id": sid, "name": f["name"], "path": None}
    stem = os.path.splitext(f["name"])[0]
    store.claim(sid, f["name"], cfg.now(), kind=KIND_SEASON, season=str(cfg.season_number),
                source_id=sid)
    season = store.get(sid)
    if not season["segments"]:
        run_ensure_source(run, src)
        total = src["duration"]
        eps = plan.plan_episodes(
            total, cfg.episode_seconds, cfg.min_last_episode, snap_tol=cfg.snap_tolerance,
            snap_fn=(lambda t: deps.find_cut(src["path"], t, cfg.snap_tolerance)))
        problems = plan.check_episode_coverage(eps, total)
        if problems:
            raise PermanentError("episode plan is inconsistent: " + "; ".join(problems))
        if len(eps) > cfg.max_episodes:
            raise PermanentError(f"this source needs {len(eps)} episodes; the safety limit is "
                                 f"{cfg.max_episodes} (MAX_EPISODES_PER_SOURCE)")
        store.upsert(sid, segments=json.dumps(eps), expected_episodes=len(eps),
                     source_duration=round(total, 3), season_status=f"PLANNED 0/{len(eps)}")
        season = store.get(sid)
    eps = json.loads(season["segments"])
    now = cfg.clock()

    for ep in eps:
        ekey = plan.episode_key(sid, ep["n"])
        erow = store.get(ekey)
        if erow and erow["status"] == COMPLETED:
            continue
        if erow and not core.is_eligible(erow, cfg, now):
            continue
        if run.over_budget():
            raise RunStop("time budget for this run is used up")
        try:
            stats = process_episode(run, src, ep, store.get(sid), os.path.join(run.workroot, safe_name(stem, sid), f"E{ep['n']:02d}"))
            for k in ("long_generated", "shorts_generated", "yt_uploaded", "yt_confirmed"):
                run.s[k] += stats[k]
            run.s["episodes_completed"] += 1
        except RunStop:
            if store.get(ekey):
                store.upsert(ekey, status=RETRY_PENDING)
            raise
        except Exception as e:  # noqa: BLE001
            kind, status, stage, msg = record_failure(store, ekey, e, cfg, log)
            run.s["failed"].append({"name": f"{f['name']} episode {ep['n']}", "stage": stage,
                                    "error": msg, "status": status})
            if kind == "quota":
                run.s["quota_exceeded"] = True
                raise RunStop("YouTube API quota is used up")
        finally:
            shutil.rmtree(os.path.join(run.workroot, safe_name(stem, sid), f"E{ep['n']:02d}"),
                          ignore_errors=True)

    rows = [store.get(plan.episode_key(sid, ep["n"])) or {} for ep in eps]
    done = sum(1 for r in rows if r.get("status") == COMPLETED)
    store.upsert(sid, season_status=f"EPISODES {done}/{len(eps)}")
    if any(r.get("status") == FAILED for r in rows):
        store.upsert(sid, status=FAILED, season_status=f"EPISODES {done}/{len(eps)} - an episode gave up")
        return "failed"
    if done < len(eps):
        store.upsert(sid, status=RETRY_PENDING)
        return "partial"

    # every episode is completed and recorded
    if len(eps) > 1 and cfg.mega_enabled:
        mrow = store.get(plan.mega_key(sid))
        if not (mrow and mrow["status"] == COMPLETED):
            try:
                st = process_mega(run, src, rows, os.path.join(run.workroot, safe_name(stem, sid), "MEGA"))
                run.s["yt_uploaded"] += st["yt_uploaded"]
                run.s["yt_confirmed"] += st["yt_confirmed"]
                run.s["mega_completed"] += 1
            except MegaBlocked as e:
                store.upsert(sid, status=FAILED, season_status=f"MEGA_BLOCKED: {scrub(e)[:200]}",
                             error=scrub(e)[:300])
                run.s["blocked"].append({"name": f["name"], "reason": scrub(e)[:300]})
                return "blocked"
            except RunStop:
                store.upsert(plan.mega_key(sid), status=RETRY_PENDING)
                raise
            except Exception as e:  # noqa: BLE001
                kind, status, stage, msg = record_failure(store, plan.mega_key(sid), e, cfg, log)
                run.s["failed"].append({"name": f"{f['name']} mega video", "stage": stage,
                                        "error": msg, "status": status})
                store.upsert(sid, status=RETRY_PENDING if status != FAILED else FAILED,
                             season_status="MEGA " + status)
                if kind == "quota":
                    run.s["quota_exceeded"] = True
                    raise RunStop("YouTube API quota is used up")
                return "mega_failed"
            finally:
                shutil.rmtree(os.path.join(run.workroot, safe_name(stem, sid), "MEGA"), ignore_errors=True)
    store.upsert(sid, status=COMPLETED, completed_at=cfg.now(), season_status="COMPLETE",
                 error="", failed_stage="")
    run.s["seasons_completed"] += 1
    try:
        core.with_retries(lambda: deps.move_to_done(sid), attempts=cfg.attempts,
                          base_delay=cfg.base_delay, sleep=cfg.sleep, label="move to Done", log=log)
    except Exception as e:  # noqa: BLE001
        log(f"[warn] season completed but moving the source to Done failed (retried next run): {scrub(e)[:120]}")
    return "completed"


def season_eligible(rec, cfg, now):
    if rec is None:
        return True
    if rec.get("status") == FAILED and str(rec.get("season_status", "")).startswith("MEGA_BLOCKED") \
            and not cfg.mega_enabled:
        return True
    return core.is_eligible(rec, cfg, now)


# ======================================================================
# Whole run, summary
# ======================================================================
def final_status(s):
    if s["quota_exceeded"]:
        return "QUOTA_EXCEEDED"
    bad = bool(s["failed"] or s["blocked"])
    if bad and (s["episodes_completed"] or s["seasons_completed"]):
        return "PARTIAL"
    if bad:
        return "FAILED"
    return "SUCCESS"


def run_all(deps, store, cfg, workdir="work", log=print, monotonic=time.monotonic):
    run = Run(deps, store, cfg, workdir, log, monotonic)
    s = run.s
    store.load()
    found = deps.list_incoming()
    s["discovered"] = len(found)
    for f in found:
        rec = store.get(f["id"])
        if rec is None:
            store.upsert(f["id"], source_id=f["id"], kind=KIND_SEASON, filename=f["name"],
                         date_received=cfg.now(), status=PENDING, season=str(cfg.season_number))
            s["new"] += 1
        elif rec["status"] == COMPLETED:
            s["duplicates_skipped"] += 1
            log(f"[duplicate] already completed, not processing again: {f['name']}")
            try:
                deps.move_to_done(f["id"])
            except Exception as e:  # noqa: BLE001
                log(f"[warn] could not move duplicate to Done: {scrub(e)[:120]}")
    now = cfg.clock()
    eligible = [f for f in found if season_eligible(store.get(f["id"]), cfg, now)]
    for f in eligible[:cfg.max_videos]:
        if run.over_budget():
            s["stopped"] = "time budget used up"
            break
        try:
            process_season(run, f)
        except RunStop as e:
            s["stopped"] = str(e)
            store.upsert(f["id"], status=RETRY_PENDING)
            log(f"[stop] {e}")
            break
        except Exception as e:  # noqa: BLE001
            kind, status, stage, msg = record_failure(store, f["id"], e, cfg, log)
            s["failed"].append({"name": f["name"], "stage": stage, "error": msg, "status": status})
            if kind == "quota":
                s["quota_exceeded"] = True
                break
        finally:
            shutil.rmtree(os.path.join(workdir, safe_name(os.path.splitext(f["name"])[0], f["id"])),
                          ignore_errors=True)
            p = os.path.join(workdir, safe_name(os.path.splitext(f["name"])[0], f["id"]) + "_source.mp4")
            if os.path.exists(p):
                os.remove(p)
    s["awaiting_retry"] = sum(1 for r in store.rows.values()
                              if r["status"] == RETRY_PENDING and r["kind"] != KIND_SEASON)
    s["gave_up"] = sum(1 for r in store.rows.values()
                       if r["status"] == FAILED and r["kind"] != KIND_SEASON)
    s["final_status"] = final_status(s)
    return s


def build_summary_text(s):
    lines = [
        "Anime Recap daily summary",
        f"Final status: {s['final_status']}",
        f"Sources found in Incoming: {s['discovered']} (new: {s['new']}, "
        f"already completed and skipped: {s['duplicates_skipped']})",
        f"Seasons (sources) completed: {s['seasons_completed']}",
        f"Episodes completed: {s['episodes_completed']}",
        f"Mega videos completed: {s['mega_completed']}",
        f"Long videos generated: {s['long_generated']}",
        f"Shorts generated: {s['shorts_generated']}",
        f"YouTube uploads (ID returned): {s['yt_uploaded']}",
        f"YouTube uploads confirmed by YouTube: {s['yt_confirmed']}",
        f"Failures this run: {len(s['failed'])}",
        f"Jobs awaiting retry: {s['awaiting_retry']}",
        f"Jobs that gave up (FAILED): {s['gave_up']}",
    ]
    if s["stopped"]:
        lines.append(f"Run stopped early: {s['stopped']}. Unfinished work resumes next run.")
    if s["quota_exceeded"]:
        lines.append("API quota was exceeded; remaining jobs wait for the next run.")
    for f in s["failed"]:
        lines.append(f"- {f['name']}: failed at {f['stage']} -> {f['status']} ({f['error'][:120]})")
    for b in s["blocked"]:
        lines.append(f"- {b['name']}: MEGA BLOCKED - {b['reason'][:200]}")
    if s["yt_uploaded"] > s["yt_confirmed"]:
        lines.append("Note: some uploads returned an ID but YouTube could not confirm them "
                     "(the login may lack read permission).")
    if s["duration_report"]:
        lines.append("")
        lines.append("Duration report (seconds): unit | source | final | diff | narration | sync | subtitles | scenes")
        for r in s["duration_report"]:
            lines.append(f"{r['unit'][-12:]} | {r['source']} | {r['final']} | {r['diff']:+} | "
                         f"{r['narration']} | {r['sync']} | {r['subtitles']} | {r['scene_match']}")
    return "\n".join(lines)
