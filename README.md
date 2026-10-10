# Anime Recap Automation

## Reliability fixes in this build

- Google Drive links are parsed from `/file/d/ID`, `/drive/folders/ID`, and query-style `?id=ID` URLs. Bare IDs remain accepted.
- Video segment planning samples source footage in chronological order across the source timeline.
- The renderer refuses to repeat footage when narration is longer than the source; it raises a clear `MediaError`.
- The legacy renderer no longer uses FFmpeg `-stream_loop -1`, and refuses mismatched source/narration duration.
- The editor fallback still tries subtitle rendering, then rendering without subtitles, then the safe non-looping renderer.

## Run tests

Use Python 3.10+:

```bash
python -m unittest discover -s tests -v
```

The unit tests use mocks for external services. Media integration tests require FFmpeg and FFprobe installed and available on `PATH`.

## Important operational note

If a narration is longer than its source footage, this build intentionally fails instead of repeating scenes to hide the mismatch. Shorten/regenerate the narration or provide enough source footage, then retry. Review the generated output before publishing.

Keep API keys, OAuth credentials, and tokens in GitHub Secrets or environment variables; never commit them to the repository.
