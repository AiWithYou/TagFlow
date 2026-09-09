# Changelog

## 2.0.0 — 2026-09-10

### Data safety

- Rebuilt image-family transfers around staged copies. Move sources are deleted
  only after every image/caption/metadata file has been copied and published.
- Added no-overwrite publication and case-insensitive stem reservations for
  concurrent TagFlow transfers. Windows uses same-filesystem no-clobber rename;
  POSIX uses hard links with exclusive-copy fallback.
- Failed source deletion restores original bytes. Failed rollback retains the
  destination family and a recovery manifest instead of deleting the last copy.
- Fixed dotted-name ownership: `photo.en.txt` belongs to `photo.en.png`, not
  `photo.png`, when both images exist. Transfer and audit share the same rule.
- Fixed Unicode casefold suffix slicing and reject ambiguous case-only sidecars
  and symbolic-link sources rather than silently transferring the wrong data.
- Collision renames now update known filename references in `.tagflow.json`.
  Arbitrary model/prompt fields and the source metadata are not rewritten.
- A move into the source directory is now a no-op, not an unexpected rename.

### Dataset audit

- Replaced per-sidecar stem sorting/full comparisons with reusable directory
  indexes and longest-stem lookups bounded by filename dot boundaries.
- Only equal-size image candidates are SHA-256 hashed. Reports include
  `hashed_image_count`; equal-size but different-content files remain distinct.
- Made base-caption matching case-insensitive on every supported platform.
- Report malformed metadata, ambiguous case-only captions, and directory scan
  failures. Skip special files such as named pipes.
- Added a programmatic `should_stop` callback, checked between files and hash
  chunks. Cancellation raises `AuditCancelled`, never a partial clean report.

### Maintenance

- Added a single release-version source and `python tagflow_entry.py --version`.
- Added failure-injection, concurrent-transfer, Unicode, read-only, audit and
  launcher regression tests. Retained the existing tests, adapting the old move
  rollback fault injection to the new copy-then-delete implementation.
- Expanded CI to Windows/Linux with Python 3.11/3.13 and portable compilation.
- AI providers, presets, authentication, billing paths and saved settings remain
  unchanged. No new runtime dependencies were added.

### Compatibility and limits

Transfers now require temporary destination space and can be slower than an
in-place rename on the same drive. Symbolic-link families and ambiguous
case-only sidecars are intentionally rejected. This is exception-safe recovery,
not an atomic multi-file transaction or a power-loss guarantee. Do not edit
source/destination files from another application during a transfer.

See [v2 data safety and verification](docs/v2_data_safety_ja.md).
