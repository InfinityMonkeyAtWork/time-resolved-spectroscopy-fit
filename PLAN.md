# Active Plan

No active multi-step feature.

Cleared 2026-09-29 after the baseline-independent initialization work
(`seed_source` on `File.fit_2d` / `Project.fit_2d`, by-name seeding at every
entry point through one `File` helper, sweep validation before the output
file opens; v0.19.0). The behaviour changes are in `CHANGELOG.md`; the
provenance amendments in `docs/design/fit_archive_principles.md` and
`docs/design/fit_archive_schema.md`; the deferred coerced-value warning and
the fit-preparation follow-up in `TODO.md` (item 14).
