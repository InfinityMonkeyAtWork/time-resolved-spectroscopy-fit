# Changelog Workflow

Shared source of truth for generating or updating `CHANGELOG.md` from git
history before a release.

Update `CHANGELOG.md` with a new entry based on git history since the last
documented release. Use [Keep a Changelog](https://keepachangelog.com) format.

## Arguments

- Optional version string (e.g. `0.7.0`)
- If omitted, read the current version from `pyproject.toml` and use that

## 1. Gather context

Run these commands to understand what changed:

```bash
# Find the last version documented in CHANGELOG.md
# (first line matching "## [<version>]" or "## <version>")
head -100 CHANGELOG.md

# Get current version from pyproject.toml
grep '^version' pyproject.toml

# Find the most recent git tag
git tag --sort=-v:refname | head -5

# Get all commits since the last tag (or since the last documented version)
git log <last_tag>..HEAD --oneline --no-merges

# For richer context, also read the full commit messages
git log <last_tag>..HEAD --no-merges --format="%h %s%n%b"
```

## 2. Categorize changes

Read the commits and diffs to understand what actually changed. Group changes
into these categories and omit empty categories:

- **Added** -> new features, new functions, new API methods
- **Changed** -> behavior changes, API changes, renamed things
- **Fixed** -> bug fixes
- **Removed** -> removed features or deprecated code

## 3. Write the entry

Write human-readable changelog entries. Rules:

- **Write for users, not developers.** "Added `pGauss` profile function for
  Gaussian depth profiles" not "added pGauss to profile.py".
- **Group related commits into single entries.** Five commits that refine one
  feature = one changelog bullet.
- **Skip internal-only changes** like CI tweaks, test refactors, code style
  fixes, or pre-commit config changes unless they affect the user
  (e.g. "Tests now run on Python 3.14").
- **Mention breaking changes prominently.** If an API was renamed or removed,
  say what the old name was and what to use instead.
- **Use backticks** for code identifiers (function names, parameter names, etc).
- Each bullet should be one concise sentence, two at most.

## 4. Update `CHANGELOG.md`

- Read `CHANGELOG.md` to match the existing style and header. The file uses
  [Keep a Changelog](https://keepachangelog.com) format and
  [Semantic Versioning](https://semver.org/).
- Insert the new version entry **after the header and before any existing
  entries**. If there's an `[Unreleased]` section, replace it.
- Format:

```text
## [<version>] - <YYYY-MM-DD>

### Added
- ...

### Changed
- ...
```

- Use today's date for the release date.

## 5. Show the result

Print the new entry so the user can review it before committing.
