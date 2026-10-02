# Active Plan

No active multi-step feature.

Cleared 2026-10-01 after the ownership enforcement (TODO step 4, branch
`enforce-ownership`, v0.20.0): owned `File` inputs, corrections, baseline,
windows and noise; atomic model replacement and attachment; detached result
records; the lmfit objects and `update_value` package-internal; the simulator
drawing from the `NoiseModel` it declares; static expression chains settled
the way lmfit does. The behaviour changes are in `CHANGELOG.md`; the decisions,
mechanisms and rejected alternatives in `docs/design/api_ownership_contract.md`
(Mechanisms section, fork 3); the simulator output ownership and the
generation-time parameter specification in `TODO.md` (item 8).
