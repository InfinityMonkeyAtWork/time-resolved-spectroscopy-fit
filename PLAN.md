# Active Plan

No active multi-step feature.

Cleared 2026-09-23 after the noise-model weighting work (`File.set_noise`
with `poisson` / `gaussian` / `unknown`, weighted residuals and Jacobians,
noise in the fit identity and metrics, MCMC and CI following the declared
model, `Simulator.noise_model`, the `sensitivity` module as the independent
check, notebooks 12 and 13; v0.17.0). The design rationale, the rejected
frozen-weights alternative, the verification numbers and the example
teaching split live in `docs/design/archive/noise_model_weighting_plan.md`;
the identity and schema amendments in `docs/design/fit_archive_principles.md`
and `docs/design/fit_archive_schema.md`; follow-ups in `TODO.md`.
