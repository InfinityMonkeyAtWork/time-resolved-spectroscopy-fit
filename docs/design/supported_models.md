# Supported Models

## Supported Today

All combinations below are supported by both the 1D (energy-only) and 2D (energy × time) evaluators, with time-axis features (dynamics, subcycle, convolution) active only in 2D.

- Plain energy models.
- Energy model with dynamics on a top-level energy parameter.
- Energy model with subcycle dynamics on a top-level parameter.
- Convolution kernels inside time-domain Dynamics models, typically for IRF broadening of time-dependent traces.
- Energy model with a profile model on a top-level energy parameter.
- Energy model with a profile on one parameter and dynamics on a different top-level parameter. (Any number of top-level time-dependent and/or profiled parameters on the same energy model are supported as long as any given top-level energy parameter has no more than one directly attached model.)
- Energy model with a profile on a top-level parameter and standard dynamics on one of that profile's internal parameters. (Single-cycle only; no multi-cycle / subcycle models.)
- A single 2D fit may combine several of the features above — e.g. multiple profiled parameters, standard and subcycle dynamics on other parameters, and IRF convolution on the resulting time traces — within one model.

## Expression Support Today

Definitions:
"base parameter": a standard top-level energy-model parameter defined directly in the model, not by an expression. A base parameter may be fit-varying, fixed, time-dependent, or profiled.
"expression parameter": a top-level energy-model parameter defined by an expression that references one or more base parameters.

- Expressions linking a top-level energy parameter to one or more other top-level energy parameters.
- Static fan-out expressions, where multiple top-level energy parameters reference the same top-level base parameter.
- Expressions where a top-level energy expression parameter references a top-level time-dependent parameter.
- Expressions where a top-level energy expression parameter references a top-level profiled parameter.
- Expressions where a top-level energy expression parameter references a top-level profiled parameter whose internal profile parameters are themselves time-dependent via standard dynamics, so the top-level expression inherits the resulting profile variation. (Single-cycle only; no multi-cycle / subcycle models.)

## Explicitly Excluded Today

- Adding dynamics directly to a top-level parameter that already has a profile.
- Adding a profile directly to a top-level parameter that already has dynamics.
- Adding a profile to a parameter inside an attached dynamics model.
- Adding multi-cycle / subcycle dynamics to a parameter inside an attached profile model.
- Adding dynamics to an expression parameter.
- Adding a profile to an expression parameter.
- Adding a convolution kernel as a top-level energy-model component. Convolution is supported only in Dynamics/time models; the main supported use case is IRF broadening.
- Transitive expression chains that pass through time-varying or profile-varying parameters are not supported; users should reference the varying base parameter directly instead of chaining through another expression.

## Lowering Notes

The sections above describe model semantics. The graph intermediate representation (GIR) backend lowers most supported single-file energy and 2D models, but a few cases still use the model/component/parameter (MCP) reference evaluator:

- Standalone ``TIME_1D`` dynamics models are graph-valid and supported by MCP,
  but remain outside the lowered backend scope for now.
- Expression parameters lower only when the expression is arithmetic-only.
  Function calls, attribute access, subscripts, and other non-arithmetic AST
  forms fall back to MCP.
- Project-level fitting is still wired through ``fit_project_mcp`` even when
  the underlying per-file models are lowerable.

## Notes

Static energy-only expression chains are handled by lmfit, so they may continue to work in interpreter-backed workflows. They are not the model shape we use as
the backend-portability contract, because the same chain pattern becomes invalid
or ambiguous once dynamics or profiles enter the dependency path. The recommended
pattern is direct fan-out from base parameters instead of multi-step chains.
