# Model Design Rules

## Supported Today

- Plain energy models.
- Energy model with dynamics on a top-level energy parameter.
- Energy model with subcycle dynamics on a top-level parameter.
- Energy model with a profile model on a top-level energy parameter.
- Energy model with a profile on one parameter and dynamics on a different top-level parameter. (Any number of top-level time-dependent and/or profiled parameters on the same energy model are supported as long as any given top-level energy parameter has no more than one directly attached model.)
- Energy model with a profile on a top-level parameter and standard dynamics on one of that profile's internal parameters. (Single-cycle only; no multi-cycle / subcycle models.)

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
- Transitive expression chains that pass through time-varying or profile-varying parameters are not supported; users should reference the varying base parameter directly instead of chaining through another expression.

## Notes/ Future Changes

We may choose to disallow all transitive expression chains in the future. Static transitive expression chains in energy models are currently allowed. However there is a user experience issue: a chain that works in the static case can become invalid once dynamics or a profile is added. This is surprising and hard to document/ communicate clearly.
If this change is implemented the recommended pattern would be direct fan-out expressions from the base parameter rather than multi-step chains.
