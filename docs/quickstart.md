# Quick Start

This guide shows three common patterns:
1. basic energy-model loading,
2. adding dynamics, and
3. adding profiles (and recommended profile+dynamics composition).

## Basic
```python
from trspecfit import Project, File

# Create a project and file wrapper
project = Project(path='my_project', name='my_experiment')
file = File(parent_project=project, path='my_dataset')

# Load an energy model (this becomes file.model_active)
# models_energy.yaml can define generic components like:
#   some_background_function, some_peak_function
file.load_model('models_energy.yaml', ['some_energy_model'])

# See which energy parameters are available in the loaded model
file.describe_model()
```

## Dynamics
```python
# Fix well-known energy parameters via a baseline fit
file.define_baseline(t_ind=0)
file.fit_baseline()

# Add time dependence to one parameter in the active energy model
# target_parameter must match a loaded parameter name exactly
file.add_time_dependence(
    target_model='some_energy_model',
    target_parameter='some_base_parameter',
    dynamics_yaml='models_time.yaml',
    dynamics_model=['some_dynamics_model'],
)

# Inspect updated parameters
file.describe_model()

# Global 2D fit
file.fit_2d()
```

## Profile
```python
import numpy as np
from trspecfit import Project, File

project = Project(path='my_project', name='my_experiment')
file = File(
    parent_project=project,
    path='my_dataset',
    aux_axis=np.linspace(0, 10, 21),  # required for profile models
)

file.load_model('models_energy.yaml', ['some_energy_model'])

# Attach profile to a base parameter
file.add_par_profile(
    target_model='some_energy_model',
    target_parameter='some_base_parameter',
    profile_yaml='models_profile.yaml',
    profile_model=['some_profile_model'],
)

# Add dynamics to a base parameter (most common use case)
file.add_time_dependence(
    target_model='some_energy_model',
    target_parameter='some_base_parameter',
    dynamics_yaml='models_time.yaml',
    dynamics_model=['some_dynamics_model'],
)

# Optionally: attach dynamics to a profile parameter (series composition)
file.add_time_dependence(
    target_model='some_energy_model',
    target_parameter='some_profile_parameter',
    dynamics_yaml='models_time.yaml',
    dynamics_model=['another_dynamics_model'],
)
```

## Supported and Disallowed Paths
- Supported: `base parameter -> profile`, then `profile parameter -> dynamics`.
- Disallowed: attaching both profile and dynamics directly to the same base parameter (disabled to avoid strongly correlated fits).

## Notes
- `add_time_dependence(...)` must be called before `create_value_2d()` if you want dynamics in one or more parameters.
- `dynamics_model` in `add_time_dependence(...)` can contain multiple names for multi-cycle dynamics.
- For repeating dynamics, pass `frequency=<Hz>` to `add_time_dependence(...)`.

## Next Steps

See [Examples](examples/index.rst) and [API Reference](api/index.rst) for full workflows and parameter naming details.
