# Quick Start

This guide shows the basic workflow to:
1. load an energy model,
2. make one energy parameter time-dependent, and
3. create a 2D (time x energy) model value.

## Basic Usage
```python
from trspecfit import Project, File

# Create a project and file wrapper
project = Project(path='examples/simulator', name='local-test')
file = File(parent_project=project, path='simulated_dataset')

# Load an energy model (this becomes file.model_active)
file.load_model('models_energy.yaml', ['ModelName'])

# See which energy parameters are available in the loaded model
file.describe_model()

# Add time dependence to one parameter in the active energy model.
# par_name must exactly match a parameter name in the loaded energy model.
file.add_time_dependence(
    model_yaml='models_time.yaml',
    model_info=['TimeModelName'],
    par_name='EnergyModelComponent_NN_par',
)

# Check that time-dependent parameters were added correctly
file.describe_model()

# Create and retrieve the full 2D model value (shape: n_time x n_energy)
file.model_active.create_value2D()
value_2d = file.model_active.value2D
```

## Notes
- `add_time_dependence(...)` must be called before `create_value2D()` if you want dynamics in one or more parameters.
- `model_info` in `add_time_dependence(...)` can contain multiple names for multi-cycle dynamics.
- For repeating dynamics, pass `frequency=<Hz>` to `add_time_dependence(...)`.

## Next Steps

See the [Examples](examples/index.rst) section for detailed tutorials.
