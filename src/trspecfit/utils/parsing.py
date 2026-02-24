"""
YAML parsing and mcp utilities for trspecfit.
Model validation, Component naming and numbering, etc.
"""

import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor
from ruamel.yaml.error import YAMLError

from trspecfit.config.functions import (
    all_functions,
    energy_functions,
    get_function_parameters,
    numbering_exceptions,
)


#
#
class ModelValidationError(ValueError):
    """Exception raised for errors in model YAML validation."""



#
def construct_yaml_map(self, node) -> Generator[list[tuple[str, Any]], None, None]:
    """
    Enable multiple components of the same type with automatic numbering.

    This function modifies YAML parsing to allow duplicate keys (multiple components
    of the same type) by automatically numbering them. Components get numbered
    starting from _01: GLP -> GLP_01, GLP_02, etc. Background functions, convolutions,
    and offset functions are exceptions that don't get numbered.

    Parameters
    ----------
    self : SafeConstructor
        YAML constructor instance
    node : yaml.Node
        YAML node being constructed

    Returns
    -------
    list of tuple
        List of (component_name, parameters) tuples with automatic numbering applied
    """

    data: list[tuple[str, Any]] = []
    yield data

    # Get all available function names
    available_functions = all_functions()
    # Get exceptions (functions that don't get numbered)
    exceptions = numbering_exceptions()

    # Track component names to handle duplicates
    component_counts: dict[str, int] = {}

    for key_node, value_node in node.value:
        key = self.construct_object(key_node, deep=True)
        val = self.construct_object(value_node, deep=True)

        # Check if this key is a component name (function name)
        if isinstance(key, str) and key in available_functions:
            # Check if this is an exception (background/offset function)
            if key in exceptions:
                # Don't number exceptions, just use the original name
                data.append((key, val))
            else:
                # This is a regular component, always number it
                if key in component_counts:
                    component_counts[key] += 1
                else:
                    component_counts[key] = 1

                numbered_key = f"{key}_{component_counts[key]:02d}"
                data.append((numbered_key, val))
        else:
            # This is a model name or other key, don't number it
            data.append((key, val))


SafeConstructor.add_constructor("tag:yaml.org,2002:map", construct_yaml_map)
yaml = YAML(typ="safe")


#
def parse_component_name(comp_name: str) -> tuple[str, int | None]:
    """
    Parse a component name into base function name and number.
    Component names follow the pattern: function_name or function_name_NN
    where NN is a two-digit number. This function extracts both parts.

    Parameters
    ----------
    comp_name : str
        Component name to parse (e.g., 'GLP_01', 'expFun_02', 'Offset')

    Returns
    -------
    base_name : str
        Function name without number (e.g., 'GLP', 'expFun', 'Offset')
    number : int or None
        Component number (e.g., 1, 2) or None if unnumbered
    """

    # numbered component
    if "_" in comp_name and comp_name.split("_")[-1].isdigit():
        parts = comp_name.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            base_name = "_".join(parts[:-1])
            number: int | None = int(parts[-1])
        else:
            base_name = comp_name
            number = None
    # unnumbered component (see config/functions.py numbering_exceptions)
    else:
        base_name = comp_name
        number = None

    return base_name, number


#
def validate_model_components(
    model_info_dict: dict[str, dict[str, dict[str, Any]]],
    model_info: list[str],
    model_yaml_path: Path,
) -> None:
    """
    Validate model components and parameters for common errors:
      - Invalid component names (not in available functions)
      - Invalid parameter structure (must be dict)
      - Wrong number of parameters for component type
      - Invalid parameter names for component type
      - Invalid parameter format (must be one of
        [value, vary, min, max], [value, vary], or ['expr'])
      - Invalid 'vary' flag (must be boolean)
      - Invalid bounds (min > max)
      - Values outside bounds

    Parameters
    ----------
    model_info_dict : dict
        Nested dictionary of model definitions:
        {model_name: {component_name: {param_name: param_value}}}
    model_info : list of str
        List of model names to validate
    model_yaml_path : Path
        Path to YAML file being validated (for error messages)

    Raises
    ------
    ModelValidationError
        If any validation check fails, with detailed error message
    """

    available_functions = all_functions()
    example_dir = Path(__file__).parent.parent.parent / "examples"

    # Only validate models that are being loaded
    for model_name in model_info:
        if model_name not in model_info_dict:
            continue

        components = model_info_dict[model_name]

        for comp_name, params in components.items():
            # Extract base component name (remove _01, _02 suffixes)
            base_comp_name, _ = parse_component_name(comp_name)

            # Check 1: Component type exists
            if base_comp_name not in available_functions:
                raise ModelValidationError(
                    f"Unknown component type '{base_comp_name}' in model "
                    f"'{model_name}' in {model_yaml_path}\n"
                    f"Available components: {sorted(available_functions)}\n"
                    f"Check for typos in component name."
                )

            # Check 2: Parameters should be a dictionary
            if not isinstance(params, dict):
                raise ModelValidationError(
                    f"Parameters for '{comp_name}' in model '{model_name}'"
                    f" must be a dictionary.\n"
                    f"Found: {type(params).__name__}\n"
                    f"See 'models_energy.yaml' in example directory: {example_dir}"
                )

            # Get expected parameters for this component type
            expected_params = get_function_parameters(base_comp_name)

            # Check parameter count matches
            if len(params) != len(expected_params):
                raise ModelValidationError(
                    f"Component '{comp_name}' (type: {base_comp_name}) in model "
                    f"'{model_name}' has wrong number of parameters.\n"
                    f"Expected {len(expected_params)} parameters: {expected_params}\n"
                    f"Got {len(params)} parameters: {list(params.keys())}\n"
                    f"Check {model_yaml_path}"
                )

            # Check 3: Validate each parameter
            for param_name, param_value in params.items():
                # Check if parameter name is valid for this component
                if param_name not in expected_params:
                    raise ModelValidationError(
                        f"Invalid parameter '{param_name}' for component "
                        f"'{comp_name}' (type: {base_comp_name}) in model "
                        f"'{model_name}'.\n"
                        f"Expected parameters: {expected_params}\n"
                        f"Check for typos or wrong component type."
                    )

                # Parameter value can be a list [value, vary, min, max] or [value, vary]
                # or a single expression string
                if isinstance(param_value, list):
                    if (len(param_value) == 4) or (len(param_value) == 2):
                        if len(param_value) == 4:  # Standard: [value, vary, min, max]
                            value, vary, min_val, max_val = param_value
                        elif len(param_value) == 2:  # Unbound format: [value, vary]
                            value, vary = param_value
                            min_val = -np.inf
                            max_val = np.inf

                        # Check that 'vary' is boolean
                        if not isinstance(vary, bool):
                            raise ModelValidationError(
                                f"Parameter '{param_name}' in '{comp_name}'"
                                f" (model '{model_name}'):\n"
                                f"'vary' (2nd element) must be True or False.\n"
                                f"Got: {vary} ({type(vary).__name__})"
                            )

                        # Check bounds validity
                        if isinstance(min_val, (int, float)) and isinstance(
                            max_val, (int, float)
                        ):
                            if min_val > max_val:
                                raise ModelValidationError(
                                    f"Parameter '{param_name}' in '{comp_name}'"
                                    f" (model '{model_name}'):\n"
                                    f"min ({min_val}) is greater than max ({max_val})"
                                )

                            # Check if value is within bounds
                            if isinstance(value, (int, float)):
                                if value < min_val or value > max_val:
                                    raise ModelValidationError(
                                        f"Parameter '{param_name}' in"
                                        f" '{comp_name}' (model '{model_name}'):\n"
                                        f"value ({value}) is outside"
                                        f" bounds [{min_val}, {max_val}]"
                                    )

                    elif len(param_value) == 1:
                        # Expression format: ["expression"]
                        if not isinstance(param_value[0], str):
                            raise ModelValidationError(
                                f"Parameter '{param_name}' in '{comp_name}'"
                                f" (model '{model_name}'):\n"
                                f"Single-element list must contain"
                                f" a string expression.\n"
                                f"Got: {param_value[0]}"
                                f" ({type(param_value[0]).__name__})\n"
                                f'Example: ["GLP_01_x0 + 3.6"]'
                            )
                    else:
                        raise ModelValidationError(
                            f"Parameter '{param_name}' in '{comp_name}'"
                            f" (model '{model_name}') has invalid format.\n"
                            f"Expected: [value, vary, min, max] or"
                            f' [value, vary] or ["expr"]\n'
                            f"Got: {param_value} ({len(param_value)} elements)\n"
                            f"See 'models_energy.yaml' in example directory:"
                            f" {example_dir}"
                        )

                else:
                    raise ModelValidationError(
                        f"Parameter '{param_name}' in '{comp_name}'"
                        f" (model '{model_name}') has invalid format.\n"
                        f"Expected either:\n"
                        f"  - [value, vary, min, max] for standard parameters\n"
                        f"  - [value, vary] for unbound parameters\n"
                        f"  - ['expression'] for linked parameters\n"
                        f"Got: {param_value}\n"
                        f"See 'models_energy.yaml' in example directory: {example_dir}"
                    )


#
def load_and_number_yaml_components(
    model_yaml_path: Path,
    model_info: list[str],
    is_dynamics: bool = False,
    debug: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load model YAML file and apply appropriate component numbering strategy.

    For energy models (is_dynamics=False), component numbering is applied by
    construct_yaml_map during parsing.
    For dynamics models (is_dynamics=True), additional conflict resolution is
    performed to ensure unique numbering across subcycles.

    Parameters
    ----------
    model_yaml_path : Path or str
        Full path to model YAML file
    model_info : list of str
        Model names to load from the YAML file
    is_dynamics : bool, default=False
        If True, applies dynamics numbering conflict resolution across subcycles.
        If False, treats as energy model (numbering already complete after parsing).
    debug : bool, default=False
        If True, print detailed information during loading and numbering

    Returns
    -------
    dict
        Nested dictionary of model definitions:
        {model_name: {component_name: {param_name: param_value}}}

    Raises
    ------
    FileNotFoundError
        If YAML file doesn't exist at specified path
    ModelValidationError
        If validation fails (invalid components, parameters, etc.)
    ValueError
        If YAML structure is malformed or contains duplicate model names
    YAMLError
        If YAML syntax is invalid
    """

    model_yaml_path = Path(model_yaml_path)  # Ensure Path object

    try:
        with open(model_yaml_path) as f_yaml:
            # Load YAML file with custom constructor for numbering
            model_info_ALL = yaml.load(f_yaml)

            # Convert YAML structure to dictionary format
            if isinstance(model_info_ALL, list):
                model_info_dict = {}
                for model_entry in model_info_ALL:
                    if not (isinstance(model_entry, tuple) and len(model_entry) == 2):
                        raise ValueError(f"Malformed model entry: {model_entry}")
                    model_name, components = model_entry
                    if model_name in model_info_dict:
                        raise ValueError(f"Duplicate model name found: '{model_name}'")
                    # Convert components to dict format
                    model_info_dict[model_name] = (
                        dict(components) if isinstance(components, list) else components
                    )
                    # Convert parameters to dict format
                    for comp_name, params in model_info_dict[model_name].items():
                        if isinstance(params, list):
                            model_info_dict[model_name][comp_name] = dict(params)
            else:  # should never happen unless construct_yaml_map is broken
                raise ValueError(f"Unexpected YAML structure in {model_yaml_path}")

            if debug:
                print("model_info_ALL:")
                print(model_info_ALL)
                print("model_info_dict:")
                print(model_info_dict)

            # Apply appropriate numbering strategy
            if is_dynamics:
                # Resolve numbering conflicts across subcycles
                model_info_dict = resolve_dynamics_numbering_conflicts(
                    model_info_dict, model_info, debug
                )

            # Validate the loaded model structure
            validate_model_components(model_info_dict, model_info, model_yaml_path)

            # For energy models, numbering is already complete from construct_yaml_map
            return model_info_dict

    except FileNotFoundError:
        raise FileNotFoundError(
            f"FileNotFound: model yaml file input\n"
            f"File should be located in: {model_yaml_path}\n"
            f"Check file name for typos"
        ) from None
    except ModelValidationError:
        # Validator errors are already user-friendly, just pass through
        raise
    except ValueError as e:
        # Structural errors (malformed entries, duplicates)
        if "Malformed model entry" in str(e) or "Duplicate model name" in str(e):
            raise
        # Unexpected YAML parsing error
        raise ValueError(
            f"Unexpected error parsing {model_yaml_path}\n"
            f"Original error: {e}\n\n"
            f"This may be a bug in the YAML parser.\n"
            f"Please report this error at: https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit/issues\n"
            f"Include your YAML file and this error message. Thank you!"
        ) from e
    except YAMLError as exc:
        raise RuntimeError(
            f"YAML syntax error in {model_yaml_path}\n"
            f"Please check for:\n"
            f"  - Proper indentation (use spaces, not tabs)\n"
            f"  - Matching brackets and quotes\n"
            f"  - Valid YAML syntax\n"
            f"Original error: {exc}"
        ) from exc


#
def resolve_dynamics_numbering_conflicts(
    model_info_dict: dict[str, dict[str, dict[str, Any]]],
    model_info: list[str],
    debug: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Resolve numbering conflicts for dynamics models across subcycles.

    For dynamics models with multiple subcycles, components with the same base
    name may appear in different subcycles with the same number from YAML parsing.
    This function tracks used numbers globally and reassigns conflicting numbers
    to the next available number, preserving existing numbering where possible.

    Parameters
    ----------
    model_info_dict : dict
        Nested dictionary of model definitions before conflict resolution
    model_info : list of str
        List of model names to process
    debug : bool, default=False
        If True, print detailed information during conflict resolution

    Returns
    -------
    dict
        Model dictionary with all numbering conflicts resolved
    """

    if debug:
        print("=== STARTING CONFLICT RESOLUTION ===")
        print(f"model_info: {model_info}")
        print("\nmodel_info_dict BEFORE resolution:")
        for submodel, comps in model_info_dict.items():
            if submodel in model_info:
                print(f"  {submodel}: {list(comps.keys())}")

    # Get all available function names and exceptions
    available_functions = all_functions()
    exceptions = numbering_exceptions()

    # Track the next available number for each function type globally
    global_next_available: dict[str, int] = {}
    # Track all used numbers for each function type
    used_numbers: dict[str, set[int]] = {}

    # First pass: collect all existing numbers and find conflicts
    for submodel in model_info:
        if submodel not in model_info_dict:
            continue

        for comp_name in model_info_dict[submodel].keys():
            base_name, number = parse_component_name(comp_name)

            if base_name in available_functions and base_name not in exceptions:
                if number is None:
                    number = 1  # Default numbering

                # Track used numbers
                if base_name not in used_numbers:
                    used_numbers[base_name] = set()
                    global_next_available[base_name] = 1

                used_numbers[base_name].add(number)
                global_next_available[base_name] = max(
                    global_next_available[base_name], number + 1
                )

    if debug:
        print(f"\nAfter first pass - used_numbers: {used_numbers}")
        print(f"global_next_available: {global_next_available}")

    # Second pass: resolve conflicts by reassigning duplicate numbers
    processed_dict: dict[str, dict[str, Any]] = {}
    # Track what we've already assigned in this pass
    assigned_numbers: dict[str, set[int]] = {}

    for submodel in model_info:
        if submodel not in model_info_dict:
            continue

        processed_dict[submodel] = {}

        for comp_name, comp_params in model_info_dict[submodel].items():
            base_name, current_number = parse_component_name(comp_name)

            if base_name in available_functions and base_name not in exceptions:
                if current_number is None:
                    current_number = 1  # Default numbering

                # Initialize tracking for this base name
                if base_name not in assigned_numbers:
                    assigned_numbers[base_name] = set()

                # Check if this number is already assigned in this dynamics model
                if current_number in assigned_numbers[base_name]:
                    # Conflict! Find next available number
                    while (
                        global_next_available[base_name] in assigned_numbers[base_name]
                    ):
                        global_next_available[base_name] += 1
                    new_number = global_next_available[base_name]
                    global_next_available[base_name] += 1

                    if debug:
                        print(
                            f"Conflict resolved: {comp_name} -> "
                            f"{base_name}_{new_number:02d} in {submodel}"
                        )
                else:
                    # No conflict, use current number
                    new_number = current_number

                # Mark this number as assigned
                assigned_numbers[base_name].add(new_number)

                # Create the final component name
                final_name = f"{base_name}_{new_number:02d}"
                processed_dict[submodel][final_name] = comp_params

            else:
                # Not a component function, keep as-is
                processed_dict[submodel][comp_name] = comp_params

        if debug:
            print(f"\nProcessed submodel: {submodel}")
            print(f"  {submodel}: {list(processed_dict[submodel].keys())}")

    if debug:
        print("\nFINAL processed_dict:")
        for submodel in model_info:
            if submodel in processed_dict:
                print(f"  {submodel}: {list(processed_dict[submodel].keys())}")

    return processed_dict


#
def extract_expression_parameters(expr_string: str) -> list[str]:
    """
    Extract parameter names referenced in an expression string.

    Parses expression string to find parameter names by looking for strings that
    start with known function names. Parameter naming follows the pattern
    function_name_NN_paramname (e.g., GLP_01_A, expFun_02_tau).

    Parameters
    ----------
    expr_string : str
        Expression to parse (e.g., "GLP_01_A * 0.75 + GLP_02_x0")

    Returns
    -------
    list of str
        Parameter names found in expression (e.g., ['GLP_01_A', 'GLP_02_x0'])
    """

    # Pattern to match parameter names
    # (letters, numbers, underscores, but not starting with number)
    pattern = r"\b[A-Za-z_][A-Za-z0-9_]*\b"
    matches = re.findall(pattern, expr_string)

    # Filter to keep only strings that start with known function names
    # This catches parameter names like GLP_01_A, GLP_02_x0, etc.
    parameter_refs = []
    for match in matches:
        # $% Does not work for mcp.Dynamics Par referencing another mcp.Dynamics Par!
        for func_name in energy_functions():
            if match.startswith(func_name + "_"):
                parameter_refs.append(match)
                break  # Found a match, no need to check other function names

    return parameter_refs
