"""
Test plotting functions and configuration system.

Strategy:
- Test without displaying or saving(save_img=-2)
- Verify no crashes, figure object creation, file creation
- Check config propagation through hierarchy
- Test edge cases and common user patterns
- Do not verify visual correctness (that requires manual inspection)
"""

# add test for new feature:
# def test_my_new_feature(self, sample_1D_data, default_config):
#     """Test description."""
#     x, y_list = sample_1D_data
#     plot_1d(
#         y_list, x=x, config=default_config,
#         my_new_param=value,
#         save_img=-2 # don't display, don't save
#     )
#     plt.close('all')

import tempfile
from pathlib import Path

# Run without opening windows for plots
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from trspecfit import File, Project
from trspecfit.config.plot import PlotConfig

# Local imports
from trspecfit.utils.plot import plot_1d, plot_2d


#
#
class TestPlotConfig:
    """Test PlotConfig creation and manipulation"""

    #
    def test_default_creation(self):
        """Test creating config with defaults"""

        config = PlotConfig()
        assert config.x_label == "x axis"
        assert config.dpi_plot == 100
        assert config.z_colormap == "viridis"

    #
    def test_custom_creation(self):
        """Test creating config with custom values"""

        config = PlotConfig(x_label="Custom X", dpi_plot=200, x_dir="rev")
        assert config.x_label == "Custom X"
        assert config.dpi_plot == 200
        assert config.x_dir == "rev"

    #
    def test_from_project(self):
        """Test creating config from Project"""

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(path=tmpdir, name="test")
            config = PlotConfig.from_project(project)
            assert config.x_label == project.e_label
            assert config.y_label == project.t_label
            assert config.dpi_plot == project.dpi_plt
            assert config.z_type == project.z_type

    #
    def test_from_project_with_overrides(self):
        """Test creating config from Project with overrides"""

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(path=tmpdir, name="test")
            config = PlotConfig.from_project(
                project, x_label="Overridden", dpi_plot=250
            )
            assert config.x_label == "Overridden"
            assert config.dpi_plot == 250
            # Other values should still come from project
            assert config.y_label == project.t_label

    #
    def test_update(self):
        """Test updating config attributes"""

        config = PlotConfig()
        config.update(x_label="Updated", dpi_plot=175)
        assert config.x_label == "Updated"
        assert config.dpi_plot == 175

    #
    def test_update_invalid_attribute(self):
        """Test that updating invalid attribute raises error"""

        config = PlotConfig()
        with pytest.raises(AttributeError):
            config.update(invalid_attr="value")

    #
    def test_copy(self):
        """Test copying config"""

        config = PlotConfig(x_label="Energy (eV)", y_label="Time (ps)")
        new_config = config.copy(x_label="Modified")
        assert new_config.x_label == "Modified"
        assert new_config.y_label == config.y_label  # Unchanged
        assert config.x_label == "Energy (eV)"  # Original unchanged


#
#
class TestPlot1D:
    """Test 1D plotting function"""

    #
    def test_basic_plot(self):
        """Test basic 1D plot creation"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        plot_1d(y_list, x=x, config=config, save_img=0)
        ax = plt.gca()
        assert len(ax.get_lines()) >= 2
        assert ax.get_xlabel() == "x axis"  # default label
        plt.close("all")

    #
    def test_plot_with_custom_config(self):
        """Test 1D plot with custom config"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig(x_label="Energy (eV)", x_dir="rev", dpi_plot=150)

        plot_1d(y_list, x=x, config=config, save_img=0)
        ax = plt.gca()
        assert ax.get_xlabel() == "Energy (eV)"
        assert ax.xaxis_inverted()
        plt.close("all")

    #
    def test_plot_override_config(self):
        """Test overriding config parameters in plot call"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        plot_1d(
            y_list,
            x=x,
            config=config,
            x_label="Override Label",
            x_dir="rev",
            save_img=0,
        )
        ax = plt.gca()
        assert ax.get_xlabel() == "Override Label"
        assert ax.xaxis_inverted()
        plt.close("all")

    #
    def test_plot_no_x_axis(self):
        """Test plotting without explicit x-axis"""

        y_list = [np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100))]
        config = PlotConfig()

        plot_1d(y_list, config=config, save_img=-2)
        plt.close("all")

    #
    def test_plot_single_trace(self):
        """Test plotting single trace"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x)]
        config = PlotConfig()

        plot_1d(y_list, x=x, config=config, save_img=0)
        ax = plt.gca()
        assert len(ax.get_lines()) >= 1
        plt.close("all")

    #
    def test_plot_with_limits(self):
        """Test plot with axis limits"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        plot_1d(
            y_list, x=x, config=config, x_lim=(2, 8), y_lim=(-1.5, 1.5), save_img=-2
        )
        plt.close("all")

    #
    def test_plot_reversed_axis(self):
        """Test plot with reversed x-axis"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        plot_1d(y_list, x=x, config=config, x_dir="rev", save_img=0)
        ax = plt.gca()
        assert ax.xaxis_inverted()
        plt.close("all")

    #
    def test_plot_log_scale(self):
        """Test plot with log scale"""

        x = np.linspace(0, 10, 100)
        # Use positive data for log scale
        y_list = [np.abs(np.sin(x)) + 0.1, np.abs(np.cos(x)) + 0.1]
        config = PlotConfig()

        plot_1d(y_list, x=x, config=config, y_type="log", save_img=0)
        ax = plt.gca()
        assert ax.get_yscale() == "log"
        plt.close("all")

    #
    def test_plot_with_vlines(self):
        """Test plot with vertical lines"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        plot_1d(y_list, x=x, config=config, vlines=[3, 7], save_img=0)
        ax = plt.gca()
        assert len(ax.get_lines()) >= 2  # data lines
        # vlines rendered as LineCollection
        from matplotlib.collections import LineCollection

        vline_collections = [
            c for c in ax.get_children() if isinstance(c, LineCollection)
        ]
        assert len(vline_collections) >= 1
        plt.close("all")

    #
    def test_plot_waterfall(self):
        """Test waterfall plot"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        plot_1d(y_list, x=x, config=config, waterfall=0.5, save_img=-2)
        plt.close("all")

    #
    def test_plot_save(self):
        """Test saving plot"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            plot_1d(y_list, x=x, config=config, save_img=-1, save_path=str(save_path))
            assert save_path.exists()
        plt.close("all")

    #
    def test_plot_custom_styling(self):
        """Test plot with custom styling"""

        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()

        plot_1d(
            y_list,
            x=x,
            config=config,
            colors=["red", "blue"],
            linestyles=["-", "--"],
            linewidths=[2, 1],
            legend=["Trace 1", "Trace 2"],
            save_img=-2,
        )
        plt.close("all")


#
#
class TestPlot2D:
    """Test 2D plotting function"""

    #
    def test_basic_plot(self):
        """Test basic 2D plot creation"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(data, x=x, y=y, config=config, save_img=0)
        fig = plt.gcf()
        assert len(fig.axes) >= 1  # at least one axes (data panel)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "x axis"  # default label
        plt.close("all")

    #
    def test_plot_with_custom_config(self):
        """Test 2D plot with custom config"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig(z_colormap="plasma", x_dir="rev")

        plot_2d(data, x=x, y=y, config=config, save_img=0)
        ax = plt.gcf().axes[0]
        assert ax.xaxis_inverted()
        plt.close("all")

    #
    def test_plot_override_config(self):
        """Test overriding config parameters"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(
            data, x=x, y=y, config=config, z_colormap="plasma", x_dir="rev", save_img=0
        )
        ax = plt.gcf().axes[0]
        assert ax.xaxis_inverted()
        plt.close("all")

    #
    def test_plot_z_type_log(self):
        """Test 2D plot with logarithmic color scale"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.abs(np.outer(y + 1, np.sin(x) + 2))
        config = PlotConfig(z_type="log")

        plot_2d(data, x=x, y=y, config=config, save_img=-2)
        plt.close("all")

    #
    def test_plot_no_axes(self):
        """Test plotting without explicit axes"""

        data = np.random.default_rng().standard_normal((30, 50))
        config = PlotConfig()

        plot_2d(data, config=config, save_img=-2)
        plt.close("all")

    #
    def test_plot_with_limits(self):
        """Test plot with axis limits"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(data, x=x, y=y, config=config, x_lim=(2, 8), y_lim=(1, 4), save_img=-2)
        plt.close("all")

    #
    def test_plot_with_z_limits(self):
        """Test plot with color scale limits"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(data, x=x, y=y, config=config, z_lim=(-2, 2), save_img=-2)
        plt.close("all")

    #
    def test_plot_data_slice(self):
        """Test plotting with data slice"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(
            data, x=x, y=y, config=config, data_slice=[[10, 40], [5, 25]], save_img=-2
        )
        plt.close("all")

    #
    def test_plot_with_lines(self):
        """Test plot with vertical and horizontal lines"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(
            data, x=x, y=y, config=config, vlines=[3, 7], hlines=[1, 4], save_img=-2
        )
        plt.close("all")

    #
    def test_plot_reversed_axes(self):
        """Test plot with reversed axes"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(data, x=x, y=y, config=config, x_dir="rev", y_dir="rev", save_img=0)
        ax = plt.gcf().axes[0]
        assert ax.xaxis_inverted()
        assert ax.yaxis_inverted()
        plt.close("all")

    #
    def test_plot_colorbar_horizontal(self):
        """Test plot with horizontal colorbar"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        plot_2d(data, x=x, y=y, config=config, z_colorbar="hor", save_img=-2)
        plt.close("all")

    #
    def test_plot_save(self):
        """Test saving 2D plot"""

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        rng = np.random.default_rng()
        data = rng.standard_normal((30, 50)) + np.outer(y, np.sin(x))
        config = PlotConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_2d_plot.png"
            plot_2d(
                data, x=x, y=y, config=config, save_img=-1, save_path=str(save_path)
            )
            assert save_path.exists()
        plt.close("all")


#
#
class TestPlotConfigHierarchy:
    """Test config propagation through Project -> File -> Model hierarchy"""

    #
    def test_file_inherits_from_project(self):
        """Test that File inherits plot config from Project"""

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project with custom settings
            project = Project(path=tmpdir, name="test")
            project.e_label = "Binding Energy (eV)"
            project.t_label = "Delay (ps)"
            project.x_dir = "rev"

            # Create file
            x = np.linspace(0, 10, 50)
            y = np.linspace(0, 5, 30)
            data = np.random.default_rng().standard_normal((30, 50))
            file = File(parent_project=project, data=data, energy=x, time=y)

            # Check that file inherits project settings
            config = file.plot_config
            assert config.x_label == "Binding Energy (eV)"
            assert config.y_label == "Delay (ps)"
            assert config.x_dir == "rev"

    #
    def test_file_can_customize_config(self):
        """Test that File can customize its config persistently"""

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(path=tmpdir, name="test")
            x = np.linspace(0, 10, 50)
            data = np.random.default_rng().standard_normal(50)
            file = File(parent_project=project, data=data, energy=x)

            # Customize file's config
            file.plot_config.update(x_label="Custom Energy", dpi_plot=200)

            # Verify customization persists
            assert file.plot_config.x_label == "Custom Energy"
            assert file.plot_config.dpi_plot == 200

            # Verify project unchanged
            assert project.e_label != "Custom Energy"


#
#
class TestPlotConfigFromYAML:
    """Regression: project.yaml values must propagate through the full hierarchy."""

    YAML_CONTENT = (
        "e_label: 'Binding energy (eV)'\n"
        "t_label: 'Delay (ps)'\n"
        "z_label: 'Counts'\n"
        "x_dir: 'rev'\n"
        "z_colormap: 'RdBu'\n"
    )

    EXPECTED = {
        "x_label": "Binding energy (eV)",
        "y_label": "Delay (ps)",
        "z_label": "Counts",
        "x_dir": "rev",
        "z_colormap": "RdBu",
    }

    #
    def _make_project_dir(self, tmpdir):
        """Write project.yaml and a minimal model YAML into *tmpdir*."""

        (Path(tmpdir) / "project.yaml").write_text(self.YAML_CONTENT)
        (Path(tmpdir) / "models").mkdir(exist_ok=True)

        # Copy test model YAML so load_model can find it
        import shutil

        src = Path("tests") / "models/file_energy.yaml"
        shutil.copy(src, Path(tmpdir) / "models/file_energy.yaml")

    #
    def test_project_loads_yaml_values(self):
        """Project attributes reflect non-default YAML values."""

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_project_dir(tmpdir)
            project = Project(path=tmpdir, name="test")

            assert project.e_label == "Binding energy (eV)"
            assert project.t_label == "Delay (ps)"
            assert project.z_label == "Counts"
            assert project.x_dir == "rev"
            assert project.z_colormap == "RdBu"

    #
    def test_file_plot_config_inherits_yaml(self):
        """File.plot_config must carry every non-default YAML value."""

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_project_dir(tmpdir)
            project = Project(path=tmpdir, name="test")
            file = File(
                parent_project=project,
                data=np.random.default_rng(0).standard_normal((30, 50)),
                energy=np.linspace(80, 90, 50),
                time=np.linspace(0, 100, 30),
            )

            config = file.plot_config
            for attr, expected in self.EXPECTED.items():
                assert getattr(config, attr) == expected, (
                    f"File.plot_config.{attr}: "
                    f"expected {expected!r}, got {getattr(config, attr)!r}"
                )

    #
    def test_model_plot_config_inherits_yaml(self):
        """Model.plot_config must match File.plot_config."""

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_project_dir(tmpdir)
            project = Project(path=tmpdir, name="test")
            file = File(
                parent_project=project,
                data=np.random.default_rng(0).standard_normal((30, 201)),
                energy=np.linspace(80, 90, 201),
                time=np.linspace(0, 100, 30),
            )
            file.load_model(
                model_yaml="models/file_energy.yaml",
                model_info="single_glp",
            )
            model = file.model_active
            assert model is not None  # type guard

            for attr, expected in self.EXPECTED.items():
                assert getattr(model.plot_config, attr) == expected, (
                    f"Model.plot_config.{attr}: "
                    f"expected {expected!r}, got {getattr(model.plot_config, attr)!r}"
                )

    #
    def test_component_plot_sees_yaml_config(self):
        """Component.plot() must use config values from project.yaml."""

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_project_dir(tmpdir)
            project = Project(path=tmpdir, name="test")
            file = File(
                parent_project=project,
                data=np.random.default_rng(0).standard_normal((30, 201)),
                energy=np.linspace(80, 90, 201),
                time=np.linspace(0, 100, 30),
            )
            file.load_model(
                model_yaml="models/file_energy.yaml",
                model_info="single_glp",
            )
            model = file.model_active
            assert model is not None  # type guard
            component = model.components[0]

            component.plot()
            ax = plt.gca()
            assert ax.get_xlabel() == "Binding energy (eV)"
            assert ax.xaxis_inverted(), "x_dir='rev' from YAML not applied"
            plt.close("all")


#
#
class TestPlotConfigPropagation:
    """Test that PlotConfig propagates to mcp.py and simulator."""

    #
    def _make_file_with_model(self, *, x_dir="rev"):
        """Create a project/file/model with a reversed energy axis."""

        project = Project(path="tests")
        project.x_dir = x_dir
        project.e_label = "Binding Energy (eV)"

        file = File(parent_project=project)
        file.energy = np.linspace(80, 90, 201)
        file.time = np.linspace(-10, 100, 111)
        file.data = np.random.default_rng(42).normal(
            size=(len(file.time), len(file.energy))
        )
        file.dim = 2

        file.load_model(
            model_yaml="models/file_energy.yaml",
            model_info="single_glp",
        )
        return file

    #
    def test_component_plot_respects_x_dir(self):
        """Component.plot() should reverse x-axis when config.x_dir='rev'."""

        file = self._make_file_with_model(x_dir="rev")
        assert file.model_active is not None  # type guard
        component = file.model_active.components[0]

        component.plot()
        ax = plt.gca()
        assert ax.xaxis_inverted(), "x-axis should be inverted for x_dir='rev'"
        plt.close("all")

    #
    def test_component_plot_uses_project_label(self):
        """Component.plot() should use x_label from PlotConfig."""

        file = self._make_file_with_model()
        assert file.model_active is not None  # type guard
        component = file.model_active.components[0]

        component.plot()
        ax = plt.gca()
        assert ax.get_xlabel() == "Binding Energy (eV)"
        plt.close("all")

    #
    def test_component_plot_default_dir(self):
        """Component.plot() should not invert x-axis when config.x_dir='def'."""

        file = self._make_file_with_model(x_dir="def")
        assert file.model_active is not None  # type guard
        component = file.model_active.components[0]

        component.plot()
        ax = plt.gca()
        assert not ax.xaxis_inverted(), "x-axis should not be inverted for x_dir='def'"
        plt.close("all")

    #
    def test_simulator_2d_respects_x_dir(self):
        """Simulator.plot_comparison(dim=2) should reverse x-axis when x_dir='rev'."""

        from trspecfit import Simulator

        file = self._make_file_with_model(x_dir="rev")
        model = file.model_active
        assert model is not None  # type guard
        sim = Simulator(model, noise_level=0.05)
        sim.simulate_2d()

        sim.plot_comparison(dim=2)
        fig = plt.gcf()
        for ax in fig.axes:
            # Skip colorbar axes (they don't have energy on x-axis)
            if ax.get_xlabel() == "Binding Energy (eV)":
                assert ax.xaxis_inverted(), "x-axis should be inverted in 2D sim plot"
        plt.close("all")

    #
    def test_simulator_2d_respects_axis_labels(self):
        """Simulator.plot_comparison(dim=2) should use labels from PlotConfig."""

        from trspecfit import Simulator

        file = self._make_file_with_model()
        model = file.model_active
        assert model is not None  # type guard
        sim = Simulator(model, noise_level=0.05)
        sim.simulate_2d()

        sim.plot_comparison(dim=2)
        fig = plt.gcf()
        # Find a main panel axis (not colorbar)
        main_axes = [ax for ax in fig.axes if ax.get_xlabel() != ""]
        assert any(ax.get_xlabel() == "Binding Energy (eV)" for ax in main_axes)
        plt.close("all")


#
#
class TestEdgeCases:
    """Test edge cases and potential regressions"""

    #
    def test_single_point(self):
        """Test plotting single data point"""

        config = PlotConfig()
        plot_1d([[1]], x=[0], config=config, save_img=-2)
        plt.close("all")

    #
    def test_nan_in_data(self):
        """Test plotting with NaN values"""

        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        y[10] = np.nan
        config = PlotConfig()

        plot_1d([y], x=x, config=config, save_img=-2)
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
