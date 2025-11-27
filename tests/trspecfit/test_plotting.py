"""
Test plotting functions and configuration system.

Strategy:
- Test without displaying (plt.close() or save_img=-1)
- Verify no crashes, figure object creation, file creation
- Check config propagation through hierarchy
- Test edge cases and common user patterns
- Do not verify visual correctness (that requires manual inspection)
"""

# add test for new feature:
# def test_my_new_feature(self, sample_1D_data, default_config):
#     """Test description."""
#     x, y_list = sample_1D_data
#     plot_1D(
#         y_list, x=x, config=default_config,
#         my_new_param=value,
#         save_img=-1
#     )
#     plt.close('all')

import pytest
import numpy as np
from pathlib import Path
import tempfile
# Run without opening windows for plots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
# Local imports
from trspecfit.utils.plot import plot_1D, plot_2D
from trspecfit.config.plot import PlotConfig
from trspecfit import Project, File


class TestPlotConfig:
    """Test PlotConfig creation and manipulation"""
    
    def test_default_creation(self):
        """Test creating config with defaults"""
        config = PlotConfig()
        assert config.x_label == 'x axis'
        assert config.dpi_plot == 100
        assert config.z_colormap == 'viridis'
    
    def test_custom_creation(self):
        """Test creating config with custom values"""
        config = PlotConfig(
            x_label='Custom X',
            dpi_plot=200,
            x_dir='rev'
        )
        assert config.x_label == 'Custom X'
        assert config.dpi_plot == 200
        assert config.x_dir == 'rev'
    
    def test_from_project(self):
        """Test creating config from Project"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(path=tmpdir, name='test')
            config = PlotConfig.from_project(project)
            assert config.x_label == project.e_label
            assert config.y_label == project.t_label
            assert config.dpi_plot == project.dpi_plt
    
    def test_from_project_with_overrides(self):
        """Test creating config from Project with overrides"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(path=tmpdir, name='test')
            config = PlotConfig.from_project(
                project,
                x_label='Overridden',
                dpi_plot=250
            )
            assert config.x_label == 'Overridden'
            assert config.dpi_plot == 250
            # Other values should still come from project
            assert config.y_label == project.t_label
    
    def test_update(self):
        """Test updating config attributes"""
        config = PlotConfig()
        config.update(x_label='Updated', dpi_plot=175)
        assert config.x_label == 'Updated'
        assert config.dpi_plot == 175
    
    def test_update_invalid_attribute(self):
        """Test that updating invalid attribute raises error"""
        config = PlotConfig()
        with pytest.raises(AttributeError):
            config.update(invalid_attr='value')
    
    def test_copy(self):
        """Test copying config"""
        config = PlotConfig(x_label='Energy (eV)', y_label='Time (ps)')
        new_config = config.copy(x_label='Modified')
        assert new_config.x_label == 'Modified'
        assert new_config.y_label == config.y_label  # Unchanged
        assert config.x_label == 'Energy (eV)'  # Original unchanged


class TestPlot1D:
    """Test 1D plotting function"""
    
    def test_basic_plot(self):
        """Test basic 1D plot creation"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        plot_1D(y_list, x=x, config=config, save_img=-1)
        plt.close('all')
    
    def test_plot_with_custom_config(self):
        """Test 1D plot with custom config"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig(x_label='Energy (eV)', x_dir='rev', dpi_plot=150)
        
        plot_1D(y_list, x=x, config=config, save_img=-1)
        plt.close('all')
    
    def test_plot_override_config(self):
        """Test overriding config parameters in plot call"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        plot_1D(
            y_list, x=x, config=config,
            x_label='Override Label',
            x_dir='rev',
            save_img=-1
        )
        plt.close('all')
    
    def test_plot_no_x_axis(self):
        """Test plotting without explicit x-axis"""
        y_list = [np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100))]
        config = PlotConfig()
        
        plot_1D(y_list, config=config, save_img=-1)
        plt.close('all')
    
    def test_plot_single_trace(self):
        """Test plotting single trace"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x)]
        config = PlotConfig()
        
        plot_1D(y_list, x=x, config=config, save_img=-1)
        plt.close('all')
    
    def test_plot_with_limits(self):
        """Test plot with axis limits"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        plot_1D(
            y_list, x=x, config=config,
            x_lim=(2, 8),
            y_lim=(-1.5, 1.5),
            save_img=-1
        )
        plt.close('all')
    
    def test_plot_reversed_axis(self):
        """Test plot with reversed x-axis"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        plot_1D(y_list, x=x, config=config, x_dir='rev', save_img=-1)
        plt.close('all')
    
    def test_plot_log_scale(self):
        """Test plot with log scale"""
        x = np.linspace(0, 10, 100)
        # Use positive data for log scale
        y_list = [np.abs(np.sin(x)) + 0.1, np.abs(np.cos(x)) + 0.1]
        config = PlotConfig()
        
        plot_1D(y_list, x=x, config=config, y_type='log', save_img=-1)
        plt.close('all')
    
    def test_plot_with_vlines(self):
        """Test plot with vertical lines"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        plot_1D(y_list, x=x, config=config, vlines=[3, 7], save_img=-1)
        plt.close('all')
    
    def test_plot_waterfall(self):
        """Test waterfall plot"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        plot_1D(y_list, x=x, config=config, waterfall=0.5, save_img=-1)
        plt.close('all')
    
    def test_plot_save(self):
        """Test saving plot"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_plot.png'
            plot_1D(y_list, x=x, config=config, save_img=-1, save_path=str(save_path))
            assert save_path.exists()
        plt.close('all')
    
    def test_plot_custom_styling(self):
        """Test plot with custom styling"""
        x = np.linspace(0, 10, 100)
        y_list = [np.sin(x), np.cos(x)]
        config = PlotConfig()
        
        plot_1D(
            y_list, x=x, config=config,
            colors=['red', 'blue'],
            linestyles=['-', '--'],
            linewidths=[2, 1],
            legend=['Trace 1', 'Trace 2'],
            save_img=-1
        )
        plt.close('all')


class TestPlot2D:
    """Test 2D plotting function"""
    
    def test_basic_plot(self):
        """Test basic 2D plot creation"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(data, x=x, y=y, config=config, save_img=-1)
        plt.close('all')
    
    def test_plot_with_custom_config(self):
        """Test 2D plot with custom config"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig(z_colormap='plasma', x_dir='rev')
        
        plot_2D(data, x=x, y=y, config=config, save_img=-1)
        plt.close('all')
    
    def test_plot_override_config(self):
        """Test overriding config parameters"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(
            data, x=x, y=y, config=config,
            z_colormap='plasma',
            x_dir='rev',
            save_img=-1
        )
        plt.close('all')
    
    def test_plot_no_axes(self):
        """Test plotting without explicit axes"""
        data = np.random.randn(30, 50)
        config = PlotConfig()
        
        plot_2D(data, config=config, save_img=-1)
        plt.close('all')
    
    def test_plot_with_limits(self):
        """Test plot with axis limits"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(
            data, x=x, y=y, config=config,
            x_lim=(2, 8),
            y_lim=(1, 4),
            save_img=-1
        )
        plt.close('all')
    
    def test_plot_with_z_limits(self):
        """Test plot with color scale limits"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(data, x=x, y=y, config=config, z_lim=(-2, 2), save_img=-1)
        plt.close('all')
    
    def test_plot_data_slice(self):
        """Test plotting with data slice"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(
            data, x=x, y=y, config=config,
            data_slice=[[10, 40], [5, 25]],
            save_img=-1
        )
        plt.close('all')
    
    def test_plot_with_lines(self):
        """Test plot with vertical and horizontal lines"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(
            data, x=x, y=y, config=config,
            vlines=[3, 7],
            hlines=[1, 4],
            save_img=-1
        )
        plt.close('all')
    
    def test_plot_reversed_axes(self):
        """Test plot with reversed axes"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(
            data, x=x, y=y, config=config,
            x_dir='rev',
            y_dir='rev',
            save_img=-1
        )
        plt.close('all')
    
    def test_plot_colorbar_horizontal(self):
        """Test plot with horizontal colorbar"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        plot_2D(data, x=x, y=y, config=config, z_colorbar='hor', save_img=-1)
        plt.close('all')
    
    def test_plot_save(self):
        """Test saving 2D plot"""
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 5, 30)
        data = np.random.randn(30, 50) + np.outer(y, np.sin(x))
        config = PlotConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_2d_plot.png'
            plot_2D(data, x=x, y=y, config=config, save_img=-1, save_path=str(save_path))
            assert save_path.exists()
        plt.close('all')


class TestPlotConfigHierarchy:
    """Test config propagation through Project -> File -> Model hierarchy"""
    
    def test_file_inherits_from_project(self):
        """Test that File inherits plot config from Project"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project with custom settings
            project = Project(path=tmpdir, name='test')
            project.e_label = 'Binding Energy (eV)'
            project.t_label = 'Delay (ps)'
            project.x_dir = 'rev'
            
            # Create file
            x = np.linspace(0, 10, 50)
            y = np.linspace(0, 5, 30)
            data = np.random.randn(30, 50)
            file = File(parent_project=project, data=data, energy=x, time=y)
            
            # Check that file inherits project settings
            config = file.plot_config
            assert config.x_label == 'Binding Energy (eV)'
            assert config.y_label == 'Delay (ps)'
            assert config.x_dir == 'rev'
    
    def test_file_can_customize_config(self):
        """Test that File can customize its config persistently"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(path=tmpdir, name='test')
            x = np.linspace(0, 10, 50)
            data = np.random.randn(50)
            file = File(parent_project=project, data=data, energy=x)
            
            # Customize file's config
            file.plot_config.update(x_label='Custom Energy', dpi_plot=200)
            
            # Verify customization persists
            assert file.plot_config.x_label == 'Custom Energy'
            assert file.plot_config.dpi_plot == 200
            
            # Verify project unchanged
            assert project.e_label != 'Custom Energy'


class TestEdgeCases:
    """Test edge cases and potential regressions"""
    
    def test_single_point(self):
        """Test plotting single data point"""
        config = PlotConfig()
        plot_1D([[1]], x=[0], config=config, save_img=-1)
        plt.close('all')
    
    def test_nan_in_data(self):
        """Test plotting with NaN values"""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        y[10] = np.nan
        config = PlotConfig()
        
        plot_1D([y], x=x, config=config, save_img=-1)
        plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])