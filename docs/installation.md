# Installation

These instructions assume Python 3.12+, `pip`, and virtual environment support are already installed. If you are new to Python or the command line, start with the step-by-step guide for [macOS and Linux](#step-by-step-for-macos-and-linux) or [Windows](#step-by-step-for-windows) at the bottom of this page.

## From PyPI
```bash
python -m pip install trspecfit
```

## For Included Example Notebooks
```bash
python -m pip install "trspecfit[lab]"
```

## From GitHub
```bash
python -m pip install git+https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
```

## For Development
```bash
git clone https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
cd time-resolved-spectroscopy-fit
pip install -e ".[dev]"
```

## Step-by-step for macOS and Linux

This walkthrough assumes you have none of the tools installed yet. It uses [VS Code](https://code.visualstudio.com/) as the editor.

**1. Install Python.** Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest Python 3.12+ installer for macOS (or use your Linux distro's package manager, e.g. `sudo apt install python3 python3-venv` on Debian/Ubuntu). Run the installer and accept the defaults. Then open Terminal and check your Python version with `python3 --version`. Make sure it reports 3.12 or newer before continuing.

**2. Install Git.** On macOS, open Terminal (`Cmd+Space`, type "Terminal") and run `git --version` — macOS will offer to install the Xcode Command Line Tools, which include Git. On Linux, run `sudo apt install git` (or the equivalent for your distro).

**3. Install VS Code.** Go to [code.visualstudio.com](https://code.visualstudio.com/), download the installer for your OS, and run it.

**4. Clone the repository.** Open Terminal, navigate to the folder where you want to keep the project (e.g. `cd ~/Documents`), and run:
```bash
git clone https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
cd time-resolved-spectroscopy-fit
```

**5. Open the project in VS Code.** From inside the project folder, run:
```bash
code .
```
If the `code` command is not found, open VS Code the normal way (from your Applications folder) and use *File → Open Folder…* to select the `time-resolved-spectroscopy-fit` folder.

**6. Open the integrated terminal.** Inside VS Code, press ``Cmd+` `` (Command + backtick) to open a terminal at the project root.

**7. Create and activate a virtual environment, then install the package.**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[lab]"
```
You should see `(.venv)` at the start of your prompt once activated.

**8. Open an example notebook.** Press `Cmd+Shift+E` to open the file explorer sidebar. Expand `examples/fitting_workflows/`, pick the first notebook (`01_basic_fitting`), and open it. When VS Code asks which kernel to use, select the `.venv` interpreter.

## Step-by-step for Windows

This walkthrough assumes you have none of the tools installed yet. It uses [VS Code](https://code.visualstudio.com/) as the editor.

**1. Install Python.** Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest Python 3.12+ Windows installer. **Important:** on the first installer screen, tick the box labeled *"Add python.exe to PATH"* before clicking *Install Now*. This is the one non-default step you must take. After installation, open PowerShell and run `python --version`. Make sure it reports 3.12 or newer before continuing.

**2. Install Git.** Go to [git-scm.com/download/win](https://git-scm.com/download/win), download the installer, and run it. The defaults are fine.

**3. Install VS Code.** Go to [code.visualstudio.com](https://code.visualstudio.com/), download the Windows installer (`.exe`), and run it.

**4. Clone the repository.** Open PowerShell (press the Windows key, type "PowerShell", hit Enter), navigate to the folder where you want to keep the project (e.g. `cd $HOME\Documents`), and run:
```powershell
git clone https://github.com/InfinityMonkeyAtWork/time-resolved-spectroscopy-fit.git
cd time-resolved-spectroscopy-fit
```

**5. Open the project in VS Code.** From inside the project folder, run:
```powershell
code .
```
If the `code` command is not found, open VS Code the normal way (from the Start menu) and use *File → Open Folder…* to select the `time-resolved-spectroscopy-fit` folder.

**6. Open the integrated terminal.** Inside VS Code, press ``Ctrl+` `` (Control + backtick) to open a PowerShell terminal at the project root.

**7. Create and activate a virtual environment, then install the package.**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install ".[lab]"
```
You should see `(.venv)` at the start of your prompt once activated. If PowerShell blocks the activation script with an execution-policy error, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once and try again.

**8. Open an example notebook.** Press `Ctrl+Shift+E` to open the file explorer sidebar. Expand `examples/fitting_workflows/`, pick the first notebook (`01_basic_fitting`), and open it. When VS Code asks which kernel to use, select the `.venv` interpreter.
