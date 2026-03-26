# Installation

These instructions assume Python 3.12+, `pip`, and virtual environment support are already installed.

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

## Requirements

- Python >= 3.12
- Dependencies: lmfit, numpy, scipy, matplotlib, and others (installed automatically)
