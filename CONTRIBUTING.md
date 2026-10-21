# Contributing to siggi

Thanks for your interest in improving `siggi`! This guide covers setting up a
development environment, running the tests, and building the documentation.

## Development environment

Clone the repository and install it in editable mode with the test and
documentation extras:

```bash
git clone https://github.com/dirac-institute/siggi.git
cd siggi
pip install -e ".[test,docs]"
```

This installs the runtime dependencies plus `pytest`, `pytest-cov`, and the
Sphinx documentation toolchain.

## Running the tests

The test suite uses `unittest`-style tests and runs under `pytest`:

```bash
pytest tests
```

With coverage:

```bash
pytest tests --cov=siggi --cov-report=term-missing
```

Note: some tests (e.g. `tests/test_siggi.py`) write temporary files such as
`test.pkl` to the current working directory, so run the suite from a writable
location.

## Building the documentation

Build the HTML documentation locally with Sphinx:

```bash
sphinx-build -b html docs docs/_build
```

Then open `docs/_build/index.html` in a browser. The documentation is also
built automatically on [Read the Docs](https://siggi.readthedocs.io).

## Code style

Code is linted with `flake8` in continuous integration. Please keep new code
free of lint errors (`flake8 siggi tests`).
