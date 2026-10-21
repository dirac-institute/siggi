# Installation

`siggi` is a pure-Python package. Clone the repository and install it with
`pip`:

```bash
git clone https://github.com/dirac-institute/siggi.git
cd siggi
pip install .
```

For a development install (editable, with the test and documentation extras):

```bash
pip install -e ".[test,docs]"
```

## Dependencies

The runtime dependencies are installed automatically:

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [scikit-optimize](https://scikit-optimize.github.io/)
- [joblib](https://joblib.readthedocs.io/)

Optional extras:

- `test` — `pytest`, `pytest-cov`
- `docs` — `sphinx`, `myst-parser`, `sphinx-rtd-theme`, `sphinx-autodoc-typehints`

## Vendored LSST photometry code

`siggi` bundles a subset of the LSST `sims_photUtils` package under
`siggi/lsst_utils/` (SED, Bandpass, signal-to-noise, and photometric
parameter utilities). Because this code is vendored, **you do not need an LSST
stack installation** to use `siggi`. The bundled copy is a snapshot and may
lag the upstream package.

## Supported Python versions

`siggi` targets Python 3.9 and newer.
