# siggi
[![siggi](https://github.com/dirac-institute/siggi/workflows/siggi/badge.svg)](https://github.com/dirac-institute/siggi/actions)
[![codecov.io](https://codecov.io/github/dirac-institute/siggi/coverage.svg?branch=master)](https://codecov.io/github/dirac-institute/siggi?branch=master)
[![Documentation Status](https://readthedocs.org/projects/siggi/badge/?version=latest)](https://siggi.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/118051192.svg)](https://zenodo.org/badge/latestdoi/118051192)

**Spectra Information Gain (SIGgi)**

`siggi` designs astronomical filters (bandpasses) that maximize the information
gained about galaxy redshifts from photometric colors. Given a set of galaxy
template spectra (SEDs), a redshift prior, and a parameterized family of filter
shapes, it searches for the filter throughput curves that maximize the mutual
information — in bits — between observed colors and galaxy redshift, accounting
for realistic LSST-style photometric noise.

Read the paper! [Kalmbach, VanderPlas, & Connolly (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...890...74K/abstract)

## Installation

```bash
git clone https://github.com/dirac-institute/siggi.git
cd siggi
pip install .
```

For a development install with the test and documentation extras:

```bash
pip install -e ".[test,docs]"
```

`siggi` bundles a subset of the LSST `sims_photUtils` code under
`siggi/lsst_utils/`, so **no LSST stack installation is required**. Runtime
dependencies (`numpy`, `scipy`, `matplotlib`, `scikit-learn`,
`scikit-optimize`, `joblib`) are installed automatically.

## Quick example

```python
import numpy as np
from siggi import siggi, spectra

specs = spectra()
sed_list = [specs.get_red_spectrum(), specs.get_blue_spectrum()]
weights = [0.5, 0.5]

def z_prior(z):
    return np.ones_like(z)

s = siggi(sed_list, weights, z_prior, z_min=0.0, z_max=2.5, z_steps=51)
opt = s.optimize_filters(num_filters=3, filt_type="trap",
                         n_opt_points=50, rand_state=42)

best_idx = np.argmin(opt.yi)
print("Best information gain:", np.abs(opt.yi[best_idx]), "bits")
```

## Documentation

Full documentation is hosted on [Read the Docs](https://siggi.readthedocs.io).
Worked examples, including reproductions of figures from the paper, live in the
[`examples/`](examples/) directory as Jupyter notebooks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development
environment, run the tests, and build the documentation locally.

## License

`siggi` is released under the GNU General Public License v3. See
[LICENSE](LICENSE).
