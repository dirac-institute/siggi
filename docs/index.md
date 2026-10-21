# siggi

**Spectra Information Gain (SIGgi)** is a Python package for designing
astronomical filters (bandpasses) that maximize the information gained about
galaxy redshifts from photometric colors.

Given a set of galaxy template spectra (SEDs), a redshift prior, and a
parameterized family of filter shapes, `siggi` searches for the set of filter
throughput curves that maximizes the **information gain** — the mutual
information, in bits, between the observed colors and galaxy redshift. The
optimization accounts for realistic LSST-style photometric noise.

`siggi` accompanies the paper
[Kalmbach, VanderPlas & Connolly (2020), ApJ 890, 74](https://ui.adsabs.harvard.edu/abs/2020ApJ...890...74K/abstract).
See {doc}`concepts` for the underlying method.

## Quick links

```{toctree}
:maxdepth: 2

installation
quickstart
concepts
api/index
```

## At a glance

```python
import numpy as np
from siggi import siggi, spectra

# Example template SEDs shipped with the package.
specs = spectra()
sed_list = [specs.get_red_spectrum(), specs.get_blue_spectrum()]
weights = [0.5, 0.5]

# A simple redshift prior.
def z_prior(z):
    return np.ones_like(z)

s = siggi(sed_list, weights, z_prior, z_min=0.0, z_max=2.5, z_steps=51)
opt = s.optimize_filters(num_filters=3, filt_type="trap", n_opt_points=20)
```

See {doc}`quickstart` for a fuller walk-through.
