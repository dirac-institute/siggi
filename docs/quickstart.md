# Quickstart

This walk-through mirrors the `examples/Intro_Examples.ipynb` notebook and
shows a minimal end-to-end optimization run. It assumes `siggi` is
{doc}`installed <installation>`.

## 1. Load template spectra

`siggi` ships two example galaxy template SEDs (a red and a blue galaxy) and a
dark-sky background, exposed through the
{class}`~siggi.spectra.spectra` helper class.

```python
from siggi import spectra

specs = spectra()
red = specs.get_red_spectrum()
blue = specs.get_blue_spectrum()
sed_list = [red, blue]
```

Each template is returned as an LSST Sims `Sed` object. You can also supply
your own `Sed` objects.

## 2. Define weights and a redshift prior

Provide a weight per template (commonly uniform) and a redshift prior as a
function of redshift. The prior is evaluated on the redshift grid built from
`z_min`, `z_max`, and `z_steps`.

```python
import numpy as np

weights = [0.5, 0.5]

def z_prior(z):
    # Flat prior for illustration; use a realistic prior in practice.
    return np.ones_like(z)
```

A commonly used parametric form is
$p(z) \propto z^{a}\,\exp\!\left[-(z/z_0)^{b}\right]$ (see the example
scripts under `examples/scripts/`).

## 3. Build the optimizer object

{class}`~siggi.siggi.siggi` redshifts and flux-normalizes the templates
across the redshift grid when constructed.

```python
from siggi import siggi

s = siggi(sed_list, weights, z_prior,
          z_min=0.0, z_max=2.5, z_steps=51)
```

## 4. Run the optimization

{meth}`~siggi.siggi.siggi.optimize_filters` runs a Bayesian optimization
(scikit-optimize) over the filter-edge parameters, evaluating the information
gain at each candidate filter set.

```python
opt = s.optimize_filters(
    num_filters=3,
    filt_type="trap",       # 'trap' (trapezoid) or 'comb'
    filt_min=300.0,
    filt_max=1100.0,
    n_opt_points=50,
    procs=1,
    rand_state=42,          # for reproducible results
)
```

The call returns a scikit-optimize `Optimizer`. The information-gain values
are `np.abs(opt.yi)` (stored negated for minimization) and the corresponding
filter parameters are in `opt.Xi`.

```python
best_idx = np.argmin(opt.yi)
best_point = opt.Xi[best_idx]
best_info_gain = np.abs(opt.yi[best_idx])
print(f"Best information gain: {best_info_gain:.3f} bits")
```

## 5. Plot the result

{class}`~siggi.plotting.plotting` rebuilds the best filter set and provides
diagnostic figures.

```python
from siggi import plotting

p = plotting(sed_list, best_point, filt_type="trap")
p.plot_filters()                 # filters over the template SEDs
```

## Reusing a run

Pass `save_optimizer="run.pkl"` to checkpoint the optimizer state, then
resume later with `load_optimizer=...`. See
{meth}`~siggi.siggi.siggi.optimize_filters` for the full set of options,
including fixing filter shapes (`set_ratio`, `set_width`), including frozen
filters (`frozen_filt_dict`), and parallelism (`procs`).
