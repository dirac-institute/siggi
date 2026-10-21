# Concepts

This page summarizes the method behind `siggi`. For the full treatment, see
[Kalmbach, VanderPlas & Connolly (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...890...74K/abstract).

## The problem

Photometric surveys measure galaxy brightness through a handful of broad
filters. The resulting **colors** (magnitude differences between filters)
encode information about a galaxy's redshift. Different filter designs capture
different amounts of redshift information. `siggi` asks: *which set of filters
extracts the most redshift information for a given population of galaxies?*

## Information gain

`siggi` scores a candidate filter set by the **information gain**: how much the
uncertainty in redshift is reduced once the colors are observed. In terms of
Shannon entropy $H$,

$$ \mathrm{IG} = H(z) - H(z \mid \text{colors}), $$

the difference between the entropy of the redshift prior and the expected
entropy of the redshift posterior given the colors. This is the mutual
information between redshift and color, measured in bits.

Maximizing the information gain over filter designs yields the filter set that
is most informative about redshift for the assumed galaxy population and prior.

## How siggi estimates it

1. **Redshift grid and templates.** The input template SEDs are redshifted and
   flux-normalized across a redshift grid
   ({class}`~siggi.siggi.siggi` constructor).
2. **Colors with noise.** For a candidate filter set, the observed colors of
   each template are computed through the filters — including LSST hardware and
   atmosphere via `addSystemBandpass` — together with realistic photometric
   magnitude errors from the sky background
   ({meth}`~siggi.calcIG.calcIG.calc_colors`).
3. **Color distributions.** Each redshift produces a multivariate-normal
   distribution of colors (mean from the template, covariance from the
   magnitude errors).
4. **Monte-Carlo entropy estimate.** Samples are drawn weighted by the redshift
   prior, and the entropy terms $H(z)$ and $H(z\mid\text{colors})$ are estimated
   to give the information gain
   ({meth}`~siggi.calcIG.calcIG.calc_gain`,
   {meth}`~siggi.calcIG.calcIG.calc_IG`), using the entropy and
   hyperellipsoid-volume helpers in {class}`~siggi.mathUtils.mathUtils`.
5. **Optimization.** {meth}`~siggi.siggi.siggi.optimize_filters` drives a
   Bayesian optimizer (scikit-optimize) over the filter-edge parameters,
   proposing filter sets, validating that they are physically sensible
   ({meth}`~siggi.siggiBase._siggiBase.create_and_validate_filter_dict`), and
   maximizing the information gain.

## Filter parameterizations

Filters are described by a small number of shape parameters and turned into
throughput curves by the classes in {mod}`siggi.filters`:

- **Trapezoid** ({class}`~siggi.filters.trapezoidFilter.trapezoidFilter`):
  each filter is a trapezoid defined by four corner wavelengths (lower-left,
  upper-left, upper-right, lower-right). Fixing `set_ratio` and/or `set_width`
  reduces the number of free parameters.
- **Comb** ({class}`~siggi.filters.combFilter.combFilter`): several trapezoid
  "teeth" are summed into a single comb bandpass.

The {class}`~siggi.filters.filterFactory.filterFactory` selects the family by
name (`'trap'` or `'comb'`).
