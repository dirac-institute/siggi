import os
import numpy as np
from . import Sed

__all__ = ["spectra"]


class spectra(object):

    """
    Convenience loaders for example and template spectra.

    This class provides helper methods that return LSST Sims ``Sed``
    objects for the galaxy template SEDs and the dark-sky background that
    ship with ``siggi``, plus a synthetic sigmoid spectrum for testing.
    """

    def __init__(self):

        return

    def get_red_spectrum(self):

        """
        Load the bundled example red galaxy template SED.

        Returns
        -------
        Sed
            A red (old, quiescent) galaxy template SED loaded from the
            packaged ``Inst.10E10.1Z.spec.gz`` file.
        """

        sed_obj = Sed()
        sed_obj.readSED_flambda(os.path.join(os.path.dirname(__file__),
                                             'data/Inst.10E10.1Z.spec.gz'))

        return sed_obj

    def get_blue_spectrum(self):

        """
        Load the bundled example blue galaxy template SED.

        Returns
        -------
        Sed
            A blue (young, star-forming) galaxy template SED loaded from the
            packaged ``Inst.64E08.1Z.spec.gz`` file.
        """

        sed_obj = Sed()
        sed_obj.readSED_flambda(os.path.join(os.path.dirname(__file__),
                                             'data/Inst.64E08.1Z.spec.gz'))

        return sed_obj

    def get_dark_sky_spectrum(self):

        """
        Load the LSST baseline dark-sky background SED.

        Returns
        -------
        Sed
            The dark-sky emission SED loaded from the packaged
            ``lsst_baseline_throughputs/darksky.dat`` file, used to set the
            sky background when computing photometric errors.
        """

        sed_obj = Sed()
        sky_sed = os.path.join(os.path.dirname(__file__),
                               'data/lsst_baseline_throughputs',
                               'darksky.dat')
        sed_obj.readSED_flambda(sky_sed)

        return sed_obj

    def get_sigmoid_spectrum(self, lam_0=364.6):

        """
        Build a synthetic sigmoid (smoothed step) SED.

        Parameters
        ----------
        lam_0 : float, optional
            Wavelength, in nanometres, of the sigmoid's inflection point
            (the half-maximum of the step). Default is 364.6.

        Returns
        -------
        Sed
            An SED whose flux rises as ``1 / (1 + exp(lam_0 - wavelen))``
            over a wavelength grid from 9 to 2400 nm. Useful as a simple
            test spectrum.
        """

        wavelen = np.arange(9., 2400.05, 0.1)
        spec = 1 / (1 + np.exp(lam_0-wavelen))
        sed_obj = Sed()
        sed_obj.setSED(wavelen=wavelen, flambda=spec)

        return sed_obj
