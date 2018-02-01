import numpy as np
from astroML.datasets import sdss_corrected_spectra
from . import Sed

__all__ = ["spectra"]


class spectra(object):

    """
    This class will include methods to get spectra for examples.
    """

    def __init__(self):

        return

    def get_red_spectrum(self):

        sed_obj = Sed()
        sed_obj.readSED_flambda('../data/Inst.10E10.1Z.spec.gz')

        return sed_obj

    def get_blue_spectrum(self):

        sed_obj = Sed()
        sed_obj.readSED_flambda('../data/Inst.64E08.1Z.spec.gz')

        return sed_obj

    def get_sigmoid_spectrum(self, lam_0=364.6):

        wavelen = np.arange(99., 2400.05, 0.1)
        spec = 1 / np.exp(wavelen - lam_0)
        sed_obj = Sed()
        sed_obj.setSED(wavelen=wavelen, flambda=spec)

        return sed_obj
