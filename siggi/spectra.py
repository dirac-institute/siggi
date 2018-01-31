import numpy as np
from astroML.datasets import sdss_corrected_spectra
from . import Sed

__all__ = ["spectra"]


class spectra(object):

    """
    This class will include methods to get spectra for examples.
    """

    def __init__(self):

        data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
        self.spectra = sdss_corrected_spectra.reconstruct_spectra(data)
        self.wavelen = sdss_corrected_spectra.compute_wavelengths(data)

        return

    def get_red_spectrum(self):

        spec = self.spectra[3100]/np.max(self.spectra[3100])
        sed_obj = Sed()
        sed_obj.setSED(self.wavelen/10., spec)

        return sed_obj

    def get_blue_spectrum(self):

        spec = self.spectra[684]/np.max(self.spectra[684])
        sed_obj = Sed()
        sed_obj.setSED(self.wavelen/10., spec)

        return sed_obj
