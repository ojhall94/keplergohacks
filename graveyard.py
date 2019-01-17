    def _spectrum_width(self, numax, a=0.66, b=0.88, factor=1):
        """Returns the estimated Full Width Half Maximum of the envelope of
        seismic oscillation modes centered at a given numax.
        Equation taken from Mosser et al. (2010)

        Parameters
        ----------
        numax : float
            A hypothetical or observed numax.
        a : float
            A linear coefficient in the equation
        b : float
            A power coefficient in the equation
        factor : float
            A factor by which to multiply the width of the Full Width Half
            Maximum.

        Returns
        -------
        fwhm : float
            The Full Width Half Maximum of a seismic mode envelope centered at
            a given numax. Will have same units as the input value for numax.
        """
        fwhm = a * numax**b * factor
        return fwhm

    def estimate_background(self, skips=50):
        """Estimates background noise of the power spectrum, via moving filter
        in steps of `skips`. Starting at a given frequency value, it calculates
        the approximate width of a seismic mode envelope were it centered at
        that frequency value, which means that the bin widht increases at larger
        frequencies (smaller periods). It then takes the median of values within
        that width as the value for that bin. Finally, it inerpolates over the
        binned values in order to obtain an estimate for the noise background.

        Parameters
        ----------
        skips : int
            The step size (in array indices) of the moving filter.

        Returns
        -------
        bkg : array-like
            An estimate of the noise background of the power spectrum. Has the
            same units as the `power` attribute.
        """
        med = [np.median(self.power[np.abs(self.frequency.value - d) < self._spectrum_width(d, factor=1)].value) for d in self.frequency[::skips].value]
        f = interpolate.interp1d(self.frequency[::skips].value, med, bounds_error=False)
        bkg = f(self.frequency.value)
        return bkg

        
