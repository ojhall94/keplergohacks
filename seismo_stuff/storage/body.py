from .asteroseismology import dnu_mass_prior, estimate_mass, estimate_radius, estimate_logg

    def estimate_numax(self, method = 'simple'):
        """Estimates the peak of the envelope of seismic oscillation modes,
        numax, using a choice of method.

        Method `simple` smoothes the periodogram power using a broad Gaussian
        filter. The power excess around the mode envelope creates a hump in the
        smoothed spectrum, the peak of which is taken as numax.

        Method `autocorrelate` first creates an array of possible numax values.
        It then estimates the width of the mode envelope at each numax using a
        standard relation, and autocorrelates this region. For a numax around
        the true numax, this will provide the correlation of the mode envelope
        with itself. Due to the equally spaced pattern of modes, the correlation
        at this numax will be large, revealing the likely numax value.

        Parameters:
        -----------
        method : str
            {'simple', 'autocorrelation'}. Default: 'simple'.

        Returns:
        --------
        numax : float
            The numax of the periodogram. In the units of the periodogram object
            frequency.
        numax_err : float
            The uncertainty on the numax estimate.
        """
        if method == 'simple':
            numax, numax_err, _, _, _ = self._numax_simple()
            return numax, numax_err

        if method == 'autocorrelate':
            raise NotImplementedError('Autocorrelation numax calculation not yet impelmented')

    def _numax_simple(self):
        """Smoothes the periodogram using a broad Gaussian filter, and returns
        the frequency of highest power in the smoothed spectrum"""
        smoothed_ps = gaussian_filter(self.power.value, 1000)
        best_numax = self.frequency[np.argmax(smoothed_ps)]

        #Fit a simple gaussian to the peak
        fwhm = 0.66 * best_numax.value**0.88                    #Predicted FWHM of the envelope
        #Relationship between FWHM and standard deviation. We use half the FWHM to generate sigma.
        sigma_guess = 0.5 * fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
        sel = np.where((self.frequency.value > (best_numax.value-fwhm))
                        * ( self.frequency.value < (best_numax.value+fwhm)))
        popt, pcov = curve_fit(self._gaussian, self.frequency[sel], smoothed_ps[sel],
                                p0 = [sigma_guess, np.max(smoothed_ps), best_numax.value])

        numax = u.Quantity(popt[2], self.frequency.unit)
        numax_err = u.Quantity(popt[0], self.frequency.unit)

        return numax, numax_err, smoothed_ps, popt, sel

    def _numax_autocorrelate(self):
        raise NotImplementedError('Not yet implemented')

    def plot_numax_diagnostics(self, method='simple', **kwargs):
        """Estimates the numax of the oscillation modes, and plots a number of
        diagnostics used in the estimation method.

        For full details on the method please see the docstring for the
        `estimate_numax()` function.

        Parameters:
        -----------
        method : str
            {'simple', 'autocorrelation'}. Default: 'simple'.

        **kwargs : dict
            Dictionary of arguments ot be passed to `Periodogram.plot`.

        Returns:
        --------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.

        """
        if method == 'simple':
            numax, numax_err, smoothed_ps, popt, sel = self._numax_simple()

            ax  = self.plot(**kwargs)
            ax.plot(self.frequency, smoothed_ps, label = 'Filtered PS')
            ax.plot(self.frequency[sel], self._gaussian(self.frequency[sel].value, *popt),
                    label = 'Gaussian Fit')
            ax.axvline(numax.value, linestyle='-.',
                    label="$\\nu_\mathrm{{max}} = {0:.2f}${1}".format(numax.value, numax.unit.to_string('latex')))
            ax.legend()

            return ax

        elif method == 'autocorrelate':
            raise NotImplementedError('Not yet implemented')

    def estimate_dnu(self, method='empirical', numax=None):
        """ Estimates the average value of the large frequency spacing, DeltaNu,
        of the seismic oscillations of the target.

        If method = `empirical`, Dnu will be calculated using an empirical
        relation for numax taken from Stello et al. 2009, as

        dnu = 0.294 * numax^0.772,

        with a 15% uncertainty.

        If method = `autocorrelate`, it will autocorrelate the region around
        the estimated numax expected to contain seismic oscillation modes.
        Repeating peaks in the autocorrelation implies an evenly spaced structure
        of modes. The peak closest to an empirical estimate of dnu is taken as
        the true value.

        If `numax` is None, a simple `numax` is calculated using the
        estimate_numax(method='simple') function.

        Parameters:
        ----------
        method : str
            {'empirical', 'autocorrelation'}. Default: 'simple'.

        numax : float
            An estimated numax value of the mode envelope in the periodogram. If
            not given units it is assumed to be in units of the periodogram
            frequency attribute.

        Returns:
        -------
        deltanu : float
            The average large frequency spacing of the seismic oscillation modes.
            In units of the periodogram frequency attribute.
        """
        if numax is None:
            numax, _ = self.estimate_numax(method='simple')

        if method == 'empirical':
            #Calcluate dnu using the method by Stello et al. 2009
            dnu = u.Quantity(0.294 * numax.value ** 0.772, numax.unit)
            dnu_err = 0.15 * dnu
            return dnu, dnu_err

        if method == 'autocorrelate':
            dnu, dnu_err, _, _, _, _, _, _  = self._dnu_autocorrelate(numax)
            return dnu, dnu_err

    def _dnu_autocorrelate(self, numax):
        """Calculates delta nu by correlating the region expected to contain
        seismic modes with itself."""
        #Calculate the ACF for the best numax estimate
        acf = self.autocorrelate(numax)
        #Finding Mass-prior limits on dnu values for the best numax esimate
        lower, upper = dnu_mass_prior(numax.value)

        #Note that this is only functional for evenly spaced grid of frequencies
        #An exception is already built into self.autocorrelate to check for this, however
        fs = np.median(np.diff(self.frequency))

        #Calculating the correpsonding indices
        lo = int(np.floor(lower / fs.value))
        up = int(np.floor(upper / fs.value))

        #Building list of possible dnus
        lags = np.arange(len(acf)) * fs.value
        acfrange = acf[lo:up]     #The acf range to look for dnu
        lagrange = lags[lo:up]    #The range of lags to look for dnu

        #The best dnu value is at the position of maximum acf power within the range
        best_dnu = lagrange[np.argmax(acfrange)]
        sigma_guess = 0.05 * best_dnu
        # best_dnu, sigma_guess = self.estimate_dnu(method='empirical', numax=numax)

        #Fit a Gaussian to the peak
        sel = np.where((lagrange > (best_dnu - sigma_guess))
                        & (lagrange < (best_dnu + sigma_guess)))
        popt, pcov = curve_fit(self._gaussian, lagrange[sel], acfrange[sel],
                                p0 = [sigma_guess, np.max(acfrange), best_dnu])

        dnu = u.Quantity(popt[2], self.frequency.unit)
        dnu_err = u.Quantity(popt[0], self.frequency.unit)
        return dnu, dnu_err, lags, acf, lo, up, popt, sel

    def plot_dnu_diagnostics(self, method='autocorrelate', numax=None):
        if numax is None:
            numax, _ = self.estimate_numax(method='simple')

        if method == 'empirical':
            raise NotImplentedError('No diagnostic plots for `empirical` method.')

        elif method == 'autocorrelate':
            dnu, dnu_err, lags, acf, lo, up, popt, sel = self._dnu_autocorrelate(numax)

            #TODO: Make these fit the style

            fig, ax = plt.subplots(2,figsize=(12,12))
            self.plot(ax=ax[0])

            ax[1].plot(lags, acf)
            ax[1].plot(lags[lo:up][sel], self._gaussian(lags[lo:up][sel], popt[0], popt[1], popt[2]))
            ax[1].set_xlabel(r'$\Delta\nu$')
            ax[1].set_ylabel(r'ACF')
            ax[1].axvline(lo*np.median(np.diff(self.frequency)).value)
            ax[1].axvline(up*np.median(np.diff(self.frequency)).value)

            fig.tight_layout()
            plt.show()
            return ax

    def estimate_stellar_parameters(self, Teff, Teff_err = None,
                                    numax=None, numax_err=None,
                                    dnu=None, dnu_err=None,
                                    fdnu=1., fnumax=1.):
        """Returns stellar parameters calculated using asteroseismology and a
        value of temperature.

        If no numax or delta nu values are passed, it will calculate them using
        the most precise methods implemented.

        Parameters:
        -----------
        Teff : float
            The effective temperature of the star. In units of Kelvin.
        Teff_error : float
            Error on the Teff value,
        numax : float
            The frequency of maximum power of the seismic mode envelope. In units of
            microhertz.
        numax_error : float
            Error on the numax value.
        dnu : float
            The frequency spacing between two consecutive overtones of equal radial
            degree. In units of microhertz.
        dnu_error : float
            Error on the dnu value.
        fdnu : int
            A correction to the seismic scaling relation for Delta Nu. Effectively
            rescales the solar value for Delta Nu.
        fnumax : int
            A correction to the seismic scaling relation for Numax. Effectively
            rescales the solar value for Numax.

        Returns:
        -------
        stellar_paramaters : dict
            A dictionary containing Mass, Radius, Numax, DeltaNu, and log(g)
            (where `g` is the surface gravity of the star), all with errors.
            Also includes input Teff, and uncertainty on Teff.
        """
        #Make checks
        if (numax is None) & (numax_err is not None):
            raise ValueError('You cant pass a numax error without a numax value!')
        if (dnu is None) & (dnu_err is not None):
            raise ValueError('You cant pass a dnu error without a dnu value!')
        if fdnu <= 0.:
            raise ValueError('fdnu must be larger than 0.')
        if fnumax <= 0.:
            raise ValueError('fnumax must be larger than 0.')

        if numax is None:
            numax, numax_err = self.estimate_numax(method='simple')
        if dnu is None:
            dnu, dnu_err = self.estimate_dnu(method='autocorrelate')

        mass, mass_err =  estimate_mass(numax, dnu, Teff,
                    numax_err=numax_err, dnu_err = dnu_err, Teff_err=Teff_err,
                    fdnu=fdnu, fnumax=fnumax)
        radius, radius_err =  estimate_radius(numax, dnu, Teff,
                    numax_err=numax_err, dnu_err = dnu_err, Teff_err=Teff_err,
                    fdnu=fdnu, fnumax=fnumax)
        logg, logg_err =  estimate_logg(numax, Teff,
                    numax_err=numax_err, Teff_err=Teff_err,
                    fnumax=fnumax)

        stellar_parameters = {'numax': numax,
                                'numax_err' : numax_err,
                                'dnu' : dnu,
                                'dnu_err' : dnu_err,
                                'mass' : mass,
                                'mass_err' : mass_err,
                                'radius' : radius,
                                'radius_err' : radius_err,
                                'logg' : logg,
                                'logg_err' : logg_err,
                                'Teff' : Teff}
        if Teff_err is not None:
            stellar_parameters['Teff_err'] = Teff_err

        return stellar_parameters

    def autocorrelate(self, numax, width_factor=1):
        """An autocorrelation function for seismic mode envelopes.
        For a given numax, the method calculates the expected Full Width Half
        Maximum of the seismic mode envelope as (Mosser et al 2010)

        fwhm = 0.66 * numax^0.88 .

        Strictly speaking, this is intended for red giants, but will suffice for
        our purposes here for all stars. It then correlates a region of one
        fwhm either side of the estimated numax with itself.

        Before autocorrelating, it also multiplies the section with a hanning
        window, which will increase the autocorrelation power if the region
        has a Gaussian shape, as we'd expect for seismic oscillations.

        Parameters:
        ----------
            numax : float
                The estimated position of the numax of the power spectrum. This
                is used to calculated the region autocorrelated with itself.

            width_factor : float
                This factor is multiplied with the estimated fwhm of the
                oscillation modes, effectively increasing or decreasing the
                autocorrelation range.

        Returns:
        --------
            acf : array-like
                The autocorrelation power calculated for the given numax
        """
        fs = np.median(np.diff(self.frequency))

        if np.isclose(np.median(np.diff(self.frequency.value)), fs.value):
            #Calculate the index FWHM for a given numax
            fwhm = int(np.floor(width_factor * 0.66 * numax.value**0.88 / fs.value)) #Express the FWHM in indices
            fwhm -= fwhm % 2                                    # Make the FWHM value even (%2 = 0 if even, 1 if odd)
            x = int(numax / fs)                                 #Find the index value of numax
            s = np.hanning(len(self.power[x-fwhm:x+fwhm]))      #Define the hanning window for the evaluated frequency space
            C = self.power[x-fwhm:x+fwhm].value * s             #Multiply the evaluated SNR space by the hanning window
            result = np.correlate(C, C, mode='full')            #Correlated the resulting SNR space with itself

        else:
            raise NotImplementedError("The autocorrelate() function requires a grid of evenly spaced frequencies at this time.")

        return result[int(len(result)/2):]      #Return one half of the autocorrelation function

    def _gaussian(self, x, sigma, height, mu):
        """A simple Gaussian function for fitting to autocorrelation peaks."""
        return height * np.exp(-(x - mu)**2 / (2.0 * sigma**2))
