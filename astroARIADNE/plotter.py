"""plot_utils module for plotting SEDs."""

__all__ = ['SEDPlotter']

import functools
import glob
import logging
import pickle
from random import choice

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from extinction import apply
from isochrones.interp import DFInterpolator
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from PyAstronomy import pyasl
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm

import corner
from dynesty import plotting as dyplot

from .config import (filesdir, filter_labels, gridsdir, modelsdir,
                     spectra_cache)
from .isochrone import get_isochrone
from .phot_utils import *
from .sed_library import *
from .utils import *


# Model grid names as shown in plot legends.
_GRID_LABELS = {
    'phoenix': 'PHOENIX', 'btsettl': 'BT-Settl', 'btnextgen': 'BT-NextGen',
    'btcond': 'BT-Cond', 'ck04': 'Castelli & Kurucz', 'kurucz': 'Kurucz',
    'coelho': 'Coelho', 'bosz': 'BOSZ', 'tlusty': 'TLUSTY',
    'sphinx': 'SPHINX',
}


_FLUX_LABEL = r'$\lambda F_\lambda$ (erg cm$^{-2}$ s$^{-1}$)'
_WAVE_LABEL = r'$\lambda$ ($\mu$m)'
_SIGMA_TOT_LABEL = r'$\sigma_{\rm tot} = \sqrt{\sigma_{\rm obs}^2 + s^2}$'


def filter_label(name):
    """Plot label of a band: its ``config.filter_labels`` entry (mathtext,
    in the survey's own notation), else the name with spaces for
    underscores."""
    name = str(name)
    return filter_labels.get(name, name.replace('_', ' '))


def _grid_label(name):
    """Legend name of a grid, or of a binary pair ('bosz+tlusty')."""
    return ' + '.join(_GRID_LABELS.get(g.lower(), g)
                      for g in str(name).split('+'))


def _styled(method):
    """Run a plotting method under the plotter's font settings (STIX text
    and mathtext by default) without changing the caller's rcParams."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with plt.rc_context(self._rc_params()):
            return method(self, *args, **kwargs)
    return wrapper


class SEDPlotter:
    """Artist class for all things SED.

    Parameters
    ----------
    input_files : str
        Directory containing the code's output files.
    out_folder : type
        Directory where to put the output plots.
    pdf : type
        Set to True to output plots in pdf.
    png : type
        Set to True to output plots in png.
    model : type
        Set to override the SED model that's going to be plotted.
        Possible values are:
            - Phoenix
            - BTSettl
            - NextGen
            - CK04 (Castelli & Kurucz 2004)
            - Kurucz (Kurucz 1993)

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    Attributes
    ----------
    chain_out : str
        Output directory for chain plot.
    like_out : str
        Output directory for likelihood plot.
    post_out : str
        Output directory for posteriors plot.
    moddir : type
        Directory wheere the SED models are located.
    out : dict
        SED fitting routine output.
    engine : str
        Selected fitting engine.
    star : Star
        The fitted Star object.
    coordinator : array_like
        Array coordinating fixed parameters.
    fixed : array_like
        Array coordinating fixed parameters.
    norm : bool
        norm is set to True if a normalization constant is fitted instead of
        radius + distance.
    grid : str
        Selected model grid.
    av_law : function
        Exticntion law chosen for the fit.
    order : array_like
        Array coordinating parameter order.
    interpolator : function
        Interpolator function.
    theta : array_like
        `Best fit` parameter vector

    """

    __wav_file = 'PHOENIXv2/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

    def __init__(self, input_files, out_folder, pdf=False,
                 model=None, settings=None, method='averaged',
                 save_model=False, ir_excess=False):
        """See class docstring."""
        print('\nInitializing plotter.\n')
        # General setup
        self.pdf = pdf
        png = True if not pdf else False
        self.png = png
        self.out_folder = out_folder
        self.bma = False
        self.method = method
        self.save = save_model
        self.irx = ir_excess
        if self.irx:
            model = 'btsettl'

        traces = f'{out_folder}/traces'
        histograms = f'{out_folder}/histograms'
        self.traces_out = traces
        self.hist_out = histograms
        self.moddir = modelsdir
        self.settings_dir = settings

        # Read output files.
        if input_files != 'raw':
            with open(input_files, 'rb') as fh:
                # older outputs name the excess-scatter terms <band>_noise
                out = upgrade_legacy_names(pickle.load(fh))
            self.out = out
            self.engine = out['engine']
            self.star = out['star']
            self.coordinator = out['coordinator']
            self.fixed = out['fixed']
            self.norm = out['norm']
            self.binary = bool(out.get('binary', False))
            if model is None:
                if self.engine != 'Bayesian Model Averaging':
                    self.grid = out['model_grid']
                else:
                    self.bma = True
                    zs = np.array([out['lnZ'][key]
                                   for key in out['lnZ'].keys()])
                    keys = np.array([key for key in out['lnZ'].keys()])
                    grid = keys[np.argmax(zs)]
                    self.grid = grid
            else:
                self.grid = model
            # A binary fit's grid is a pair, e.g. 'bosz+tlusty'.
            self.grid_name = self.grid
            if self.binary:
                g1, g2 = split_grid_name(self.grid)
                self.grid, self.grid2 = g1, (g2 or g1)
            else:
                self.grid2 = None
            self.av_law = out['av_law']

            # Create target folders
            create_dir(out_folder)
            if self.engine != 'Bayesian Model Averaging':
                create_dir(traces)
            create_dir(histograms)

            self.star.load_grid(self.grid)

            if not self.norm:
                self.order = np.array(
                    [
                        'teff', 'logg', 'z',
                        'dist', 'rad', 'Av',
                    ]
                )
            else:
                self.order = np.array(
                    ['teff', 'logg', 'z', 'norm', 'Av'])
            if self.binary:
                self.order = np.append(self.order, BINARY_PARAMS)

            mask = self.star.filter_mask
            flxs = self.star.flux[mask]
            errs = self.star.flux_er[mask]
            filters = self.star.filter_names[mask]
            wave = self.star.wave[mask]
            if self.irx:
                irx_mask = self.star.irx_filter_mask
                irx_filt = self.star.filter_names[irx_mask]
                irx_wave = self.star.wave[irx_mask]
            for filt, flx, flx_e in zip(filters, flxs, errs):
                self.order = np.append(self.order, excess_scatter_name(filt))

            # Model fluxes as the fit computed them: outputs from before
            # band-integrated extinction used the central-wavelength one.
            band_ext = bool(out.get('band_extinction', False))
            self.interpolator = load_band_grid(self.grid, band_ext)
            if self.binary:
                self.interpolator = (self.interpolator,
                                     load_band_grid(self.grid2, band_ext))

            # Get best fit parameters. A single-grid fit has no BMA
            # 'samples' summary; its one summary serves both methods.
            bf_avg = out['best_fit_averaged']
            bf_sam = out.get('best_fit_samples', bf_avg)
            if 'uncertainties_samples' not in out:
                for k in ('best_fit', 'uncertainties', 'confidence_interval'):
                    out.setdefault(f'{k}_samples', out[f'{k}_averaged'])
            theta_samples = np.zeros(self.order.shape[0])
            theta_average = np.zeros(self.order.shape[0])
            for i, param in enumerate(self.order):
                if param != 'inflation':
                    theta_samples[i] = bf_sam[param]
                    theta_average[i] = bf_avg[param]
            # The reported best-fit values (titles and lines of the corner
            # plot), before any max-likelihood override below.
            self.theta_best = (theta_samples if method == 'samples'
                               else theta_average).copy()
            if self.binary:
                # The secondary's Teff and radius are strongly correlated
                # (a hotter, smaller star gives the same UV flux), so the
                # per-parameter best values need not form a good model.
                # Plot the highest-likelihood posterior sample instead.
                theta_ml = self._max_like_theta()
                if theta_ml is not None:
                    theta_samples = theta_average = theta_ml

            # Calculate best fit model.
            model_samples = model_grid(theta_samples, filters, wave,
                                       self.interpolator, self.norm,
                                       self.av_law, binary=self.binary)
            model_average = model_grid(theta_average, filters, wave,
                                       self.interpolator, self.norm,
                                       self.av_law, binary=self.binary)
            if self.irx:
                irx_model_sam = model_grid(theta_samples, irx_filt, irx_wave,
                                           self.interpolator, self.norm,
                                           self.av_law, binary=self.binary)
                irx_model_avg = model_grid(theta_average, irx_filt, irx_wave,
                                           self.interpolator, self.norm,
                                           self.av_law, binary=self.binary)
            if method == 'averaged':
                self.theta = theta_average
                self.model = model_average
                if self.irx:
                    self.irx_model = irx_model_avg
            elif method == 'samples':
                self.theta = theta_samples
                self.model = model_samples
                if self.irx:
                    self.irx_model = irx_model_sam

            # Get archival fluxes.
            self.__extract_info()
        else:
            self.star = None

        # Setup plots.
        self.__read_config()
        print('\nPlotter initialized.\n')

    def _max_like_theta(self):
        """Parameter vector (``self.order``) of the highest-likelihood
        posterior sample of the plotted grid (pair), or None."""
        if self.bma:
            samples = self.out.get('originals', {}).get(self.grid_name)
        else:
            samples = self.out.get('posterior_samples')
        if not samples or 'loglike' not in samples:
            return None
        ll = np.asarray(samples['loglike'], dtype=float)
        if not ll.size or not np.isfinite(ll).any():
            return None
        i = int(np.nanargmax(ll))
        theta = np.zeros(self.order.shape[0])
        for j, param in enumerate(self.order):
            v = samples[param]
            theta[j] = v[i] if np.ndim(v) else v
        logger.info('Binary fit: plotting the highest-likelihood posterior '
                    'sample of %s', self.grid_name)
        return theta

    def __extract_info(self):
        self.flux = []
        self.flux_er = []
        self.wave = []
        self.bandpass = []

        for i, f in zip(self.star.used_filters, self.star.flux):
            if i:
                self.flux.append(f)
        for i, e in zip(self.star.used_filters, self.star.flux_er):
            if i:
                self.flux_er.append(e)
        for i, w in zip(self.star.used_filters, self.star.wave):
            if i:
                self.wave.append(w)
        for i, bp in zip(self.star.used_filters, self.star.bandpass):
            if i:
                self.bandpass.append(bp)

        self.flux = np.array(self.flux)
        self.flux_er = np.array(self.flux_er)
        self.wave = np.array(self.wave)
        self.bandpass = np.array(self.bandpass).T
        # Do the same for IR bandpasses
        self.irx_flux = []
        self.irx_flux_er = []
        self.irx_wave = []
        self.irx_bandpass = []

        for i, f in zip(self.star.irx_used_filters, self.star.flux):
            if i:
                self.irx_flux.append(f)
        for i, e in zip(self.star.irx_used_filters, self.star.flux_er):
            if i:
                self.irx_flux_er.append(e)
        for i, w in zip(self.star.irx_used_filters, self.star.wave):
            if i:
                self.irx_wave.append(w)
        for i, bp in zip(self.star.irx_used_filters, self.star.bandpass):
            if i:
                self.irx_bandpass.append(bp)

        self.irx_flux = np.array(self.irx_flux)
        self.irx_flux_er = np.array(self.irx_flux_er)
        self.irx_wave = np.array(self.irx_wave)
        self.irx_bandpass = np.array(self.irx_bandpass).T

    def _load_from_cache(self, model_key, teff=None, logg=None, z=None):
        """Return (wavelength, flux) from pre-computed HDF5 cache, or None.

        At ``teff``, ``logg``, ``z`` (default: the primary's best fit)."""
        if not spectra_cache:
            return None
        import h5py
        import os as _os
        if not _os.path.isfile(spectra_cache):
            return None
        teff = self.theta[0] if teff is None else teff
        logg = self.theta[1] if logg is None else logg
        z = self.theta[2] if z is None else z
        try:
            with h5py.File(spectra_cache, 'r') as f:
                if model_key not in f:
                    return None
                g = f[model_key]
                teffs = g['teff'][:]
                loggs = g['logg'][:]
                zs = g['z'][:]
                t_val = np.unique(teffs)[np.argmin(np.abs(np.unique(teffs) - teff))]
                l_val = np.unique(loggs)[np.argmin(np.abs(np.unique(loggs) - logg))]
                z_val = np.unique(zs)[np.argmin(np.abs(np.unique(zs) - z))]
                mask = (teffs == t_val) & (loggs == l_val) & (zs == z_val)
                if mask.any():
                    idx = np.where(mask)[0][0]
                else:
                    # Sparse grids (e.g. TLUSTY, whose lowest log g rises
                    # with Teff) need not hold the per-axis nearest
                    # combination: take the nearest node that exists, with
                    # Teff in relative units and log g / [Fe/H] in dex.
                    d2 = (((teffs - teff) / teff / 0.05) ** 2
                          + ((loggs - logg) / 0.5) ** 2
                          + ((zs - z) / 0.5) ** 2)
                    idx = int(np.argmin(d2))
                wave = g['wavelength'][:]
                flux = g['flux'][idx]
            return wave.astype(float), flux.astype(float)
        except Exception as e:
            logger.warning(f'Failed to load spectra cache for {model_key}: {e}')
            return None

    def _sed_xmin(self):
        """Lower wavelength limit (um) of the SED plots: 0.25 um, or bluer
        when fitted bands lie below it (GALEX, HST UV, IUE / TD-1)."""
        w = np.asarray(self.star.wave)[self.star.filter_mask]
        if w.size and w.min() < 0.14:
            return 0.1
        if w.size and w.min() < 0.25:
            return 0.125
        return 0.25

    # --- styling helpers ----------------------------------------------------

    def _rc_params(self):
        """rcParams every plot is drawn under (see ``plot_settings.dat``)."""
        return {
            'font.family': self.fontname,
            'mathtext.fontset': self.mathtext_fontset,
            'legend.fontsize': self.legend_fontsize,
            'legend.title_fontsize': self.legend_fontsize,
            'axes.labelsize': self.fontsize,
            'xtick.labelsize': self.tick_labelsize,
            'ytick.labelsize': self.tick_labelsize,
        }

    def _filter_colors(self, waves):
        """One colour per band, in the order given.

        Bands are ranked by wavelength and spread evenly over the
        ``filter_cmap`` colormap (violet = bluest, red = reddest), so
        neighbouring bands get distinct colours however close their
        wavelengths are. Colours too pale to read on white are darkened.
        """
        waves = np.asarray(waves, dtype=float)
        n = waves.size
        if n == 0:
            return np.empty((0, 4))
        rank = np.empty(n, dtype=int)
        rank[np.argsort(waves, kind='stable')] = np.arange(n)
        pos = 0.04 + 0.92 * rank / max(n - 1, 1)
        cols = np.array(plt.get_cmap(self.filter_cmap)(pos))
        # Relative luminance (WCAG); darken until the contrast on white is
        # at least 2.2:1, i.e. luminance <= 0.43.
        for c in cols:
            for _ in range(30):
                lin = np.where(c[:3] <= 0.04045, c[:3] / 12.92,
                               ((c[:3] + 0.055) / 1.055) ** 2.4)
                if lin @ [0.2126, 0.7152, 0.0722] <= 0.43:
                    break
                c[:3] *= 0.95
        return cols

    def _param_label(self, param, units=True):
        """Mathtext symbol (``units=False``) or axis label of a parameter.

        In a binary fit the primary's parameters carry the subscript 1.
        """
        if is_excess_scatter(param):
            return excess_scatter_label(param)
        b = getattr(self, 'binary', False)
        sym = {
            'teff': r'T_{\rm eff,1}' if b else r'T_{\rm eff}',
            'logg': r'\log g_1' if b else r'\log g',
            'z': r'{\rm [Fe/H]}',
            'dist': r'd',
            'rad': r'R_1' if b else r'R',
            'norm': r'(R/d)^2',
            'Av': r'A_V',
            'inflation': r'\sigma',
            'teff2': r'T_{\rm eff,2}',
            'logg2': r'\log g_2',
            'rad2': r'R_2',
            'mass': r'M_1' if b else r'M',
        }.get(param)
        if sym is None:
            return param
        unit = {'teff': 'K', 'teff2': 'K', 'dist': 'pc',
                'rad': r'R$_{\odot}$', 'rad2': r'R$_{\odot}$',
                'mass': r'M$_{\odot}$', 'Av': 'mag'}.get(param)
        if not units:
            return sym
        return f'${sym}$' + (f' ({unit})' if unit else '')

    def _value_title(self, param, value, upper, lower):
        """Corner-plot title: value with asymmetric errors, quoted to two
        significant figures of the smaller error."""
        sym = self._param_label(param, units=False)
        ep, em = upper - value, value - lower
        scale = 0
        if value != 0 and abs(value) < 1e-3:
            scale = int(np.floor(np.log10(abs(value))))
        v, ep, em = (x / 10 ** scale for x in (value, ep, em))
        err = min(e for e in (ep, em, np.inf) if e > 0)
        nd = 2 if not np.isfinite(err) else \
            int(np.clip(1 - np.floor(np.log10(err)), 0, 8))
        body = f'{v:.{nd}f}^{{+{ep:.{nd}f}}}_{{-{em:.{nd}f}}}'
        if scale:
            body = rf'({body})\times10^{{{scale}}}'
        return f'${sym} = {body}$'

    def _wave_axis(self, ax, xlims, labels=True):
        """Log wavelength axis with plain-number ticks (0.2, 0.5, 1, 2...)
        and unlabelled minor ticks."""
        ax.set_xscale('log', nonpositive='clip')
        if xlims[1] > 10:
            cand = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        else:
            cand = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5]
        ticks = [t for t in cand if xlims[0] <= t <= xlims[1]]
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=50))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        if labels:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        else:
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.set_xlim(xlims)

    # Error-bar style: the measurement error solid with small caps, in front
    # of the fit's total uncertainty, dotted and without caps.
    _CAP = 2.5

    def _error_bars(self, ax, x, y, xerr, yerr_obs, color, yerr_tot=None):
        """A point's bandpass and measurement error (solid, small caps) and,
        with ``yerr_tot``, the total uncertainty of the fit behind them
        (dotted, no caps), all in ``color``."""
        if yerr_tot is not None:
            tot = ax.errorbar(x, y, yerr=yerr_tot, fmt='none', ecolor=color,
                              elinewidth=1.5, capsize=0, zorder=1)
            tot[2][0].set_linestyle(':')
        ax.errorbar(x, y, xerr=xerr, yerr=yerr_obs, fmt='none', ecolor=color,
                    elinewidth=1.5, capsize=self._CAP, capthick=1.5,
                    zorder=1.5)

    def _error_bar_handle(self, ax, label, total=False):
        """Legend entry of an error-bar style (see ``_error_bars``)."""
        h = ax.errorbar([np.nan], [np.nan], yerr=[1.0], fmt='none',
                        ecolor='0.35', elinewidth=1.5,
                        capsize=0 if total else self._CAP, capthick=1.5,
                        label=label)
        if total:
            h[2][0].set_linestyle(':')
        return h

    def _side_legends(self, ax, handles, filters, waves, colors,
                      bottom=None):
        """Legends in a column right of ``ax``: ``handles`` (model curves and
        marker meanings) on top, then the bands, bluest first, each with the
        colour of its points. The band legend may run down beside
        ``bottom`` (the residual panel) and gets more columns if it would
        not fit."""
        fig = ax.figure
        x0, gap = 1.015, 0.02
        top = 1.0
        if handles:
            leg = ax.legend(handles=handles, loc='upper left',
                            bbox_to_anchor=(x0, top), borderaxespad=0.,
                            frameon=False, handletextpad=0.5,
                            labelspacing=0.45, handlelength=1.8)
            ax.add_artist(leg)
            # add_artist clips to the axes, which would leave this legend
            # out of the bbox_inches='tight' crop.
            leg.set_clip_on(False)
            box = leg.get_window_extent()
            top = ax.transAxes.inverted().transform((0, box.y0))[1] - gap
        order = np.argsort(np.asarray(waves, dtype=float), kind='stable')
        ms = np.sqrt(self.scatter_size)
        fhandles = [Line2D([], [], ls='', marker=self.marker, ms=ms,
                           mfc=colors[i], mec='black', mew=0.6,
                           label=filter_label(filters[i]))
                    for i in order]
        # Rows that fit between ``top`` and the bottom of the lowest panel.
        low = (bottom or ax).get_position().y0 * fig.bbox.height
        avail = ax.transAxes.transform((0, top))[1] - low
        row = self.legend_fontsize * (1 + 0.45) * fig.dpi / 72
        nrow = max(int(avail // row) - 1, 1)
        ncol = int(np.ceil(len(fhandles) / nrow))
        ax.legend(handles=fhandles, loc='upper left',
                  bbox_to_anchor=(x0, top), borderaxespad=0., ncol=ncol,
                  title='Filter', frameon=False, handletextpad=0.3,
                  labelspacing=0.45, columnspacing=1.0, alignment='left')

    def _get_raw_spectrum(self, model_key, teff, logg, z):
        """Fetch broadened spectrum for given model and stellar params.

        Returns (wave, flux) trimmed to the standard wavelength range and
        instrumentally broadened where applicable, but NOT extincted or
        normalized.  Returns None on failure.
        """
        orig_theta = self.theta.copy()
        self.theta[0], self.theta[1], self.theta[2] = teff, logg, z
        try:
            cached = self._load_from_cache(model_key)
            if cached is not None:
                return cached

            if self.moddir is None:
                return None

            if model_key == 'phoenix':
                if not hasattr(self, '_phoenix_wave'):
                    self._phoenix_wave = fits.open(
                        self.moddir + self.__wav_file
                    )[0].data * u.angstrom.to(u.um)
                wave = self._phoenix_wave
                mask = (0.125 < wave) & (wave < 4.629296073126975)
                flux = self.fetch_Phoenix()
                new_w = wave[mask]
                new_ww = np.linspace(new_w[0], new_w[-1], len(new_w))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_ww, flux, 1500,
                    edgeHandling="firstlast", fullout=True, maxsig=8
                )
                return new_w, brf[mask]

            elif model_key in ('btsettl', 'btnextgen', 'btcond'):
                fetch = {
                    'btsettl': self.fetch_btsettl,
                    'btnextgen': self.fetch_btnextgen,
                    'btcond': self.fetch_btcond,
                }[model_key]
                wave, flux = fetch()
                mask = (0.125 < wave) & (wave < 4.629296073126975)
                wave, flux = wave[mask], flux[mask]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast", fullout=True, maxsig=8
                )
                return wave, brf

            elif model_key in ('ck04', 'kurucz', 'coelho'):
                fetch = {
                    'ck04': self.fetch_ck04,
                    'kurucz': self.fetch_kurucz,
                    'coelho': self.fetch_coelho,
                }[model_key]
                wave, flux = fetch()
                lo = 0.15 if model_key in ('kurucz', 'coelho') else 0.125
                mask = (lo < wave) & (wave < 4.629296073126975)
                return wave[mask], flux[mask]

        except Exception as e:
            logger.warning(
                f'Failed to get spectrum for {model_key} '
                f'(Teff={teff}, logg={logg}, z={z}): {e}'
            )
            return None
        finally:
            self.theta[:] = orig_theta

    def _binary_SED(self, ax):
        """Model spectrum of a binary fit: each component (in
        ``model_color_primary`` / ``model_color_secondary``, with opacity
        ``model_component_alpha``) and their sum (``model_color``), all solid,
        extincted and distance-scaled."""
        Rv = 3.1
        th = self.theta
        teff, logg, z, dist, rad, Av, teff2, logg2, rad2 = th[:9]
        dist_sr = dist * u.pc.to(u.solRad)
        comps = []
        for grid, t, g, r in ((self.grid, teff, logg, rad),
                              (self.grid2, teff2, logg2, rad2)):
            sp = self._get_raw_spectrum(grid, t, g, z)
            if sp is None:
                logger.warning(
                    f"No spectrum of grid '{grid}' for the SED plot (spectra "
                    f"cache or ARIADNE_MODELS); drawing the other component "
                    f"only.")
                comps.append(None)
                continue
            wave, flux = sp
            flux = apply(self.av_law(wave * 1e4, Av, Rv), flux.copy())
            comps.append((wave, flux * (r / dist_sr) ** 2))
        styles = ((self.model_color_primary,
                   f'Primary: {_grid_label(self.grid)}, '
                   rf'$T_{{\rm eff}}$ = {teff:.0f} K'),
                  (self.model_color_secondary,
                   f'Secondary: {_grid_label(self.grid2)}, '
                   rf'$T_{{\rm eff}}$ = {teff2:.0f} K'))
        for comp, (col, lab) in zip(comps, styles):
            if comp is not None:
                w, f = comp
                ax.plot(w, f * w, lw=1.2, ls='-', color=col,
                        alpha=self.model_component_alpha, zorder=0,
                        label=lab)
        if all(c is not None for c in comps):
            (w1, f1), (w2, f2) = comps
            ftot = f1 + np.interp(w1, w2, f2, left=np.nan, right=np.nan)
            ax.plot(w1, ftot * w1, lw=1.25, color=self.model_color, zorder=0,
                    label='Sum')
            if self.save:
                np.savetxt(f'{self.out_folder}/SED.dat',
                           np.vstack((w1, ftot * w1)).T, fmt='%s',
                           header='wavelength(mu m) wave*flux(erg cm-2 s-2)')

    def _grid_axes(self, grid):
        """Unique Teff, log g and [Fe/H] node values of ``grid``."""
        self.star.load_grid(grid)
        return (np.unique(self.star.teff), np.unique(self.star.logg),
                np.unique(self.star.z))

    def _plot_bma_gray_spectra(self, ax, n_samples=100):
        """Draw gray sampled model spectra weighted by BMA weights."""
        rng = np.random.default_rng()
        Rv = 3.1

        for model_name, weight in self.out['weights'].items():
            if weight < 1e-4:
                continue
            n_model = max(1, round(weight * n_samples))
            samples = self.out['originals'][model_name]
            n_avail = len(samples['teff'])
            indices = rng.choice(n_avail, size=n_model, replace=True)

            if getattr(self, 'binary', False):
                self._binary_gray_draws(ax, model_name, samples, indices, Rv)
                continue

            self.star.load_grid(model_name)

            # Snap samples to grid points and deduplicate
            grouped = {}  # (teff_grid, logg_grid, z_grid) -> [(norm, Av)]
            unique_teff = np.unique(self.star.teff)
            unique_logg = np.unique(self.star.logg)
            unique_z = np.unique(self.star.z)

            for idx in indices:
                t_snap = unique_teff[
                    np.argmin(np.abs(unique_teff - samples['teff'][idx]))]
                g_snap = unique_logg[
                    np.argmin(np.abs(unique_logg - samples['logg'][idx]))]
                z_snap = unique_z[
                    np.argmin(np.abs(unique_z - samples['z'][idx]))]

                if not self.norm:
                    rad = samples['rad'][idx]
                    dist = samples['dist'][idx] * u.pc.to(u.solRad)
                    norm_val = (rad / dist) ** 2
                    Av_val = samples['Av'][idx]
                else:
                    norm_val = samples['norm'][idx]
                    Av_val = samples['Av'][idx]

                key = (t_snap, g_snap, z_snap)
                grouped.setdefault(key, []).append((norm_val, Av_val))

            for (t, g, z_val), param_list in grouped.items():
                result = self._get_raw_spectrum(model_name, t, g, z_val)
                if result is None:
                    continue
                wave, flux_raw = result
                for norm_val, Av in param_list:
                    ext = self.av_law(wave * 1e4, Av, Rv)
                    flx = apply(ext, flux_raw.copy()) * norm_val * wave
                    ax.plot(wave, flx, color='gray', alpha=0.15,
                            lw=0.5, zorder=-1)

        # Restore winning model's grid
        self.star.load_grid(self.grid)

    def _binary_gray_draws(self, ax, pair, samples, indices, Rv):
        """Gray posterior-draw spectra of one binary grid pair: both
        components at their nearest grid nodes, summed per draw."""
        g1, g2 = split_grid_name(pair)
        g2 = g2 or g1
        axes = {g: self._grid_axes(g) for g in dict.fromkeys((g1, g2))}
        spectra = {}

        def _snap(grid, t, g, z):
            at, ag, az = axes[grid]
            key = (grid, at[np.argmin(np.abs(at - t))],
                   ag[np.argmin(np.abs(ag - g))],
                   az[np.argmin(np.abs(az - z))])
            if key not in spectra:
                spectra[key] = self._get_raw_spectrum(*key)
            return spectra[key]

        def _val(k, i):
            v = samples[k]
            return v[i] if np.ndim(v) else v

        for idx in indices:
            z = _val('z', idx)
            s1 = _snap(g1, _val('teff', idx), _val('logg', idx), z)
            s2 = _snap(g2, _val('teff2', idx), _val('logg2', idx), z)
            if s1 is None or s2 is None:
                continue
            d = _val('dist', idx) * u.pc.to(u.solRad)
            Av = _val('Av', idx)
            w1, f1 = s1
            w2, f2 = s2
            ftot = (f1 * (_val('rad', idx) / d) ** 2
                    + np.interp(w1, w2, f2, left=np.nan, right=np.nan)
                    * (_val('rad2', idx) / d) ** 2)
            ftot = apply(self.av_law(w1 * 1e4, Av, Rv), ftot)
            ax.plot(w1, ftot * w1, color='gray', alpha=0.15, lw=0.5,
                    zorder=-1)

    @_styled
    def plot_SED_no_model(self, s=None):
        """Plot raw photometry."""
        if self.star is None:
            self.star = s
        self.__extract_info()
        # Get plot ylims.
        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        f, ax = plt.subplots(figsize=self.figsize)

        # Model plot
        used_f = self.star.filter_names[self.star.filter_mask]
        # Bands flagged by the blackbody QC are outlined in red so outliers
        # can be eyeballed against the BB fit drawn below.
        flagged = set(getattr(self.star, '_qc_bb_flagged_bands', None) or [])

        # One colour per band, violet (bluest) to red (reddest); the filter
        # legend beside the plot names them.
        point_colors = self._filter_colors(self.wave)

        for c, w, fl, fe, bp, fi in zip(
                point_colors,
                self.wave, self.flux, self.flux_er,
                self.bandpass, used_f):
            self._error_bars(ax, w, fl * w, bp, fe * w, c)

            ax.scatter(w, fl * w,
                       edgecolors='black',
                       linewidths=0.6,
                       marker=self.marker,
                       c=[c],
                       s=self.scatter_size,
                       alpha=self.scatter_alpha, zorder=3)

            # QC-flagged bands get a hollow red halo drawn on top, which stays
            # visible regardless of the point's wavelength colour (the colormap
            # is itself red at long wavelengths).
            if fi in flagged:
                ax.scatter(w, fl * w,
                           edgecolors='red', facecolors='none',
                           linewidths=1.8, marker=self.marker,
                           s=self.scatter_size * 2.4, zorder=4)

        # Blackbody fit from the photometry QC, log-scaled to the non-flagged
        # bands so it overlays the data. Lets the user judge visually whether a
        # red-outlined band genuinely departs from the stellar continuum.
        bb_T = getattr(self.star, '_qc_bb_teff', None)
        if bb_T:
            h_, c_, k_ = 6.62607015e-34, 2.99792458e8, 1.380649e-23

            def _planck(lam_um_arr):
                lam_m = np.asarray(lam_um_arr) * 1e-6
                x = h_ * c_ / (lam_m * k_ * bb_T)
                return 1.0 / (lam_m ** 5 * (np.exp(np.clip(x, 0, 500)) - 1.0))

            wave_um = np.asarray(self.wave)
            F_lam = np.asarray(self.flux)
            trusted = np.array([fi not in flagged for fi in used_f])
            if trusted.sum() < 2:
                trusted = np.ones(len(used_f), dtype=bool)
            B_bands = _planck(wave_um)
            log_scale = np.median(np.log10(F_lam[trusted])
                                  - np.log10(B_bands[trusted]))
            scale = 10.0 ** log_scale
            lam_curve = np.logspace(np.log10(max(wave_um.min() * 0.5, 0.1)),
                                    np.log10(wave_um.max() * 2.0), 400)
            F_curve = _planck(lam_curve) * scale
            ax.plot(lam_curve, F_curve * lam_curve,
                    color='0.4', lw=1.0, ls='--', zorder=1,
                    label=rf'Blackbody, $T$ = {bb_T:.0f} K')

        ax.set_ylim([ymin * .8, ymax * 1.25])
        ax.set_yscale('log', nonpositive='clip')
        ax.set_ylabel(_FLUX_LABEL, fontsize=self.fontsize)
        ax.set_xlabel(_WAVE_LABEL, fontsize=self.fontsize)
        ax.tick_params(axis='both', which='major',
                       labelsize=self.tick_labelsize)
        # Same wavelength axis as the model SED plot.
        self._wave_axis(ax, [self._sed_xmin(), 6])

        handles, _ = ax.get_legend_handles_labels()
        if flagged & set(used_f):
            handles.append(Line2D([], [], ls='', marker=self.marker,
                                  ms=np.sqrt(self.scatter_size * 2.4),
                                  mfc='none', mec='red', mew=1.8,
                                  label='Flagged by QC'))
        self._side_legends(ax, handles, used_f, self.wave, point_colors)

        if self.pdf:
            plt.savefig(self.out_folder + '/SED_no_model.pdf',
                        bbox_inches='tight')
        if self.png:
            plt.savefig(self.out_folder + '/SED_no_model.png',
                        bbox_inches='tight')
        pass

    @_styled
    def plot_SED(self):
        """Create the plot of the SED."""
        if self.moddir is None and not spectra_cache:
            logger.warning('Models directory not provided and no spectra cache set, skipping SED plot')
            return
        print('Plotting SED')
        # Get plot ylims.
        ymin = (self.flux * self.wave).min()
        ymax = (self.flux * self.wave).max()

        if self.irx:
            ymin = (self.irx_model * self.irx_wave).min()

        n_filt = self.star.used_filters.sum()
        n_pars = int(len(self.theta) - n_filt)

        # Get models residuals
        mask = self.star.filter_mask
        mags = self.star.mags[mask]
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        filters = self.star.filter_names[mask]
        wave = self.star.wave[mask]

        # The excess scatter s of the plotted parameter vector, magnitudes ->
        # flux, in a copy: self.theta keeps magnitudes, so a second call (and
        # plot_like's trace-plot truths) see the same values.
        theta = np.array(self.theta, dtype=float)
        for i, th in enumerate(theta[n_pars:]):
            mag = mags[i]
            filt = filters[i]
            _, er = mag_to_flux(
                mag, th, filt,
                (getattr(self.star, 'mag_system', None) or {}).get(filt))
            theta[n_pars + i] = er

        # errors: sigma_tot = sqrt(sigma_obs^2 + s^2), the uncertainty the fit
        # used; residuals are in units of it.
        residuals, errors = get_residuals(
            theta, flxs, errs, wave, filters, self.interpolator, self.norm,
            self.av_law, binary=self.binary)

        norm_res = residuals / errors

        # Create plot layout

        f = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

        ax = f.add_subplot(gs[0])
        ax_r = f.add_subplot(gs[1])

        if self.bma:
            self._plot_bma_gray_spectra(ax)
        n_lines = len(ax.lines)
        self.SED(ax)
        if not self.binary:
            for ln in ax.lines[n_lines:]:
                ln.set_label(f'{_grid_label(self.grid)}, '
                             rf'$T_{{\rm eff}}$ = {self.theta[0]:.0f} K')

        # Model plot: observed fluxes as circles coloured by band (see the
        # filter legend), synthetic model fluxes as open diamonds. Each point
        # has its measurement error (solid) in front of the total uncertainty
        # of the fit (dotted). The axis is lambda F_lambda, so the F_lambda
        # errors are scaled by lambda too.
        colors = self._filter_colors(self.wave)
        for w, fl, bp, eo, et, c in zip(self.wave, self.flux, self.bandpass,
                                        self.flux_er, errors, colors):
            self._error_bars(ax, w, fl * w, bp, eo * w, c, yerr_tot=et * w)
        ax.scatter(self.wave, self.flux * self.wave,
                   edgecolors='black', linewidths=0.6, marker=self.marker,
                   c=colors, s=self.scatter_size, zorder=2,
                   alpha=self.scatter_alpha)
        ax.scatter(self.wave, self.model * self.wave,
                   marker=self.marker_model,
                   edgecolors=self.marker_colors_model,
                   s=self.scatter_size * 1.6,
                   facecolor='none', zorder=3, lw=2)

        if self.irx:
            ax.errorbar(self.irx_wave, self.irx_flux * self.irx_wave,
                        xerr=self.irx_bandpass,
                        yerr=self.irx_flux_er * self.irx_wave,
                        fmt=',', ecolor=self.irx_error_color, zorder=0,
                        marker=None)
            ax.scatter(self.irx_wave, self.irx_flux * self.irx_wave,
                       edgecolors='black', marker=self.marker,
                       c=self.marker_colors_irx, s=self.scatter_size, zorder=1,
                       alpha=self.scatter_alpha)
            ax.scatter(self.irx_wave, self.irx_model * self.irx_wave,
                       marker=self.marker_model,
                       edgecolors=self.marker_colors_model_irx,
                       s=self.scatter_size, facecolor='none', zorder=3, lw=3)

        # Residual plot
        ax_r.axhline(y=0, lw=2, ls='--', c='k', alpha=.7)

        # In units of sigma_tot the total uncertainty is +-1 and the
        # measurement error sigma_obs / sigma_tot.
        for w, r, bp, eo, et, c in zip(self.wave, norm_res, self.bandpass,
                                       self.flux_er, errors, colors):
            self._error_bars(ax_r, w, r, bp, eo / et, c, yerr_tot=1.0)
        ax_r.scatter(self.wave, norm_res, zorder=3,
                     edgecolors='black', linewidths=0.6, marker=self.marker,
                     c=colors, s=self.scatter_size,
                     alpha=self.scatter_alpha)
        # ax_r.scatter(self.wave, norm_res,
        #              marker=self.marker_model,
        #              edgecolors=self.marker_colors_model, s=self.scatter_size,
        #              facecolor='none', lw=3, zorder=10)

        if self.irx:
            irx_res = (self.irx_flux - self.irx_model) / self.irx_flux_er
            ax_r.errorbar(self.irx_wave, np.zeros(self.irx_wave.shape[0]),
                          xerr=self.irx_bandpass, yerr=self.irx_flux_er,
                          fmt=',', ecolor=self.irx_error_color, marker=None)
            ax_r.scatter(self.irx_wave, np.zeros(self.irx_wave.shape[0]),
                         edgecolors='black', marker=self.marker,
                         c=self.marker_colors_irx, s=self.scatter_size,
                         alpha=self.scatter_alpha)
            ax_r.scatter(self.irx_wave, irx_res,
                         marker=self.marker_model,
                         edgecolors=self.marker_colors_model_irx,
                         s=self.scatter_size, facecolor='none', lw=3, zorder=10)

        # Formatting
        res_std = norm_res.std()
        ax.set_ylim([ymin * 0.6, ymax * 1.5])
        # +-5 standard deviations of the residuals, widened to show them all
        # with their +-1 sigma_tot bars.
        res_lim = max(5 * res_std, 1.15 * np.max(np.abs(norm_res) + 1))
        ax_r.set_ylim([-res_lim, res_lim])
        ax.set_yscale('log', nonpositive='clip')
        ax_r.set_xlabel(_WAVE_LABEL, fontsize=self.fontsize)
        ax.set_ylabel(_FLUX_LABEL, fontsize=self.fontsize)
        ax_r.set_ylabel('Residuals\n' + r'($\sigma_{\rm tot}$)',
                        fontsize=self.tick_labelsize)
        # The unit, spelt out under the panel's right end (the y label has no
        # room for it).
        ax_r.annotate(_SIGMA_TOT_LABEL, xy=(1, 0), xycoords='axes fraction',
                      xytext=(0, -0.62 * self.tick_labelsize - 12),
                      textcoords='offset points', ha='right', va='top',
                      fontsize=self.legend_fontsize)
        for a in (ax, ax_r):
            a.tick_params(axis='both', which='major',
                          labelsize=self.tick_labelsize)
        ax_r.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,
                                                        symmetric=True))
        ax_r.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        xlims = [self._sed_xmin(), 250 if self.irx else 6]
        self._wave_axis(ax, xlims, labels=False)
        self._wave_axis(ax_r, xlims)

        # Legends: model curves and marker meanings in the plot, the bands
        # beside it.
        handles, _ = ax.get_legend_handles_labels()
        if self.bma:
            handles.append(Line2D([], [], color='gray', alpha=0.5, lw=1,
                                  label='Posterior draws'))
        ms = np.sqrt(self.scatter_size)
        handles += [
            Line2D([], [], ls='', marker=self.marker, ms=ms, mfc='0.8',
                   mec='black', mew=0.6, label='Observed'),
            self._error_bar_handle(ax, r'$\pm\sigma_{\rm obs}$ (measured)'),
            self._error_bar_handle(
                ax, r'$\pm\sigma_{\rm tot}$ (with fitted excess scatter $s$)',
                total=True),
            Line2D([], [], ls='', marker=self.marker_model, ms=ms * 1.1,
                   mfc='none', mec=self.marker_colors_model, mew=2,
                   label='Model (synthetic photometry)'),
        ]
        if self.irx:
            handles += [
                Line2D([], [], ls='', marker=self.marker, ms=ms,
                       mfc=self.marker_colors_irx, mec='black', mew=0.6,
                       label='Observed, IR excess'),
                Line2D([], [], ls='', marker=self.marker_model, ms=ms * 1.1,
                       mfc='none', mec=self.marker_colors_model_irx, mew=2,
                       label='Model, IR excess'),
            ]
        self._side_legends(ax, handles, filters, self.wave, colors,
                           bottom=ax_r)

        if self.pdf:
            plt.savefig(f'{self.out_folder}/SED.pdf', bbox_inches='tight')
        if self.png:
            plt.savefig(f'{self.out_folder}/SED.png', bbox_inches='tight')
        if self.save:
            data = np.vstack((self.wave, self.model * self.wave)).T
            np.savetxt(f'{self.out_folder}/synthetic.dat', data, fmt='%s',
                       header='wavelength(mu m) wave*flux(erg cm-2 s-2)')
        pass

    def SED(self, ax):
        """Plot the SED model."""
        if getattr(self, 'binary', False):
            self._binary_SED(ax)
            return
        # Only these grids ship full spectra. BOSZ and TLUSTY have them only
        # in a cache extended with tools/extend_spectra_cache.py (the Zenodo
        # cache lacks them) and SPHINX has none, so for those the continuous
        # model line can't always be drawn. Degrade gracefully: plot_SED
        # still shows the photometry and the synthetic-flux markers; we just
        # skip (and explain) the model curve. (Also guards the if/elif chain
        # below, which has no else and would otherwise hit undefined
        # wave/flux in the save branch.)
        _spectra_grids = ('phoenix', 'btsettl', 'btnextgen', 'btcond',
                          'ck04', 'kurucz', 'coelho')
        _cache_only_grids = ('bosz', 'tlusty')
        cached_only = None
        if self.grid.lower() in _cache_only_grids:
            cached_only = self._load_from_cache(self.grid.lower())
            if cached_only is None:
                logger.warning(
                    f"Grid '{self.grid}' has no spectra in the spectra cache "
                    f"(add them with tools/extend_spectra_cache.py); plotting "
                    f"the SED without the continuous model spectrum "
                    f"(photometry and synthetic model fluxes are still "
                    f"shown).")
                return
        elif self.grid.lower() not in _spectra_grids:
            logger.warning(
                f"Grid '{self.grid}' has no spectra in the cache; plotting the "
                f"SED without the continuous model spectrum (photometry and "
                f"synthetic model fluxes are still shown).")
            return
        Rv = 3.1  # For extinction.
        if not self.norm:
            rad = self.theta[4]
            dist = self.theta[3] * u.pc.to(u.solRad)
            norm = (rad / dist) ** 2
            Av = self.theta[5]
        else:
            norm = self.theta[3]
            Av = self.theta[4]

        # SED plot.
        if self.grid == 'phoenix':
            cached = self._load_from_cache('phoenix')
            if cached is not None and not self.irx:
                new_w, brf = cached
            else:
                wave = fits.open(self.moddir + self.__wav_file)[0].data
                wave *= u.angstrom.to(u.um)
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                flux = self.fetch_Phoenix()
                new_w = wave[lower_lim * upper_lim]
                new_ww = np.linspace(new_w[0], new_w[-1], len(new_w))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_ww, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
                brf = brf[lower_lim * upper_lim]
            ext = self.av_law(new_w * 1e4, Av, Rv)
            brf = apply(ext, brf)
            flx = brf * norm * new_w
            ax.plot(new_w, flx, lw=1.25, color=self.model_color,
                    zorder=0)

        elif self.grid == 'btsettl':
            cached = self._load_from_cache('btsettl')
            if cached is not None and not self.irx:
                wave, brf = cached
            else:
                wave, flux = self.fetch_btsettl()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
            ext = self.av_law(wave * 1e4, Av, Rv)
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'btnextgen':
            cached = self._load_from_cache('btnextgen')
            if cached is not None and not self.irx:
                wave, brf = cached
            else:
                wave, flux = self.fetch_btnextgen()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
            ext = self.av_law(wave * 1e4, Av, Rv)
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'btcond':
            cached = self._load_from_cache('btcond')
            if cached is not None and not self.irx:
                wave, brf = cached
            else:
                wave, flux = self.fetch_btcond()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
                new_w = np.linspace(wave[0], wave[-1], len(wave))
                brf, _ = pyasl.instrBroadGaussFast(
                    new_w, flux, 1500,
                    edgeHandling="firstlast",
                    fullout=True, maxsig=8
                )
            ext = self.av_law(wave * 1e4, Av, Rv)
            flx = apply(ext, brf)
            flx *= wave * norm
            ax.plot(wave, flx, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'ck04':
            cached = self._load_from_cache('ck04')
            if cached is not None and not self.irx:
                wave, flux = cached
            else:
                wave, flux = self.fetch_ck04()
                lower_lim = 0.125 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'kurucz':
            cached = self._load_from_cache('kurucz')
            if cached is not None and not self.irx:
                wave, flux = cached
            else:
                wave, flux = self.fetch_kurucz()
                lower_lim = 0.15 < wave
                upper_lim = wave < 4.629296073126975
                if self.irx:
                    upper_lim = wave < wave[-1]
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)

        elif cached_only is not None:
            # BOSZ / TLUSTY: spectra from the extended cache only.
            wave, flux = cached_only
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)

        elif self.grid == 'coelho':
            cached = self._load_from_cache('coelho')
            if cached is not None:
                wave, flux = cached
            else:
                wave, flux = self.fetch_coelho()
                lower_lim = 0.15 < wave
                upper_lim = wave < 4.629296073126975
                wave = wave[lower_lim * upper_lim]
                flux = flux[lower_lim * upper_lim]
            ext = self.av_law(wave * 1e4, Av, Rv)
            flux = apply(ext, flux)
            flux *= wave * norm
            ax.plot(wave, flux, lw=1.25, color=self.model_color, zorder=0)
        if self.save:
            data = np.vstack((wave, flux)).T
            np.savetxt(f'{self.out_folder}/SED.dat', data, fmt='%s',
                       header='wavelength(mu m) wave*flux(erg cm-2 s-2)')
        pass

    @_styled
    def plot_trace(self):
        """Plot SED chains."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                f, ax = plt.subplots(figsize=(12, 4))
                ax.step(range(len(samples[param])), samples[param],
                        color='k', alpha=0.8)
                ax.set_ylabel(self._param_label(param),
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.set_xlabel('Steps',
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                best = self.out['best_fit'][param]
                # ax.axhline(np.median(samples[param]), color='red', lw=2)
                ax.axhline(best, color='red', lw=2)
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig(self.chain_out + '/' + param +
                            '.png', bbox_inches='tight')
        plt.close('all')
        pass

    @_styled
    def plot_like(self):
        """Plot Likelihoods."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                f, ax = plt.subplots(figsize=(12, 4))
                ax.scatter(samples[param], samples['loglike'], alpha=0.5, s=40)
                best = self.out['best_fit'][param]
                # ax.axvline(np.median(samples[param]), color='red', lw=1.5)
                ax.axvline(best, color='red', lw=1.5)
                ax.set_ylabel('log likelihood',
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.set_xlabel(self._param_label(param),
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig(self.like_out + '/' + param + '.png',
                            bbox_inches='tight')
        plt.close('all')
        if self.engine == 'dynesty':
            fig, axes = dyplot.traceplot(
                self.out['dynesty'],
                truths=self.theta,
                show_titles=True, trace_cmap='plasma',
            )
            plt.savefig(self.like_out + '/dynesty_trace.png')
        pass

    @_styled
    def plot_post(self):
        """Plot posteriors."""
        samples = self.out['posterior_samples']
        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                f, ax = plt.subplots(figsize=(12, 4))
                ax.scatter(samples[param], samples['posteriors'], alpha=0.5,
                           s=40)
                best = self.out['best_fit'][param]
                # ax.axvline(np.median(samples[param]), color='red', lw=1.5)
                ax.axvline(best, color='red', lw=1.5)
                ax.set_ylabel('log posterior',
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.set_xlabel(self._param_label(param),
                              fontsize=self.fontsize,
                              fontname=self.fontname
                              )
                ax.tick_params(
                    axis='both', which='major',
                    labelsize=self.tick_labelsize
                )
                plt.savefig(self.post_out + '/' + param + '.png',
                            bbox_inches='tight')
        plt.close('all')
        pass

    def plot_hist(self):
        pass

    @_styled
    def plot_bma_hist(self):
        """Plot histograms."""
        print('Plotting BMA histograms.')
        colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown'
        ]
        models = [key for key in self.out['originals'].keys()]
        for i, param in enumerate(self.order):
            if is_excess_scatter(param):
                continue
            if not self.coordinator[i]:
                f1, ax1 = plt.subplots(figsize=(12, 6))
                f2, ax2 = plt.subplots(figsize=(12, 6))
                for j, m in enumerate(models):
                    # Get samples
                    samp = self.out['originals'][m][param]
                    # Plot sample histogram
                    label = '{}, $P$ = {:.3f}'.format(
                        _grid_label(m), self.out['weights'][m])
                    # Normal
                    n, bins1, patches = ax1.hist(samp, alpha=.3, bins=20,
                                                 label=label, density=True,
                                                 color=colors[j % len(colors)])
                    # Weighted
                    n, bins2, patches = ax2.hist(
                        samp, alpha=.3, bins=20, label=label,
                        weights=[self.out['weights'][m]] * len(samp)
                    )
                    # Fit a KDE to data
                    kde = gaussian_kde(samp)
                    # Estimate amplitude of the weighted distributions
                    mu, sig = norm.fit(samp)
                    try:
                        bc = bins2[:-1] + np.diff(bins2)
                        popt, pcov = curve_fit(norm_fit, xdata=bc, ydata=n,
                                               p0=[mu, sig, n.max()],
                                               maxfev=50000)
                    except RuntimeError:
                        popt = (mu, sig, n.max())
                    xx1 = np.linspace(bins1[0], bins1[-1], 1000)
                    xx2 = np.linspace(bins2[0], bins2[-1], 1000)
                    # Plot best fit
                    ax1.plot(xx1, kde(xx2), lw=2, alpha=1,
                             color=colors[j % len(colors)])
                    ax2.plot(xx1, kde(xx2) * popt[2], lw=2, alpha=1,
                             color=colors[j % len(colors)])
                # The same but for the weighted samples
                n, bins, patches = ax1.hist(
                    self.out['weighted_samples'][param], alpha=.3,
                    bins=20, label='Weighted sampling', density=True,
                    color='tab:cyan'
                )
                kde = gaussian_kde(self.out['weighted_samples'][param])
                xx = np.linspace(bins[0], bins[-1], 300)
                ax1.plot(xx, kde(xx), color='tab:cyan', lw=2, alpha=1, ls='--')

                # Now ditto for the weighted average
                n, bins, patches = ax1.hist(
                    self.out['weighted_average'][param], alpha=.3,
                    bins=20, label='Weighted average', density=True,
                    color='tab:pink'
                )
                kde = gaussian_kde(self.out['weighted_average'][param])
                xx = np.linspace(bins[0], bins[-1], 300)
                ax1.plot(xx, kde(xx), color='tab:pink', lw=2, alpha=1, ls='-.')
                lab = self._param_label(param)
                # Normal
                ax1.set_ylabel('PDF',
                               fontsize=self.fontsize,
                               fontname=self.fontname
                               )
                # Weighted
                ax2.set_ylabel('N',
                               fontsize=self.fontsize,
                               fontname=self.fontname
                               )
                axes = [ax1, ax2]
                for ax in axes:
                    ax.set_xlabel(lab,
                                  fontsize=self.fontsize,
                                  fontname=self.fontname
                                  )
                    for tick in ax.get_yticklabels():
                        tick.set_fontname(self.fontname)
                    for tick in ax.get_xticklabels():
                        tick.set_fontname(self.fontname)

                    ax.tick_params(
                        axis='both', which='major',
                        labelsize=self.tick_labelsize
                    )
                    ax.legend(loc=0)

                if param == 'z':
                    param = 'Fe_H'
                if self.png:
                    f1.savefig(self.hist_out + '/' + param + '.png',
                               bbox_inches='tight')
                    f2.savefig(self.hist_out + '/weighted_' + param + '.png',
                               bbox_inches='tight')
                if self.pdf:
                    f1.savefig(self.hist_out + '/' + param + '.pdf',
                               bbox_inches='tight')
                    f2.savefig(self.hist_out + '/weighted_' + param + '.pdf',
                               bbox_inches='tight')
                plt.close(f1)
                plt.close(f2)

        if self.bma:
            mist_samp = self.out.get('mist_samples', {})
            # Age hist
            if 'age' in mist_samp:
                samp = mist_samp['age']
                samp = samp[np.isfinite(samp)]
                if len(samp) == 0:
                    pass
                else:
                    f, ax = plt.subplots(figsize=(12, 4))
                    n, bins, patches = ax.hist(
                        samp, alpha=.3, bins=20, label='MIST', density=True
                    )
                    kde = gaussian_kde(samp)
                    xx = np.linspace(bins[0], bins[-1], 10000)
                    ax.plot(xx, kde(xx), color='k', lw=2, alpha=.7)
                    ax.set_ylabel('PDF',
                                  fontsize=self.fontsize,
                                  fontname=self.fontname)
                    ax.set_xlabel('Age (Gyr)',
                                  fontsize=self.fontsize,
                                  fontname=self.fontname)

                    for tick in ax.get_yticklabels():
                        tick.set_fontname(self.fontname)
                    for tick in ax.get_xticklabels():
                        tick.set_fontname(self.fontname)

                    ax.tick_params(
                        axis='both', which='major',
                        labelsize=self.tick_labelsize
                    )
                    plt.legend(loc=0)
                    if self.png:
                        plt.savefig(self.hist_out + '/age.png',
                                    bbox_inches='tight')
                    if self.pdf:
                        plt.savefig(self.hist_out + '/age.pdf',
                                    bbox_inches='tight')
                    plt.close(f)
            # Mass hist
            if 'iso_mass' in mist_samp:
                samp = mist_samp['iso_mass']
                samp = samp[np.isfinite(samp)]
                if len(samp) == 0:
                    pass
                else:
                    f, ax = plt.subplots(figsize=(12, 4))
                    n, bins, patches = ax.hist(
                        samp, alpha=.3, bins=20, label='MIST', density=True
                    )
                    kde = gaussian_kde(samp)
                    xx = np.linspace(bins[0], bins[-1], 10000)
                    ax.plot(xx, kde(xx), color='k', lw=2, alpha=.7)
                    ax.set_ylabel('PDF',
                                  fontsize=self.fontsize,
                                  fontname=self.fontname)
                    ax.set_xlabel(self._param_label('mass'),
                                  fontsize=self.fontsize,
                                  fontname=self.fontname)

                    for tick in ax.get_yticklabels():
                        tick.set_fontname(self.fontname)
                    for tick in ax.get_xticklabels():
                        tick.set_fontname(self.fontname)

                    ax.tick_params(
                        axis='both', which='major',
                        labelsize=self.tick_labelsize
                    )
                    plt.legend(loc=0)
                    if self.png:
                        plt.savefig(self.hist_out + '/iso_mass.png',
                                    bbox_inches='tight')
                    if self.pdf:
                        plt.savefig(self.hist_out + '/iso_mass.pdf',
                                    bbox_inches='tight')
                    plt.close(f)
            # EEP hist
            if 'eep' in mist_samp:
                samp = mist_samp['eep']
                samp = samp[np.isfinite(samp)]
                if len(samp) == 0:
                    pass
                else:
                    f, ax = plt.subplots(figsize=(12, 4))
                    n, bins, patches = ax.hist(
                        samp, alpha=.3, bins=20, label='MIST', density=True
                    )
                    kde = gaussian_kde(samp)
                    xx = np.linspace(bins[0], bins[-1], 10000)
                    ax.plot(xx, kde(xx), color='k', lw=2, alpha=.7)
                    ax.set_ylabel('PDF',
                                  fontsize=self.fontsize,
                                  fontname=self.fontname)
                    ax.set_xlabel('EEP',
                                  fontsize=self.fontsize,
                                  fontname=self.fontname)

                    for tick in ax.get_yticklabels():
                        tick.set_fontname(self.fontname)
                    for tick in ax.get_xticklabels():
                        tick.set_fontname(self.fontname)

                    ax.tick_params(
                        axis='both', which='major',
                        labelsize=self.tick_labelsize
                    )
                    plt.legend(loc=0)
                    if self.png:
                        plt.savefig(self.hist_out + '/EEP.png',
                                    bbox_inches='tight')
                    if self.pdf:
                        plt.savefig(self.hist_out + '/EEP.pdf',
                                    bbox_inches='tight')
                    plt.close(f)

    @_styled
    def plot_bma_HR(self, nsamp):
        """Plot HR diagram for the star."""
        bf_key = f'best_fit_{self.method}'
        if 'age' not in self.out.get(bf_key, {}):
            return
        age = self.out[bf_key]['age']
        if not np.isfinite(age):
            # No MIST age estimate available (e.g. bolometric correction
            # grids could not be loaded), so the HR diagram cannot be made.
            return
        print('Plotting HR diagram')
        # Get necessary info from the star.
        age = self.out[bf_key]['age']
        feh = self.out[f'best_fit_{self.method}']['z']
        teff = np.log10(self.out[f'best_fit_{self.method}']['teff'])
        lum = np.log10(self.out[f'best_fit_{self.method}']['lum'])
        teff_lo, teff_hi = self.out[f'uncertainties_{self.method}']['teff']
        lum_lo, lum_hi = self.out[f'uncertainties_{self.method}']['lum']
        teff_lo = teff_lo / (10 ** teff * np.log(10))
        teff_hi = teff_hi / (10 ** teff * np.log(10))
        lum_lo = lum_lo / (10 ** lum * np.log(10))
        lum_hi = lum_hi / (10 ** lum * np.log(10))
        ages = self.out['mist_samples']['age']
        m = self.method if self.method == 'samples' else 'average'
        fehs = self.out[f'weighted_{m}']['z']

        if feh > 0.5:
            feh = 0.5

        iso_bf = get_isochrone(np.log10(age) + 9, feh)

        logteff = iso_bf['logTeff'].values
        loglum = iso_bf['logL'].values
        mass = iso_bf['mass'].values

        fig, ax = plt.subplots(figsize=self.hr_figsize)

        points = np.array([logteff, loglum]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(mass.min(), mass.max())
        lc = LineCollection(segments, cmap=self.hr_cmap, norm=norm,
                            linewidths=5)

        lc.set_array(mass)
        line = ax.add_collection(lc)
        line.zorder = 1000
        cbar = fig.colorbar(line, ax=ax, pad=0.01)
        cbar.set_label(r'Isochrone mass (M$_{\odot}$)',
                       rotation=270,
                       fontsize=self.fontsize,
                       labelpad=28)

        for i in range(nsamp):
            a = np.log10(choice(ages)) + 9
            z = choice(fehs)
            if z > 0.5:
                z = 0.5
            iso = get_isochrone(a, z)

            logt = iso['logTeff'].values
            logl = iso['logL'].values
            ax.plot(logt, logl, color='gray')

        ax.errorbar(teff, lum, xerr=[[teff_lo], [teff_hi]],
                    yerr=[[lum_lo], [lum_hi]], color=self.hr_color,
                    zorder=1001)
        ax.scatter(teff, lum, s=350, color=self.hr_color, zorder=1002,
                   edgecolors='k', marker=self.hr_marker,
                   label='Primary' if getattr(self, 'binary', False) else None)
        if getattr(self, 'binary', False):
            # The isochrones are the primary's; the secondary is only placed.
            bf = self.out[f'best_fit_{self.method}']
            un = self.out[f'uncertainties_{self.method}']
            t2, l2 = np.log10(bf['teff2']), np.log10(bf['lum2'])
            t2e = np.array(un['teff2']) / (bf['teff2'] * np.log(10))
            l2e = np.array(un['lum2']) / (bf['lum2'] * np.log(10))
            ax.errorbar(t2, l2, xerr=[[t2e[0]], [t2e[1]]],
                        yerr=[[l2e[0]], [l2e[1]]], color='tab:blue',
                        zorder=1001)
            ax.scatter(t2, l2, s=350, color='tab:blue', zorder=1002,
                       edgecolors='k', marker='*', label='Secondary')
            ax.legend(loc='best', markerscale=0.6)

        ax.invert_xaxis()
        ax.set_xlabel(r'$\log\,(T_{\rm eff}/{\rm K})$',
                      fontsize=self.fontsize)
        ax.set_ylabel(r'$\log\,(L/{\rm L}_{\odot})$',
                      fontsize=self.fontsize)
        ax.tick_params(
            axis='both', which='major',
            labelsize=self.tick_labelsize
        )
        cbar.ax.tick_params(labelsize=self.tick_labelsize)

        if self.png:
            plt.savefig(self.out_folder + '/HR_diagram.png',
                        bbox_inches='tight')
        if self.pdf:
            plt.savefig(self.out_folder + '/HR_diagram.pdf',
                        bbox_inches='tight')

    @_styled
    def plot_corner(self):
        """Make corner plot."""
        print('Plotting corner.')
        m = self.method if self.method == 'samples' else 'average'
        samples = self.out.get(f'weighted_{m}',
                               self.out.get('posterior_samples'))
        all_samps = []
        theta_lo = []
        theta_up = []

        for i, o in enumerate(self.order):
            if is_excess_scatter(o):
                self.coordinator[i] = 1

        theta = getattr(self, 'theta_best', self.theta)
        theta = theta[self.coordinator == 0]
        used_params = self.order[self.coordinator == 0]

        for i, param in enumerate(self.order):
            if not self.coordinator[i]:
                if is_excess_scatter(param):
                    continue
                _, lo, up = credibility_interval(samples[param])
                theta_lo.append(lo)
                theta_up.append(up)
                all_samps.append(samples[param])

        corner_samp = np.vstack(all_samps)

        titles = [self._value_title(p, v, up, lo) for p, v, up, lo
                  in zip(used_params, theta, theta_up, theta_lo)]
        labels = [self._param_label(p) for p in used_params]

        fig = corner.corner(
            corner_samp.T,
            plot_contours=True,
            fill_contours=False,
            plot_datapoints=True,
            no_fill_contours=True,
            max_n_ticks=4
        )
        # Wider panels than corner's 2 inches, so the titles fit.
        fig.set_size_inches(fig.get_size_inches() * 1.1)

        axes = np.array(fig.axes).reshape((theta.shape[0], theta.shape[0]))

        for i in range(theta.shape[0]):
            ax = axes[i, i]
            ax.axvline(theta[i], color=self.corner_med_c,
                       linestyle=self.corner_med_style)
            ax.axvline(theta_lo[i], color=self.corner_v_c,
                       linestyle=self.corner_v_style)
            ax.axvline(theta_up[i], color=self.corner_v_c,
                       linestyle=self.corner_v_style)
            t = titles[i]

            ax.set_title(t, fontsize=self.corner_title_fontsize, pad=10)

        for yi in range(theta.shape[0]):
            for xi in range(yi):
                ax = axes[yi, xi]
                if xi == 0:
                    ax.tick_params(axis='y',
                                   labelsize=self.corner_tick_fontsize)
                    ax.set_ylabel(labels[yi],
                                  labelpad=self.corner_labelpad,
                                  fontsize=self.corner_fontsize)
                if yi == theta.shape[0] - 1:
                    ax.tick_params(axis='x',
                                   labelsize=self.corner_tick_fontsize)
                    ax.set_xlabel(labels[xi],
                                  labelpad=self.corner_labelpad,
                                  fontsize=self.corner_fontsize)
                ax.axvline(theta[xi], color=self.corner_med_c,
                           linestyle=self.corner_med_style)
                ax.axhline(theta[yi], color=self.corner_med_c,
                           linestyle=self.corner_med_style)
                ax.plot(theta[xi], theta[yi], self.corner_marker)
        axes[-1, -1].set_xlabel(labels[-1],
                                labelpad=self.corner_labelpad,
                                fontsize=self.corner_fontsize)
        axes[-1, -1].tick_params(axis='x',
                                 labelsize=self.corner_tick_fontsize)

        if self.pdf:
            plt.savefig(f'{self.out_folder}/CORNER.pdf', bbox_inches='tight')
        if self.png:
            plt.savefig(f'{self.out_folder}/CORNER.png', bbox_inches='tight')

    def clean(self):
        """Close opened figures."""
        plt.close('all')

    def fetch_Phoenix(self):
        """Fetch correct Phoenixv2 SED file.

        The directory containing the Phoenix spectra must be called PHOENIXv2
        Within PHOENIXv2 there should be the wavelength file called
        WAVE_PHOENIX-ACES-AGSS-COND-2011.fits and several folders called
        Z[-/+]X.X where X.X are the metallicities (e.g. Z-0.0, Z+1.0, etc)
        """
        # Change hdd to a class variable depending on an env param.
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        selected_SED = self.moddir + 'PHOENIXv2/Z'
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED += metal_add
        selected_SED += '/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 5 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + '0'
        selected_SED += metal_add
        selected_SED += '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        flux = fits.open(selected_SED)[0].data
        flux *= (u.erg / u.s / u.cm ** 2 / u.cm).to(
            u.erg / u.s / u.cm ** 2 / u.um)
        return flux

    def fetch_btsettl(self):
        """Fetch correct BT-Settl SED file.

        The directory containing the BT-Settl spectra must be called BTSettl
        Within BTSettl there should be yet another directory
        called AGSS2009, within BTSettl/AGSS2009 there should be the SED fits
        files with the following naming convention:

        lteTTT-G.G[-/+]Z.Za+0.0.BT-Settl.AGSS2009.fits

        where TTT are the first 3 digits of the effective temperature if it's a
        number over 10000, else it's the first 2 digit prepended by a 0.
        G.G is the log g and Z.Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff]) // 100
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.moddir + 'BTSettl/AGSS2009/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+*'
        gl = glob.glob(selected_SED)
        selected_SED = gl[0]
        tab = Table(fits.open(selected_SED)[1].data)
        flux = np.array(tab['FLUX'].tolist()) * conversion
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        return wave, flux

    def fetch_btnextgen(self):
        """Fetch correct BT-NextGen SED file.

        The directory containing the BT-NextGen spectra must be called
        BTNextGen. Within BTNextGen there should be yet another directory
        called AGSS2009, within BTNextGen/AGSS2009 there should be the SED fits
        files with the following naming convention:

        lteTTT-G.G[-/+]Z.Za+0.0..BT-NextGen.AGSS2009.fits

        where TTT are the first 3 digits of the effective temperature if it's a
        number over 10000, else it's the first 2 digit prepended by a 0.
        G.G is the log g and Z.Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff]) // 100
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.moddir + 'BTNextGen/AGSS2009/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+*'
        gl = glob.glob(selected_SED)
        selected_SED = gl[0]
        tab = Table(fits.open(selected_SED)[1].data)
        flux = np.array(tab['FLUX'].tolist()) * conversion
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        return wave, flux

    def fetch_btcond(self):
        """Fetch correct BT-COND SED file.

        The directory containing the BT-COND spectra must be called
        BTCOND. Within BTCOND there should be yet another directory
        called CIFIST2011, within BTCOND/CIFIST2011 there should be the SED
        fits files with the following naming convention:

        lteTTT-G.G[-/+]Z.Za+0.0..BT-Cond.CIFIST2011.fits

        where TTT are the first 3 digits of the effective temperature if it's a
        number over 10000, else it's the first 2 digit prepended by a 0.
        G.G is the log g and Z.Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff]) // 100
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = str(sel_z)
        if sel_z == 0:
            metal_add = '-0.0'
        if sel_z > 0:
            metal_add = '+' + str(sel_z)
        selected_SED = self.moddir + 'BTCond/CIFIST2011/lte'
        selected_SED += str(sel_teff) if len(str(sel_teff)) == 3 else \
            '0' + str(sel_teff)
        selected_SED += '-' + str(sel_logg) + metal_add + 'a+*'
        gl = glob.glob(selected_SED)
        selected_SED = gl[0]
        tab = Table(fits.open(selected_SED)[1].data)
        flux = np.array(tab['FLUX'].tolist()) * conversion
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        return wave, flux

    def fetch_ck04(self):
        """Fetch correct Castelli-Kurucz 2004 SED file.

        The directory containing the Castelli-Kurucz spectra must be called
        Castelli_Kurucz. Within Castelli_Kurucz there should be a group of
        directories called ck[pm]ZZ where ZZ is the metalicity without the dot.
        Within each directory there are fits files named:

        ck[pm]ZZ_TTTT.fits

        where ZZ is metalicity as previous and TTTT is the effective
        temperature.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = 'm' + str(-sel_z).replace('.', '')
        if sel_z == 0:
            metal_add = 'p00'
        if sel_z > 0:
            metal_add = 'p' + str(sel_z).replace('.', '')
        name = 'ck' + metal_add
        lgg = 'g{:02.0f}'.format(sel_logg * 10)
        selected_SED = self.moddir + 'Castelli_Kurucz/' + name + '/' + name
        selected_SED += '_' + str(sel_teff) + '.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        flux = np.array(tab[lgg].tolist()) * conversion
        return wave, flux

    def fetch_kurucz(self):
        """Fetch correct Kurucz 1993 SED file.

        The directory containing the Kurucz spectra must be called
        Kurucz. Within Kurucz there should be a group of
        directories called k[pm]ZZ where ZZ is the metalicity without the dot.
        Within each directory there are fits files named:

        k[pm]ZZ_TTTT.fits

        where ZZ is metalicity as previous and TTTT is the effective
        temperature.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        metal_add = ''
        if sel_z < 0:
            metal_add = 'm' + str(-sel_z).replace('.', '')
        if sel_z == 0:
            metal_add = 'p00'
        if sel_z > 0:
            metal_add = 'p' + str(sel_z).replace('.', '')
        name = 'k' + metal_add
        lgg = 'g{:02.0f}'.format(sel_logg * 10)
        selected_SED = self.moddir + 'Kurucz/' + name + '/' + name
        selected_SED += '_' + str(sel_teff) + '.fits'
        tab = Table(fits.open(selected_SED)[1].data)
        wave = np.array(tab['WAVELENGTH'].tolist()) * u.angstrom.to(u.um)
        flux = np.array(tab[lgg].tolist()) * conversion
        return wave, flux

    def fetch_coelho(self):
        """Fetch correct Coelho 2014 SED file.

        The directory containing the Coelho spectra must be called
        Coelho14. Within Coelho14 there should be a group of
        files called t[0X]XXXX_g[+-]Y.Y_[mp]ZZp0[14]_sed.fits
        where X is the temperature, Y is the logg and Z the metallicity.
        """
        conversion = (u.erg / u.s / u.cm ** 2 / u.angstrom)
        conversion = conversion.to(u.erg / u.s / u.cm ** 2 / u.um)
        teff = self.theta[0]
        logg = self.theta[1]
        z = self.theta[2]
        select_teff = np.argmin((abs(teff - np.unique(self.star.teff))))
        select_logg = np.argmin((abs(logg - np.unique(self.star.logg))))
        select_z = np.argmin((abs(z - np.unique(self.star.z))))
        sel_teff = int(np.unique(self.star.teff)[select_teff])
        sel_logg = np.unique(self.star.logg)[select_logg]
        sel_z = np.unique(self.star.z)[select_z]
        sel_teff = str(sel_teff) if sel_teff >= 1e5 else '0{}'.format(sel_teff)
        selected_SED = self.moddir + 'Coelho14/t' + sel_teff + '_g'
        sel_logg = '+{:.1f}'.format(sel_logg) if sel_logg > 0 else '-0.5'
        selected_SED += sel_logg
        if sel_z < 0:
            selected_SED += '_m{:02.0f}'.format(-sel_z * 10)
        else:
            selected_SED += '_p{:02.0f}'.format(sel_z * 10)

        selected_SED = glob.glob(selected_SED + 'p0[04]_sed.fits')
        selected_SED = selected_SED[0]
        hdul = fits.open(selected_SED)
        head = hdul[0].header
        data = hdul[0].data
        flux = data * conversion
        CRVAL1 = head['CRVAL1']
        CDEL1 = head['CDELT1']
        wave = 10 ** np.array(
            [CRVAL1 + CDEL1 * i for i in range(data.shape[0])])
        wave *= u.angstrom.to(u.um)
        return wave, flux

    def __read_config(self):
        """Read plotter configuration file."""
        # Defaults for keys that older settings files lack.
        self.mathtext_fontset = 'stix'
        self.legend_fontsize = 16
        self.filter_cmap = 'turbo'
        self.corner_title_fontsize = 18
        self.model_color_primary = '#1f4e79'
        self.model_color_secondary = '#c1432e'
        self.model_component_alpha = 0.8
        if self.settings_dir is None:
            settings = open(filesdir + '/plot_settings.dat', 'r')
        else:
            settings = open(self.settings_dir, 'r')
        for line in settings.readlines():
            if line[0] == '#' or line[0] == '\n':
                continue
            splt = line.split(' ')
            attr = splt[0]
            if attr == 'figsize':
                vals = splt[1].split('\n')[0].split(',')
                val = (int(vals[0]), int(vals[1]))
            elif attr == 'hr_figsize':
                vals = splt[1].split('\n')[0].split(',')
                val = (int(vals[0]), int(vals[1]))
            elif 'alpha' in attr:
                val = float(splt[1].split('\n')[0])
            else:
                try:
                    val = int(splt[1].split('\n')[0])
                except ValueError:
                    val = splt[1].split('\n')[0]
            setattr(self, attr, val)
