"""Main driver of the fitting routine."""

__all__ = ['Fitter', 'dynesty_log_like', 'dynesty_loglike_bma', 'pt_dynesty',
           'pt_multinest']

import copy
import logging
import os
import pickle
import time
import warnings

logger = logging.getLogger(__name__)
import multiprocessing as _mp
from tqdm import tqdm

import extinction
import astropy.units as u
import pandas as pd
import numpy as np
import scipy.stats as st
from numpy.random import choice
from astropy.constants import sigma_sb
from isochrones.interp import DFInterpolator
from termcolor import colored

from .config import (filesdir, gridsdir, priorsdir, filter_names, colors,
                     iso_mask, iso_bands, grid_wave_coverage)
from .error import *
from .isochrone import estimate
from .phot_utils import *
from .sed_library import *
from .sed_library import _make_transform
from .utils import *

try:
    import dynesty
    from dynesty.utils import resample_equal

    bma_flag = True
except ModuleNotFoundError:
    wrn = 'Dynesty package not found.\n'
    wrn += 'Install dynesty with `pip install dynesty`'
    warnings.warn(wrn)
    bma_flag = False
try:
    # pymultinest calls sys.exit() (SystemExit) when libmultinest, or a
    # library it links against (e.g. MKL), cannot be loaded; that must not
    # take the whole package down, since dynesty does not need it.
    import pymultinest
except (ImportError, OSError, SystemExit):
    pymultinest = None
    warnings.warn(
        '(py)MultiNest installation (or libmultinest.so/.dylib) not '
        'detected; the multinest engine is unavailable.'
    )

# Every Pool here hands its workers the photometry, priors and interpolator
# through module globals set just before the Pool is created, so the workers
# must be forked. Use a fork context rather than the process-wide start method:
# set_start_method(force=True) at import time can be undone by any later
# import or by user code, and spawned workers then die with
# "NameError: name 'flux' is not defined" (issue #58, PR #81). Passing the
# state through logl_args/ptform_args instead would re-pickle the model grid
# with every pool batch.
_MP_CTX = _mp.get_context('fork')
Pool = _MP_CTX.Pool


_BOUND_CHECKED = ('teff', 'logg', 'z', 'dist', 'rad', 'norm', 'Av')


def _missing_bands(interp, bands):
    """Bands in ``bands`` that the band-grid interpolator ``interp`` has no
    model flux for: no column, or a column that is NaN at every node.

    Not every grid carries every ARIADNE band (e.g. only BOSZ and TLUSTY have
    the IUE and TD-1 ultraviolet bands), and the likelihood looks bands up
    by name, so a band the grid lacks must be caught before fitting.
    """
    cols = interp.column_index
    out = []
    for b in bands:
        j = cols.get(str(b))
        if j is None or not np.isfinite(interp.grid[..., j]).any():
            out.append(str(b))
    return out


_ALL_GRIDS = ('phoenix', 'btsettl', 'btnextgen', 'btcond', 'ck04', 'kurucz',
              'coelho', 'bosz', 'sphinx', 'tlusty')

# Binary-fit state shared with the sampler wrappers (set by
# Fitter.initialize). The defaults keep single-star fits unchanged.
binary = False
contrast = None


def _pair_name(grid1, grid2):
    """BMA/output name of a binary grid pair, e.g. ``'bosz+tlusty'``."""
    return f'{grid1}+{grid2}'


def _grid_teff_range(interp):
    """(min, max) Teff of a band grid."""
    t = np.asarray(interp.index_columns[1], dtype=float)
    return float(t.min()), float(t.max())


def _prior_mass_inside(prior, lo, hi, n=4001):
    """Fraction of a 1-D prior's mass inside [lo, hi] (from its inverse CDF)."""
    try:
        q = np.asarray(_make_transform(prior)(np.linspace(0.0005, 0.9995, n)),
                       dtype=float)
    except Exception:                                  # noqa: BLE001
        return 1.0
    q = q[np.isfinite(q)]
    if not q.size:
        return 1.0
    return float(np.mean((q >= lo) & (q <= hi)))


def _prior_bound_flags(samples, priors, frac=0.01):
    """Parameters whose posterior is truncated by a finite prior bound.

    A parameter is flagged when the gap between the upper (lower) prior
    bound and the 99.85th (0.15th) percentile, the edge of the 3-sigma
    interval, is less than ``frac`` of that interval's width: the posterior
    runs into the wall instead of tailing off before it. Posteriors wider
    than half the prior are prior-dominated and skipped. The prior span is
    no yardstick, as it is orders of magnitude wider than the posterior for
    rad or norm. A lower bound at exactly zero is a physical floor (Av,
    norm, dist), not a truncation, and is not checked. Only frozen scipy
    priors have a known support; spline priors are skipped.
    """
    flags = []
    for p in _BOUND_CHECKED:
        prior, s = priors.get(p), samples.get(p)
        if prior is None or s is None or not hasattr(prior, 'support'):
            continue
        s = np.asarray(s, dtype=float).ravel()
        s = s[np.isfinite(s)]
        if s.size < 2:
            continue
        try:
            lo, hi = (float(b) for b in prior.support())
        except Exception:
            continue
        q_lo, q_hi = np.percentile(s, [0.15, 99.85])
        span = q_hi - q_lo
        # A posterior that fills its prior (an unconstrained logg) is
        # prior-dominated, not railed.
        if span <= 0 or span > 0.5 * (hi - lo):
            continue
        if np.isfinite(hi) and q_hi >= hi - frac * span:
            flags.append({'param': p, 'edge': 'upper', 'quantile': q_hi,
                          'bound': hi})
        if np.isfinite(lo) and lo != 0 and q_lo <= lo + frac * span:
            flags.append({'param': p, 'edge': 'lower', 'quantile': q_lo,
                          'bound': lo})
    return flags


def _prior_flag_message(f):
    pct = '99.85' if f['edge'] == 'upper' else '0.15'
    return (f"{f['param']} posterior truncated by the {f['edge']} prior "
            f"bound ({pct}th percentile {f['quantile']:.4g}, bound "
            f"{f['bound']:.4g}): the fit and everything derived from it "
            f"may be biased. Widen the prior via prior_setup.")


def _upgrade_prior_setup(prior_setup):
    """Accept the pre-rename ``<band>_noise`` keys of ``prior_setup``.

    The per-band excess-scatter terms used to be called ``<band>_noise``.
    Such keys are read as ``<band>_excess_scatter`` with a ``FutureWarning``;
    if both spellings are given for a band, the new one wins.
    """
    legacy = [k for k in (prior_setup or {})
              if isinstance(k, str) and k.endswith('_noise')]
    if not legacy:
        return prior_setup
    renamed = {k: k[:-len('_noise')] + EXCESS_SCATTER_SUFFIX for k in legacy}
    warnings.warn(
        'prior_setup keys ending in "_noise" are deprecated: the per-band '
        'terms are now named "<band>_excess_scatter" (e.g. '
        f'{legacy[0]!r} -> {renamed[legacy[0]]!r}). Reading '
        f'{", ".join(map(repr, legacy))} as such.', FutureWarning,
        stacklevel=3)
    out = {}
    for k, v in prior_setup.items():
        if k in renamed:
            if renamed[k] in prior_setup:
                continue
            k = renamed[k]
        out[k] = v
    return out


def _check_excess_scatter_setup(prior_setup, scatter):
    """Keys of ``prior_setup`` to use, after checking the excess-scatter ones.

    ``scatter`` lists the excess-scatter terms of the fitted bands. A term of
    a band that is not fitted (e.g. one a catalogue no longer returns) is
    skipped with a warning. Only ``'default'`` and ``('fixed', value)`` apply
    to these terms; any other prior is replaced by the default, with a
    warning.
    """
    keys = []
    for k, v in prior_setup.items():
        if is_excess_scatter(k):
            if k not in scatter:
                warnings.warn(
                    f'prior_setup: {k!r} is not the excess-scatter term of a '
                    f'fitted band; ignoring it. Fitted bands: '
                    f'{", ".join(scatter)}.', UserWarning, stacklevel=3)
                continue
            if v != 'default' and (isinstance(v, str) or v[0] != 'fixed'):
                warnings.warn(
                    f'prior_setup: excess-scatter terms take only "default" '
                    f'or ("fixed", value); using the default prior for '
                    f'{k!r} instead of {v!r}.', UserWarning, stacklevel=3)
                continue
        keys.append(k)
    return keys


def _prior_signature(prior):
    """A comparable description of one prior: frozen scipy distributions by
    name and parameters, spline priors by their coefficients."""
    if prior is None:
        return None
    if hasattr(prior, 'dist'):
        return (prior.dist.name,
                tuple(float(a) for a in getattr(prior, 'args', ())),
                tuple(sorted((k, float(v)) for k, v in
                             (getattr(prior, 'kwds', {}) or {}).items())))
    if hasattr(prior, 'get_coeffs'):
        return (type(prior).__name__,
                tuple(np.round(prior.get_coeffs(), 12).tolist()))
    return repr(prior)


class Fitter:
    """The Fitter class handles the fitting routines and parameter estimation.

    Examples
    --------
    The fitter isn't instantiaded with any arguments, rather you instantiate a
    Fitter object, then you set up the configurations and finally you
    initialize the object by running the initialize method.
    >>> f = Fitter()
    >>> f.star = s  # s must be a valid Star object.
    >>> f.initialize()

    Attributes
    ----------
    star : Star
        The ``Star`` object to fit.
    setup : list
        Sampler configuration. The first element selects the engine
        (``'dynesty'``); the remaining elements are
        ``[nlive, dlogz, bound, sample, threads, dynamic, walks]``.
    out_folder : str
        Folder where results and plots are written. Defaults to the star name.
    verbose : bool
        Program verbosity. Default ``True``.
    norm : bool
        If ``True``, fit a single normalisation constant instead of radius and
        distance; the radius is recovered afterwards from the Gaia parallax.
    grid : str
        Atmosphere grid for a single-grid fit (e.g. ``'phoenix'``,
        ``'btsettl'``, ``'ck04'``, ``'kurucz'``, ``'coelho'``, ``'bosz'``,
        ``'sphinx'``, ``'tlusty'``).
    bma : bool
        If ``True``, run Bayesian Model Averaging across several grids. This
        loads every model interpolator and fits all of them, so it is slower.
    models : list
        The grids to average over when ``bma`` is ``True``.
    binary : bool
        Fit the SED as the sum of two stars sharing distance, extinction and
        [Fe/H] (adds ``teff2``, ``logg2``, ``rad2``). Default ``False``.
    grid2 : str
        Secondary's grid for a single-grid binary fit (default: ``grid``).
    models2 : list or None
        Secondary grids for a binary BMA: every ``models`` x ``models2`` pair
        is fitted. ``None`` pairs each grid in ``models`` with itself.
    contrast : list or None
        Component contrasts ``(band, m2 - m1, sigma)`` for a binary fit; add
        them with ``add_contrast``.
    sequential : bool
        If ``True``, fit the BMA grids one after another. See also
        ``n_grid_jobs`` to fit grids concurrently. Default ``True``.
    n_grid_jobs : int
        Number of BMA grids fit concurrently, one core each. Default ``1``
        (sequential). Each job loads its own grid, so size this to the
        machine's core count and memory.
    n_samples : int or None
        Number of samples drawn from the BMA-averaged posterior. ``None`` uses
        the combined chain length.
    walks : int
        Number of ``rwalk`` MCMC steps per dynesty proposal. Default ``25``;
        also settable as ``setup[7]``.
    av_law : str
        Extinction law. Default ``'fitzpatrick'``.
    estimate_logg : bool
        If ``True``, treat log g as a derived quantity via the isochrone step.
        Default ``False``.
    estimate_age : bool
        If ``True``, run the MIST isochrone fit for age, mass and EEP after the
        SED fit. Default ``True``.
    isochrone_dlogz : float
        Evidence tolerance for the MIST isochrone age/mass fit (the dominant
        cost of a BMA run). Lower is more precise but slower. Default ``0.01``.
    prior_setup : dict or None
        Manual prior specification consumed by ``create_priors_from_setup``.
    experimental : bool
        Enable experimental code paths. Default ``False``.
    synthetic_filters : list or None
        Extra filters to predict at the best-fit SED, written to
        ``synthetic_user_filters.dat`` (and appended to
        ``synthetic_magnitudes.dat`` / ``synthetic_fluxes.dat`` with
        ``observed=0``). Entries may be a pyphot library filter name, a
        ``pyphot.Filter`` or a ``(name, wavelength_um, transmission)`` tuple
        (photon-counting detector unless a 4th element ``'energy'`` is given).
        ARIADNE grid bands are read from the band grid; any other filter is
        synthesized from the spectra cache (``fetch_spectra_cache()``) and is
        NaN where the cached BT-Settl spectra do not span it. Default ``None``.

    """

    colors = colors
    # Single-star defaults (also for instances built without __init__).
    _binary = False
    _grid2 = None
    _interpolator2 = None
    _bma_models2 = None
    contrast = None

    def __init__(self):

        # Default values for attributes
        self._interpolators = []
        self._grids = []
        # Binary (two-star) fit: off by default. See ``binary``, ``grid2``,
        # ``models2`` and ``add_contrast``.
        self._binary = False
        self._grid2 = None
        self._interpolator2 = None
        self._bma_models2 = None
        self.contrast = None
        self.out_folder = None
        self.verbose = True
        self.star = None
        self.setup = ['dynesty']
        self.norm = False
        self.grid = 'phoenix'
        self.estimate_logg = False
        self.estimate_age = True
        self.av_law = 'fitzpatrick'
        self.n_samples = None
        self.bma = False
        self.prior_setup = None
        self.sequential = True
        self.experimental = False
        # Number of MCMC steps per proposal for dynesty's rwalk sampler.
        self.walks = 25
        # Number of BMA model grids to fit concurrently, each on a single
        # core. 1 (default) keeps the original sequential behaviour. Set this
        # to len(models) to run every grid at once. CAUTION: this spawns
        # ``n_grid_jobs`` processes, so choose a value your machine's core
        # count and memory can handle (each grid loads its own model grid).
        self.n_grid_jobs = 1
        # Re-fit every BMA grid even when a compatible ``<grid>_out.pkl``
        # already exists in ``out_folder`` (see ``refit``).
        self._refit = False
        # Evidence tolerance for the MIST isochrone age/mass nested sampling
        # (the dominant cost of a BMA run). Lower is more precise but slower;
        # the isochrone fit itself is parallelised across ``threads``.
        self.isochrone_dlogz = 0.01
        # Extra filters to predict at the best-fit SED (issue #88).
        self.synthetic_filters = None

    @property
    def star(self):
        """Star to fit for."""
        return self._star

    @star.setter
    def star(self, star):
        self._star = star

    @property
    def setup(self):
        """Set up options."""
        return self._setup

    @setup.setter
    def setup(self, setup):
        err_msg = 'The setup has to contain at least the fitting engine'
        err_msg += f', multinest or dynesty.\nThe setup was {setup}'
        if len(setup) < 1:
            InputError(err_msg).__raise__()
        self._setup = setup
        self._engine = setup[0]
        defaults = False
        if len(setup) == 1:
            defaults = True
        if self._engine == 'multinest':
            warnings.warn(
                "The 'multinest' engine is deprecated and will be removed "
                "entirely in astroARIADNE v2.0. It only supports single-grid "
                "fits (not BMA); use engine='dynesty' instead.",
                DeprecationWarning, stacklevel=2,
            )
            if defaults:
                self._nlive = 500
                self._dlogz = 0.5
            else:
                self._nlive = setup[1]
                self._dlogz = setup[2]
        if self._engine == 'dynesty':
            if defaults:
                self._nlive = 500
                self._dlogz = 0.5
                self._bound = 'multi'
                self._sample = 'rwalk'
                self._threads = 1
                self._dynamic = False
            else:
                self._nlive = setup[1]
                self._dlogz = setup[2]
                self._bound = setup[3]
                self._sample = setup[4]
                self._threads = setup[5]
                self._dynamic = setup[6]
                # Optional 8th element: number of rwalk MCMC steps per
                # proposal. Omitted -> keep the current value (default 25),
                # so existing 7-element setups are unaffected.
                if len(setup) > 7:
                    self.walks = setup[7]

    @property
    def norm(self):
        """Bool to decide if a normalization constant will be fitted.

        Set as True to not fit for radius and distance and fit for a
        normalization constant. After the fit a radius is calculated using
        the Gaia parallax.
        """
        return self._norm

    @norm.setter
    def norm(self, norm):
        if type(norm) is not bool:
            InputError('norm must be True or False.').__raise__()
        self._norm = norm

    @property
    def grid(self):
        """Model grid selected."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        assert type(grid) == str
        self._grid = grid
        self._interpolator = self.load_interpolator(grid)

    @property
    def binary(self):
        """Fit the SED as the sum of two stars (an unresolved binary).

        The components share distance, extinction and [Fe/H]; each has its
        own ``teff``/``logg``/``rad`` (``teff2``, ``logg2``, ``rad2`` for the
        secondary, with priors in ``prior_setup`` like any other parameter).
        Each component can use its own grid: ``grid2`` for a single-grid fit,
        ``models2`` for BMA (every primary x secondary grid pair is fitted and
        averaged). Needs ``norm = False``. Optional flux-ratio constraints:
        ``add_contrast``. Default ``False``.
        """
        return self._binary

    @binary.setter
    def binary(self, binary):
        self._binary = bool(binary)

    @property
    def grid2(self):
        """Grid of the secondary in a single-grid binary fit (default: the
        primary's ``grid``)."""
        return self._grid2 if self._grid2 is not None else self._grid

    @grid2.setter
    def grid2(self, grid):
        if grid is None:
            self._grid2, self._interpolator2 = None, None
            return
        assert type(grid) == str
        self._grid2 = grid
        self._interpolator2 = self.load_interpolator(grid)

    def _secondary_interpolator(self):
        return (self._interpolator2 if self._grid2 is not None
                else self._interpolator)

    def add_contrast(self, band, delta_mag, sigma):
        """Constrain a binary fit with a measured component contrast.

        ``delta_mag`` is ``m2 - m1`` in ARIADNE band ``band`` (positive when
        the secondary is fainter), e.g. from speckle/AO imaging or a
        spectroscopic light ratio, with uncertainty ``sigma``. The band need
        not be one of the fitted photometric bands, but both components'
        grids must model it. Only used when ``binary`` is True.
        """
        if self.contrast is None:
            self.contrast = []
        self.contrast.append((str(band), float(delta_mag), float(sigma)))

    @property
    def bma(self):
        """Bayesian Model Averaging (BMA).

        Set to True if BMA is wanted. This loads every model grid interpolator
        and fits an SED to all of them, so the runtime will be slower!
        """
        return self._bma

    @bma.setter
    def bma(self, bma):
        self._bma = bma if bma_flag else False

    @property
    def models(self):
        """Models to be used in BMA."""
        return self._bma_models

    @models.setter
    def models(self, mods):
        self._bma_models = mods

    @property
    def models2(self):
        """Secondary grids of a binary BMA fit. ``None`` (default) pairs each
        grid in ``models`` with itself; a list fits every ``models`` x
        ``models2`` pair (e.g. ``models = ['bosz']``, ``models2 =
        ['tlusty']`` for a B star with a hot subdwarf companion)."""
        return self._bma_models2

    @models2.setter
    def models2(self, mods):
        self._bma_models2 = None if mods is None else list(mods)

    @property
    def refit(self):
        """Force ``fit_bma`` to re-fit every grid. Default ``False``.

        With ``False``, a grid whose ``<out_folder>/<grid>_out.pkl`` exists and
        was produced from the same photometry and parameter layout is loaded
        instead of re-fitted, so the BMA can be recomputed for a new
        ``models`` list without re-running dynesty.
        """
        return self._refit

    @refit.setter
    def refit(self, refit):
        self._refit = bool(refit)

    def _fit_signature(self):
        """Everything besides the photometry that shapes a grid's posterior:
        priors, sampler setup, extinction law and the pinned parameters.
        Saved in each ``<grid>_out.pkl`` so ``fit_bma`` only reuses a fit made
        with the same choices."""
        return {
            'priors': {k: _prior_signature(v) for k, v in
                       sorted((getattr(self, 'priors', None) or {}).items())},
            'setup': (self._engine, self._nlive, self._dlogz,
                      getattr(self, '_bound', None),
                      getattr(self, '_sample', None),
                      getattr(self, '_dynamic', None)),
            'av_law': str(getattr(self, '_av_law', None)),
            'fixed': [float(x) for x in np.ravel(self.fixed)],
            'coordinator': [int(x) for x in np.ravel(self.coordinator)],
            'binary': bool(self._binary),
            'contrast': [list(c) for c in (self.contrast or [])]
            if self._binary else [],
        }

    def _reuse_problem(self, grid):
        """Return why ``<grid>_out.pkl`` cannot be reused, or None if it can."""
        path = f'{self.out_folder}/{grid}_out.pkl'
        if not os.path.isfile(path):
            return 'no previous output'
        try:
            with open(path, 'rb') as fh:
                prev = upgrade_legacy_names(pickle.load(fh))
            prev_star = prev['star']
            prev_filts = list(prev_star.filter_names[prev_star.filter_mask])
            prev_norm = prev['norm']
            prev['global_lnZ'], prev['posterior_samples']
        except Exception as e:
            return f'unreadable previous output ({type(e).__name__})'
        if prev.get('model_grid', grid) != grid:
            return f'previous output is for grid {prev["model_grid"]}'
        cur_filts = list(self.star.filter_names[self.star.filter_mask])
        if prev_filts != cur_filts:
            return 'photometry/filters changed since the previous fit'
        prev_flux = np.asarray(prev_star.flux)[prev_star.filter_mask]
        cur_flux = np.asarray(self.star.flux)[self.star.filter_mask]
        if not np.allclose(prev_flux, cur_flux, rtol=1e-10, atol=0,
                           equal_nan=True):
            return 'photometry values changed since the previous fit'
        sig = prev.get('fit_signature')
        self._reuse_note = ('' if sig is not None else
                            ' (older output: priors and sampler settings '
                            'not recorded, so not checked)')
        if sig is not None:
            cur = self._fit_signature()
            for key, what in (('priors', 'priors'),
                              ('setup', 'sampler settings'),
                              ('av_law', 'extinction law'),
                              ('fixed', 'fixed parameters'),
                              ('coordinator', 'fixed parameters')):
                if sig.get(key) != cur[key]:
                    return f'{what} changed since the previous fit'
            # Outputs written before binary fits existed are single-star.
            if (sig.get('binary', False) != cur['binary']
                    or sig.get('contrast', []) != cur['contrast']):
                return 'binary setup changed since the previous fit'
        if bool(prev_norm) != bool(self.norm):
            return ('fit parameters changed (norm vs dist/rad) since the '
                    'previous fit')
        return None

    @property
    def sequential(self):
        """Set to True to make BMA sequentially instead of parallel."""
        return self._sequential

    @sequential.setter
    def sequential(self, sequential):
        self._sequential = sequential

    @property
    def n_samples(self):
        """Set number of samples for BMA."""
        return self._nsamp

    @n_samples.setter
    def n_samples(self, nsamp):
        self._nsamp = nsamp

    @property
    def verbose(self):
        """Program verbosity. Default is True."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if type(verbose) is not bool:
            InputError('Verbose must be True or False.').__raise__()
        self._verbose = verbose

    @property
    def out_folder(self):
        """Output folder.

        If none is provided the default will be the starname.
        """
        return self._out_folder

    @out_folder.setter
    def out_folder(self, out_folder):
        if type(out_folder) is not str and out_folder is not None:
            err_msg = 'Output folder must be an address or None.'
            InputError(err_msg).__raise__()
        self._out_folder = out_folder

    @property
    def av_law(self):
        """Select extinction law."""
        return self._av_law

    @av_law.setter
    def av_law(self, law):
        laws = [
            'cardelli',
            'odonnell',
            'calzetti',
            'fitzpatrick'
        ]
        law = law.lower()
        if law not in laws:
            err_msg = f'Extinction law {law} not recognized.'
            err_msg += 'Available extinction  laws are: `cardelli`, `odonnell`'
            err_msg += ', `calzetti`, and `fitzpatrick`'
            InputError(err_msg).__raise__()
        law_f = None
        if law == laws[0]:
            law_f = extinction.ccm89
        if law == laws[1]:
            law_f = extinction.odonnell94
        if law == laws[2]:
            law_f = extinction.calzetti00
        if law == laws[3]:
            law_f = extinction.fitzpatrick99
        self._av_law = law_f

    def _apply_grid_coverage(self):
        """Drop bands outside the current grid's native wavelength coverage.

        Some grids (e.g. Coelho 2014) only cover the optical; their redder
        photometry was extrapolated when the grid was built and is unreliable.
        For a standalone fit we therefore exclude those bands. The user's Star
        is left untouched: a shallow copy with a restricted ``used_filters`` /
        ``filter_mask`` is substituted, so every downstream consumer
        (dimension count, priors, likelihood, output) sees the reduced set.
        """
        grids = [self._grid] + ([self.grid2] if self._binary else [])
        covs = [grid_wave_coverage.get(g.lower()) for g in grids]
        covs = [c for c in covs if c is not None]
        if not covs:
            return
        cov = min(covs)
        used = self.star.used_filters == 1
        outside = used & (self.star.wave > cov)
        if not outside.any():
            return
        dropped = self.star.filter_names[outside]
        s = copy.copy(self.star)
        s.used_filters = self.star.used_filters.copy()
        s.used_filters[outside] = 0
        s.filter_mask = np.where(s.used_filters == 1)[0]
        self.star = s
        logger.warning(
            "Grid '%s' only covers <= %.2f um; excluding %d out-of-coverage "
            "band(s) from the fit: %s", '+'.join(dict.fromkeys(grids)), cov,
            len(dropped), ', '.join(dropped))

    def _used_bands(self):
        return [str(b) for b in self.star.filter_names[self.star.filter_mask]]

    def _check_grid_bands(self):
        """Standalone fit: stop if a grid (either component's, for a binary)
        lacks model fluxes for a fitted band, naming the grids that have
        them."""
        comps = [(self._grid, self._interpolator)]
        if self._binary:
            comps.append((self.grid2, self._secondary_interpolator()))
        for name, intp in comps:
            missing = _missing_bands(intp, self._used_bands())
            if not missing:
                continue
            others = [g for g in _ALL_GRIDS if g != name.lower() and
                      not _missing_bands(self.load_interpolator(g), missing)]
            err_msg = (f"Grid '{name}' has no model fluxes for the band(s) "
                       f"{', '.join(missing)}. ")
            err_msg += (f"Grids that have them: {', '.join(others)}. "
                        if others else 'No ARIADNE grid has them. ')
            err_msg += 'Fit with one of those grids or remove the band(s).'
            InputError(err_msg).__raise__()

    def _prep_contrast(self, interps):
        """Validate ``contrast`` for the component grids ``interps`` and pack
        it as ``(bands, dmags, sigmas)`` arrays for the likelihood."""
        if not self._binary or not self.contrast:
            return None
        bands = [str(b) for b, _, _ in self.contrast]
        for intp in interps:
            missing = _missing_bands(intp, bands)
            if missing:
                return missing
        sig = np.array([float(x) for _, _, x in self.contrast])
        if np.any(~np.isfinite(sig)) or np.any(sig <= 0):
            InputError('add_contrast: sigma must be positive.').__raise__()
        return (np.array(bands),
                np.array([float(d) for _, d, _ in self.contrast]), sig)

    def initialize(self):
        """Initialize the fitter.

        To be run only after every input is added.
        This function calculates the number of dimensions, runs the prior
        creation, creates output directory, initializes coordinators and sets
        up global variables.
        """
        global prior_dict, coordinator, fixed, order, star, use_norm, av_law
        global binary, contrast
        self.start = time.time()
        err_msg = 'No star is detected. Please create an instance of Star.'
        if self.star is None:
            er = InputError(err_msg)
            er.log(self.out + '/output.log')
            er.__raise__()
        binary = self._binary
        contrast = None
        if binary:
            if self.norm:
                InputError('A binary fit needs the distance/radius '
                           'parametrisation: set norm = False.').__raise__()
            if self._engine != 'dynesty':
                InputError("A binary fit needs the 'dynesty' engine."
                           ).__raise__()
        # For a standalone fit with a grid of limited wavelength coverage,
        # exclude out-of-coverage bands (e.g. Coelho's extrapolated IR). Not
        # applied in BMA, where every grid must share the same band set.
        if not self._bma:
            self._apply_grid_coverage()
            self._check_grid_bands()
        star = self.star
        if not self._bma:
            global interpolator
            interpolator = self._interpolator
            if binary:
                interpolator = (self._interpolator,
                                self._secondary_interpolator())
                contrast = self._prep_contrast(interpolator)
                if isinstance(contrast, list):
                    InputError(f"The grids '{self._grid}' / '{self.grid2}' "
                               f"have no model fluxes for the contrast "
                               f"band(s) {', '.join(contrast)}.").__raise__()
        use_norm = self.norm
        # Extinction law
        av_law = self._av_law

        # Declare order of parameters.
        if not self.norm:
            order = np.array(
                [
                    'teff', 'logg', 'z',
                    'dist', 'rad', 'Av'
                ]
            )
        else:
            order = np.array(['teff', 'logg', 'z', 'norm', 'Av'])
        if binary:
            order = np.append(order, BINARY_PARAMS)

        # Create output directory
        if self.out_folder is None:
            self.out_folder = self.star.starname + '/'
        create_dir(self.out_folder)

        self.star.save_mags(self.out_folder + '/')

        # Parameter coordination.
        # Order for the parameters are:
        # teff, logg, z, dist, rad, Av, [teff2, logg2, rad2,] excess scatter
        # or
        # teff, logg, z, norm, Av, excess scatter
        # (one excess-scatter term per fitted band)
        npars = n_physical(self.norm, binary)
        npars += self.star.used_filters.sum()
        npars = int(npars)
        self.coordinator = np.zeros(npars)  # 1 for fixed params
        self.fixed = np.zeros(npars)
        coordinator = self.coordinator
        fixed = self.fixed

        # Setup priors.
        self.default_priors = self._default_priors()
        self.create_priors_from_setup()
        prior_dict = self.priors

        # Get dimensions.
        self.ndim = self.get_ndim()

        # warnings
        if len(self._setup) == 1:
            logger.warning('Using default setup values')

        # BMA settings
        # if BMA is used, load all interpolators requested.
        if self._bma:
            if self.n_samples is None:
                self.n_samples = 'max'

            if self.star.offline:
                if self.star.temp is None:
                    self.star.temp = 4001
                off_msg = 'Offline mode assumes that the stellar'
                off_msg += ' temperature is greater than 4000 K'
                off_msg += '. If you believe this is not the case then please '
                off_msg += 'add a temperature to the Star constructor. '
                off_msg += f'The input temperature is {self.star.temp}'
                print(colored(off_msg, 'yellow'))

            used = self._used_bands()
            lacking = []
            loaded = {}

            def _load(mod):
                if mod.lower() not in loaded:
                    loaded[mod.lower()] = self.load_interpolator(mod.lower())
                return loaded[mod.lower()]

            primaries = []
            for mod in self._bma_models:
                # We'll assume that if ARIADNE is running in offline mode
                # Then the star will have > 4000 K
                if (mod.lower() in ['btcond', 'btnextgen'] and
                    self.star.temp > 4000) or \
                        (mod.lower() in ['ck04', 'kurucz'] and
                         self.star.temp < 4000) or \
                        (mod.lower() == 'coelho' and self.star.temp < 3500) or \
                        (mod.lower() == 'bosz' and
                         (self.star.temp < 2800 or self.star.temp > 16000)) or \
                        (mod.lower() == 'sphinx' and self.star.temp > 4000) or \
                        (mod.lower() == 'tlusty' and self.star.temp < 15000):
                    continue
                primaries.append(mod)

            # Single star: one fit per grid. Binary: one fit per
            # (primary grid, secondary grid) pair; the secondary is not
            # temperature-gated (star.temp describes the primary), so pick
            # ``models2`` for the companion you expect.
            if binary:
                if self._bma_models2 is None:
                    members = [(m, m) for m in primaries]
                else:
                    members = [(m, m2) for m in primaries
                               for m2 in self._bma_models2]
            else:
                members = [(m, None) for m in primaries]

            for mod, mod2 in members:
                name = mod if mod2 is None else _pair_name(mod, mod2)
                # Every BMA member must model every fitted band (the
                # evidences are compared), so a grid without model fluxes
                # for one of them (e.g. the IUE / TD-1 bands, which only
                # BOSZ and TLUSTY carry) is left out.
                comps = [(mod, _load(mod))]
                if mod2 is not None:
                    comps.append((mod2, _load(mod2)))
                missing = []
                for g, df in comps:
                    missing += [b for b in _missing_bands(df, used)
                                if b not in missing]
                if missing:
                    lacking.append((name, missing))
                    continue
                if mod2 is None:
                    self._interpolators.append(comps[0][1])
                else:
                    pair = (comps[0][1], comps[1][1])
                    c = self._prep_contrast(pair)
                    if isinstance(c, list):
                        lacking.append((name, c))
                        continue
                    self._interpolators.append(pair)
                self._grids.append(name)
            for mod, missing in lacking:
                logger.warning("Grid '%s' has no model fluxes for %s; leaving "
                               "it out of the BMA.", mod, ', '.join(missing))
            if not self._interpolators:
                err_msg = 'No BMA grid can be fitted: '
                err_msg += ('every grid left after the temperature gating '
                            'lacks model fluxes for '
                            f"{', '.join(sorted({b for _, m in lacking for b in m}))}"
                            if lacking else
                            f'the temperature gating (star.temp = '
                            f'{self.star.temp}) excludes all of '
                            f'{self._bma_models}')
                err_msg += '.'
                InputError(err_msg).__raise__()
            if binary and self.contrast:
                contrast = self._prep_contrast(self._interpolators[0])
            thr = self._threads if self._sequential else len(
                self._interpolators)
        else:
            thr = self._threads
        en = 'Bayesian Model Averaging' if self._bma else self._engine
        if binary:
            en += ' (binary)'
            self._warn_secondary_prior_coverage()
        display_routine(en, self._nlive, self._dlogz, self.ndim, self._bound,
                        self._sample, thr, self._dynamic)
        self.show_priors()

    def _warn_secondary_prior_coverage(self):
        """Warn when much of the teff2 prior lies outside a secondary grid:
        the model there is NaN (zero likelihood), which truncates the prior
        and lowers that grid's evidence."""
        prior = (getattr(self, 'priors', None) or {}).get('teff2')
        if prior is None:
            return
        if self._bma:
            grids = {}
            for name, intp in zip(self._grids, self._interpolators):
                g2 = split_grid_name(name)[1] or name
                grids[g2] = intp[1] if isinstance(intp, tuple) else intp
        else:
            grids = {self.grid2: self._secondary_interpolator()}
        for g2, intp in grids.items():
            lo, hi = _grid_teff_range(intp)
            frac = _prior_mass_inside(prior, lo, hi)
            if frac < 0.99:
                logger.warning(
                    "Only %.0f%% of the teff2 prior lies inside the "
                    "secondary grid '%s' (%.0f-%.0f K); outside it the model "
                    "is undefined, which truncates the prior and lowers the "
                    "evidence. Narrow the teff2 prior or choose another "
                    "secondary grid.", 100 * frac, g2, lo, hi)

    def get_ndim(self):
        """Calculate number of dimensions."""
        ndim = n_physical(self.norm, self._binary)
        ndim += self.star.used_filters.sum()
        ndim -= self.coordinator.sum()
        return int(ndim)

    def _default_priors(self):
        global order
        defaults = dict()

        # Initialize prior source tracking
        if not hasattr(self, 'prior_sources'):
            self.prior_sources = {}

        # Resolve spectroscopic parameters: prefer spectroscopic_params,
        # fall back to rave_params for backward compatibility.
        spec = getattr(self.star, 'spectroscopic_params', None)
        if spec is None:
            spec = getattr(self.star, 'rave_params', None)
        spec_source = spec.get('source', 'spectroscopic') if spec else None

        # Per-field null-safe helper: a spectroscopic field is usable only if
        # the spec dict exists and both the value and its error are not None.
        # This handles partial spec dicts (e.g. Hypatia [Fe/H]-only results
        # where teff/logg are None) by falling back per-field to the
        # population/default prior rather than crashing.
        def _spec_has(field):
            return (spec is not None and
                    spec.get(field) is not None and
                    spec.get(field + '_err') is not None)

        # Teff: hot-star grids (TLUSTY) need a wide uniform prior — the FGK
        # population prior assigns ~zero probability above ~12000 K, so it
        # cannot reach the O/B regime. Applied for standalone TLUSTY fits.
        if (getattr(self, '_grid', '') or '').lower() == 'tlusty':
            defaults['teff'] = st.uniform(loc=15000, scale=40000)  # 15-55 kK
            self.prior_sources['teff'] = 'tlusty_uniform'
            logger.info('Using uniform Teff prior for the TLUSTY hot-star grid')
        # Teff: spectroscopic star-specific or population prior
        elif _spec_has('teff'):
            defaults['teff'] = st.norm(loc=spec['teff'],
                                       scale=spec['teff_err'])
            self.prior_sources['teff'] = spec_source
            logger.info('Using %s Teff prior (star-specific)', spec_source)
        else:
            with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
                defaults['teff'] = pickle.load(jar)
            self.prior_sources['teff'] = 'rave_population'
            logger.info('Using population Teff prior (no spectroscopic match)')

        # logg: spectroscopic, isochrone, or population prior
        if _spec_has('logg'):
            defaults['logg'] = st.norm(loc=spec['logg'],
                                       scale=spec['logg_err'])
            self.prior_sources['logg'] = spec_source
            logger.info('Using %s logg prior (star-specific)', spec_source)
        elif self.star.get_logg:
            defaults['logg'] = st.norm(
                loc=self.star.logg, scale=self.star.logg_e)
            self.prior_sources['logg'] = 'isochrone'
            logger.info('Using isochrone logg estimate')
        else:
            defaults['logg'] = st.uniform(loc=3.5, scale=2.5)
            self.prior_sources['logg'] = 'uniform_default'
            logger.info('Using population logg prior')

        # [Fe/H]: spectroscopic or population prior
        if _spec_has('feh'):
            defaults['z'] = st.norm(loc=spec['feh'], scale=spec['feh_err'])
            self.prior_sources['z'] = spec_source
            logger.info('Using %s [Fe/H] prior (star-specific)', spec_source)
        else:
            defaults['z'] = st.norm(loc=-0.125, scale=0.234)
            self.prior_sources['z'] = 'population'
            logger.info('Using population [Fe/H] prior')
        # Distance prior setup.
        if not self._norm:
            if self.star.dist != -1:
                _d_scale = 3 * self.star.dist_e
                defaults['dist'] = st.truncnorm(
                    a=-self.star.dist / _d_scale, b=1e100,
                    loc=self.star.dist, scale=_d_scale,
                )
                self.prior_sources['dist'] = 'gaia'
            else:
                defaults['dist'] = st.uniform(loc=1, scale=3000)
                self.prior_sources['dist'] = 'uniform_default'
            # Radius prior setup.
            defaults['rad'] = st.uniform(loc=0.05, scale=100)
            self.prior_sources['rad'] = 'uniform_default'
        # Normalization prior setup.
        else:
            # norm = (R / D)^2. A fixed 1e-20 ceiling (R/D < 1e-10) rails
            # every star nearer than ~225 pc * (R / R_sun) against the bound.
            # With a distance, scale the ceiling to the rad prior's upper
            # end, 100.05 R_sun, at the nearest plausible distance
            # (D - 5 sigma, floored at 1 pc). Without one, 1e-16 still covers
            # a 1 R_sun star out to ~2 pc.
            if self.star.dist != -1:
                d_min = max(self.star.dist - 5 * self.star.dist_e, 1.)
                up = (100.05 * u.solRad.to(u.pc) / d_min) ** 2
                defaults['norm'] = st.uniform(loc=0, scale=up)
                self.prior_sources['norm'] = 'uniform_distance'
            else:
                defaults['norm'] = st.uniform(loc=0, scale=1e-16)
                self.prior_sources['norm'] = 'uniform_default'
        # Extinction prior setup.
        if self.star.Av == 0.:
            av_idx = 4 if self._norm else 5
            self.coordinator[av_idx] = 1
            self.fixed[av_idx] = 0
            defaults['Av'] = None
            self.prior_sources['Av'] = 'fixed_zero'
        else:
            # Av passed to Star by the user, or read from a dust map.
            user = getattr(self.star, 'Av_source', None) == 'user'
            if self.star.Av_e is None:
                defaults['Av'] = st.uniform(loc=0, scale=self.star.Av)
                self.prior_sources['Av'] = ('user_uniform' if user
                                            else 'dustmap_uniform')
            elif self.star.Av_e == 0:
                # Known exactly: pin it.
                av_idx = 4 if self._norm else 5
                self.coordinator[av_idx] = 1
                self.fixed[av_idx] = self.star.Av
                defaults['Av'] = None
                self.prior_sources['Av'] = ('user_fixed' if user
                                            else 'dustmap_fixed')
            else:
                mu = self.star.Av
                sig = self.star.Av_e
                low = 0
                up = 100
                b, a = (up - mu) / sig, (low - mu) / sig
                defaults['Av'] = st.truncnorm(loc=mu, scale=sig, a=a, b=b)
                self.prior_sources['Av'] = ('user_truncnorm' if user
                                            else 'dustmap')
        # Secondary star of a binary fit: the primary's Teff and radius priors
        # (a TLUSTY secondary gets TLUSTY's 15-55 kK range instead), and a
        # uniform log g. Set teff2 / logg2 / rad2 in prior_setup to override.
        if self._binary:
            g2s = ([self.grid2] if not self._bma else
                   (self._bma_models2 or self._bma_models or []))
            if g2s and all(str(g).lower() == 'tlusty' for g in g2s):
                defaults['teff2'] = st.uniform(loc=15000, scale=40000)
                self.prior_sources['teff2'] = 'tlusty_uniform'
            else:
                defaults['teff2'] = defaults['teff']
                self.prior_sources['teff2'] = self.prior_sources['teff']
            defaults['logg2'] = st.uniform(loc=3.5, scale=2.5)
            self.prior_sources['logg2'] = 'uniform_default'
            defaults['rad2'] = defaults['rad']
            self.prior_sources['rad2'] = 'uniform_default'
        # Per-band excess scatter: an extra uncertainty (flux units) added in
        # quadrature to each band's error. Half-normal prior, scale 10 errors.
        mask = self.star.filter_mask
        flxs = self.star.flux[mask]
        errs = self.star.flux_er[mask]
        for filt, flx, flx_e in zip(self.star.filter_names[mask], flxs, errs):
            p_ = excess_scatter_name(filt)
            mu = 0
            sigma = flx_e * 10
            b = (1 - flx) / flx_e
            defaults[p_] = st.truncnorm(loc=mu, scale=sigma, a=0, b=b)
            # defaults[p_] = st.uniform(loc=0, scale=5)
            order = np.append(order, p_)
        return defaults

    def create_priors_from_setup(self):
        """Create priors from the manual setup."""
        prior_dict = dict()
        self.prior_setup = _upgrade_prior_setup(self.prior_setup)
        mask = self.star.filter_mask
        scatter = [excess_scatter_name(filt)
                   for filt in self.star.filter_names[mask]]
        keys = _check_excess_scatter_setup(self.prior_setup, scatter)
        prior_out = 'Parameter\tPrior\tValues\n'
        if 'norm' in keys and ('rad' in keys or 'dist' in keys):
            er = PriorError('rad or dist', 1)
            er.log(self.out_folder + '/output.log')
            er.__raise__()
        for k in keys:
            if type(self.prior_setup[k]) == str:
                if self.prior_setup[k] == 'default':
                    prior_dict[k] = self.default_priors[k]
                    prior_out += k + '\tdefault\n'
                if self.prior_setup[k].lower() == 'rave':
                    # RAVE prior only available for teff and logg. It's already
                    # the default for [Fe/H]
                    if k == 'logg' or k == 'teff':
                        if k == 'teff':
                            with open(priorsdir + '/teff_ppf.pkl', 'rb') as jar:
                                prior_dict[k] = pickle.load(jar)
                        else:  # k == 'logg'
                            with open(priorsdir + '/logg_ppf.pkl', 'rb') as jar:
                                prior_dict[k] = pickle.load(jar)
                        prior_out += k + '\tRAVE\n'

            else:
                prior = self.prior_setup[k][0]
                if prior == 'fixed':
                    value = self.prior_setup[k][1]
                    idx = np.where(k == order)[0]
                    self.coordinator[idx] = 1
                    self.fixed[idx] = value
                    prior_out += k + '\tfixed\t{}\n'.format(value)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_fixed'
                if prior == 'normal':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    prior_dict[k] = st.norm(loc=mu, scale=sig)
                    prior_out += k + '\tnormal\t{}\t{}\n'.format(mu, sig)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_normal'
                if prior == 'truncnorm':
                    mu = self.prior_setup[k][1]
                    sig = self.prior_setup[k][2]
                    low = self.prior_setup[k][3]
                    up = self.prior_setup[k][4]
                    b, a = (up - mu) / sig, (low - mu) / sig
                    prior_dict[k] = st.truncnorm(a=a, b=b, loc=mu, scale=sig)
                    prior_out += k
                    prior_out += '\ttruncatednormal\t{}\t{}\t{}\t{}\n'.format(
                        mu, sig, low, up)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_truncnorm'
                if prior == 'uniform':
                    low = self.prior_setup[k][1]
                    up = self.prior_setup[k][2]
                    prior_dict[k] = st.uniform(loc=low, scale=up - low)
                    prior_out += k + '\tuniform\t{}\t{}\n'.format(low, up)
                    if hasattr(self, 'prior_sources'):
                        self.prior_sources[k] = 'user_uniform'
        for par in scatter:
            prior_dict[par] = self.default_priors[par]
        # A binary's secondary parameters missing from prior_setup take their
        # defaults (see _default_priors).
        if self._binary:
            for par in BINARY_PARAMS:
                if par not in keys:
                    prior_dict[par] = self.default_priors[par]
                    prior_out += par + '\tdefault\n'
                    logger.info('No prior for %s in prior_setup: using the '
                                'default', par)
        ff = open(self.out_folder + '/prior.dat', 'w')
        ff.write(prior_out)
        ff.close()
        del ff
        self.priors = prior_dict
        pass

    def show_priors(self, return_string=False):
        """Display priors in statistical notation.

        Parameters
        ----------
        return_string : bool, optional
            If True, return the formatted string instead of printing.
            Default: False

        Returns
        -------
        str or None
            Formatted prior information if return_string=True, else None
        """
        global order
        _T = '\t\t\t'
        c = np.random.choice(colors)

        # Physical parameters to display (skip the excess-scatter priors)
        physical_params = []
        for param_name in self.priors.keys():
            if is_excess_scatter(param_name):
                continue
            physical_params.append(param_name)

        # Format each prior
        prior_strs = {}
        for param_name in physical_params:
            prior_obj = self.priors[param_name]
            param_idx = np.where(order == param_name)[0]
            if len(param_idx) > 0 and self.coordinator[param_idx[0]] == 1:
                fixed_value = self.fixed[param_idx[0]]
                prior_str = f"Fixed({fixed_value:.4f})"
            else:
                prior_str = self._format_prior_notation(param_name, prior_obj)
            prior_strs[param_name] = prior_str

        # Sort parameters in logical order
        param_order = ['teff', 'logg', 'z', 'dist', 'rad', 'norm', 'Av',
                       'teff2', 'logg2', 'rad2']
        sorted_params = [p for p in param_order if p in physical_params]
        sorted_params.extend([p for p in physical_params if p not in param_order])

        # Build table
        lines = []
        lines.append(f'{_T}{"Parameter":24s}  {"Prior"}')
        lines.append(f'{_T}{"-" * 52}')
        for param_name in sorted_params:
            lines.append(f'{_T}{param_name:24s}  {prior_strs[param_name]}')

        output = '\n'.join(lines)

        if return_string:
            return output
        else:
            print(colored(output, c))
            print()
            return None

    def _format_prior_notation(self, param_name, prior_obj):
        """Format scipy.stats distribution in statistical notation.

        Parameters
        ----------
        param_name : str
            Parameter name (e.g., 'teff', 'logg')
        prior_obj : scipy.stats distribution
            The prior distribution object

        Returns
        -------
        str
            Formatted statistical notation
        """
        # RAVE population prior: loaded as an InterpolatedUnivariateSpline
        # from teff_ppf.pkl / logg_ppf.pkl. Detect by type since manual
        # prior setups don't populate prior_sources.
        if 'Spline' in type(prior_obj).__name__:
            return "RAVE (population)"
        if hasattr(self, 'prior_sources') and param_name in self.prior_sources:
            if self.prior_sources[param_name] == 'rave_population':
                return "RAVE (population)"

        # Frozen scipy distributions expose .dist (the class) and .kwds/.args
        # (the frozen parameters). Use the underlying class name to identify
        # the distribution.
        if hasattr(prior_obj, 'dist'):
            inner_name = type(prior_obj.dist).__name__
            kwds = dict(getattr(prior_obj, 'kwds', {}) or {})
            args = list(getattr(prior_obj, 'args', ()) or ())
        else:
            inner_name = type(prior_obj).__name__
            kwds, args = {}, []

        def _pop(key, idx, default=0.0):
            if key in kwds:
                return kwds[key]
            if idx < len(args):
                return args[idx]
            return default

        def _num(x, d=2):
            # Fixed point hides norm-sized values (1e-20 -> "0.00").
            if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e6):
                return f"{x:.{d}e}"
            return f"{x:.{d}f}"

        try:
            if 'norm_gen' in inner_name and 'truncnorm' not in inner_name:
                mu = _pop('loc', 0)
                sigma = _pop('scale', 1, 1.0)
                return f"N({_num(mu, 3)}, {_num(sigma, 3)})"

            if 'uniform_gen' in inner_name:
                a = _pop('loc', 0)
                scale = _pop('scale', 1, 1.0)
                b = a + scale
                return f"U({_num(a)}, {_num(b)})"

            if 'truncnorm_gen' in inner_name:
                a_std = _pop('a', 0)
                b_std = _pop('b', 1)
                mu = _pop('loc', 2)
                sigma = _pop('scale', 3, 1.0)
                a_act = mu + a_std * sigma
                b_act = mu + b_std * sigma
                a_str = "-inf" if a_std < -1e10 else _num(a_act)
                b_str = "inf" if b_std > 1e10 else _num(b_act)
                return f"TN({_num(mu)}, {_num(sigma)}, [{a_str}, {b_str}])"

            return inner_name

        except (AttributeError, TypeError):
            return f"{inner_name} (custom)"

    def fit(self):
        """Run fitting routine."""
        if self._engine == 'multinest':
            self.fit_multinest()
        else:
            self.fit_dynesty()
        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time,
            self.out_folder, self._engine, self.norm,
            out_file=getattr(self, '_last_out_file', None))
        pass

    def fit_bma(self):
        """Perform the fit with different models and the average the output.

        Only works with dynesty.
        """
        if len(self.star.filter_names[self.star.filter_mask]) <= 5:
            print(colored('\t\t\tNOT ENOUGH POINTS TO MAKE THE FIT! !', 'red'))
            return

        global interpolator, _BMA_FITTER

        # Reuse compatible ``<grid>_out.pkl`` files instead of re-running
        # dynesty (unless ``refit``); only the grids in ``todo`` are fitted.
        todo = []
        for i, gr in enumerate(self._grids):
            why = None if self.refit else self._reuse_problem(gr)
            if not self.refit and why is None:
                print(f'\t\t\tREUSING EXISTING FIT : {gr}'
                      f'{getattr(self, "_reuse_note", "")}')
                continue
            if not self.refit and why != 'no previous output':
                print(f'\t\t\tREFITTING {gr} : {why}')
            todo.append(i)

        n_jobs = min(max(int(self.n_grid_jobs), 1), max(len(todo), 1))
        if n_jobs > 1:
            # Across-grid parallelism: the grids are independent fits, so run
            # several at once (one core each). Each worker writes its own
            # ``_out.pkl``, which we read back below for averaging.
            ncpu = os.cpu_count() or 1
            if n_jobs > ncpu:
                logger.warning(
                    'n_grid_jobs=%d exceeds the %d available CPU cores; '
                    'this will oversubscribe the machine.', n_jobs, ncpu)
            _BMA_FITTER = self
            print(f'\t\t\tFITTING {len(todo)} MODELS, '
                  f'{n_jobs} CONCURRENTLY')
            with Pool(n_jobs) as pool:
                for gr, err in pool.map(_bma_grid_worker, todo):
                    if err is not None:
                        dump_out = self.out_folder + '/' + gr + '_DUMP.pkl'
                        DynestyError(dump_out, gr, err).__raise__()
        else:
            for i in todo:
                intp, gr = self._interpolators[i], self._grids[i]
                interpolator = intp
                # The interpolator(s) are already loaded; only the name is
                # needed (save() records it as out['model_grid']).
                self._grid = gr
                out_file = self.out_folder + '/' + gr + '_out.pkl'
                print('\t\t\tFITTING MODEL : ' + gr)
                try:
                    self.fit_dynesty(out_file=out_file)
                except ValueError as e:
                    dump_out = self.out_folder + '/' + gr + '_DUMP.pkl'
                    pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                    DynestyError(dump_out, gr, e).__raise__()
                    continue

        # Now that the fitting finished, read the outputs and average
        # the posteriors
        outs = []
        for g in self._grids:
            in_folder = f'{self.out_folder}/{g}_out.pkl'
            outs.append(in_folder)
            # with open(in_folder, 'rb') as out:
            #     outs.append(pickle.load(out))
        c = np.random.choice(self.colors)
        avgd = self.bayesian_model_average(outs, self._grids, self._norm,
                                           self.n_samples, c)
        self.save_bma(avgd)

        elapsed_time = execution_time(self.start)
        end(self.coordinator, elapsed_time, self.out_folder,
            'Bayesian Model Averaging', self.norm)
        pass

    def _bma_dynesty(self, intp, grid):
        global interpolator
        interpolator = intp

        # Parallel parallelized routine experiment
        if self.experimental:
            if self._dynamic:
                with Pool(self._threads) as executor:
                    sampler = dynesty.DynamicNestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz_init=self._dlogz,
                                       nlive_batch=self._nlive,
                                       wt_kwargs={'pfrac': .95})
            else:
                with Pool(self._threads) as executor:
                    sampler = dynesty.NestedSampler(
                        dynesty_loglike_bma, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample, pool=executor,
                        queue_size=self._threads, logl_args=([intp])
                    )
                    sampler.run_nested(dlogz=self._dlogz)

        elif self._dynamic:
            sampler = dynesty.DynamicNestedSampler(
                dynesty_loglike_bma, pt_dynesty, self.ndim,
                bound=self._bound, sample=self._sample, logl_args=([intp])

            )
            sampler.run_nested(dlogz_init=self._dlogz,
                               nlive_init=self._nlive,
                               wt_kwargs={'pfrac': .95})
        else:
            try:
                self.sampler = dynesty.NestedSampler(
                    dynesty_loglike_bma, pt_dynesty, self.ndim,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample,
                    logl_args=([intp])
                )
                self.sampler.run_nested(dlogz=self._dlogz)
            except Error:
                dump_out = self.out_folder + '/' + grid + '_DUMP.pkl'
                pickle.dump(self.sampler.results, open(dump_out, 'wb'))
                er = DynestyError(dump_out, grid)
                er.log(self.out + '/output.log')
                er.__raise__()

        results = self.sampler.results
        out_file = self.out_folder + '/' + grid + '_out.pkl'
        self.save(out_file, results=results)
        pass

    def fit_multinest(self, out_file=None):
        """Run MultiNest."""
        # Set up some globals
        global mask, flux, flux_er, filts, wave
        mask = star.filter_mask
        flux = star.flux[mask]
        flux_er = star.flux_er[mask]
        filts = star.filter_names[mask]
        wave = star.wave[mask]
        if pymultinest is None:
            raise ImportError(
                'pymultinest/libmultinest could not be loaded: build MultiNest '
                'and add its lib/ directory (and that of any library it links '
                'against, e.g. MKL) to LD_LIBRARY_PATH, or use the dynesty '
                'engine.')
        path = self.out_folder + '/mnest/'
        create_dir(path)  # Create multinest path.
        pymultinest.run(
            multinest_log_like, pt_multinest, self.ndim,
            n_params=self.ndim,
            sampling_efficiency=0.8,
            evidence_tolerance=self._dlogz,
            n_live_points=self._nlive,
            outputfiles_basename=path + 'chains',
            max_modes=100,
            verbose=self.verbose,
            resume=False
        )
        if out_file is None:
            out_file = f'{self.out_folder}/{self._out_name()}_out.pkl'
        self._last_out_file = out_file
        self.save(out_file=out_file)
        pass

    def fit_dynesty(self, out_file=None):
        """Run dynesty."""
        # Set up some globals
        global mask, flux, flux_er, filts, wave
        mask = star.filter_mask
        flux = star.flux[mask]
        flux_er = star.flux_er[mask]
        filts = star.filter_names[mask]
        wave = star.wave[mask]
        if self._dynamic:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.DynamicNestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        bound=self._bound, sample=self._sample,
                        pool=executor, walks=self.walks,
                        queue_size=self._threads
                    )
                    self.sampler.run_nested(dlogz_init=self._dlogz,
                                            nlive_init=self._nlive,
                                            wt_kwargs={'pfrac': 1})
            else:
                self.sampler = dynesty.DynamicNestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=self.walks,
                    bound=self._bound, sample=self._sample

                )
                self.sampler.run_nested(dlogz_init=self._dlogz,
                                        nlive_init=self._nlive,
                                        wt_kwargs={'pfrac': 1})
        else:
            if self._threads > 1:
                with Pool(self._threads) as executor:
                    self.sampler = dynesty.NestedSampler(
                        dynesty_log_like, pt_dynesty, self.ndim,
                        nlive=self._nlive, bound=self._bound,
                        sample=self._sample,
                        pool=executor, walks=self.walks,
                        queue_size=self._threads,
                    )
                    self.sampler.run_nested(dlogz=self._dlogz)
            else:
                self.sampler = dynesty.NestedSampler(
                    dynesty_log_like, pt_dynesty, self.ndim, walks=self.walks,
                    nlive=self._nlive, bound=self._bound,
                    sample=self._sample
                )
                self.sampler.run_nested(dlogz=self._dlogz)
        results = self.sampler.results
        if out_file is None:
            out_file = f'{self.out_folder}/{self._out_name()}_out.pkl'
        self._last_out_file = out_file
        self.save(out_file, results=results)
        pass

    def _out_name(self):
        """Name of this fit's grid in outputs: ``'bosz'``, or the pair
        ``'bosz+tlusty'`` for a single-grid binary fit."""
        if self._binary and not self._bma:
            return _pair_name(self._grid, self.grid2)
        return self._grid

    def save(self, out_file, results=None):
        """Save multinest/dynesty output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ: The global evidence.
        lnZerr: The global evidence error.
        posterior_samples: A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed: An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit: The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star: The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)
        engine: The fitting engine used (i.e. MultiNest or Dynesty)

        Also creates a log file with the best fit parameters and 1 sigma
        error bars.

        """
        out = dict()
        logdat = '#Parameter\tmedian\tupper\tlower\t3sig_CI\n'
        log_out = self.out_folder + '/' + 'best_fit.dat'
        if self._engine == 'multinest':
            lnz, lnzer, posterior_samples = self.multinest_results(
                self.out_folder,
                self.ndim
            )
        else:
            lnz, lnzer, posterior_samples = self.dynesty_results(results)

        mask = self.star.filter_mask

        # Save global evidence

        if self._engine == 'dynesty':
            out['dynesty'] = results
        out['global_lnZ'] = lnz
        out['global_lnZerr'] = lnzer

        # Create raw samples holder

        out['posterior_samples'] = dict()
        j = 0
        k = 0  # filter counter
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                samples = posterior_samples[:, j]
                if is_excess_scatter(param):
                    filt = self.star.filter_names[mask][k]
                    flx = self.star.flux[mask][k]
                    _, samples = flux_to_mag(flx, samples, filt)
                    k += 1
                out['posterior_samples'][param] = samples
                j += 1
            else:
                out['posterior_samples'][param] = self.fixed[i]

        # Save loglike, priors and posteriors.

        out['posterior_samples']['loglike'] = np.zeros(
            posterior_samples.shape[0]
        )

        # If normalization constant was fitted, create a distribution of radii
        # only if there's a distance available.

        if use_norm and star.dist != -1:
            rad = self._get_rad(
                out['posterior_samples']['norm'], star.dist, star.dist_e
            )
            out['posterior_samples']['rad'] = rad

        # Create a distribution of masses.

        logg_samp = out['posterior_samples']['logg']
        rad_samp = out['posterior_samples']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        out['posterior_samples']['grav_mass'] = mass_samp

        # Create a distribution of luminosities.

        teff_samp = out['posterior_samples']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        out['posterior_samples']['lum'] = lum_samp

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['posterior_samples']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            out['posterior_samples']['AD'] = ad_samp

        if binary:
            self._add_secondary_derived(out['posterior_samples'])
            frac2 = np.zeros((posterior_samples.shape[0], len(filts)))

        for i in range(posterior_samples.shape[0]):
            theta = build_params(
                posterior_samples[i, :], flux, flux_er, filts, coordinator,
                fixed, self.norm)
            out['posterior_samples']['loglike'][i] = log_likelihood(
                theta, flux, flux_er, wave, filts, interpolator, self.norm,
                av_law, binary=binary, contrast=contrast)
            if binary:
                m1, m2 = binary_components(theta, filts, wave, interpolator,
                                           av_law)
                frac2[i] = m2 / (m1 + m2)
        if binary:
            # Fraction of the observed flux from the secondary in each
            # fitted band (posterior samples; averaged across grids by BMA).
            for j, f_ in enumerate(filts):
                out['posterior_samples'][f'frac2_{f_}'] = frac2[:, j]

        # A posterior railed against a prior bound looks like a normal fit;
        # say so.
        out['prior_flags'] = _prior_bound_flags(
            out['posterior_samples'], getattr(self, 'priors', None) or {})
        for f_ in out['prior_flags']:
            warnings.warn(_prior_flag_message(f_), stacklevel=2)

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.
        # do only if not bma

        if not self.bma:
            out['best_fit_averaged'] = dict()
            out['uncertainties_averaged'] = dict()
            out['confidence_interval_averaged'] = dict()
            best_theta = np.zeros(order.shape[0])

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if is_excess_scatter(param):
                        continue
                    samp = out['posterior_samples'][param]

                    if param == 'z':
                        logdat = out_filler(samp, logdat, param, '[Fe/H]', out)
                    elif param == 'norm':
                        logdat = out_filler(samp, logdat, param, '(R/D)^2',
                                            out, fmt='e')
                        if star.dist != 1:
                            logdat = out_filler(
                                out['posterior_samples']['rad'], logdat, 'rad',
                                'R', out
                            )
                    else:
                        logdat = out_filler(samp, logdat, param, param, out)
                else:
                    logdat = out_filler(
                        0, logdat, param, param, out, fixed=self.fixed[i]
                    )
                best_theta[i] = out['best_fit_averaged'][param]

            # Add derived mass to best fit dictionary.

            samp = out['posterior_samples']['grav_mass']
            logdat = out_filler(
                samp, logdat, 'grav_mass', 'grav_mass', out
            )

            # Add derived luminosity to best fit dictionary.

            samp = out['posterior_samples']['lum']
            logdat = out_filler(samp, logdat, 'lum', 'lum', out)

            # Add derived angular diameter to best fit dictionary.

            if not use_norm:
                samp = out['posterior_samples']['AD']
                logdat = out_filler(samp, logdat, 'AD', 'AD', out)

            if binary:
                for p_ in ('grav_mass2', 'lum2', 'AD2'):
                    logdat = out_filler(out['posterior_samples'][p_], logdat,
                                        p_, p_, out)

            for i, param in enumerate(order):
                if not self.coordinator[i]:
                    if not is_excess_scatter(param):
                        continue
                    samp = out['posterior_samples'][param]
                    logdat = out_filler(samp, logdat, param,
                                        param, out, fmt='f')

            # Fill in best loglike, prior and posterior.

            out['best_fit_averaged']['loglike'] = log_likelihood(
                best_theta, flux, flux_er, wave,
                filts, interpolator, self.norm, av_law,
                binary=binary, contrast=contrast
            )

            # Spectral type
            # Load Mamajek spt table
            mamajek_spt = np.loadtxt(
                filesdir + '/mamajek_spt.dat', dtype=str, usecols=[0])
            mamajek_temp = np.loadtxt(
                filesdir + '/mamajek_spt.dat', usecols=[1])

            # Find spt
            spt_idx = np.argmin(abs(mamajek_temp - out['best_fit_averaged']['teff']))
            spt = mamajek_spt[spt_idx]
            out['spectral_type'] = spt
            if binary:
                out['spectral_type2'] = mamajek_spt[np.argmin(
                    abs(mamajek_temp - out['best_fit_averaged']['teff2']))]

        # Utilities for plotting.

        out['fit_signature'] = self._fit_signature()
        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['engine'] = self._engine
        out['norm'] = self.norm
        out['model_grid'] = self._out_name()
        out['av_law'] = av_law
        out['binary'] = bool(binary)
        out['contrast'] = list(self.contrast or []) if binary else []
        if not self.bma:
            with open(log_out, 'w') as logfile:
                logfile.write(logdat)
            if binary:
                self._write_light_fractions(out['posterior_samples'])
            # Predicted observed photometry for the full band set (+ any
            # Fitter.synthetic_filters); BMA does this in save_bma.
            self._write_synthetic(out, out['posterior_samples'])
        pickle.dump(out, open(out_file, 'wb'))
        pass

    def _add_secondary_derived(self, samples):
        """Add the secondary's gravity mass, luminosity and angular diameter
        (``grav_mass2``, ``lum2``, ``AD2``) to a samples dict in place."""
        samples['grav_mass2'] = self._get_mass(samples['logg2'],
                                               samples['rad2'])
        samples['lum2'] = self._get_lum(samples['teff2'], samples['rad2'])
        samples['AD2'] = self._get_angular_diameter(samples['rad2'],
                                                    samples['dist'])

    def _write_light_fractions(self, samples, fname='binary_light_fractions.dat'):
        """Per fitted band: the secondary's share of the observed flux and the
        observed magnitude split into the two components (medians, with the
        16th-84th percentile range of the light fraction)."""
        mask = self.star.filter_mask
        lines = ['# Binary fit: fraction of the observed flux from the '
                 'secondary in each fitted band,\n'
                 '# and the observed magnitude split into the components '
                 '(m1 = m - 2.5 log10(1 - f2), m2 = m - 2.5 log10(f2)).\n'
                 '# Filter\twavelength_um\tf2_median\tf2_p16\tf2_p84\t'
                 'mag\tmag1\tmag2\n']
        for f_, w_, m_ in zip(self.star.filter_names[mask],
                              self.star.wave[mask], self.star.mags[mask]):
            k = f'frac2_{f_}'
            if k not in samples:
                continue
            fr = np.asarray(samples[k], dtype=float)
            fr = fr[np.isfinite(fr)]
            if not fr.size:
                continue
            f50, f16, f84 = np.percentile(fr, [50, 16, 84])
            with np.errstate(divide='ignore'):
                m1 = m_ - 2.5 * np.log10(1 - f50)
                m2 = m_ - 2.5 * np.log10(f50)
            lines.append(f'{f_}\t{w_:.4f}\t{f50:.4f}\t{f16:.4f}\t'
                         f'{f84:.4f}\t{m_:.4f}\t{m1:.4f}\t{m2:.4f}\n')
        with open(f'{self.out_folder}/{fname}', 'w') as fh:
            fh.write(''.join(lines))

    def _synth_thetas(self, samples, n_draws=3000):
        """Posterior draws (subsampled to ``n_draws``) packed as the theta
        rows ``model_grid`` expects: (ndraws, 5|6). Deterministic."""
        use_norm = self.norm

        def _col(key):
            return np.atleast_1d(np.asarray(samples[key], dtype=float))
        cols = ['teff', 'logg', 'z', 'Av'] + (['norm'] if use_norm
                                              else ['dist', 'rad'])
        arrs = {k: _col(k) for k in cols}
        n = max(a.size for a in arrs.values())
        arrs = {k: np.broadcast_to(a, (n,)) for k, a in arrs.items()}
        if n > n_draws:
            sel = np.random.default_rng(0).choice(n, n_draws, replace=False)
        else:
            sel = np.arange(n)
        if use_norm:
            return np.column_stack([arrs['teff'], arrs['logg'], arrs['z'],
                                    arrs['norm'], arrs['Av']])[sel]
        cols = [arrs['teff'], arrs['logg'], arrs['z'], arrs['dist'],
                arrs['rad'], arrs['Av']]
        if self._binary:
            cols += [np.broadcast_to(_col(k), (n,)) for k in BINARY_PARAMS]
        return np.column_stack(cols)[sel]

    def _synthetic_photometry(self, samples, n_draws=3000):
        """Predicted observed photometry for the FULL filter set at the fitted SED.

        Propagates the posterior (subsampled to ``n_draws``) through the forward
        model on the BT-Settl grid -- which spans every ARIADNE band -- so every
        filter gets an observed synthetic flux and magnitude with 1-sigma errors,
        distance-, radius- and extinction-scaled exactly as the fit models the
        data. The point: predict a band the star lacks (a missing G, K, W1, ...)
        from the bands it has. Bands actually used in the fit are flagged
        ``observed``; the rest are predictions. For white-dwarf / hot-subdwarf
        fits the BT-Settl photosphere is only a rough proxy (noted in the header).

        Returns ``{filter: (wave_um, flux, flux_err, mag, mag_err, observed)}``
        in ``erg s-1 cm-2 um-1`` and the band's native (Vega or AB) mag system.
        """
        from .sed_library import model_grid
        from .phot_utils import flux_to_mag, get_effective_wavelength
        use_norm = self.norm
        av_law = self._av_law
        intp = self.load_interpolator('btsettl')
        # Only the bands BT-Settl models: it lacks e.g. the IUE / TD-1 bands,
        # and asking for those raised KeyError, which skipped the whole step.
        names = [str(f) for f in filter_names if str(f) in intp.column_index]
        wave = np.array([get_effective_wavelength(f) for f in names])
        thetas = self._synth_thetas(samples, n_draws)
        # One array object for every draw: sed_library memoises the sliced
        # grid on the identity of the filter array.
        band_arr = np.array(names)
        fl = np.array([model_grid(th, band_arr, wave, intp, use_norm, av_law,
                                  binary=self._binary)
                       for th in thetas])                 # (ndraws, nbands)
        observed = {str(b) for b in
                    self.star.filter_names[self.star.used_filters == 1]}
        photom = {}
        for j, f in enumerate(names):
            c = fl[:, j]
            c = c[np.isfinite(c) & (c > 0)]
            if c.size < 10:
                continue
            fmed = float(np.median(c))
            ferr = float(0.5 * (np.percentile(c, 84) - np.percentile(c, 16)))
            mag, mag_err = flux_to_mag(fmed, ferr, f)
            if not (np.isfinite(mag) and np.isfinite(mag_err)):
                continue
            photom[f] = (float(wave[j]), fmed, ferr, float(mag), float(mag_err),
                         f in observed)
        return photom

    def _synthetic_user_filters(self, samples, grid_photom=None, n_draws=3000):
        """Predicted photometry in ``self.synthetic_filters`` (issue #88).

        Names that are ARIADNE grid bands are read from the band-grid result
        (``grid_photom``, source ``grid``); every other filter is synthesized
        from the spectra cache's BT-Settl spectra (source ``spectra_cache``)
        and the SAME posterior draws are pushed through the same forward model.
        Returns ``({name: record}, header_note)``; records are NaN where the
        filter is not covered by the cached spectra. Never raises for a missing
        cache: it warns once with the fetch command and skips those filters.
        """
        from . import synth_phot as sp
        from . import config
        from .sed_library import model_grid
        specs = getattr(self, 'synthetic_filters', None)
        if specs is None or (hasattr(specs, '__len__') and len(specs) == 0):
            return {}, ''
        filts = sp.resolve_filters(specs)
        gridnames = {str(f) for f in filter_names}
        todo = [sf for sf in filts if sf.name not in gridnames]
        out = {}
        gp = grid_photom if grid_photom is not None else {}
        for sf in filts:
            if sf.name in gridnames:
                if sf.name in gp:
                    _, fx, fxe, _, _, _ = gp[sf.name]
                    out[sf.name] = sp.user_record(sf, fx, fxe, 'grid')
                else:
                    out[sf.name] = sp.user_record(sf, np.nan, np.nan, 'grid')
        note = 'grid bands: BT-Settl band grid'
        if todo:
            path = getattr(config, 'spectra_cache', None)
            if not path or not os.path.isfile(path):
                logger.warning(
                    'synthetic_filters: %s are not ARIADNE grid bands and need '
                    'the spectra cache, which is missing. Fetch it with: %s '
                    '(or set $ARIADNE_SPECTRA_CACHE). Those filters are '
                    'skipped.', [s.name for s in todo], sp.FETCH_CMD)
            else:
                key = (path, tuple(s.key for s in todo))
                store = self.__dict__.setdefault('_synth_cube_cache', {})
                if key not in store:
                    store[key] = sp.build_cube(path, todo)
                cube = store[key]
                names = [s.name for s in todo]
                wave = np.array([s.wave_um for s in todo])
                thetas = self._synth_thetas(samples, n_draws)
                fl = np.array([model_grid(th, names, wave, cube, self.norm,
                                          self._av_law, binary=self._binary)
                               for th in thetas])
                for j, sf in enumerate(todo):
                    c = fl[:, j]
                    c = c[np.isfinite(c) & (c > 0)]
                    if c.size < 10:
                        out[sf.name] = sp.user_record(sf, np.nan, np.nan,
                                                      'spectra_cache')
                        continue
                    fmed = float(np.median(c))
                    ferr = float(0.5 * (np.percentile(c, 84)
                                        - np.percentile(c, 16)))
                    out[sf.name] = sp.user_record(sf, fmed, ferr,
                                                  'spectra_cache')
                note += (f'; other filters: BT-Settl spectra cache '
                         f'({os.path.basename(path)}, groups '
                         f'{"+".join(sp.CACHE_GROUPS)})')
        # keep the user's order
        order = [s.name for s in filts]
        return {n: out[n] for n in order if n in out}, note

    def _write_synthetic(self, out, samples):
        """Compute and write the synthetic photometry files; never raises."""
        try:
            synphot = self._synthetic_photometry(samples)
        except Exception as e:
            logger.warning('synthetic photometry skipped: %s', e)
            synphot = {}
        user, note = {}, ''
        try:
            user, note = self._synthetic_user_filters(samples, synphot)
        except Exception as e:
            logger.warning('user-filter synthetic photometry skipped: %s', e)
        extra = {n: r for n, r in user.items()
                 if r['source'] == 'spectra_cache' and np.isfinite(r['mag_vega'])}
        photom = dict(synphot)
        for n, r in extra.items():
            photom[n] = (r['wavelength_um'], r['flux'], r['flux_err'],
                         r['mag_vega'], r['mag_vega_err'], False)
        out['synthetic_photometry'] = photom
        out['synthetic_user_filters'] = user
        try:
            synthdat, magdat = self._synth_phot_tables(photom, user_names=extra)
            with open(f'{self.out_folder}/synthetic_fluxes.dat', 'w') as fh:
                fh.write(synthdat)
            with open(f'{self.out_folder}/synthetic_magnitudes.dat', 'w') as fh:
                fh.write(magdat)
            if user:
                from . import synth_phot as sp
                with open(f'{self.out_folder}/synthetic_user_filters.dat',
                          'w') as fh:
                    fh.write(sp.user_table(user, note))
        except Exception as e:
            logger.warning('writing synthetic photometry files failed: %s', e)

    @staticmethod
    def _synth_phot_tables(photom, user_names=()):
        """(flux_table, mag_table) strings for the synthetic-photometry files.

        ``user_names``: names of user-specified (spectra-cache) filters in
        ``photom``; their magnitudes are Vega-system (noted in the header).
        """
        flux = 'Filter\tFlux\n'
        mag = ('# Predicted observed photometry from the best-fit SED (BT-Settl '
               'grid).\n# observed=1: band used in the fit; observed=0: '
               'prediction. Mag in the band native (Vega/AB) system.\n')
        if len(user_names):
            mag += ('# User-specified filters (Fitter.synthetic_filters) from '
                    'the spectra cache: Vega mags, see '
                    'synthetic_user_filters.dat for AB.\n')
        mag += ('# Filter\twavelength_um\tmag\tmag_err\tflux_erg_cm2_s_um\t'
                'flux_err\tobserved\n')
        for f, (w, fx, fxe, mg, mge, ob) in photom.items():
            flux += f'{f}\t{fx:.6e}\n'
            mag += (f'{f}\t{w:.4f}\t{mg:.4f}\t{mge:.4f}\t{fx:.6e}\t{fxe:.6e}\t'
                    f'{int(ob)}\n')
        return flux, mag

    def save_bma(self, avgd):
        """Save BMA output and relevant information.

        Saves a dictionary as a pickle file. The dictionary contains the
        following:

        lnZ : The global evidences.
        posterior_samples : A dictionary containing the samples of each
                            parameter (even if it's fixed), the evidence,
                            log likelihood, the prior, and the posterior
                            for each set of sampled parameters.
        fixed : An array with the fixed parameter values.
        coordinator : An array with the status of each parameter (1 for fixed
                      0 for free)
        best_fit : The best fit is chosen to be the median of each sample.
                   It also includes the log likelihood of the best fit.
        star : The Star object containing the information of the star (name,
               magnitudes, fluxes, coordinates, etc)

        Also creates a log file with the best fit parameters, 1 sigma
        error bars and 3 sigma CIs.

        """
        out = dict()
        logdat_samples = '#Parameter\tmedian\tupper\tlower\t3sig_low\t3sig_up\n'
        logdat_average = '#Parameter\tmedian\tupper\tlower\t3sig_low\t3sig_up\n'
        log_out_samples = f'{self.out_folder}/best_fit_sample.dat'
        log_out_average = f'{self.out_folder}/best_fit_average.dat'
        prob_out = f'{self.out_folder}/model_probabilities.dat'

        # Save global evidence of each model.
        out['lnZ'] = avgd['evidences']

        # Save original samples.
        out['originals'] = avgd['originals']

        # Save weights.
        out['weights'] = avgd['weights']

        # Create raw samples holder
        out['weighted_samples'] = dict()
        out['weighted_average'] = dict()
        j = 0
        for i, par in enumerate(order):
            if not self.coordinator[i]:
                out['weighted_samples'][par] = avgd['weighted_samples'][par]
                out['weighted_average'][par] = avgd['weighted_average'][par]
                j += 1
            else:
                out['weighted_samples'][par] = self.fixed[i]
                out['weighted_average'][par] = self.fixed[i]

        # If normalization constant was fitted, create a distribution of radii.

        if use_norm and star.dist != -1:
            rad_sampled = self._get_rad(
                out['weighted_samples']['norm'], star.dist, star.dist_e
            )
            rad_averageed = self._get_rad(
                out['weighted_average']['norm'], star.dist, star.dist_e
            )
            out['weighted_samples']['rad'] = rad_sampled
            out['weighted_average']['rad'] = rad_averageed

        # Create a distribution of masses

        logg_samp = out['weighted_samples']['logg']
        logg_average = out['weighted_average']['logg']
        rad_samp = out['weighted_samples']['rad']
        rad_average = out['weighted_average']['rad']
        mass_samp = self._get_mass(logg_samp, rad_samp)
        mass_average = self._get_mass(logg_average, rad_average)
        out['weighted_samples']['grav_mass'] = mass_samp
        out['weighted_average']['grav_mass'] = mass_average

        # Create a distribution of luminosities.

        teff_samp = out['weighted_samples']['teff']
        teff_average = out['weighted_average']['teff']
        lum_samp = self._get_lum(teff_samp, rad_samp)
        lum_average = self._get_lum(teff_average, rad_average)
        out['weighted_samples']['lum'] = lum_samp
        out['weighted_average']['lum'] = lum_average

        # Create a distribution of angular diameters.

        if not use_norm:
            dist_samp = out['weighted_samples']['dist']
            dist_average = out['weighted_average']['dist']
            ad_samp = self._get_angular_diameter(rad_samp, dist_samp)
            ad_average = self._get_angular_diameter(rad_average, dist_average)
            out['weighted_samples']['AD'] = ad_samp
            out['weighted_average']['AD'] = ad_average

        if binary:
            self._add_secondary_derived(out['weighted_samples'])
            self._add_secondary_derived(out['weighted_average'])
            # Light fractions: averaged across the grid pairs like any other
            # sampled quantity (bayesian_model_average).
            for k, v in avgd['weighted_samples'].items():
                if k.startswith('frac2_'):
                    out['weighted_samples'][k] = v
                    out['weighted_average'][k] = avgd['weighted_average'][k]

        # Best fit
        # The logic is as follows:
        # Calculate KDE for each marginalized posterior distributions
        # Find peak
        # peak is best fit.

        out['best_fit_samples'] = dict()
        out['uncertainties_samples'] = dict()
        out['confidence_interval_samples'] = dict()
        out['best_fit_averaged'] = dict()
        out['uncertainties_averaged'] = dict()
        out['confidence_interval_averaged'] = dict()
        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if is_excess_scatter(param):
                    continue
                samp = out['weighted_samples'][param]
                sampw = out['weighted_average'][param]

                if param == 'z':
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                '[Fe/H]', out, method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                '[Fe/H]', out,
                                                method='averaged')
                elif param == 'norm':
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                '(R/D)^2', out, fmt='e',
                                                method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                '(R/D)^2', out, fmt='e',
                                                method='averaged')
                    if star.dist != 1:
                        logdat_samples = out_filler(
                            out['weighted_samples']['rad'], logdat_samples,
                            'rad', 'R', out, method='samples'
                        )
                        logdat_average = out_filler(
                            out['weighted_average']['rad'], logdat_average,
                            'rad', 'R', out, method='averaged'
                        )

                else:
                    logdat_samples = out_filler(samp, logdat_samples, param,
                                                param, out, method='samples')
                    logdat_average = out_filler(sampw, logdat_average, param,
                                                param, out, method='averaged')
            else:
                logdat_samples = out_filler(
                    0, logdat_samples, param, param, out, fixed=self.fixed[i],
                    method='samples'
                )
                logdat_average = out_filler(
                    0, logdat_average, param, param, out, fixed=self.fixed[i],
                    method='averaged'
                )

        # Add derived mass to best fit dictionary.

        samp = out['weighted_samples']['grav_mass']
        sampw = out['weighted_average']['grav_mass']
        logdat_samples = out_filler(samp, logdat_samples, 'grav_mass',
                                    'grav_mass', out, method='samples')
        logdat_average = out_filler(sampw, logdat_average, 'grav_mass',
                                    'grav_mass', out, method='averaged')

        # Add derived luminosity to best fit dictionary.

        samp = out['weighted_samples']['lum']
        sampw = out['weighted_average']['lum']
        logdat_samples = out_filler(samp, logdat_samples, 'lum', 'lum', out,
                                    method='samples')
        logdat_average = out_filler(sampw, logdat_average, 'lum', 'lum', out,
                                    method='averaged')

        # Add derived angular diameter to best fit dictionary.

        if not use_norm:
            samp = out['weighted_samples']['AD']
            sampw = out['weighted_average']['AD']
            logdat_samples = out_filler(samp, logdat_samples, 'AD', 'AD', out,
                                        method='samples')
            logdat_average = out_filler(sampw, logdat_average, 'AD', 'AD', out,
                                        method='averaged')

        if binary:
            for p_ in ('grav_mass2', 'lum2', 'AD2'):
                logdat_samples = out_filler(
                    out['weighted_samples'][p_], logdat_samples, p_, p_, out,
                    method='samples')
                logdat_average = out_filler(
                    out['weighted_average'][p_], logdat_average, p_, p_, out,
                    method='averaged')

        # Add estimated age to best fit dictionary. This is done with the wider
        # sampled distribution instead of the averaged one in order to save time
        # (for a binary: the primary's, from photometry with the secondary's
        # light removed).

        if self.estimate_age:
            age_samp, mass_samp, eep_samp = self._run_estimate_age(
                out['best_fit_samples'],
                out['uncertainties_samples'],
                c=choice(self.colors),
                samples=out['weighted_samples']
            )
            # Create new thingy for MIST samples. Sadly now everything done before
            # this update will be incompatible :(
            out['mist_samples'] = dict()
            out['mist_samples']['age'] = age_samp
            out['mist_samples']['iso_mass'] = mass_samp
            out['mist_samples']['eep'] = eep_samp
            logdat_samples = out_filler(age_samp, logdat_samples, 'age', 'age', out,
                                        method='samples')
            logdat_samples = out_filler(mass_samp, logdat_samples, 'iso_mass',
                                        'iso_mass', out, method='samples')
            logdat_samples = out_filler(eep_samp, logdat_samples, 'eep', 'eep', out,
                                        method='samples')
            # Ugly... but faster than doing the kde 2x times...
            age = out['best_fit_samples']['age']
            age_unc = out['uncertainties_samples']['age']
            age_ci = out['confidence_interval_samples']['age']
            iso_mass = out['best_fit_samples']['iso_mass']
            iso_mass_unc = out['uncertainties_samples']['iso_mass']
            iso_mass_ci = out['confidence_interval_samples']['iso_mass']
            eep = out['best_fit_samples']['eep']
            eep_unc = out['uncertainties_samples']['eep']
            eep_ci = out['confidence_interval_samples']['eep']
            out['best_fit_averaged']['age'] = age
            out['best_fit_averaged']['iso_mass'] = iso_mass
            out['best_fit_averaged']['eep'] = eep
            out['uncertainties_averaged']['age'] = age_unc
            out['uncertainties_averaged']['iso_mass'] = iso_mass_unc
            out['uncertainties_averaged']['eep'] = eep_unc
            out['confidence_interval_averaged']['age'] = age_ci
            out['confidence_interval_averaged']['iso_mass'] = iso_mass_ci
            out['confidence_interval_averaged']['eep'] = eep_ci
            logdat_average += f'age\t{age:.4f}\t'
            logdat_average += f'{age_unc[1]:.4f}\t{age_unc[0]:.4f}\t'
            logdat_average += f'{age_ci[0]:.4f}\t{age_ci[1]}\n'
            logdat_average += f'iso_mas\t{iso_mass:.4f}\t'
            logdat_average += f'{iso_mass_unc[1]:.4f}\t{iso_mass_unc[0]:.4f}\t'
            logdat_average += f'{iso_mass_ci[0]:.4f}\t{iso_mass_ci[1]}\n'
            logdat_average += f'eep\t{eep:.4f}\t'
            logdat_average += f'{eep_unc[1]:.4f}\t{eep_unc[0]:.4f}\t'
            logdat_average += f'{eep_ci[0]:.4f}\t{eep_ci[1]}\n'
        else:
            out['mist_samples'] = {}
        ###
        probdat = ''

        max_prob = 0
        max_prob_mod = ''
        for k in avgd['weights'].keys():
            if avgd['weights'][k] > max_prob:
                max_prob = avgd['weights'][k]
                max_prob_mod = k
            probdat += f'{k}_probability\t{avgd["weights"][k]:.4f}\n'

        # Predicted OBSERVED synthetic photometry (flux + magnitude) for the
        # full band set at the fitted SED of the max-probability model,
        # propagated through the forward model (distance/radius/extinction-
        # scaled). Never breaks the save.
        self._write_synthetic(out, avgd['originals'][max_prob_mod])

        for i, param in enumerate(order):
            if not self.coordinator[i]:
                if not is_excess_scatter(param):
                    continue
                samp = out['weighted_samples'][param]
                sampw = out['weighted_average'][param]
                logdat_samples = out_filler(samp, logdat_samples, param, param,
                                            out, fmt='f', method='samples')
                logdat_average = out_filler(sampw, logdat_average, param, param,
                                            out, fmt='f', method='averaged')

        out['fixed'] = self.fixed
        out['coordinator'] = self.coordinator
        out['star'] = self.star
        out['norm'] = self.norm
        out['engine'] = 'Bayesian Model Averaging'
        out['av_law'] = av_law
        out['binary'] = bool(binary)
        out['contrast'] = list(self.contrast or []) if binary else []

        # Spectral type

        # Load Mamajek spt table
        mamajek_spt, mamajek_temp = np.loadtxt(f'{filesdir}/mamajek_spt.dat',
                                               dtype=str, usecols=[0, 1],
                                               unpack=True)

        mamajek_temp = mamajek_temp.astype(float)

        # Find spt
        spt_idx = np.argmin(
            abs(mamajek_temp - out['best_fit_averaged']['teff']))
        spt = mamajek_spt[spt_idx]
        out['spectral_type'] = spt
        if binary:
            out['spectral_type2'] = mamajek_spt[np.argmin(
                abs(mamajek_temp - out['best_fit_averaged']['teff2']))]
            self._write_light_fractions(out['weighted_samples'])
        out_file = f'{self.out_folder}/BMA.pkl'
        with open(log_out_samples, 'w') as logfile:
            logfile.write(logdat_samples)
        with open(log_out_average, 'w') as logfile:
            logfile.write(logdat_average)
        with open(prob_out, 'w') as logfile:
            logfile.write(probdat)
        pickle.dump(out, open(out_file, 'wb'))
        self.out = out
        nc_file = f'{self.out_folder}/ariadne_result.nc'
        self.to_netcdf(nc_file)

    @staticmethod
    def bayesian_model_average(outputs, grids, norm, nsamples, c='white'):
        """Perform Bayesian Model Averaging.

        Parameters
        ----------
        outputs: array_like
            An array or list of output pickle files from the individual model
            fits.
        grids: array_like
            An array or list of the model grids used to model.
        norm: bool
            Flag indicating if the normalization factor was fit for.
        nsamples: int
            The number of samples to sample from the averaged distribution.
        c: str, optional
            Termcolor color.

        """
        evidences = []
        post_samples = []
        model_posteriors = []
        # Read and extract model posterior information.
        for o in outputs:
            with open(o, 'rb') as f:
                model_posteriors.append(upgrade_legacy_names(pickle.load(f)))
        # Extract the evidences of each model.
        for o in model_posteriors:
            evidences.append(o['global_lnZ'])
            post_samples.append(o['posterior_samples'])
        # Convert evidences to weights/probabilities via a softmax of the
        # log-evidences. Shift by the MAX (not the min) so the largest exponent
        # is 0: softmax is shift-invariant, so the weights are unchanged in the
        # normal regime, but shifting by the min lets exp() overflow to inf ->
        # nan when one grid's log-evidence dominates (delta lnZ > ~709).
        evidences = np.array(evidences)
        weights = evidences - evidences.max()
        weights = [np.exp(e) / np.exp(weights).sum() for e in weights]
        weights = np.array(weights)
        # We're not averaging these
        ban = ['loglike', 'priors', 'posteriors']
        if norm:  # if normalization was used, we won't average the radius
            ban.append('rad')
        # Create an output dictionary for the averaged samples.
        out = dict()
        out['originals'] = dict()
        out['weights'] = dict()
        # Populate the dict with the probabilities and individual model
        # posteriors
        for i, o in enumerate(model_posteriors):
            out['weights'][o['model_grid']] = weights[i]
            out['originals'][o['model_grid']] = o['posterior_samples']
        out['weighted_samples'] = dict()
        out['weighted_average'] = dict()

        print(colored('\t\t*** AVERAGING POSTERIOR SAMPLES ***', c))
        for k in tqdm(post_samples[0].keys()):
            if k in ban:  # Skip things that are not main parameters.
                continue
            try:  # Skip fixed params.
                len(post_samples[0][k])
            except TypeError:
                continue
            traces = []
            extended_weights = []
            out['weighted_samples'][k] = np.zeros(nsamples)
            out['weighted_average'][k] = np.zeros(nsamples)
            for i, o in enumerate(post_samples):
                # This is for the weighted sampling.
                traces.append(o[k])
                extended_weights.append(np.ones(len(o[k])) * weights[i])
                # This is for the weighted averaging.
                # The weighted averaging consists of taking the weighted
                # average of the posterior samples, supersampled to maked them
                # coincide in length.
                weighted_samples = choice(o[k], nsamples) * weights[i]
                out['weighted_average'][k] += weighted_samples
            # Do the weighted sampling. For this we're going to estimate the
            # KDE of the 'master' posterior built by taking random samples
            # from each model where the number of samples is proportional
            # to the model's relative probability. This is equivalent
            # to taking the KDE of each model and then do the weighted average
            avg_kde = st.gaussian_kde(np.concatenate(traces),
                                   weights=np.concatenate(extended_weights))
            out['weighted_samples'][k] = avg_kde.resample(nsamples)[0]
        # Now we save the evidences.
        out['evidences'] = dict()
        for e, g in zip(evidences, grids):
            out['evidences'][g] = e
        return out

    @staticmethod
    def multinest_results(out_folder, ndim):
        """Extract posterior samples, global evidence and its error."""
        path = f'{out_folder}/mnest/chains'
        output = pymultinest.Analyzer(outputfiles_basename=path, n_params=ndim)
        posterior_samples = output.get_equal_weighted_posterior()[:, :-1]
        lnz = output.get_stats()['global evidence']
        lnzer = output.get_stats()['global evidence error']
        return lnz, lnzer, posterior_samples

    @staticmethod
    def dynesty_results(results):
        """Extract posterior samples, global evidence and its error."""
        weights = np.exp(results['logwt'] - results['logz'][-1])
        posterior_samples = resample_equal(results.samples, weights)
        lnz = results.logz[-1]
        lnzer = results.logzerr[-1]
        return lnz, lnzer, posterior_samples

    @staticmethod
    def _get_mass(logg, rad):
        """Calculate mass from logg and radius."""
        # Solar logg = 4.437
        # g = g_Sol * M / R**2
        mass = logg + 2 * np.log10(rad) - 4.437
        mass = 10 ** mass
        return mass

    @staticmethod
    def _get_lum(teff, rad):
        sb = sigma_sb.to(u.solLum / u.K ** 4 / u.solRad ** 2).value
        L = 4 * np.pi * rad ** 2 * sb * teff ** 4
        return L

    @staticmethod
    def _get_rad(samples, dist, dist_e):
        """Calculate radius from the normalization constant and distance."""
        norm = samples
        # Create a synthetic distribution for distance.
        # N = (R / D) ** 2
        b, a = (np.inf - dist) / dist_e, (0 - dist) / dist_e
        d = st.truncnorm(loc=dist, scale=dist_e, a=a, b=b).rvs(
            size=norm.shape[0])
        n = np.sqrt(norm)
        r = n * d  # This is in pc
        r *= u.pc.to(u.solRad)  # Transform to Solar radii
        return r

    @staticmethod
    def _get_angular_diameter(rad, dist):
        diameter = 2 * rad
        ad = (diameter / (dist * u.pc.to(u.solRad))) * u.rad.to(u.marcsec)
        return ad

    @staticmethod
    def load_interpolator(model):
        """Load a DFInterpolator."""
        return load_band_grid(model)

    def _run_estimate_age(self, bf, unc, c='white', samples=None):
        """Estimate age using MIST isochrones.

        Parameters
        ----------
        bf: dict
            A dictionary with the best fit parameters.
        unc: dict
            A dictionary with the uncertainties.

        """
        print(
            colored(
                '\t\t*** ESTIMATING AGE AND MASS USING MIST ISOCHRONES ***', c
            )
        )
        params = dict()  # params for isochrones.
        for i, k in enumerate(order):
            # The secondary of a binary (teff2, ...) is not fitted to MIST.
            if k == 'logg' or is_excess_scatter(k) or k in BINARY_PARAMS:
                continue
            if k == 'teff':
                par = 'Teff'
            if k == 'z':
                par = 'feh'
            if k == 'dist':
                par = 'distance'
            if k == 'rad':
                par = 'radius'
            if k == 'Av':
                par = 'AV'
            if not self.coordinator[i]:
                if k != 'norm':
                    params[par] = (bf[k], max(unc[k]))
                if k == 'norm' and self.star.dist != -1:
                    params['distance'] = (self.star.dist, self.star.dist_e)
                if par == 'distance':
                    err = max(unc[k])
                    params['parallax'] = (1000 / bf[k], 1000 * err / bf[k] ** 2)
            else:
                continue

        params['mass'] = (bf['grav_mass'], max(unc['grav_mass']))
        if self.star.lum != 0 and self.star.lum_e != 0:
            params['logL'] = (np.log10(bf['lum']),
                              abs(np.log10(max(unc['lum']))))
        mags = self.star.mags[iso_mask == 1]
        mags_e = self.star.mag_errs[iso_mask == 1]
        names = self.star.filter_names[iso_mask == 1]
        used_bands = []
        for m, e, b, fn in zip(mags, mags_e, iso_bands, names):
            if m == 0:
                continue
            if self._binary:
                # The primary's magnitude: remove the secondary's share of
                # the light (posterior), and add its spread to the error. A
                # band without a light fraction (not fitted) is left out.
                fr = (samples or {}).get(f'frac2_{fn}')
                if fr is None:
                    continue
                fr = np.asarray(fr, dtype=float)
                fr = fr[np.isfinite(fr) & (fr < 1)]
                if fr.size < 10:
                    continue
                m1 = m - 2.5 * np.log10(1 - fr)
                m, e = float(np.median(m1)), float(np.hypot(e, np.std(m1)))
            params[b] = (m, e)
            used_bands.append(b)

        age_samp, mass_samp, eep_samp = estimate(
            used_bands, params, logg=False,
            out_folder=self.out_folder,
            threads=self._threads, dlogz=self.isochrone_dlogz)

        return age_samp, mass_samp, eep_samp

    def to_dict(self):
        """Return BMA results as a structured dict matching the ecosystem spec.

        Returns a dictionary with keys ``posterior``, ``observed_data``, and
        ``sample_stats``, each containing numpy arrays ready for conversion to
        ``arviz.InferenceData``.

        Raises
        ------
        RuntimeError
            If the fitter has not been run yet (``self.out`` is not populated).
        """
        if not hasattr(self, 'out') or self.out is None:
            raise RuntimeError(
                'No results available. Run the fitter before calling to_dict().'
            )
        out = self.out
        ws = out.get('weighted_samples', {})

        # Determine number of draws from any free parameter
        n_draws = None
        for v in ws.values():
            if hasattr(v, '__len__') and len(v) > 1:
                n_draws = len(v)
                break
        if n_draws is None:
            n_draws = 100_000  # fallback

        def _to_chain(arr_or_scalar):
            """Reshape to (1, n_draws) for single-chain arviz convention."""
            if hasattr(arr_or_scalar, '__len__') and len(arr_or_scalar) > 1:
                return np.array(arr_or_scalar).reshape(1, -1)
            return np.full((1, n_draws), float(arr_or_scalar))

        # Map internal names → ecosystem spec names
        param_map = {
            'teff': 'Teff',
            'logg': 'logg',
            'z': 'feh',
            'rad': 'radius',
            'lum': 'luminosity',
            'dist': 'distance',
            'Av': 'Av',
            'teff2': 'Teff2',
            'logg2': 'logg2',
            'rad2': 'radius2',
            'lum2': 'luminosity2',
        }

        posterior = {}
        for internal, spec in param_map.items():
            if internal in ws:
                posterior[spec] = _to_chain(ws[internal])

        # Observed photometry
        mask = self.star.filter_mask
        observed_data = {
            'wavelength': self.star.wave[mask],  # already in micron
            'flux': self.star.flux[mask],
            'flux_err': self.star.flux_er[mask],
        }

        # Model evidence and weights
        weights = out.get('weights', {})
        lnZ = out.get('lnZ', {})

        model_names = list(weights.keys())
        weight_vals = np.array([weights[m] for m in model_names])
        lnZ_vals = np.array([lnZ.get(m, 0.0) for m in model_names])
        bma_log_evidence = float(np.sum(weight_vals * lnZ_vals))

        sample_stats = {
            'log_evidence': bma_log_evidence,
            'model_names': model_names,
            'model_weights': weight_vals,
        }

        # Per-model posteriors from originals
        originals = out.get('originals', {})
        model_posteriors = {}
        for model_name, model_samples in originals.items():
            n_model = None
            for v in model_samples.values():
                if hasattr(v, '__len__') and len(v) > 1:
                    n_model = len(v)
                    break
            if n_model is None:
                continue

            def _to_model_chain(a, nd=n_model):
                if hasattr(a, '__len__') and len(a) > 1:
                    return np.array(a).reshape(1, -1)
                return np.full((1, nd), float(a))

            mp = {}
            for internal, spec in param_map.items():
                if internal in model_samples:
                    mp[spec] = _to_model_chain(model_samples[internal])
            if 'loglike' in model_samples:
                mp['log_likelihood'] = _to_model_chain(
                    model_samples['loglike'])
            model_posteriors[model_name] = mp

        # MIST isochrone samples
        mist = out.get('mist_samples', {})
        mist_posterior = {}
        for k in ('age', 'iso_mass', 'eep'):
            if k in mist and hasattr(mist[k], '__len__') and len(mist[k]) > 1:
                mist_posterior[k] = _to_chain(mist[k])

        # Filter names and bandwidths for SED reconstruction
        observed_data['filter_names'] = np.array(
            self.star.filter_names[mask].tolist()
            if hasattr(self.star.filter_names[mask], 'tolist')
            else list(self.star.filter_names[mask])
        )
        observed_data['bandwidths'] = self.star.bandpass[mask]

        # Model SED fluxes (injected by worker from SEDPlotter)
        model_sed = {}
        if 'model_sed' in out:
            ms = out['model_sed']
            if 'model_flux' in ms:
                model_sed['model_flux'] = np.array(ms['model_flux'])

        # Summary statistics for quick display
        summary = {}
        for src_key in ('best_fit_averaged', 'uncertainties_averaged',
                        'confidence_interval_averaged'):
            if src_key in out:
                summary[src_key] = {}
                for k, v in out[src_key].items():
                    if hasattr(v, '__len__'):
                        summary[src_key][k] = np.array(
                            [float(x) for x in v])
                    else:
                        summary[src_key][k] = float(v)

        return {
            'posterior': posterior,
            'observed_data': observed_data,
            'sample_stats': sample_stats,
            'model_posteriors': model_posteriors,
            'mist_posterior': mist_posterior,
            'model_sed': model_sed,
            'summary': summary,
        }

    def to_netcdf(self, path):
        """Export BMA results as arviz DataTree in netCDF4 format.

        Produces an ``arviz.InferenceData``-compatible ``.nc`` file
        following the ecosystem output spec.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``'ariadne_result.nc'``).
        """
        import arviz as az

        d = self.to_dict()

        ss = d['sample_stats']
        data = {
            'posterior': d['posterior'],
            'observed_data': d['observed_data'],
            'constant_data': {
                'log_evidence': np.array(ss['log_evidence']),
                'model_weights': ss['model_weights'],
                'model_names': np.array(ss['model_names']),
            },
        }

        # Add per-model posteriors as separate groups
        for model_name, mp in d.get('model_posteriors', {}).items():
            data[f'posterior_{model_name}'] = mp

        # Add MIST isochrone posteriors
        if d.get('mist_posterior'):
            data['posterior'].update(d['mist_posterior'])

        # Add model SED fluxes to observed_data
        if d.get('model_sed') and 'model_flux' in d['model_sed']:
            data['observed_data']['model_flux'] = d['model_sed']['model_flux']

        # Add summary stats to constant_data
        for stat_key in ('best_fit_averaged', 'uncertainties_averaged',
                         'confidence_interval_averaged'):
            if stat_key in d.get('summary', {}):
                for param, val in d['summary'][stat_key].items():
                    key = f'{stat_key}__{param}'
                    data['constant_data'][key] = np.array(val)

        if int(az.__version__.split('.')[0]) >= 1:
            dt = az.from_dict(data)
        else:
            # arviz < 1 takes one keyword per standard group; a positional
            # dict would become the posterior. The per-model posteriors are
            # custom groups, added afterwards.
            std = ('posterior', 'observed_data', 'constant_data')
            dt = az.from_dict(**{g: data[g] for g in std})
            extra = {g: v for g, v in data.items() if g not in std}
            if extra:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', message='.*not defined in the InferenceData')
                    dt.add_groups(extra)
        dt.to_netcdf(path)
        logger.info(f'Wrote InferenceData to {path}')


#####################
# Dynesty and multinest wrappers


# Set by fit_bma before forking the grid-parallel Pool; read by the workers.
_BMA_FITTER = None


def _bma_grid_worker(index):
    """Fit one BMA model grid single-threaded and write its ``_out.pkl``.

    Runs in a forked worker process: the star photometry, priors, coordinator
    and other module globals are inherited from the parent, so only the
    per-grid interpolator is swapped in. The fit result is delivered via the
    on-disk pickle (the parent reads it back for averaging), so nothing needs
    to be pickled back to the parent. Returns ``(grid, None)`` on success or
    ``(grid, error)`` on failure.
    """
    global interpolator
    fitter = _BMA_FITTER
    intp = fitter._interpolators[index]
    grid = fitter._grids[index]
    out_file = f'{fitter.out_folder}/{grid}_out.pkl'
    # Forked processes inherit the parent's RNG state; reseed so each grid
    # draws independently.
    np.random.seed()
    interpolator = intp
    fitter._threads = 1   # grids are the unit of parallelism; no inner Pool
    fitter._grid = grid   # used by save() for out['model_grid']
    try:
        fitter.fit_dynesty(out_file=out_file)
    except ValueError as e:
        dump_out = f'{fitter.out_folder}/{grid}_DUMP.pkl'
        pickle.dump(fitter.sampler.results, open(dump_out, 'wb'))
        return grid, e
    return grid, None


def dynesty_loglike_bma(cube, interpolator):
    """Dynesty log likelihood wrapper for BMA."""
    mask = star.filter_mask
    flux = star.flux[mask]
    flux_er = star.flux_er[mask]
    filts = star.filter_names[mask]
    wave = star.wave[mask]
    theta = build_params(
        cube, flux, flux_er, filts, coordinator, fixed, use_norm
    )
    return log_likelihood(theta, flux, flux_er, wave,
                          filts, interpolator, use_norm, av_law,
                          binary=binary, contrast=contrast)


def dynesty_log_like(cube):
    """Dynesty log likelihood wrapper."""
    theta = build_params(
        cube, flux, flux_er, filts, coordinator, fixed, use_norm
    )
    return log_likelihood(theta, flux, flux_er, wave,
                          filts, interpolator, use_norm, av_law,
                          binary=binary, contrast=contrast)


def pt_dynesty(cube):
    """Dynesty prior transform."""
    return prior_transform_dynesty(cube, flux, flux_er, filts, prior_dict,
                                   coordinator, use_norm, binary)


def multinest_log_like(cube, ndim, nparams):
    """Multinest log likelihood wrapper."""
    theta = [cube[i] for i in range(ndim)]
    theta = build_params(
        theta, flux, flux_er, filts, coordinator, fixed, use_norm
    )
    return log_likelihood(theta, flux, flux_er, wave,
                          filts, interpolator, use_norm, av_law)


def pt_multinest(cube, ndim, nparams):
    """Multinest prior transform."""
    prior_transform_multinest(cube, flux, flux_er, filts, prior_dict,
                              coordinator, use_norm)
