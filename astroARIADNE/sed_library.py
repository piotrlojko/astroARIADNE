"""sed_library contain the model, prior and likelihood to be used."""

import numba as nb
import numpy as np
from extinction import apply
from scipy.special import ndtr, ndtri

from .utils import excess_scatter_name

# The parameter order and the extinction curve are constant for a given fit
# (they depend only on the filter set, the use_norm flag and the extinction
# law), yet the prior transform and the model are evaluated millions of times
# during nested sampling. Cache them so the per-call cost is just a dict lookup
# instead of rebuilding arrays and re-running the extinction law every time.
_ORDER_CACHE = {}
_EXT_CACHE = {}
_TRANSFORM_CACHE = {}
_GRID_CACHE = {}
_BAND_EXT_CACHE = {}


@nb.njit(cache=True, nogil=True)
def _searchsorted(arr, x):
    """Binary search matching isochrones' searchsorted: returns (L, eq).

    ``eq`` is True when ``x`` exactly equals a grid node, in which case ``L``
    is that node's index; otherwise ``L`` is the insertion index (upper
    bracket). Ported verbatim so bracket selection is identical.
    """
    N = arr.shape[0]
    L = 0
    R = N - 1
    eq = False
    m = (L + R) // 2
    done = False
    while not done:
        xm = arr[m]
        if xm < x:
            L = m + 1
        elif xm > x:
            R = m - 1
        else:
            L = m
            eq = True
            done = True
        m = (L + R) // 2
        if L > R:
            done = True
    return L, eq


@nb.njit(cache=True, nogil=True)
def _trilinear3d(x0, x1, x2, ii0, ii1, ii2, cube):
    """GIL-free trilinear interpolation over a pre-sliced (n0,n1,n2,nf) cube.

    Replicates isochrones' ``interp_value_3d`` exactly: NaN input or any axis
    out of bounds returns NaN for every filter; otherwise the 8 corner values
    are combined with the same edge weights and accumulation order. Edge
    indices are clamped, which only ever touches the zero-weight corner of an
    exact-node hit (so the result is unchanged) while avoiding out-of-bounds
    reads.
    """
    nf = cube.shape[3]
    out = np.empty(nf, dtype=nb.float64)
    n0 = ii0.shape[0]
    n1 = ii1.shape[0]
    n2 = ii2.shape[0]
    if (x0 != x0 or x1 != x1 or x2 != x2
            or x0 < ii0[0] or x0 > ii0[n0 - 1]
            or x1 < ii1[0] or x1 > ii1[n1 - 1]
            or x2 < ii2[0] or x2 > ii2[n2 - 1]):
        for k in range(nf):
            out[k] = np.nan
        return out

    l0, eq0 = _searchsorted(ii0, x0)
    if eq0:
        i0 = l0
        y0 = 0.0
    else:
        i0 = l0 - 1
        y0 = (x0 - ii0[i0]) / (ii0[l0] - ii0[i0])
    l1, eq1 = _searchsorted(ii1, x1)
    if eq1:
        i1 = l1
        y1 = 0.0
    else:
        i1 = l1 - 1
        y1 = (x1 - ii1[i1]) / (ii1[l1] - ii1[i1])
    l2, eq2 = _searchsorted(ii2, x2)
    if eq2:
        i2 = l2
        y2 = 0.0
    else:
        i2 = l2 - 1
        y2 = (x2 - ii2[i2]) / (ii2[l2] - ii2[i2])

    for k in range(nf):
        out[k] = 0.0
    for e in range(8):
        d0 = (e >> 2) & 1
        d1 = (e >> 1) & 1
        d2 = e & 1
        w = (y0 if d0 else 1.0 - y0) \
            * (y1 if d1 else 1.0 - y1) \
            * (y2 if d2 else 1.0 - y2)
        j0 = i0 + d0 if i0 + d0 < n0 else n0 - 1
        j1 = i1 + d1 if i1 + d1 < n1 else n1 - 1
        j2 = i2 + d2 if i2 + d2 < n2 else n2 - 1
        for k in range(nf):
            out[k] += cube[j0, j1, j2, k] * w
    return out


@nb.njit(cache=True, nogil=True)
def _bracket(ii, x):
    """Lower node index and fractional position of ``x`` on axis ``ii``, as
    in ``_trilinear3d`` (an exact node hit gives that node and 0)."""
    l, eq = _searchsorted(ii, x)
    if eq:
        return l, 0.0
    i = l - 1
    return i, (x - ii[i]) / (ii[l] - ii[i])


@nb.njit(cache=True, nogil=True)
def _band_ext_factor(x0, x1, x2, av, ii0, ii1, ii2, avn, cube):
    """``10**(-0.4 dA)`` per band from a band-extinction table.

    ``cube[n0, n1, n2, band, i_av, 0]`` is dA at extinction ``avn[i_av]`` and
    ``[..., 1]`` its derivative in A_V (see ``band_extinction``). Trilinear in
    the grid axes like ``_trilinear3d``; cubic Hermite in A_V between the
    nodes, linear beyond them. Returns ones where the grid lookup is out of
    bounds or NaN, where the band flux itself is NaN.
    """
    nbands = cube.shape[3]
    out = np.ones(nbands)
    n0 = ii0.shape[0]
    n1 = ii1.shape[0]
    n2 = ii2.shape[0]
    nav = avn.shape[0]
    if (x0 != x0 or x1 != x1 or x2 != x2 or av != av
            or x0 < ii0[0] or x0 > ii0[n0 - 1]
            or x1 < ii1[0] or x1 > ii1[n1 - 1]
            or x2 < ii2[0] or x2 > ii2[n2 - 1]):
        return out
    i0, y0 = _bracket(ii0, x0)
    i1, y1 = _bracket(ii1, x1)
    i2, y2 = _bracket(ii2, x2)
    hermite = False
    j = 0
    dt = 0.0
    h00 = h10 = h01 = h11 = 0.0
    if av <= avn[0]:
        j = 0
        dt = av - avn[0]
    elif av >= avn[nav - 1]:
        j = nav - 1
        dt = av - avn[nav - 1]
    else:
        j, t = _bracket(avn, av)
        if j >= nav - 1:          # exact hit on the last node
            j = nav - 1
            dt = 0.0
        else:
            hermite = True
            h = avn[j + 1] - avn[j]
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = (t3 - 2.0 * t2 + t) * h
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = (t3 - t2) * h
    acc = np.zeros(nbands)
    for e in range(8):
        d0 = (e >> 2) & 1
        d1 = (e >> 1) & 1
        d2 = e & 1
        w = (y0 if d0 else 1.0 - y0) \
            * (y1 if d1 else 1.0 - y1) \
            * (y2 if d2 else 1.0 - y2)
        if w == 0.0:
            continue
        j0 = i0 + d0 if i0 + d0 < n0 else n0 - 1
        j1 = i1 + d1 if i1 + d1 < n1 else n1 - 1
        j2 = i2 + d2 if i2 + d2 < n2 else n2 - 1
        for k in range(nbands):
            if hermite:
                v = (h00 * cube[j0, j1, j2, k, j, 0]
                     + h10 * cube[j0, j1, j2, k, j, 1]
                     + h01 * cube[j0, j1, j2, k, j + 1, 0]
                     + h11 * cube[j0, j1, j2, k, j + 1, 1])
            else:
                v = cube[j0, j1, j2, k, j, 0] + dt * cube[j0, j1, j2, k, j, 1]
            acc[k] += w * v
    for k in range(nbands):
        out[k] = 10.0 ** (-0.4 * acc[k])
    return out


def _band_ext_arrays(interp, filts, av_law):
    """Band-extinction table of ``interp`` for ``filts`` (see
    ``band_extinction.interpolator_table``), or None when ``interp`` has it
    switched off (``band_extinction`` attribute, set by ``load_band_grid``)
    or none can be built. Memoised on object identity like
    ``_grid_arrays``."""
    on = getattr(interp, 'band_extinction', False)
    if not on:
        return None
    key = (id(interp), id(filts), id(av_law))
    cached = _BAND_EXT_CACHE.get(key)
    if (cached is not None and cached[0] is interp and cached[1] is filts
            and cached[2] is av_law):
        return cached[3]
    from .band_extinction import interpolator_table
    arrays = interpolator_table(interp, filts, av_law)
    if len(_BAND_EXT_CACHE) >= 32:
        _BAND_EXT_CACHE.pop(next(iter(_BAND_EXT_CACHE)))
    _BAND_EXT_CACHE[key] = (interp, filts, av_law, arrays)
    return arrays


def prepare_band_extinction(interp, filts, av_law):
    """Load or build the band-extinction table of ``interp`` for ``filts``
    now (e.g. before forking sampler workers). True if one is in use."""
    return _band_ext_arrays(interp, filts, av_law) is not None


def band_extinction_factor(teff, logg, z, Av, filts, interp, av_law):
    """Correction from central-wavelength to band-integrated extinction:
    ``10**(-0.4 dA)`` per band of ``filts`` (1 where not tabulated)."""
    arrays = _band_ext_arrays(interp, filts, av_law)
    if arrays is None:
        return 1.0
    ii0, ii1, ii2, avn, cube = arrays
    return _band_ext_factor(logg, teff, z, Av, ii0, ii1, ii2, avn, cube)


def forget_interpolator(interp):
    """Drop the memoised arrays of ``interp`` (sliced flux cubes and
    band-extinction tables), so a temporary interpolator can be freed."""
    for cache in (_GRID_CACHE, _BAND_EXT_CACHE):
        for k in [k for k, v in cache.items() if v[0] is interp]:
            del cache[k]
    interp.__dict__.pop('_band_extinction_tables', None)


def set_band_extinction(interpolators, on):
    """Switch band-integrated extinction on or off for band-grid
    interpolator(s) (a single one or a binary pair)."""
    its = (interpolators if isinstance(interpolators, (tuple, list))
           else (interpolators,))
    for it in its:
        it.band_extinction = bool(on)


def _grid_arrays(interp, filts):
    """Return (and cache) the axis arrays and the filter-sliced flux cube.

    The DFInterpolator's per-call cost (``column_index`` lookups, ``icols``
    construction, broadcasting checks) is hoisted here: the cube is sliced to
    the requested filters once per (interpolator, filter set), so the hot path
    is a single nogil numba call. Memoised on object identity to avoid building
    keys on each call; references are held so ids cannot be reused.
    """
    # Keyed by (interpolator, filter array): a binary fit asks the same grid
    # for several filter sets (fitted bands, flux-ratio bands), and a
    # one-slot-per-grid cache would rebuild the cube on every alternation.
    key = (id(interp), id(filts))
    cached = _GRID_CACHE.get(key)
    if (cached is not None and cached[0] is interp and cached[1] is filts):
        return cached[2]
    ii0, ii1, ii2 = interp.index_columns
    icols = np.array([interp.column_index[f] for f in filts])
    cube = np.ascontiguousarray(interp.grid[:, :, :, icols], dtype=np.float64)
    arrays = (
        np.ascontiguousarray(ii0, dtype=np.float64),
        np.ascontiguousarray(ii1, dtype=np.float64),
        np.ascontiguousarray(ii2, dtype=np.float64),
        cube,
    )
    if len(_GRID_CACHE) >= 32:          # bounded: drop the oldest slice
        _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
    _GRID_CACHE[key] = (interp, filts, arrays)
    return arrays


def _make_transform(prior):
    """Build a fast unit-cube -> parameter transform for a single prior.

    scipy frozen-distribution ``.ppf`` is extremely slow (tens of us per call,
    and it is called once per free parameter on every nested-sampling
    proposal). For the distributions ARIADNE actually uses we substitute the
    closed-form inverse CDF, which is exact and ~100-1000x faster. Anything
    else (e.g. the callable spline population priors) is used directly, and
    unknown frozen distributions fall back to ``.ppf``.
    """
    # scipy frozen distributions expose ``.dist.name``; normalise the shape
    # params, loc and scale via ``_parse_args`` so this works whether the prior
    # was built with keywords (``norm(loc=.., scale=..)``) or positionally
    # (``norm(5500, 120)`` / ``truncnorm(a, b, loc, scale)``). Reading ``.kwds``
    # alone silently dropped positional loc/scale and raised KeyError on a
    # positionally-built truncnorm.
    if hasattr(prior, 'dist') and hasattr(prior.dist, 'name'):
        name = prior.dist.name
        shapes, loc, scale = prior.dist._parse_args(*prior.args, **prior.kwds)
        if name == 'uniform':
            return lambda u, loc=loc, scale=scale: loc + u * scale
        if name == 'norm':
            return lambda u, loc=loc, scale=scale: loc + scale * ndtri(u)
        if name == 'truncnorm':
            a, b = shapes
            phi_a = ndtr(a)
            span = ndtr(b) - phi_a
            return lambda u, loc=loc, scale=scale, pa=phi_a, span=span: \
                loc + scale * ndtri(pa + u * span)
        return prior.ppf
    # Callable priors (e.g. the InterpolatedUnivariateSpline Teff prior).
    if callable(prior):
        return prior
    return prior.ppf


def _free_param_transforms(prior_dict, coordinator, order):
    """Return (and cache) the per-free-parameter transforms, in order.

    The inputs are the same global objects on every call within a fit, so we
    memoise on object identity and verify with ``is`` rather than building a
    hashable key (e.g. ``tuple(order)``) on each of the ~10^5 calls. Keeping a
    reference to ``prior_dict`` keeps its ``id`` from being reused.
    """
    cached = _TRANSFORM_CACHE.get(id(prior_dict))
    if (cached is not None and cached[0] is prior_dict
            and cached[1] is coordinator and cached[2] is order):
        return cached[3]
    transforms = [
        _make_transform(prior_dict[par])
        for fixed, par in zip(coordinator, order) if not fixed
    ]
    _TRANSFORM_CACHE[id(prior_dict)] = (prior_dict, coordinator, order,
                                        transforms)
    return transforms


BINARY_PARAMS = ('teff2', 'logg2', 'rad2')


def _param_order(filts, use_norm, binary=False):
    """Return (and cache) the full parameter-name order for a filter set.

    Memoised on ``(use_norm, binary)`` with an identity check on ``filts`` to
    avoid building ``tuple(filts)`` on every call.

    A binary fit (``binary=True``, which requires ``use_norm=False``) appends
    the secondary's ``teff2, logg2, rad2`` after ``Av``; the two components
    share ``z``, ``dist`` and ``Av``.
    """
    cached = _ORDER_CACHE.get((use_norm, binary))
    if cached is not None and cached[0] is filts:
        return cached[1]
    if use_norm:
        base = ['teff', 'logg', 'z', 'norm', 'Av']
    else:
        base = ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
    if binary:
        base = base + list(BINARY_PARAMS)
    order = np.array(base + [excess_scatter_name(f) for f in filts])
    _ORDER_CACHE[(use_norm, binary)] = (filts, order)
    return order


def n_physical(use_norm, binary=False):
    """Number of physical parameters ahead of the per-band excess-scatter
    terms."""
    return (5 if use_norm else 6) + (len(BINARY_PARAMS) if binary else 0)


def _ext_attenuation_base(wave, av_law):
    """Return (and cache) 10**(-0.4 * A(lambda)) evaluated at Av=1.

    All supported laws are exactly linear in Av for fixed Rv, so the
    attenuation at arbitrary Av is simply this array raised to the power Av.
    Memoised on the law with an identity check on ``wave`` to avoid building
    ``tuple(wave)`` on every call.
    """
    cached = _EXT_CACHE.get(av_law)
    if cached is not None and cached[0] is wave:
        return cached[1]
    Rv = 3.1
    unit_ext = av_law(wave * 1e4, 1.0, Rv)
    base = 10 ** (-0.4 * unit_ext)
    _EXT_CACHE[av_law] = (wave, base)
    return base


def build_params(theta, flux, flux_e, filts, coordinator, fixed, use_norm):
    """Build the parameter vector that goes into the model."""
    params = np.zeros(len(coordinator))
    i = 0
    for j in range(len(coordinator)):
        if coordinator[j]:
            params[j] = fixed[j]
        else:
            params[j] = theta[i]
            i += 1
    return params


def get_interpolated_flux(temp, logg, z, filts, interpolators):
    """Interpolate the grid of fluxes in a given teff, logg and z.

    Parameters
    ----------
    temp: float
        The effective temperature.
    logg: float
        The superficial gravity.
    z: float
        The metallicity.
    filts: str
        The desired filter.

    Returns
    -------
    flux : float
        The interpolated flux at temp, logg, z for filter filt.

    """
    ii0, ii1, ii2, cube = _grid_arrays(interpolators, filts)
    # Grid axes are (logg, teff, z), matching the original DFInterpolator call.
    return _trilinear3d(logg, temp, z, ii0, ii1, ii2, cube)


def _pair(interpolators):
    """(primary, secondary) interpolators of a binary fit: a 2-tuple/list as
    given, or the same grid for both components."""
    if isinstance(interpolators, (tuple, list)):
        return interpolators[0], interpolators[1]
    return interpolators, interpolators


def model_grid(theta, filts, wave, interpolators, use_norm, av_law,
               binary=False):
    """Return the model grid in the selected filters.

    Parameters:
    -----------
    theta : array_like
        The parameters of the fit: teff, logg, z, radius, distance
    star : Star
        The Star object containing all relevant information regarding the star.
    interpolators : dict
        A dictionary with the interpolated grid.
    use_norm : bool
        False for a full fit  (including radius and distance). True to fit
        for a normalization constant instead.
    binary : bool
        If True, add a second star (``teff2, logg2, rad2`` in ``theta[6:9]``)
        sharing the primary's z, distance and extinction. ``interpolators``
        is then a ``(primary, secondary)`` pair of grids (or one grid used for
        both). Requires ``use_norm=False``.

    Returns
    -------
    model : dict
        A dictionary whose keys are the filters and the values are the
        interpolated fluxes

    """
    if binary:
        m1, m2 = binary_components(theta, filts, wave, interpolators, av_law)
        return m1 + m2
    if use_norm:
        teff, logg, z, norm, Av = theta[:5]
    else:
        teff, logg, z, dist, rad, Av = theta[:6]
        dist *= 4.435e+7  # Transform from pc to solRad

    flux = get_interpolated_flux(teff, logg, z, filts, interpolators)

    # Extinction is linear in Av (fixed Rv), so scale the cached unit curve
    # at each band's central wavelength, then correct it to the extinction
    # integrated over the band (see band_extinction).
    atten = _ext_attenuation_base(wave, av_law) ** Av
    atten = atten * band_extinction_factor(teff, logg, z, Av, filts,
                                           interpolators, av_law)
    if use_norm:
        model = flux * atten * norm
    else:
        model = flux * atten * (rad / dist) ** 2
    return model


def binary_components(theta, filts, wave, interpolators, av_law):
    """Observed (extincted, distance-scaled) fluxes of the two components.

    ``theta`` is ``[teff, logg, z, dist, rad, Av, teff2, logg2, rad2, ...]``;
    the components share ``z``, ``dist`` and ``Av`` and each scales as
    ``(R / d)^2``. ``interpolators`` is a ``(primary, secondary)`` pair (or
    one grid for both). Returns ``(model1, model2)``.

    The extinction integrated over a band depends on the spectrum, so each
    component gets its own band-extinction correction.
    """
    teff, logg, z, dist, rad, Av, teff2, logg2, rad2 = theta[:9]
    dist = dist * 4.435e+7  # pc -> solRad
    i1, i2 = _pair(interpolators)
    atten = _ext_attenuation_base(wave, av_law) ** Av
    f1 = get_interpolated_flux(teff, logg, z, filts, i1)
    f2 = get_interpolated_flux(teff2, logg2, z, filts, i2)
    f1 = f1 * band_extinction_factor(teff, logg, z, Av, filts, i1, av_law)
    f2 = f2 * band_extinction_factor(teff2, logg2, z, Av, filts, i2, av_law)
    return (f1 * atten * (rad / dist) ** 2,
            f2 * atten * (rad2 / dist) ** 2)


def get_residuals(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                  av_law, binary=False):
    """Calculate residuals of the model."""
    model = model_grid(theta, filts, wave, interpolators, use_norm, av_law,
                       binary=binary)
    start = n_physical(use_norm, binary)
    inflation = theta[start:]
    residuals = flux - model
    errs = np.sqrt(flux_er ** 2 + inflation ** 2)
    return residuals, errs


def contrast_model(theta, interpolators, cfilts, av_law=None):
    """Model magnitude difference ``m2 - m1`` of the components in ``cfilts``.

    ``-2.5 log10[(rad2 / rad1)^2 F2 / F1]``: distance and the shared
    extinction cancel in the ratio, so it depends on the two stars only.
    Positive where the secondary is the fainter one. With ``av_law``, the
    small difference in band-integrated extinction between the two spectra
    (see ``band_extinction``) is included.
    """
    teff, logg, z, rad = theta[0], theta[1], theta[2], theta[4]
    teff2, logg2, rad2 = theta[6], theta[7], theta[8]
    i1, i2 = _pair(interpolators)
    f1 = get_interpolated_flux(teff, logg, z, cfilts, i1)
    f2 = get_interpolated_flux(teff2, logg2, z, cfilts, i2)
    if av_law is not None:
        Av = theta[5]
        f1 = f1 * band_extinction_factor(teff, logg, z, Av, cfilts, i1,
                                         av_law)
        f2 = f2 * band_extinction_factor(teff2, logg2, z, Av, cfilts, i2,
                                         av_law)
    return -2.5 * np.log10((rad2 / rad) ** 2 * f2 / f1)


def contrast_loglike(theta, interpolators, contrast, av_law=None):
    """Gaussian log-likelihood of measured component contrasts.

    ``contrast`` is ``(cfilts, dmags, sigmas)``: band names, the measured
    ``m2 - m1`` (e.g. from speckle / AO imaging or a spectroscopic light
    ratio) and its uncertainty.
    """
    cfilts, dmags, sigmas = contrast
    dm = contrast_model(theta, interpolators, cfilts, av_law)
    return -0.5 * (((dm - dmags) / sigmas) ** 2).sum()


def log_likelihood(theta, flux, flux_er, wave, filts, interpolators, use_norm,
                   av_law, binary=False, contrast=None):
    """Calculate log likelihood of the model."""
    res, ers = get_residuals(theta, flux, flux_er, wave,
                             filts, interpolators, use_norm, av_law,
                             binary=binary)

    lnl = fast_loglik(res, ers)

    if not np.isfinite(lnl):
        return -1e300

    ll = -.5 * lnl
    if binary and contrast is not None:
        c = contrast_loglike(theta, interpolators, contrast, av_law)
        if not np.isfinite(c):
            return -1e300
        ll += c
    return ll


@nb.njit(cache=True)
def fast_loglik(res, ers):
    ers2 = ers ** 2
    c = np.log(2 * np.pi * ers2)
    lnl = (c + (res ** 2 / ers2)).sum()
    return lnl


def prior_transform_dynesty(u, flux, flux_er, filts, prior_dict, coordinator,
                            use_norm, binary=False):
    """Transform the prior from the unit cube to the parameter space."""
    u2 = np.array(u)
    order = _param_order(filts, use_norm, binary)
    transforms = _free_param_transforms(prior_dict, coordinator, order)
    for i, transform in enumerate(transforms):
        u2[i] = transform(u2[i])
    return u2


def prior_transform_multinest(u, flux, flux_er, filts, prior_dict, coordinator,
                              use_norm, binary=False):
    """Transform the prior from the unit cube to the parameter space."""
    order = _param_order(filts, use_norm, binary)
    transforms = _free_param_transforms(prior_dict, coordinator, order)
    for i, transform in enumerate(transforms):
        u[i] = transform(u[i])


# Band-grid file of every model grid (Datafiles/model_grids).
GRID_FILES = {
    'phoenix': 'Phoenixv2_DF.pkl', 'btsettl': 'BTSettl_DF.pkl',
    'btnextgen': 'BTNextGen_DF.pkl', 'btcond': 'BTCond_DF.pkl',
    'ck04': 'CK04_DF.pkl', 'kurucz': 'Kurucz_DF.pkl',
    'coelho': 'Coelho_DF.pkl', 'bosz': 'BOSZ_DF.pkl',
    'sphinx': 'SPHINX_DF.pkl', 'tlusty': 'TLUSTY_DF.pkl',
}


def load_band_grid(name, band_extinction=True):
    """The ``DFInterpolator`` of model grid ``name`` (e.g. ``'bosz'``).

    It is tagged with ``grid_name`` and, unless ``band_extinction`` is False,
    set to redden its band fluxes with the extinction integrated over each
    band (``set_band_extinction``; see ``band_extinction``)."""
    import pandas as pd
    from isochrones.interp import DFInterpolator
    from .config import gridsdir
    key = str(name).lower()
    fn = GRID_FILES.get(key)
    if fn is None:
        raise ValueError(f'unknown model grid {name!r}; known grids: '
                         f'{", ".join(GRID_FILES)}')
    with open(f'{gridsdir}/{fn}', 'rb') as fh:
        interp = DFInterpolator(pd.read_pickle(fh))
    interp.grid_name = key
    interp.band_extinction = bool(band_extinction)
    return interp


def split_grid_name(name):
    """``('bosz', 'tlusty')`` for a binary grid pair named ``'bosz+tlusty'``,
    ``(name, None)`` for a single grid."""
    parts = str(name).split('+')
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(name), None
