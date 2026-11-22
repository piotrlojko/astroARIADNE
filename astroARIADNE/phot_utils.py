"""phot_utils module for SED fitting.

This module contains useful functions in order to obtain fluxes from different
broadband filters. It also has functions to convert to different units of flux

It uses the module pyphot to get the fluxes and bandpasses of different
broadband filters. Custom filters not in pyphot's library are loaded from
ASCII files in Datafiles/filters/.
"""

import astropy.constants as const
import astropy.units as u
import pyphot
import numpy as np
import os

from .config import filter_names as _FILTER_NAMES

_PYPHOT_LIB = pyphot.get_library()

# Cache for custom filters loaded from ASCII files
_CUSTOM_FILTER_CACHE = {}

# Directory containing custom filter ASCII files
_CUSTOM_FILTER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'Datafiles', 'filters'
)


def _get_custom_filter(band):
    """Load a custom filter from an ASCII file if not in pyphot's library.

    Parameters
    ----------
    band : str
        Filter name.

    Returns
    -------
    pyphot.Filter or None
        Filter object if found, None otherwise.
    """
    if band in _CUSTOM_FILTER_CACHE:
        return _CUSTOM_FILTER_CACHE[band]

    # Check if filter is in pyphot's library
    try:
        filt = _PYPHOT_LIB[band]
        _CUSTOM_FILTER_CACHE[band] = filt
        return filt
    except KeyError:
        pass

    # Try to load from ASCII file
    filepath = os.path.join(_CUSTOM_FILTER_DIR, f'{band}.dat')
    if not os.path.exists(filepath):
        return None

    try:
        filt = pyphot.Filter.from_ascii(
            filepath, dtype='csv', unit='Angstrom', name=band
        )
        _CUSTOM_FILTER_CACHE[band] = filt
        return filt
    except Exception:
        return None


def _get_filter(band):
    """Get a filter object, trying pyphot's library first, then custom files."""
    return _get_custom_filter(band)


def extract_info(magnitudes, errors, filters, systems=None):
    """Extract the flux information for a Star.

    ``systems`` maps band -> magnitude system for bands not in their default
    system (see ``mag_to_flux``)."""
    flux = dict()
    flux_er = dict()
    wave = dict()
    bandpass = dict()
    systems = systems or {}

    for mag, err, band in zip(magnitudes, errors, filters):
        # Get central wavelength
        leff = get_effective_wavelength(band)
        mag_err = err
        # get flux, flux error and bandpass
        flx, flx_err = mag_to_flux(mag, mag_err, band, systems.get(band))
        bp = get_bandpass(band)
        flux[band] = flx  # * leff
        flux_er[band] = flx_err  # * leff
        wave[band] = leff
        bandpass[band] = bp

    return wave, flux, flux_er, bandpass


def convert_jansky_to_ergs(j):
    """Convert flux from jansky to erg s-1 cm-2 Hz-1."""
    return j * 1e-23


def convert_jansky_to_ergs_lambda(j, l):
    """Convert flux from jansky to erg s-2 cm-2 lambda-1 in the units of l."""
    return j * const.c.to(u.micrometer / u.s).value / l ** 2


def convert_f_lambda_to_f_nu(f, l):
    """Convert flux from erg s-1 cm-2 lambda-1 to erg s-1 cm-2 Hz-1."""
    return f / const.c.to(u.micrometer / u.s).value * l ** 2


def convert_f_nu_to_f_lambda(f, l):
    """Convert flux from erf s-1 cm-2 Hz-1 to erg s-1 cm-2 lambda-1."""
    return f * const.c.to(u.micrometer / u.s).value / l ** 2


# Surveys whose catalogues are AB-native. Every other band defaults to Vega.
# HST photometry comes in VEGAMAG, ABMAG or STMAG depending on the source, so
# HST bands default to Vega and callers say otherwise per band (Star's
# ``mag_system``).
_AB_PREFIXES = ('PS1_', 'SDSS_', 'GALEX_', 'SkyMapper_')
_MAG_SYSTEMS = {'vega': 'Vega', 'ab': 'AB', 'st': 'ST'}


def normalize_mag_system(system):
    """'vega'/'AB'/'st' (any case) -> 'Vega'/'AB'/'ST'; None stays None."""
    if system is None:
        return None
    try:
        return _MAG_SYSTEMS[str(system).lower()]
    except KeyError:
        raise ValueError(f'unknown magnitude system {system!r}; use one of '
                         'Vega, AB, ST') from None


def default_mag_system(band):
    """The system a band's magnitudes are in unless told otherwise."""
    return 'AB' if any(p in band for p in _AB_PREFIXES) else 'Vega'


def zero_flux(band, system=None):
    """Flux of a zero-magnitude source in ``band``, erg s-1 cm-2 um-1.

    ``system`` is 'Vega', 'AB' or 'ST' (None: the band's default). AB is
    3631 Jy at the band's effective wavelength; ST is f_lambda =
    10**(-0.4 * 21.10) erg s-1 cm-2 AA-1; Vega is pyphot's Vega zero point.
    """
    system = normalize_mag_system(system) or default_mag_system(band)
    if system == 'AB':
        return convert_f_nu_to_f_lambda(3.631e-20,
                                        get_effective_wavelength(band))
    if system == 'ST':
        return 10 ** (-0.4 * 21.10) * 1e4
    return get_zero_flux(band)


def mag_to_flux(mag, mag_err, band, system=None):
    """Convert from magnitude to flux in erg s-1 cm-2 um-1.

    ``band`` must match a pyphot filter name. ``system`` is the magnitude
    system, 'Vega', 'AB' or 'ST'; None uses the band's default: AB for
    PanSTARRS, SDSS, GALEX and SkyMapper, Vega otherwise.
    """
    flux = 10 ** (-.4 * mag) * zero_flux(band, system)
    flux_err = abs(-.4 * flux * np.log(10) * mag_err)
    return flux, flux_err


def flux_to_mag(flux, flux_err, band, system=None):
    """Convert from flux (erg s-1 cm-2 um-1) to magnitude in ``system``
    (None: the band's default, as in ``mag_to_flux``)."""
    mag = -2.5 * np.log10(flux / zero_flux(band, system))
    mag_err = 2.5 * flux_err / (np.log(10) * flux)
    return mag, mag_err


# Unit of the fluxes ARIADNE fits (f_lambda).
FLUX_UNIT = u.erg / (u.s * u.cm ** 2 * u.um)


def _to_quantity(value, unit, name):
    """``value`` as a scalar Quantity; plain numbers take ``unit``."""
    if isinstance(value, u.Quantity):
        q = value
    else:
        if unit is None:
            raise ValueError(
                f'{name} {value!r} has no unit: pass an astropy Quantity '
                '(e.g. 1.2 * u.mJy) or give the unit (e.g. unit="mJy")')
        try:
            q = u.Quantity(value, u.Unit(unit))
        except (TypeError, ValueError) as e:
            raise ValueError(f'{name}: {e}') from None
    if isinstance(q.unit, u.FunctionUnitBase):
        raise ValueError(f'{name} {q} is a magnitude, not a flux; add it '
                         'with add_mag / mag_dict instead')
    if not q.isscalar:
        raise ValueError(f'{name} must be a single value, got {q}')
    return q


def flux_density_to_flux(flux, band, unit=None):
    """Apparent flux density in ``band`` -> f_lambda in erg s-1 cm-2 um-1.

    ``flux`` is an astropy Quantity in any spectral flux density unit that
    ``astropy.units.spectral_density`` understands: f_lambda (e.g.
    erg s-1 cm-2 AA-1, W m-2 nm-1), f_nu (e.g. Jy, mJy, erg s-1 cm-2 Hz-1),
    lambda f_lambda = nu f_nu (e.g. erg s-1 cm-2, W m-2) or a photon flux
    density (e.g. photon s-1 cm-2 AA-1). A plain number is read in ``unit``.
    The conversion is made at the band's effective wavelength, the wavelength
    ARIADNE assigns to the band (and the one the AB zero point uses).
    """
    return _flux_value(flux, band, unit, 'flux')


def _flux_value(value, band, unit, name):
    q = _to_quantity(value, unit, name)
    wave = get_effective_wavelength(band) * u.um
    try:
        return q.to_value(FLUX_UNIT, equivalencies=u.spectral_density(wave))
    except u.UnitsError:
        raise ValueError(
            f'{name}: cannot convert {q.unit} to a flux density; use '
            'f_lambda, f_nu, lambda*f_lambda (= nu*f_nu) or photon flux '
            'density units, e.g. erg/(s cm2 AA), Jy, erg/(s cm2)') from None


def flux_density_to_mag(flux, flux_err, band, system=None, unit=None):
    """Convert an apparent flux (density) in ``band`` to a magnitude.

    ``flux`` and ``flux_err`` are astropy Quantities in any flux density unit
    accepted by ``flux_density_to_flux`` (they need not share a unit), or
    plain numbers in ``unit``. A plain-number ``flux_err`` (e.g. 0) takes the
    unit of a Quantity ``flux``. ``system`` is the magnitude system of the
    result as in ``flux_to_mag`` (None: the band's default).

    Returns ``(mag, mag_err)``; ``mag_to_flux(mag, mag_err, band, system)``
    gives back the input converted to erg s-1 cm-2 um-1. Raises ValueError if
    the values cannot be used.
    """
    if band not in _FILTER_NAMES:
        raise ValueError(f'unknown filter {band!r}; names must match '
                         'Star.filter_names')
    if (not isinstance(flux_err, u.Quantity)
            and isinstance(flux, u.Quantity) and unit is None):
        unit = flux.unit
    f = _flux_value(flux, band, unit, 'flux')
    f_err = _flux_value(flux_err, band, unit, 'flux error')
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f'flux must be positive and finite, got {flux}')
    if not np.isfinite(f_err) or f_err < 0:
        raise ValueError(
            f'flux error must be >= 0 and finite, got {flux_err}')
    return flux_to_mag(f, f_err, band, system)


def get_zero_flux(band):
    """Look for the filter information in the pyphot library of filters.

    If the filter is not in pyphot's library, it is loaded from a custom
    ASCII file in Datafiles/filters/.
    """
    filt = _get_filter(band)
    if filt is None:
        raise KeyError(
            f"Filter '{band}' not found in pyphot library or custom filter files."
        )
    # Get Vega zero flux in erg / cm2 / s / um
    f0 = filt.Vega_zero_flux.to('erg/(um * cm ** 2 * s)').value
    return f0


def get_effective_wavelength(band):
    """Get central wavelength of a specific filter in um."""
    filt = _get_filter(band)
    if filt is None:
        raise KeyError(
            f"Filter '{band}' not found in pyphot library or custom filter files."
        )
    # Get central wavelength in um
    leff = filt.cl.to('um').value
    return leff


def get_bandpass(band):
    """Get the bandpass of a specific filter in um."""
    filt = _get_filter(band)
    if filt is None:
        raise KeyError(
            f"Filter '{band}' not found in pyphot library or custom filter files."
        )
    # Get lower and upper bandpass in um
    width = filt.width.to('um').value
    bp = width
    return bp / 2


def mag_to_flux_AB(mag, mag_err):
    """Calculate flux in erg s-1 cm-2 Hz-1."""
    flux = 10 ** (-.4 * (mag + 48.6))
    flux_err = abs(-.4 * flux * np.log(10) * mag_err)
    return flux, flux_err
