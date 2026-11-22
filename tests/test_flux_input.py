"""Photometry given as apparent fluxes (flux_dict / Star.add_flux)."""
import astropy.units as u
import numpy as np
import pytest

from astroARIADNE.phot_utils import (FLUX_UNIT, flux_density_to_flux,
                                     flux_density_to_mag,
                                     get_effective_wavelength, mag_to_flux)
from astroARIADNE.star import Star

FLAM = u.erg / (u.s * u.cm ** 2 * u.AA)
MAGS = {'2MASS_J': (9.763, 0.028), '2MASS_H': (9.135, 0.026),
        '2MASS_Ks': (8.899, 0.019)}
HST = 'HST_WFC3_F275W'


def _expected(q, band):
    """``q`` in erg s-1 cm-2 um-1 at the band's effective wavelength."""
    wave = get_effective_wavelength(band) * u.um
    return q.to_value(FLUX_UNIT, equivalencies=u.spectral_density(wave))


def _star(**kw):
    kw.setdefault('mag_dict', MAGS)
    return Star('demo', 172.56, 7.588, offline=True, dist=500., dist_e=10.,
                Av=0.05, verbose=False, **kw)


def _idx(star, band):
    return list(star.filter_names).index(band)


# --- phot_utils ------------------------------------------------------------

def test_ab_band_matches_ab_magnitude():
    """3631 Jy * 10**(-0.4 m) in an AB band is magnitude m (same wavelength
    for the conversion and for the AB zero point)."""
    mag, err = flux_density_to_mag(3631 * u.Jy * 10 ** (-0.4 * 13.751),
                                   0. * u.Jy, 'PS1_r')
    assert np.isclose(mag, 13.751, atol=1e-12)
    assert err == 0


@pytest.mark.parametrize('flux, err', [
    (1.0e-13 * FLAM, 3.0e-15 * FLAM),                          # f_lambda
    (2.0e-26 * u.W / u.m ** 2 / u.Hz, 1e-27 * u.W / u.m ** 2 / u.Hz),  # f_nu
    (0.35 * u.Jy, 10 * u.mJy),                                 # mixed units
    (4.0e-10 * u.erg / u.s / u.cm ** 2, 2e-11 * u.W / u.m ** 2),  # nu f_nu
    (0.5 * u.photon / u.s / u.cm ** 2 / u.AA,
     0.01 * u.photon / u.s / u.cm ** 2 / u.AA),                # photon flux
])
@pytest.mark.parametrize('band, system', [
    ('2MASS_H', None), ('2MASS_H', 'AB'), (HST, 'ST'), ('SDSS_g', None)])
def test_round_trip(flux, err, band, system):
    """mag_to_flux(flux_density_to_mag(F)) gives back F (and its error)."""
    mag, mag_err = flux_density_to_mag(flux, err, band, system)
    f, f_err = mag_to_flux(mag, mag_err, band, system)
    assert np.isclose(f, _expected(flux, band), rtol=1e-12)
    assert np.isclose(f_err, _expected(err, band), rtol=1e-12)


def test_plain_numbers_need_a_unit():
    a = flux_density_to_mag(2.1e-13, 1e-14, '2MASS_J', unit='erg/(s cm2 AA)')
    b = flux_density_to_mag(2.1e-13 * FLAM, 1e-14 * FLAM, '2MASS_J')
    assert np.allclose(a, b, rtol=1e-12)
    with pytest.raises(ValueError, match='no unit'):
        flux_density_to_mag(2.1e-13, 1e-14, '2MASS_J')


def test_plain_error_takes_flux_unit():
    a = flux_density_to_mag(12 * u.mJy, 0.5, '2MASS_J')
    b = flux_density_to_mag(12 * u.mJy, 0.5 * u.mJy, '2MASS_J')
    assert np.allclose(a, b, rtol=1e-12)


def test_flux_density_to_flux():
    q = 1.5 * u.Jy
    assert np.isclose(flux_density_to_flux(q, '2MASS_J'),
                      _expected(q, '2MASS_J'), rtol=1e-12)
    assert np.isclose(flux_density_to_flux(1.5, '2MASS_J', unit='Jy'),
                      _expected(q, '2MASS_J'), rtol=1e-12)


@pytest.mark.parametrize('flux, err, band, match', [
    (1 * u.K, 0.1 * u.K, '2MASS_J', 'cannot convert'),
    (1 * u.erg / u.s, 0 * u.erg / u.s, '2MASS_J', 'cannot convert'),
    (1 * u.Jy, 0.1 * u.K, '2MASS_J', 'flux error: cannot convert'),
    (12.0 * u.ABmag, 0.1 * u.mag, '2MASS_J', 'magnitude'),
    (-1 * u.mJy, 0.1 * u.mJy, '2MASS_J', 'positive'),
    (0 * u.mJy, 0.1 * u.mJy, '2MASS_J', 'positive'),
    (np.nan * u.mJy, 0.1 * u.mJy, '2MASS_J', 'positive'),
    (1 * u.mJy, -0.1 * u.mJy, '2MASS_J', '>= 0'),
    ([1, 2] * u.mJy, 0.1 * u.mJy, '2MASS_J', 'single value'),
    (1 * u.mJy, 0.1 * u.mJy, 'NOT_A_BAND', 'unknown filter'),
])
def test_bad_input_rejected(flux, err, band, match):
    with pytest.raises(ValueError, match=match):
        flux_density_to_mag(flux, err, band)


# --- Star ------------------------------------------------------------------

def test_flux_dict_equals_mag_dict():
    """A star built from the fluxes of its magnitudes is the same star."""
    ref = _star()
    fluxes = {}
    for band, (m, e) in MAGS.items():
        f, fe = mag_to_flux(m, e, band)
        fluxes[band] = (f * FLUX_UNIT, fe * FLUX_UNIT)
    s = _star(mag_dict=None, flux_dict=fluxes)
    assert not s.get_mags
    assert np.array_equal(s.used_filters, ref.used_filters)
    assert np.allclose(s.mags, ref.mags, rtol=0, atol=1e-12)
    assert np.allclose(s.mag_errs, ref.mag_errs, rtol=1e-12)
    assert np.allclose(s.flux, ref.flux, rtol=1e-12)
    assert np.allclose(s.flux_er, ref.flux_er, rtol=1e-12)


def test_flux_dict_with_mag_dict_and_units():
    s = _star(flux_dict={'WISE_RSR_W1': (25. * u.mJy, 0.8 * u.mJy),
                         'GaiaDR2v2_G': (3.2e-13, 1e-15, 'erg/(s cm2 AA)')})
    for band in MAGS:
        assert s.used_filters[_idx(s, band)] == 1
    i = _idx(s, 'WISE_RSR_W1')
    assert np.isclose(s.flux[i], _expected(25. * u.mJy, 'WISE_RSR_W1'))
    assert np.isclose(s.flux_er[i], _expected(0.8 * u.mJy, 'WISE_RSR_W1'))
    i = _idx(s, 'GaiaDR2v2_G')
    assert np.isclose(s.flux[i], _expected(3.2e-13 * FLAM, 'GaiaDR2v2_G'))


def test_flux_dict_uses_band_mag_system():
    """The stored magnitude is in the band's system; the flux is unchanged."""
    q = 2e-16 * FLAM
    vega = _star(flux_dict={HST: (q, 0 * FLAM)})
    ab = _star(flux_dict={HST: (q, 0 * FLAM)}, mag_system={HST: 'AB'})
    i = _idx(vega, HST)
    assert not np.isclose(vega.mags[i], ab.mags[i])
    assert np.isclose(vega.flux[i], ab.flux[i], rtol=1e-12)
    assert np.isclose(ab.flux[i], _expected(q, HST), rtol=1e-12)


def test_add_flux():
    s = _star()
    s.add_flux(12.3 * u.mJy, 0.4 * u.mJy, '2MASS_J')    # replaces J
    s.add_flux(2.1e-13, 1e-14, 'GaiaDR2v2_BP', unit='erg / (s cm2 AA)')
    i = _idx(s, '2MASS_J')
    assert np.isclose(s.flux[i], _expected(12.3 * u.mJy, '2MASS_J'))
    assert np.isclose(s.flux_er[i], _expected(0.4 * u.mJy, '2MASS_J'))
    i = _idx(s, 'GaiaDR2v2_BP')
    assert s.used_filters[i] == 1
    assert np.isclose(s.flux[i], _expected(2.1e-13 * FLAM, 'GaiaDR2v2_BP'))


def test_add_flux_irx_band():
    s = _star()
    s.add_flux(5. * u.mJy, 0.5 * u.mJy, 'WISE_RSR_W4')
    i = _idx(s, 'WISE_RSR_W4')
    assert s.irx_used_filters[i] == 1 and s.used_filters[i] == 0
    assert np.isclose(s.flux[i], _expected(5. * u.mJy, 'WISE_RSR_W4'))


@pytest.mark.parametrize('kw', [
    {'flux_dict': {'2MASS_J': (1 * u.Jy, 0.1 * u.Jy)}},          # in both
    {'flux_dict': {'2MASS_W': (1 * u.Jy, 0.1 * u.Jy)}},          # bad band
    {'flux_dict': {'WISE_RSR_W1': (1 * u.Jy,)}},                 # no error
    {'flux_dict': {'WISE_RSR_W1': (1, 0.1, 'Jy', 'x')}},         # too long
    {'flux_dict': {'WISE_RSR_W1': (1, 0.1)}},                    # no unit
    {'flux_dict': {'WISE_RSR_W1': (1 * u.K, 0.1 * u.K)}},        # bad unit
])
def test_bad_flux_dict_rejected(kw):
    # InputError.__raise__ logs and exits, as elsewhere in Star.
    with pytest.raises(SystemExit):
        _star(**kw)


def test_add_flux_bad_input_rejected():
    s = _star()
    with pytest.raises(SystemExit):
        s.add_flux(-1 * u.mJy, 0.1 * u.mJy, '2MASS_J')
    with pytest.raises(SystemExit):
        s.add_flux(1., 0.1, '2MASS_J')
