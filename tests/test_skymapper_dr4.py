"""SkyMapper DR4 photometry, kept band by band (TAP, VizieR fallback)."""
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table

import astroARIADNE.librarian._api as api
from astroARIADNE.librarian import Librarian, _cds
from astroARIADNE.librarian._qc import _qc_skymapper_band

GAIA = 6210817001175183104
NEIGHBOUR = 6210817005473218304
SM_ID = 113412874

# HD 133729 in DR4: clean u, v and z; saturated g, r and i, which DR4 gives
# a magnitude but no error (flag 4 set, no good measurement).
HD133729 = {'u': (9.5924, 0.0624, 0, 2), 'v': (9.2242, 0.0428, 0, 1),
            'g': (8.9066, None, 62, 0), 'r': (8.9408, None, 28, 0),
            'i': (8.9479, None, 28, 0), 'z': (9.5612, 0.0247, 0, 1)}
CLEAN = {'SkyMapper_u': (9.5924, 0.0624), 'SkyMapper_v': (9.2242, 0.0428),
         'SkyMapper_z': (9.5612, 0.0247)}


def _tap_row(gaia=GAIA):
    """``dr4.master`` as the SkyMapper TAP service returns it."""
    cols = {'object_id': [SM_ID], 'gaia_dr3_id1': [gaia]}
    for f, (mag, err, flags, ngood) in HD133729.items():
        cols[f'{f}_psf'] = [mag]
        cols[f'e_{f}_psf'] = MaskedColumn([err or 0.0], mask=[err is None])
        cols[f'{f}_flags'] = [flags]
        cols[f'{f}_ngood'] = [ngood]
    return Table(cols)


def _vizier_table():
    """II/379/smssdr4: a clean faint neighbour listed before the star."""
    cols = {'ObjectId': [999, SM_ID], 'Gaia3Id1': [NEIGHBOUR, GAIA]}
    for f, (mag, err, flags, ngood) in HD133729.items():
        cols[f'{f}PSF'] = [18.0, mag]
        cols[f'e_{f}PSF'] = MaskedColumn([0.05, err or 0.0],
                                         mask=[False, err is None])
        cols[f'flags{f}'] = [0, flags]
        cols[f'Ngood{f}'] = [3, ngood]
    return Table(cols)


def _lib(sm_id=SM_ID):
    lib = Librarian.__new__(Librarian)
    lib._gaia_id = GAIA
    lib._coord = SkyCoord(226.74555, -31.6441583, unit='deg')
    lib._search_radius = 3 * u.arcsec
    lib._magnitudes = {}
    lib._ids = {'SkyMapper': sm_id}
    return lib


def _patch(monkeypatch, tap=None, cone=None):
    """Fake both services; return the TAP queries and VizieR catalogues asked."""
    asked = {'tap': [], 'vizier': []}

    def fake_tap(service, query, *a, **k):
        asked['tap'].append(query)
        return tap

    def fake_cone(catalog, *a, **k):
        asked['vizier'].append(catalog)
        if isinstance(cone, Exception):
            raise cone
        return cone

    monkeypatch.setattr(api, '_tap_query', fake_tap)
    monkeypatch.setattr(_cds, '_cone', fake_cone)
    return asked


def _assert_clean_bands(mags):
    assert set(mags) == set(CLEAN)
    for band, (mag, err) in CLEAN.items():
        assert mags[band] == pytest.approx((mag, err), abs=1e-4)


@pytest.mark.parametrize('flags, ngood, ok', [
    (0, 1, True), (0, 2, True), (0, 0, False), (4, 3, False),
    (1, 3, False), (None, 1, False), (0, None, False)])
def test_band_needs_no_flag_and_a_good_measurement(flags, ngood, ok):
    assert _qc_skymapper_band(flags, ngood) is ok


def test_a_star_saturated_in_gri_keeps_u_v_and_z(monkeypatch):
    asked = _patch(monkeypatch, tap=_tap_row())
    lib = _lib()
    lib._fetch_skymapper()
    _assert_clean_bands(lib._magnitudes)
    assert 'FROM dr4.master' in asked['tap'][0]
    assert f'object_id = {SM_ID}' in asked['tap'][0]
    assert asked['vizier'] == []


def test_vizier_serves_dr4_when_the_tap_service_fails(monkeypatch):
    asked = _patch(monkeypatch, tap=None, cone=_vizier_table())
    lib = _lib()
    lib._fetch_skymapper()
    # the Gaia-matched row, not the neighbour listed first
    _assert_clean_bands(lib._magnitudes)
    assert asked['vizier'] == ['II/379/smssdr4']


def test_tap_row_matched_to_another_gaia_source_is_not_used(monkeypatch):
    asked = _patch(monkeypatch, tap=_tap_row(gaia=NEIGHBOUR),
                   cone=_vizier_table())
    lib = _lib()
    lib._fetch_skymapper()
    _assert_clean_bands(lib._magnitudes)
    assert asked['vizier'] == ['II/379/smssdr4']


def test_star_without_a_skymapper_id_is_looked_up_in_dr4(monkeypatch):
    asked = _patch(monkeypatch, cone=_vizier_table())
    lib = _lib(sm_id=None)
    lib._fetch_skymapper()
    _assert_clean_bands(lib._magnitudes)
    assert asked['tap'] == []


def test_star_not_in_dr4_adds_nothing(monkeypatch):
    _patch(monkeypatch, tap=Table(), cone=None)
    lib = _lib()
    lib._fetch_skymapper()
    assert lib._magnitudes == {}


def test_both_services_failing_adds_nothing(monkeypatch, caplog):
    _patch(monkeypatch, tap=None, cone=RuntimeError('VizieR down'))
    lib = _lib()
    lib._fetch_skymapper()
    assert lib._magnitudes == {}
    assert 'SkyMapper DR4 failed on TAP and on VizieR' in caplog.text
