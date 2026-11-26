"""Extinction integrated over each band (band_extinction module).

A band flux reddened over the band, ``<F 10**(-0.4 A_V a)>``, differs from
the central-wavelength approximation ``<F> 10**(-0.4 A_V a(lambda_c))`` for
wide bands. The tables correct the model to the former; checked here against
direct synthetic photometry of the cached node spectra.
"""
import os

import extinction
import h5py
import numpy as np
import pytest

import astroARIADNE.band_extinction as be
from astroARIADNE import synth_phot as sp
from astroARIADNE.config import spectra_cache
from astroARIADNE.phot_utils import get_effective_wavelength
from astroARIADNE.sed_library import (band_extinction_factor,
                                      binary_components, load_band_grid,
                                      model_grid)

LAW = extinction.fitzpatrick99
GRID = 'ck04'                      # small cache group: fast to tabulate
BANDS = np.array(['GaiaDR2v2_G', 'GALEX_NUV', '2MASS_J', 'WISE_RSR_W2'])
WAVE = np.array([get_effective_wavelength(b) for b in BANDS])
NODE = (6000., 4.5, 0.0)           # a ck04 grid node (teff, logg, z)

pytestmark = pytest.mark.skipif(not be.available(GRID),
                                reason='spectra cache with ck04 needed')


@pytest.fixture
def tables_dir(tmp_path, monkeypatch):
    d = tmp_path / 'band_ext'
    monkeypatch.setenv('ARIADNE_BAND_EXTINCTION_DIR', str(d))
    return d


def _exact_dmag(band, av, node=NODE, grid=GRID):
    """-2.5 log10 of the reddened / unreddened band flux of the node
    spectrum: the band-integrated extinction in mag."""
    with h5py.File(spectra_cache, 'r') as h:
        g = h[grid]
        t, lg, z = g['teff'][:], g['logg'][:], g['z'][:]
        i = int(np.nonzero((t == node[0]) & (lg == node[1])
                           & (z == node[2]))[0][0])
        w = g['wavelength'][:].astype(float)
        F = g['flux'][i].astype(float)
    ops = sp.coeff_matrix([sp.resolve_filter(band)], w)
    red = F * 10 ** (-0.4 * LAW(w * 1e4, av, 3.1))
    f0 = sp.convolve(F[None], *ops)[0, 0]
    f1 = sp.convolve(red[None], *ops)[0, 0]
    return -2.5 * np.log10(f1 / f0)


def _model_dmag(interp, av, node=NODE):
    th = np.array([node[0], node[1], node[2], 100., 1., av])
    th0 = th.copy()
    th0[5] = 0.0
    return -2.5 * np.log10(model_grid(th, BANDS, WAVE, interp, False, LAW)
                           / model_grid(th0, BANDS, WAVE, interp, False, LAW))


@pytest.mark.parametrize('av', [0.4, 2.0, 2.7, 11.0])
def test_matches_synthetic_photometry_of_the_reddened_spectrum(tables_dir,
                                                               av):
    model = _model_dmag(load_band_grid(GRID), av)
    for j, b in enumerate(BANDS[:3]):
        # Hermite interpolation in A_V: < 0.3 mmag up to A_V = 30
        assert model[j] == pytest.approx(_exact_dmag(b, av), abs=3e-4), b


def test_central_wavelength_extinction_is_wrong_for_wide_bands(tables_dir):
    old = _model_dmag(load_band_grid(GRID, band_extinction=False), 0.4)
    exact = [_exact_dmag(b, 0.4) for b in BANDS[:2]]
    assert abs(old[0] - exact[0]) > 0.005        # Gaia G, 6000 K
    assert abs(old[1] - exact[1]) > 0.05         # GALEX NUV on the bump


def test_switched_off_is_the_central_wavelength_extinction():
    interp = load_band_grid(GRID, band_extinction=False)
    av = 1.3
    th = np.array([NODE[0], NODE[1], NODE[2], 100., 1., av])
    f = model_grid(th, BANDS, WAVE, interp, False, LAW)
    th[5] = 0.0
    f0 = model_grid(th, BANDS, WAVE, interp, False, LAW)
    np.testing.assert_allclose(f / f0, 10 ** (-0.4 * LAW(WAVE * 1e4, av)),
                               rtol=1e-12)


def test_band_beyond_the_cached_spectra_keeps_the_central_wavelength(
        tables_dir):
    # the ck04 cache ends at 4.61 um, inside WISE W2
    fac = band_extinction_factor(*NODE, 2.0, BANDS, load_band_grid(GRID),
                                 LAW)
    assert fac[3] == 1.0 and fac[0] != 1.0


def test_off_node_teff_interpolates_between_the_node_tables(tables_dir):
    interp = load_band_grid(GRID)
    lo = band_extinction_factor(6000., 4.5, 0., 0.8, BANDS, interp, LAW)
    hi = band_extinction_factor(6250., 4.5, 0., 0.8, BANDS, interp, LAW)
    mid = band_extinction_factor(6125., 4.5, 0., 0.8, BANDS, interp, LAW)
    dm = [-2.5 * np.log10(x[:3]) for x in (lo, mid, hi)]
    np.testing.assert_allclose(dm[1], 0.5 * (dm[0] + dm[2]), atol=1e-9)


def test_each_binary_component_gets_its_own_correction(tables_dir):
    interp = load_band_grid(GRID)
    th = np.array([6000., 4.5, 0., 100., 1., 0.8, 9000., 4.0, 0.5])
    on = binary_components(th, BANDS, WAVE, interp, LAW)
    off = binary_components(th, BANDS, WAVE,
                            load_band_grid(GRID, band_extinction=False), LAW)
    f1 = band_extinction_factor(6000., 4.5, 0., 0.8, BANDS, interp, LAW)
    f2 = band_extinction_factor(9000., 4.0, 0., 0.8, BANDS, interp, LAW)
    np.testing.assert_allclose(on[0], off[0] * f1, rtol=1e-12)
    np.testing.assert_allclose(on[1], off[1] * f2, rtol=1e-12)
    assert not np.allclose(f1[:3], f2[:3])


def test_tables_are_saved_and_reused(tables_dir, monkeypatch):
    band_extinction_factor(*NODE, 1.0, BANDS, load_band_grid(GRID), LAW)
    files = sorted(os.listdir(tables_dir))
    assert files == sorted(f'{GRID}__fitzpatrick99__{b}.npz' for b in BANDS)

    def _no_rebuild(*a, **k):
        raise AssertionError('tables rebuilt instead of loaded')
    monkeypatch.setattr(be, 'grid_tables', _no_rebuild)
    fac = band_extinction_factor(*NODE, 1.0, BANDS, load_band_grid(GRID), LAW)
    assert fac[0] != 1.0
    # another law is another table
    monkeypatch.undo()
    monkeypatch.setenv('ARIADNE_BAND_EXTINCTION_DIR', str(tables_dir))
    band_extinction_factor(*NODE, 1.0, BANDS[:1], load_band_grid(GRID),
                           extinction.ccm89)
    assert f'{GRID}__ccm89__GaiaDR2v2_G.npz' in os.listdir(tables_dir)


def test_tables_are_rebuilt_when_their_inputs_change(tables_dir,
                                                     monkeypatch):
    band_extinction_factor(*NODE, 1.0, BANDS[:1], load_band_grid(GRID), LAW)
    calls = []
    orig = be.grid_tables
    monkeypatch.setattr(be, 'grid_tables',
                        lambda *a, **k: calls.append(1) or orig(*a, **k))
    monkeypatch.setattr(be, '_VERSION', be._VERSION + 1)
    band_extinction_factor(*NODE, 1.0, BANDS[:1], load_band_grid(GRID), LAW)
    assert calls == [1]


def test_grid_without_cached_spectra_keeps_the_central_wavelength(
        tables_dir, caplog):
    interp = load_band_grid('sphinx')
    fac = band_extinction_factor(4000., 4.5, 0., 1.0, BANDS[:1], interp, LAW)
    assert fac == 1.0
    assert not os.path.exists(tables_dir)
