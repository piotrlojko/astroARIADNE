"""IUE / TD-1 ultraviolet bands: carried by the BOSZ and TLUSTY grids only.

Grids without model fluxes for a fitted band leave a BMA, and a standalone
fit with such a grid stops with an error naming the grids that have it.
"""
import os
from types import SimpleNamespace

import astropy.units as u
import extinction
import numpy as np
import pandas as pd
import pytest
from isochrones.interp import DFInterpolator

import astroARIADNE.config as config
from astroARIADNE.fitter import Fitter, _missing_bands
from astroARIADNE.star import Star

UV = ['IUE_IUE.1250-1300', 'IUE_IUE.1450-1500', 'IUE_IUE.1675-1725',
      'IUE_IUE.2150-2200', 'IUE_IUE.2395-2445', 'IUE_IUE.2900-2950',
      'IUE_IUE.2900-3000', 'TD1_WIDE.1565', 'TD1_WIDE.1965',
      'TD1_WIDE.2365', 'TD1_WIDE.2740']
FLAM = u.erg / (u.s * u.cm ** 2 * u.AA)
# A late-B star (HD 133729-like) with two far-UV fluxes.
MAGS = {'GaiaDR2v2_BP': (9.112, 0.005), 'GaiaDR2v2_G': (9.111, 0.003),
        'GaiaDR2v2_RP': (9.068, 0.005), '2MASS_J': (9.010, 0.024),
        '2MASS_H': (9.076, 0.024), '2MASS_Ks': (9.072, 0.023)}
FLUXES = {'TD1_WIDE.1565': (5.25e-12 * FLAM, 4.9e-13 * FLAM),
          'IUE_IUE.1250-1300': (4.574742e-12 * FLAM, 1.830e-13 * FLAM)}


def _grid(name):
    return Fitter.load_interpolator(name)


def _star(fluxes=FLUXES, **kw):
    return Star('uvdemo', 226.74555, -31.6441583, offline=True, dist=463.,
                dist_e=14., Av=0.38, Av_e=0.02, temp=11500., temp_e=0.,
                verbose=False, mag_dict=MAGS, flux_dict=fluxes, **kw)


def _fitter(star, tmp_path, bma, models=None, grid=None):
    f = Fitter()
    f.star = star
    f.setup = ['dynesty']
    f.out_folder = str(tmp_path)
    f.verbose = False
    f.bma = bma
    if bma:
        f.models = models
    else:
        f.grid = grid
    f.prior_setup = {'teff': ('normal', 11500, 1000),
                     'logg': ('normal', 4.0, 0.5), 'z': ('default'),
                     'dist': ('default'), 'rad': ('default'),
                     'Av': ('normal', 0.38, 0.02)}
    return f


# --- helper -----------------------------------------------------------------

def test_missing_bands_helper():
    idx = pd.MultiIndex.from_product([[4.0, 4.5], [5000., 6000.], [0.0]],
                                     names=['logg', 'teff', 'feh'])
    df = pd.DataFrame({'A': np.arange(4.), 'B': np.full(4, np.nan)},
                      index=idx)
    df.loc[(4.0, 5000., 0.0), 'A'] = np.nan          # partial NaN is fine
    intp = DFInterpolator(df)
    assert _missing_bands(intp, ['A', 'B', 'C']) == ['B', 'C']


# --- grids ------------------------------------------------------------------

@pytest.mark.parametrize('grid', ['bosz', 'tlusty'])
def test_bosz_and_tlusty_carry_uv_bands(grid):
    intp = _grid(grid)
    assert _missing_bands(intp, UV) == []
    j = [intp.column_index[b] for b in UV]
    vals = intp.grid[..., j]
    # every node a grid has elsewhere has a finite, positive UV flux
    has = np.isfinite(intp.grid[..., intp.column_index['2MASS_J']])
    assert np.all(np.isfinite(vals[has]))
    assert np.all(vals[has] >= 0)


@pytest.mark.parametrize('grid', ['phoenix', 'btsettl', 'btnextgen',
                                  'btcond', 'ck04', 'kurucz', 'coelho',
                                  'sphinx'])
def test_other_grids_lack_uv_bands(grid):
    assert _missing_bands(_grid(grid), UV) == UV


def _cache_groups():
    path = config.spectra_cache
    if not path or not os.path.isfile(path):
        return set()
    import h5py
    with h5py.File(path, 'r') as h:
        return set(h.keys())


@pytest.mark.parametrize('grid', ['bosz', 'tlusty'])
def test_uv_columns_match_cache_spectra(grid):
    """The grid's UV columns (from the raw spectra) agree with the same
    bands convolved from the cache group (broadened/resampled copies)."""
    if grid not in _cache_groups():
        pytest.skip(f'spectra cache has no {grid} group')
    import h5py
    from astroARIADNE import synth_phot as sp
    filts = sp.resolve_filters(UV + ['GROUND_JOHNSON_V'])
    with h5py.File(config.spectra_cache, 'r') as h:
        g = h[grid]
        w = g['wavelength'][:].astype(float)
        t, lg, z = g['teff'][:], g['logg'][:], g['z'][:]
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(t), 40, replace=False))
        F = g['flux'][pick].astype(float)
    fl = sp.convolve(F, *sp.coeff_matrix(filts, w))
    df = pd.read_pickle(os.path.join(config.gridsdir, {
        'bosz': 'BOSZ_DF.pkl', 'tlusty': 'TLUSTY_DF.pkl'}[grid]))
    for k, i in enumerate(pick):
        row = df.xs((float(lg[i]), float(t[i])), level=(0, 1))
        row = row.iloc[np.argmin(np.abs(row.index.values - z[i]))]
        ref = row[[f.name for f in filts]].values.astype(float)
        ok = ref > 1e-3 * ref.max()                  # skip negligible FUV
        np.testing.assert_allclose(fl[k][ok], ref[ok], rtol=0.01)


# --- fitter -----------------------------------------------------------------

def test_star_takes_uv_fluxes():
    s = _star()
    names = list(s.filter_names[s.filter_mask])
    assert 'TD1_WIDE.1565' in names and 'IUE_IUE.1250-1300' in names


def test_bma_leaves_out_grids_without_the_bands(tmp_path, caplog):
    f = _fitter(_star(), tmp_path, bma=True,
                models=['phoenix', 'btsettl', 'kurucz', 'bosz'])
    with caplog.at_level('WARNING', logger='astroARIADNE.fitter'):
        f.initialize()
    assert f._grids == ['bosz']
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("Grid 'phoenix' has no model fluxes for" in m
               for m in msgs) == 1


def test_bma_with_no_capable_grid_exits(tmp_path):
    f = _fitter(_star(), tmp_path, bma=True, models=['phoenix', 'btsettl'])
    with pytest.raises(SystemExit):
        f.initialize()


def test_bma_without_uv_bands_keeps_every_grid(tmp_path):
    f = _fitter(_star(fluxes=None), tmp_path, bma=True,
                models=['phoenix', 'btsettl', 'bosz'])
    f.initialize()
    assert f._grids == ['phoenix', 'btsettl', 'bosz']


def test_standalone_grid_without_the_bands_exits(tmp_path):
    f = _fitter(_star(), tmp_path, bma=False, grid='phoenix')
    with pytest.raises(SystemExit):
        f.initialize()


def test_standalone_bosz_fits_uv_bands(tmp_path):
    f = _fitter(_star(), tmp_path, bma=False, grid='bosz')
    f.initialize()
    assert f.ndim == 6 + len(MAGS) + len(FLUXES)


def test_synthetic_photometry_skips_bands_btsettl_lacks(tmp_path):
    f = Fitter.__new__(Fitter)
    f.out_folder = str(tmp_path)
    f._norm = False
    f._av_law = extinction.fitzpatrick99
    f.star = SimpleNamespace(filter_names=config.filter_names,
                             used_filters=np.zeros(len(config.filter_names)))
    r = np.random.default_rng(1)
    n = 200
    samples = {'teff': r.normal(6000, 50, n), 'logg': r.uniform(4.2, 4.4, n),
               'z': np.zeros(n), 'Av': r.uniform(0, 0.1, n),
               'dist': r.normal(50, 0.5, n), 'rad': r.normal(1.0, 0.02, n)}
    photom = f._synthetic_photometry(samples)
    assert '2MASS_J' in photom and 'GaiaDR2v2_G' in photom
    assert not set(photom) & set(UV)


# --- plotter ------------------------------------------------------------------

def test_cache_lookup_takes_nearest_existing_node():
    """TLUSTY has no log g = 1.75 model at 55000 K: the per-axis nearest
    combination does not exist, so the nearest existing node is used."""
    if 'tlusty' not in _cache_groups():
        pytest.skip('spectra cache has no tlusty group')
    import h5py
    from astroARIADNE.plotter import SEDPlotter
    p = SEDPlotter.__new__(SEDPlotter)
    p.theta = np.array([55000., 1.75, 0.0, 400., 10., 0.1])
    wave, flux = p._load_from_cache('tlusty')
    with h5py.File(config.spectra_cache, 'r') as h:
        g = h['tlusty']
        t, lg, z = g['teff'][:], g['logg'][:], g['z'][:]
        assert not np.any((t == 55000) & (lg == 1.75))
        hot = np.where((t == 55000) & (z == 0))[0]
        i = hot[np.argmin(lg[hot])]                 # lowest log g at 55 kK
        np.testing.assert_array_equal(flux, g['flux'][i].astype(float))
    assert wave[0] < 0.1249 and np.all(np.isfinite(flux))


@pytest.mark.parametrize('waves, lo', [([0.43, 0.64, 2.16], 0.25),
                                       ([0.154, 0.64], 0.125),
                                       ([0.2245, 0.64], 0.125),
                                       ([0.1275, 0.1565, 0.64], 0.1)])
def test_sed_plot_starts_blueward_of_the_bluest_band(waves, lo):
    from astroARIADNE.plotter import SEDPlotter
    p = SEDPlotter.__new__(SEDPlotter)
    p.star = SimpleNamespace(wave=np.array(waves + [0.9]),
                             filter_mask=np.arange(len(waves)))
    assert p._sed_xmin() == lo


def test_mags_dat_keeps_long_band_names(tmp_path):
    s = _star()
    s.add_mag(15.1, 0.02, 'HST_ACS_WFC_F850LP')
    s.save_mags(str(tmp_path) + '/')
    names = [l.split()[0] for l in open(tmp_path / 'mags.dat')
             if not l.startswith('#')]
    assert 'IUE_IUE.1250-1300' in names and 'HST_ACS_WFC_F850LP' in names
