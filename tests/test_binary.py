"""Unresolved-binary SED fits: two stars sharing distance, extinction and [Fe/H].

Model and likelihood (sed_library), the Fitter setup (parameters, priors, grid
pairs, checks), a small end-to-end fit with its outputs and plots, and that
single-star fits are unchanged.
"""
import os
import pickle

import extinction
import numpy as np
import pytest

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter
from astroARIADNE.phot_utils import flux_to_mag, get_effective_wavelength
from astroARIADNE.sed_library import (BINARY_PARAMS, _param_order,
                                      binary_components, contrast_loglike,
                                      contrast_model, get_residuals,
                                      load_band_grid, log_likelihood,
                                      model_grid, n_physical, split_grid_name)
from astroARIADNE.star import Star

LAW = extinction.fitzpatrick99
BANDS = np.array(['GaiaDR2v2_BP', 'GaiaDR2v2_G', 'GaiaDR2v2_RP', '2MASS_J',
                  '2MASS_H', '2MASS_Ks', 'WISE_RSR_W1', 'WISE_RSR_W2'])
WAVE = np.array([get_effective_wavelength(b) for b in BANDS])
THETA = np.array([6000., 4.3, 0.0, 120., 1.2, 0.05, 4200., 4.6, 0.7])


@pytest.fixture(scope='module')
def grids():
    return {g: load_band_grid(g) for g in ('phoenix', 'btsettl', 'bosz',
                                           'tlusty')}


# --- model --------------------------------------------------------------------

def test_binary_model_is_the_sum_of_two_single_stars(grids):
    p, b = grids['phoenix'], grids['btsettl']
    one = model_grid(THETA[:6], BANDS, WAVE, p, False, LAW)
    two = model_grid(np.r_[THETA[6:8], THETA[2:4], THETA[8], THETA[5]],
                     BANDS, WAVE, b, False, LAW)
    tot = model_grid(THETA, BANDS, WAVE, (p, b), False, LAW, binary=True)
    np.testing.assert_allclose(tot, one + two, rtol=1e-12)
    m1, m2 = binary_components(THETA, BANDS, WAVE, (p, b), LAW)
    np.testing.assert_allclose(m1, one, rtol=1e-12)
    np.testing.assert_allclose(m2, two, rtol=1e-12)


def test_one_grid_serves_both_components(grids):
    p = grids['phoenix']
    np.testing.assert_allclose(
        model_grid(THETA, BANDS, WAVE, p, False, LAW, binary=True),
        model_grid(THETA, BANDS, WAVE, (p, p), False, LAW, binary=True))


def test_residuals_start_after_the_secondary(grids):
    p = grids['phoenix']
    scatter = np.arange(1, len(BANDS) + 1) * 1e-12
    th = np.r_[THETA, scatter]
    flux = model_grid(THETA, BANDS, WAVE, p, False, LAW, binary=True)
    res, ers = get_residuals(th, flux, np.zeros(len(BANDS)), WAVE, BANDS, p,
                             False, LAW, binary=True)
    np.testing.assert_allclose(res, 0, atol=1e-30)
    np.testing.assert_allclose(ers, scatter)


def test_contrast_is_m2_minus_m1_and_independent_of_distance(grids):
    p = grids['phoenix']
    cf = np.array(['GaiaDR2v2_G', '2MASS_Ks'])
    dm = contrast_model(THETA, p, cf, LAW)
    m1, m2 = binary_components(THETA, BANDS, WAVE, p, LAW)
    j = [list(BANDS).index(b) for b in cf]
    np.testing.assert_allclose(dm, -2.5 * np.log10(m2[j] / m1[j]), rtol=1e-10)
    assert np.all(dm > 0)                      # the K dwarf is fainter
    far = THETA.copy()
    far[3] = 900.
    np.testing.assert_allclose(contrast_model(far, p, cf, LAW), dm,
                               rtol=1e-12)
    # Extinction integrated over a band depends on the spectrum, so in the
    # wide G band the 6000 K and 4200 K stars are reddened differently (A_G /
    # A_V is known to fall with colour), in Ks hardly; without the law (or
    # with band_extinction off) A_V cancels exactly.
    far[5] = 1.5
    shift = contrast_model(far, p, cf, LAW) - dm
    assert 0.05 < abs(shift[0]) < 0.3 and abs(shift[1]) < 0.01
    np.testing.assert_allclose(contrast_model(far, p, cf),
                               contrast_model(THETA, p, cf), rtol=1e-12)
    assert contrast_loglike(THETA, p, (cf, dm, np.ones(2)), LAW) == 0


def test_likelihood_adds_the_contrast_term(grids):
    p = grids['phoenix']
    flux = model_grid(THETA, BANDS, WAVE, p, False, LAW, binary=True)
    th = np.r_[THETA, np.zeros(len(BANDS))]
    err = flux * 0.01
    base = log_likelihood(th, flux, err, WAVE, BANDS, p, False, LAW,
                          binary=True)
    cf = np.array(['GaiaDR2v2_G'])
    dm = contrast_model(THETA, p, cf, LAW)
    c = (cf, dm + 0.2, np.array([0.1]))
    with_c = log_likelihood(th, flux, err, WAVE, BANDS, p, False, LAW,
                            binary=True, contrast=c)
    assert with_c == pytest.approx(base - 0.5 * 2 ** 2)


def test_parameter_order():
    order = _param_order(BANDS, False, binary=True)
    assert list(order[:9]) == ['teff', 'logg', 'z', 'dist', 'rad', 'Av',
                               *BINARY_PARAMS]
    assert len(order) == 9 + len(BANDS)
    assert n_physical(False, True) == 9 and n_physical(False) == 6
    assert n_physical(True) == 5
    assert split_grid_name('bosz+tlusty') == ('bosz', 'tlusty')
    assert split_grid_name('bosz') == ('bosz', None)


# --- Fitter setup -------------------------------------------------------------

MAGS = {'GaiaDR2v2_BP': (9.30, 0.01), 'GaiaDR2v2_G': (9.10, 0.01),
        'GaiaDR2v2_RP': (8.80, 0.01), '2MASS_J': (8.30, 0.02),
        '2MASS_H': (8.00, 0.02), '2MASS_Ks': (7.95, 0.02),
        'WISE_RSR_W1': (7.90, 0.02), 'WISE_RSR_W2': (7.92, 0.02)}
PRIORS = {'teff': ('normal', 6000, 300), 'logg': ('uniform', 3.5, 5.0),
          'z': ('default'), 'dist': ('default'), 'rad': ('uniform', 0.3, 3),
          'Av': ('normal', 0.05, 0.01)}


def _star(mags=MAGS, **kw):
    kw.setdefault('temp', 6000.)
    return Star('bintest', 100., -20., offline=True, plx=1000 / 120.,
                plx_e=0.02, dist=120., dist_e=1.5, Av=0.05, Av_e=0.01,
                temp_e=0., verbose=False, mag_dict=mags, **kw)


def _fitter(tmp_path, star=None, priors=PRIORS, bma=False, grid='phoenix',
            grid2=None, models=None, models2=None, nlive=500):
    f = Fitter()
    f.star = star or _star()
    f.setup = ['dynesty', nlive, 0.5, 'multi', 'rwalk', 1, False]
    f.out_folder = str(tmp_path)
    f.verbose = False
    f.binary = True
    f.bma = bma
    if bma:
        f.models = models
        f.models2 = models2
    else:
        f.grid = grid
        if grid2:
            f.grid2 = grid2
    f.prior_setup = dict(priors)
    return f


def test_initialize_adds_the_secondary(tmp_path):
    f = _fitter(tmp_path, grid2='btsettl')
    f.initialize()
    assert list(fitter_mod.order[6:9]) == list(BINARY_PARAMS)
    assert f.ndim == 9 + len(MAGS)
    assert fitter_mod.binary is True
    i1, i2 = fitter_mod.interpolator
    assert i1 is f._interpolator and i2 is f._interpolator2
    # secondary priors default to the primary's *default* Teff and radius
    # priors (not a star-specific prior set for the primary) and U(3.5, 6)
    assert f.priors['teff2'] is f.default_priors['teff']
    assert f.priors['rad2'] is f.default_priors['rad']
    assert f.priors['logg2'].ppf(0) == pytest.approx(3.5)
    assert f._out_name() == 'phoenix+btsettl'


def test_tlusty_secondary_gets_the_hot_star_teff_prior(tmp_path):
    f = _fitter(tmp_path, grid='bosz', grid2='tlusty')
    f.initialize()
    assert f.priors['teff2'].ppf(0) == pytest.approx(15000)
    assert f.priors['teff2'].ppf(1) == pytest.approx(55000)


def test_user_priors_for_the_secondary(tmp_path):
    pri = dict(PRIORS, teff2=('uniform', 3000, 5000), rad2=('fixed', 0.7))
    f = _fitter(tmp_path, priors=pri)
    f.initialize()
    assert f.priors['teff2'].ppf(1) == pytest.approx(5000)
    i = list(fitter_mod.order).index('rad2')
    assert f.coordinator[i] == 1 and f.fixed[i] == 0.7
    assert f.ndim == 8 + len(MAGS)


def test_single_star_layout_is_unchanged(tmp_path):
    f = _fitter(tmp_path)
    f.binary = False
    f.initialize()
    assert list(fitter_mod.order[:6]) == ['teff', 'logg', 'z', 'dist', 'rad',
                                          'Av']
    assert 'teff2' not in fitter_mod.order and 'teff2' not in f.priors
    assert f.ndim == 6 + len(MAGS)
    assert fitter_mod.binary is False
    assert fitter_mod.interpolator is f._interpolator


def test_binary_needs_radius_and_distance(tmp_path):
    f = _fitter(tmp_path, priors={'teff': ('default'), 'logg': ('default'),
                                  'z': ('default'), 'norm': ('default'),
                                  'Av': ('default')})
    f.norm = True
    with pytest.raises(SystemExit):
        f.initialize()


def test_secondary_grid_must_have_the_bands(tmp_path):
    mags = dict(MAGS, **{'IUE_IUE.1250-1300': (6.3, 0.05)})
    f = _fitter(tmp_path, star=_star(mags, temp=11500.), grid='bosz',
                grid2='phoenix')
    with pytest.raises(SystemExit):
        f.initialize()
    g = _fitter(tmp_path, star=_star(mags, temp=11500.), grid='bosz',
                grid2='tlusty')
    g.initialize()                               # BOSZ + TLUSTY both have it


def test_contrast_band_must_be_modelled(tmp_path):
    f = _fitter(tmp_path, grid='phoenix')
    f.add_contrast('IUE_IUE.1250-1300', 1.0, 0.1)
    with pytest.raises(SystemExit):
        f.initialize()
    g = _fitter(tmp_path, grid='phoenix')
    g.add_contrast('GaiaDR2v2_G', 2.5, 0.1)
    g.initialize()
    bands, dm, sig = fitter_mod.contrast
    assert list(bands) == ['GaiaDR2v2_G'] and dm[0] == 2.5 and sig[0] == 0.1


def test_bma_fits_every_grid_pair(tmp_path):
    f = _fitter(tmp_path, bma=True, models=['phoenix', 'btsettl'],
                models2=['btsettl', 'bosz'])
    f.initialize()
    assert f._grids == ['phoenix+btsettl', 'phoenix+bosz', 'btsettl+btsettl',
                        'btsettl+bosz']
    assert all(isinstance(i, tuple) for i in f._interpolators)


def test_bma_pairs_a_grid_with_itself_by_default(tmp_path):
    f = _fitter(tmp_path, bma=True, models=['phoenix', 'btsettl'])
    f.initialize()
    assert f._grids == ['phoenix+phoenix', 'btsettl+btsettl']


def test_bma_drops_pairs_without_the_bands(tmp_path, caplog):
    mags = dict(MAGS, **{'TD1_WIDE.1565': (7.7, 0.1)})
    f = _fitter(tmp_path, star=_star(mags, temp=11500.), bma=True,
                models=['bosz', 'kurucz'], models2=['tlusty', 'btsettl'])
    with caplog.at_level('WARNING', logger='astroARIADNE.fitter'):
        f.initialize()
    assert f._grids == ['bosz+tlusty']
    assert any("'bosz+btsettl'" in r.getMessage() for r in caplog.records)


def test_changing_to_binary_invalidates_reuse(tmp_path):
    f = _fitter(tmp_path)
    f.initialize()
    sig_bin = f._fit_signature()
    f._binary = False
    assert f._fit_signature()['binary'] is False
    assert sig_bin['binary'] is True


# --- end to end ---------------------------------------------------------------

def _synthetic_mags(grids):
    th = np.array([11500., 4.0, 0.0, 460., 2.3, 0.38, 30000., 4.5, 0.25])
    bands = np.array(list(BANDS) + ['IUE_IUE.1250-1300', 'TD1_WIDE.1565'])
    wave = np.array([get_effective_wavelength(b) for b in bands])
    m1, m2 = binary_components(th, bands, wave,
                               (grids['bosz'], grids['tlusty']), LAW)
    mags = {}
    for b, f in zip(bands, m1 + m2):
        m, e = flux_to_mag(f, f * 0.02, b)
        mags[str(b)] = (float(m), float(e))
    return mags, (m2 / (m1 + m2))


def test_end_to_end_binary_fit_and_plots(tmp_path, grids):
    mags, frac2 = _synthetic_mags(grids)
    star = Star('e2e', 226.7, -31.6, offline=True, plx=1000 / 460.,
                plx_e=0.03, dist=460., dist_e=8., Av=0.38, Av_e=0.02,
                temp=11500., temp_e=0., verbose=False, mag_dict=mags)
    pri = {'teff': ('normal', 11500, 500), 'logg': ('uniform', 3.5, 4.5),
           'z': ('normal', 0.0, 0.1), 'dist': ('default'),
           'rad': ('uniform', 1.5, 3.0), 'Av': ('normal', 0.38, 0.02),
           'teff2': ('uniform', 20000, 40000), 'logg2': ('uniform', 4.0, 4.75),
           'rad2': ('uniform', 0.05, 0.6),
           'IUE.1250-1300_excess_scatter': ('fixed', 0),
           'WIDE.1565_excess_scatter': ('fixed', 0)}
    f = _fitter(tmp_path, star=star, priors=pri, grid='bosz', grid2='tlusty',
                nlive=60)
    f.initialize()
    f.fit()
    pk = tmp_path / 'bosz+tlusty_out.pkl'
    with open(pk, 'rb') as fh:
        out = pickle.load(fh)
    assert out['binary'] is True and out['model_grid'] == 'bosz+tlusty'
    assert out['band_extinction'] is True
    ps = out['posterior_samples']
    for k in ('teff2', 'logg2', 'rad2', 'lum2', 'grav_mass2', 'AD2',
              'frac2_IUE_IUE.1250-1300'):
        assert np.size(ps[k]) > 10
    # the secondary dominates the far UV, not the optical
    assert np.median(ps['frac2_IUE_IUE.1250-1300']) > 0.3
    assert np.median(ps['frac2_GaiaDR2v2_G']) < 0.2
    txt = (tmp_path / 'best_fit.dat').read_text()
    assert 'teff2' in txt and 'lum2' in txt
    lf = (tmp_path / 'binary_light_fractions.dat').read_text().splitlines()
    assert sum(not l.startswith('#') for l in lf) == len(mags)

    from astroARIADNE.plotter import SEDPlotter
    p = SEDPlotter(str(pk), str(tmp_path / 'plots'), pdf=False)
    assert p.binary and (p.grid, p.grid2) == ('bosz', 'tlusty')
    # the plot's model fluxes are reddened as the fit's were
    assert all(it.band_extinction for it in p.interpolator)
    i = int(np.argmax(ps['loglike']))
    assert p.theta[6] == ps['teff2'][i]          # max-likelihood sample
    theta = p.theta.copy()
    p.plot_SED()
    # plot_SED leaves the plotted vector alone (its excess scatter stays in
    # magnitudes), so it can be drawn again
    np.testing.assert_array_equal(p.theta, theta)
    # Each point: the measurement error solid with caps, in front of the
    # total uncertainty sigma_tot = sqrt(sigma_obs^2 + s^2) dotted without.
    # The SED axis is lambda F_lambda, so error bars are lambda * sigma_F.
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer
    from astroARIADNE.plotter import _SIGMA_TOT_LABEL

    def seg(c):          # the y bar: [(x, y - e), (x, y + e)]
        segs = c.lines[2][-1].get_segments()
        return np.asarray(segs[0], dtype=float) if segs else np.empty(0)

    def bars(ax):
        # (the legend's sample bars have no finite segment)
        cs = [c for c in ax.containers if isinstance(c, ErrorbarContainer)
              and seg(c).shape == (2, 2) and np.isfinite(seg(c)).all()]
        half = [np.ptp(seg(c)[:, 1]) / 2 for c in cs]
        dotted = [c.lines[2][-1].get_linestyle()[0][1] is not None
                  for c in cs]
        solid = [(h, c) for h, c, d in zip(half, cs, dotted) if not d]
        tot = [(h, c) for h, c, d in zip(half, cs, dotted) if d]
        return solid, tot

    ax, ax_r = plt.gcf().axes[:2]
    solid, tot = bars(ax)
    assert len(solid) == len(tot) == len(p.wave)
    for (h_o, c_o), (h_t, c_t), w, fe in zip(solid, tot, p.wave, p.flux_er):
        assert h_o == pytest.approx(fe * w, rel=1e-9)
        assert h_t >= fe * w * (1 - 1e-9)
        if w < 0.2:      # UV bands: excess scatter fixed at zero
            assert h_t == pytest.approx(fe * w, rel=1e-9)
        assert len(c_o.lines[1]) > 0 and len(c_t.lines[1]) == 0   # caps
        assert max(a.get_zorder() for a in c_o.lines[2]) > \
            c_t.lines[2][0].get_zorder()
    # residuals in units of sigma_tot: the total bar is +-1, the measurement
    # error a fraction of it, and the unit is spelt out
    solid_r, tot_r = bars(ax_r)
    assert [h for h, _ in tot_r] == pytest.approx([1.0] * len(p.wave))
    assert all(0 < h <= 1 + 1e-9 for h, _ in solid_r)
    assert r'\sigma_{\rm tot}' in ax_r.get_ylabel()
    assert _SIGMA_TOT_LABEL in [t.get_text() for t in ax_r.texts]
    legend_text = [t.get_text() for lg in ax.get_children()
                   if isinstance(lg, plt.matplotlib.legend.Legend)
                   for t in lg.get_texts()]
    assert r'$\pm\sigma_{\rm obs}$ (measured)' in legend_text
    # components in their own colours, the sum in black
    from matplotlib.colors import to_hex
    lines = {ln.get_label().split(':')[0]: to_hex(ln.get_color())
             for ln in plt.gcf().axes[0].get_lines()}
    assert lines['Primary'] == '#1f4e79'
    assert lines['Secondary'] == '#c1432e'
    assert lines['Sum'] == '#000000'
    # all solid; the components at opacity 0.8
    style = {ln.get_label().split(':')[0]: (ln.get_linestyle(), ln.get_alpha())
             for ln in plt.gcf().axes[0].get_lines()}
    assert style['Primary'] == style['Secondary'] == ('-', 0.8)
    assert style['Sum'] == ('-', None)
    plt.close('all')
    p.plot_corner()
    assert (tmp_path / 'plots' / 'SED.png').exists()
    assert 'IUE.1250-1300_excess_scatter' in ps

    # An output written before the rename (<band>_noise) still opens.
    def _legacy(d):
        if isinstance(d, dict):
            return {(k[:-len('_excess_scatter')] + '_noise'
                     if isinstance(k, str) and k.endswith('_excess_scatter')
                     else k): _legacy(v) for k, v in d.items()}
        return d
    old = tmp_path / 'old_out.pkl'
    legacy = _legacy(out)
    del legacy['band_extinction']     # nor did it have band extinction
    with open(old, 'wb') as fh:
        pickle.dump(legacy, fh)
    po = SEDPlotter(str(old), str(tmp_path / 'plots_old'), pdf=False)
    assert 'G_excess_scatter' in po.order
    assert not any(it.band_extinction for it in po.interpolator)
    assert not any(k.endswith('_noise') for k in po.out['posterior_samples'])
    po.plot_SED()
    plt.close('all')
    assert (tmp_path / 'plots' / 'CORNER.png').exists()
