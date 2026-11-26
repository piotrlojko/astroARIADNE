"""Per-band excess-scatter terms: names, and the pre-rename ``_noise`` names.

Each fitted band has an extra uncertainty added in quadrature to its error,
named ``<band>_excess_scatter`` (formerly ``<band>_noise``). The old names are
still read: in ``prior_setup`` (with a FutureWarning) and in fit outputs.
"""
import warnings

import numpy as np
import pytest

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter
from astroARIADNE.phot_utils import flux_to_mag
from astroARIADNE.star import Star
from astroARIADNE.utils import (excess_scatter_label, excess_scatter_name,
                                get_band_tag, get_noise_name,
                                is_excess_scatter, upgrade_legacy_names)

MAGS = {'GaiaDR2v2_BP': (9.30, 0.01), 'GaiaDR2v2_G': (9.10, 0.01),
        'GaiaDR2v2_RP': (8.80, 0.01), '2MASS_J': (8.30, 0.02),
        'WISE_RSR_W1': (7.90, 0.02)}
PRIORS = {'teff': ('normal', 6000, 300), 'logg': ('uniform', 3.5, 5.0),
          'z': ('default'), 'dist': ('default'), 'rad': ('uniform', 0.3, 3),
          'Av': ('normal', 0.05, 0.01)}


def _fitter(tmp_path, priors):
    f = Fitter()
    f.star = Star('scattertest', 100., -20., offline=True, plx=1000 / 120.,
                  plx_e=0.02, dist=120., dist_e=1.5, Av=0.05, Av_e=0.01,
                  temp=6000., temp_e=0., verbose=False, mag_dict=MAGS)
    f.setup = ['dynesty', 100, 0.5, 'multi', 'rwalk', 1, False]
    f.out_folder = str(tmp_path)
    f.verbose = False
    f.grid = 'phoenix'
    f.prior_setup = dict(priors)
    return f


def test_names():
    assert excess_scatter_name('WISE_RSR_W1') == 'W1_excess_scatter'
    assert excess_scatter_name('TYCHO_B_MvB') == 'BT_excess_scatter'
    assert excess_scatter_name('IUE_IUE.1250-1300') == \
        'IUE.1250-1300_excess_scatter'
    assert excess_scatter_name('TD1_WIDE.1565') == 'WIDE.1565_excess_scatter'
    assert is_excess_scatter('W1_excess_scatter')
    assert not is_excess_scatter('teff') and not is_excess_scatter('W1_noise')
    assert excess_scatter_label('IUE.1250-1300_excess_scatter') == \
        'IUE.1250-1300 excess scatter'


def test_old_helper_still_works_but_warns():
    with pytest.warns(DeprecationWarning, match='get_band_tag'):
        assert get_noise_name('WISE_RSR_W2') == get_band_tag('WISE_RSR_W2')


def test_legacy_output_keys_are_renamed_in_place_and_in_order():
    out = {'posterior_samples': {'teff': 1, 'G_noise': 2, 'W1_noise': 3},
           'fit_signature': {'priors': {'z': 0, 'G_noise': 4}},
           'engine': 'dynesty'}
    assert upgrade_legacy_names(out) is out
    assert list(out['posterior_samples']) == \
        ['teff', 'G_excess_scatter', 'W1_excess_scatter']
    assert out['fit_signature']['priors'] == {'z': 0, 'G_excess_scatter': 4}
    assert out['engine'] == 'dynesty'


def test_prior_setup_uses_the_new_names(tmp_path):
    f = _fitter(tmp_path, dict(PRIORS, G_excess_scatter=('fixed', 0)))
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        f.initialize()
    order = list(fitter_mod.order)
    assert order[6:] == [excess_scatter_name(b) for b in
                         f.star.filter_names[f.star.filter_mask]]
    i = order.index('G_excess_scatter')
    assert f.coordinator[i] == 1 and f.fixed[i] == 0
    assert 'G_excess_scatter\tfixed' in (tmp_path / 'prior.dat').read_text()


def test_old_noise_keys_in_prior_setup_still_work(tmp_path):
    f = _fitter(tmp_path, dict(PRIORS, G_noise=('fixed', 0)))
    with pytest.warns(FutureWarning, match='G_excess_scatter'):
        f.initialize()
    i = list(fitter_mod.order).index('G_excess_scatter')
    assert f.coordinator[i] == 1
    assert 'G_noise' not in f.prior_setup
    assert not any(k.endswith('_noise') for k in f.priors)


def test_new_name_wins_over_old_for_the_same_band(tmp_path):
    f = _fitter(tmp_path, dict(PRIORS, G_noise=('fixed', 0),
                               G_excess_scatter='default'))
    with pytest.warns(FutureWarning):
        f.initialize()
    i = list(fitter_mod.order).index('G_excess_scatter')
    assert f.coordinator[i] == 0
    assert np.isfinite(f.priors['G_excess_scatter'].ppf(0.5))


def test_term_of_a_band_not_fitted_is_skipped_with_a_warning(tmp_path):
    # e.g. a catalogue no longer returns that band
    f = _fitter(tmp_path, dict(PRIORS, W2_excess_scatter='default',
                               G_excess_scatter='default'))
    with pytest.warns(UserWarning, match="'W2_excess_scatter' is not"):
        f.initialize()
    assert 'W2_excess_scatter' not in f.priors
    assert 'G_excess_scatter' in f.priors


def test_unsupported_prior_type_falls_back_to_the_default(tmp_path):
    f = _fitter(tmp_path, dict(PRIORS, G_excess_scatter=('normal', 0, 1)))
    with pytest.warns(UserWarning, match='take "default"'):
        f.initialize()
    i = list(fitter_mod.order).index('G_excess_scatter')
    assert f.coordinator[i] == 0
    assert f.priors['G_excess_scatter'].dist.name == 'truncnorm'


def _band_flux(f, band):
    names = list(f.star.filter_names[f.star.filter_mask])
    return f.star.flux[f.star.filter_mask][names.index(band)]


def test_uniform_bounds_are_given_in_magnitudes(tmp_path):
    f = _fitter(tmp_path, dict(PRIORS, G_excess_scatter=('uniform', 0, 0.05),
                               J_excess_scatter='default'))
    f.initialize()
    i = list(fitter_mod.order).index('G_excess_scatter')
    assert f.coordinator[i] == 0
    prior = f.priors['G_excess_scatter']
    assert prior.dist.name == 'uniform'
    lo, hi = prior.support()
    flux = _band_flux(f, 'GaiaDR2v2_G')
    assert lo == 0
    # the flux bound is 0.05 mag, as the outputs report the scatter
    assert flux_to_mag(flux, hi, 'GaiaDR2v2_G')[1] == pytest.approx(0.05)
    # other bands keep the default half-normal
    assert f.priors['J_excess_scatter'].dist.name == 'truncnorm'
    assert 'G_excess_scatter\tuniform\t0\t0.05\tmag' in \
        (tmp_path / 'prior.dat').read_text()


def test_truncnorm_bounds_are_given_in_magnitudes(tmp_path):
    f = _fitter(tmp_path, dict(
        PRIORS, W1_excess_scatter=('truncnorm', 0, 0.05, 0, 0.2)))
    f.initialize()
    prior = f.priors['W1_excess_scatter']
    flux = _band_flux(f, 'WISE_RSR_W1')
    lo, hi = prior.support()
    assert lo == pytest.approx(0)
    assert flux_to_mag(flux, hi, 'WISE_RSR_W1')[1] == pytest.approx(0.2)
    # half-normal shape: median 0.674 sigma (well inside the bound)
    assert flux_to_mag(flux, prior.ppf(0.5), 'WISE_RSR_W1')[1] == \
        pytest.approx(0.05 * 0.6745, rel=0.02)


@pytest.mark.parametrize('spec', [
    ('uniform', 0.1, 0.05), ('uniform', -0.1, 0.1), ('uniform', 0.1),
    ('truncnorm', 0, 0, 0, 0.1), ('truncnorm', 0, 0.05, 0.2, 0.1)])
def test_bad_scatter_bounds_stop_the_fit(tmp_path, spec):
    f = _fitter(tmp_path, dict(PRIORS, G_excess_scatter=spec))
    with pytest.raises(SystemExit):
        f.initialize()


def test_reported_scatter_belongs_to_its_band_after_a_fixed_term(tmp_path):
    # BP's term is fixed, so it has no samples; the bands after it must still
    # be converted to magnitudes with their own flux, so each one's samples
    # stay inside its own bound.
    import pickle
    f = _fitter(tmp_path, dict(PRIORS, BP_excess_scatter=('fixed', 0),
                               J_excess_scatter=('uniform', 0, 0.02),
                               W1_excess_scatter=('uniform', 0, 0.05)))
    f.setup = ['dynesty', 40, 5.0, 'multi', 'rwalk', 1, False]
    f.initialize()
    f.fit()
    with open(tmp_path / 'phoenix_out.pkl', 'rb') as fh:
        ps = pickle.load(fh)['posterior_samples']
    for k, cap in (('J_excess_scatter', 0.02), ('W1_excess_scatter', 0.05)):
        s = np.asarray(ps[k])
        assert s.min() >= 0 and s.max() <= cap * (1 + 1e-9), k
        assert s.max() > 0.5 * cap, k      # the samples do fill the range
    assert ps['BP_excess_scatter'] == 0
