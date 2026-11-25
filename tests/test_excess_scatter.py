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
