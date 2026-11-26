"""fit_bma reuses compatible <grid>_out.pkl files unless refit is set (#56).

Offline: dynesty, the averaging and the output step are stubbed out.
"""
import os
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.stats as st

import astroARIADNE.fitter as fitter_mod
from astroARIADNE.fitter import Fitter

FILTS = ['2MASS_J', '2MASS_H', '2MASS_Ks', 'GaiaDR2v2_G', 'WISE_RSR_W1',
         'WISE_RSR_W2', 'SDSS_g']
FLUX = np.linspace(1e-10, 7e-10, len(FILTS))


def _star(filts=FILTS, flux=FLUX):
    return SimpleNamespace(filter_names=np.array(filts),
                           filter_mask=np.ones(len(filts), dtype=bool),
                           flux=np.asarray(flux[:len(filts)]))


def _fitter(tmp_path, grids, refit=False, norm=False, teff_hi=7000.):
    f = Fitter.__new__(Fitter)
    f.out_folder = str(tmp_path)
    f.star = _star()
    f._grids = list(grids)
    f._interpolators = [None] * len(grids)
    f._norm = norm
    f._refit = refit
    f.n_grid_jobs = 1
    f.n_samples = 10
    f.colors = ['white']
    f.coordinator = np.zeros(6)
    f.fixed = np.zeros(6)
    f.priors = {'teff': st.uniform(2300., teff_hi - 2300.)}
    f._engine, f._nlive, f._dlogz = 'dynesty', 500, 0.5
    f._bound, f._sample, f._dynamic = 'multi', 'rwalk', False
    f._av_law = 'fitzpatrick'
    f.start = 0
    f._binary = False
    f.binary = False
    return f


def _fake_out(tmp_path, grid, norm=False, star=None, signature=None,
              band_extinction=True):
    d = {'global_lnZ': 1.0, 'posterior_samples': {},
         'star': star or _star(), 'norm': norm,
         'model_grid': grid, 'binary': False}
    if band_extinction is not None:
        d['band_extinction'] = band_extinction
    if signature is not None:
        d['fit_signature'] = signature
    with open(tmp_path / f'{grid}_out.pkl', 'wb') as fh:
        pickle.dump(d, fh)


def _run(monkeypatch, f):
    fitted, averaged = [], []
    monkeypatch.setattr(Fitter, 'fit_dynesty',
                        lambda self, out_file=None: fitted.append(self.grid))
    monkeypatch.setattr(
        Fitter, 'bayesian_model_average',
        staticmethod(lambda outs, grids, *a: averaged.append(list(grids))))
    monkeypatch.setattr(Fitter, 'save_bma', lambda self, avgd: None)
    monkeypatch.setattr(fitter_mod, 'execution_time', lambda s: 0)
    monkeypatch.setattr(fitter_mod, 'end', lambda *a, **k: None)
    f.fit_bma()
    return fitted, averaged


def test_reuses_existing_fits_and_fits_missing(tmp_path, monkeypatch):
    f = _fitter(tmp_path, ['phoenix', 'btsettl'])
    _fake_out(tmp_path, 'phoenix', signature=f._fit_signature())
    fitted, averaged = _run(monkeypatch, f)
    assert fitted == ['btsettl']
    assert averaged == [['phoenix', 'btsettl']]


def test_refit_fits_everything(tmp_path, monkeypatch):
    f = _fitter(tmp_path, ['phoenix', 'btsettl'])
    _fake_out(tmp_path, 'phoenix', signature=f._fit_signature())
    f.refit = True
    fitted, _ = _run(monkeypatch, f)
    assert fitted == ['phoenix', 'btsettl']


def test_incompatible_fit_is_refitted(tmp_path, monkeypatch, capsys):
    f = _fitter(tmp_path, ['phoenix', 'btsettl', 'kurucz', 'ck04', 'bosz'])
    sig = f._fit_signature()
    _fake_out(tmp_path, 'phoenix', star=_star(FILTS[:-1]), signature=sig)
    _fake_out(tmp_path, 'btsettl', norm=True, signature=sig)
    _fake_out(tmp_path, 'kurucz', star=_star(flux=FLUX * 1.1), signature=sig)
    other = _fitter(tmp_path, [], teff_hi=8000.)._fit_signature()
    _fake_out(tmp_path, 'ck04', signature=other)               # prior differs
    _fake_out(tmp_path, 'bosz', signature=sig)                 # compatible
    fitted, _ = _run(monkeypatch, f)
    assert fitted == ['phoenix', 'btsettl', 'kurucz', 'ck04']
    out = capsys.readouterr().out
    for why in ('filters changed', 'norm vs dist/rad',
                'photometry values changed', 'priors changed'):
        assert why in out


def test_older_output_reused_with_note(tmp_path, monkeypatch, capsys):
    _fake_out(tmp_path, 'phoenix')                    # no fit_signature
    fitted, _ = _run(monkeypatch, _fitter(tmp_path, ['phoenix']))
    assert fitted == []
    assert 'not checked' in capsys.readouterr().out


def test_extinction_law_is_recorded_the_same_in_every_run(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    # The law is recorded by name. Outputs written before recorded
    # str(function), whose memory address changes from run to run, so they
    # were never reused by a later run.
    import extinction
    f = _fitter(tmp_path, ['phoenix', 'btsettl', 'kurucz'])
    f._av_law = extinction.fitzpatrick99
    sig = f._fit_signature()
    assert sig['av_law'] == 'fitzpatrick'
    _fake_out(tmp_path, 'phoenix', signature=sig)
    _fake_out(tmp_path, 'btsettl', signature=dict(        # older, another run
        sig, av_law='<cyfunction fitzpatrick99 at 0x7acfb61d9900>'))
    _fake_out(tmp_path, 'kurucz', signature=dict(         # another law
        sig, av_law='<cyfunction ccm89 at 0x7d90b43d9900>'))
    fitted, _ = _run(monkeypatch, f)
    assert fitted == ['kurucz']
    assert 'extinction law changed' in capsys.readouterr().out


def test_output_from_before_the_excess_scatter_rename_is_reused(tmp_path,
                                                                monkeypatch):
    f = _fitter(tmp_path, ['phoenix'])
    f.priors = dict(f.priors, G_excess_scatter=st.truncnorm(
        a=0, b=10, loc=0, scale=1e-11))
    sig = f._fit_signature()
    old = dict(sig, priors={k.replace('_excess_scatter', '_noise'): v
                            for k, v in sig['priors'].items()})
    assert 'G_noise' in old['priors']
    _fake_out(tmp_path, 'phoenix', signature=old)
    fitted, _ = _run(monkeypatch, f)
    assert fitted == []


def test_fit_with_another_band_extinction_setting_is_refitted(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    # Outputs written before band-integrated extinction (no
    # 'band_extinction' entry) used the central-wavelength extinction.
    f = _fitter(tmp_path, ['phoenix', 'btsettl', 'kurucz'])
    sig = f._fit_signature()
    _fake_out(tmp_path, 'phoenix', signature=sig)
    _fake_out(tmp_path, 'btsettl', signature=sig, band_extinction=None)
    _fake_out(tmp_path, 'kurucz', signature=sig, band_extinction=False)
    fitted, _ = _run(monkeypatch, f)
    assert fitted == ['btsettl', 'kurucz']
    assert 'band_extinction' in capsys.readouterr().out
    # and the other way round
    f.band_extinction = False
    fitted, _ = _run(monkeypatch, f)
    assert fitted == ['phoenix']
