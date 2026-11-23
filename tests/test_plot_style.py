"""Plot styling: STIX fonts, per-filter colours and legends, value titles.

Every SEDPlotter plot is drawn under the fonts of ``plot_settings.dat``
(STIXGeneral text and STIX mathtext by default) without touching the caller's
rcParams. SED points are coloured by band, and a legend beside the plot names
the bands, bluest first.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

from astroARIADNE.config import filesdir
from astroARIADNE.plotter import SEDPlotter, _grid_label
from astroARIADNE.star import Star

MAGS = {'2MASS_Ks': (9.072, 0.023), 'GaiaDR2v2_BP': (9.112, 0.005),
        'GaiaDR2v2_G': (9.111, 0.003), '2MASS_J': (9.010, 0.024),
        'GaiaDR2v2_RP': (9.068, 0.005), '2MASS_H': (9.076, 0.024)}


def _plotter(tmp_path, **kw):
    return SEDPlotter('raw', str(tmp_path), pdf=False, **kw)


def _star():
    return Star('styledemo', 226.74555, -31.6441583, offline=True,
                dist=463., dist_e=14., Av=0.38, Av_e=0.02, temp=11500.,
                temp_e=0., verbose=False, mag_dict=MAGS)


def _legend_texts(ax):
    legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    return {(lg.get_title().get_text() or None):
            [t.get_text() for t in lg.get_texts()] for lg in legends}


def test_default_fonts_are_stix(tmp_path):
    rc = _plotter(tmp_path)._rc_params()
    assert rc['font.family'] == 'STIXGeneral'
    assert rc['mathtext.fontset'] == 'stix'


def test_settings_file_without_new_keys_still_works(tmp_path):
    old = [l for l in open(filesdir + '/plot_settings.dat')
           if not l.startswith(('mathtext_fontset', 'legend_fontsize',
                                'filter_cmap', 'corner_title_fontsize'))]
    cfg = tmp_path / 'old_settings.dat'
    cfg.write_text(''.join(old))
    p = _plotter(tmp_path, settings=str(cfg))
    assert (p.mathtext_fontset, p.legend_fontsize, p.filter_cmap,
            p.corner_title_fontsize) == ('stix', 16, 'turbo', 18)


def test_photometry_plot_has_filter_legend_and_stix_text(tmp_path):
    before = dict(plt.rcParams)
    p = _plotter(tmp_path)
    p.plot_SED_no_model(s=_star())
    ax = plt.gcf().axes[0]
    legends = _legend_texts(ax)
    names = p.star.filter_names[p.star.filter_mask]
    assert legends['Filter'] == list(names[np.argsort(p.wave)])
    assert ax.xaxis.label.get_fontname() == 'STIXGeneral'
    assert ax.xaxis.label.get_fontproperties().get_math_fontfamily() \
        == 'stix'
    assert (tmp_path / 'SED_no_model.png').exists()
    # the caller's rcParams are left alone
    assert plt.rcParams['font.family'] == before['font.family']
    assert plt.rcParams['mathtext.fontset'] == before['mathtext.fontset']
    plt.close('all')


def test_filter_colours_follow_wavelength_rank(tmp_path):
    p = _plotter(tmp_path)
    waves = np.array([2.2, 0.44, 0.55, 1.25, 0.551])
    cols = p._filter_colors(waves)
    assert cols.shape == (5, 4)
    # colours are assigned by wavelength rank: the same set comes back for
    # sorted input, in sorted order
    np.testing.assert_allclose(cols[np.argsort(waves)],
                               p._filter_colors(np.sort(waves)))
    # bands 1 nm apart still get distinct colours
    assert np.abs(cols[2] - cols[4]).max() > 0.05
    # and none is too pale for white paper
    lin = np.where(cols[:, :3] <= 0.04045, cols[:, :3] / 12.92,
                   ((cols[:, :3] + 0.055) / 1.055) ** 2.4)
    assert np.all(lin @ [0.2126, 0.7152, 0.0722] <= 0.43 + 1e-9)


def test_value_titles_round_to_the_error(tmp_path):
    p = _plotter(tmp_path)
    assert p._value_title('teff', 11079., 11946., 10663.) == \
        r'$T_{\rm eff} = 11079^{+867}_{-416}$'
    assert p._value_title('rad2', 0.336, 0.421, 0.313) == \
        r'$R_2 = 0.336^{+0.085}_{-0.023}$'
    t = p._value_title('norm', 1.234e-20, 1.30e-20, 1.20e-20)
    assert t.endswith(r'\times10^{-20}$') and '1.23' in t
    p.binary = True
    assert p._value_title('teff', 11079., 11946., 10663.).startswith(
        r'$T_{\rm eff,1} = ')
    assert p._param_label('rad') == r'$R_1$ (R$_{\odot}$)'


def test_grid_labels():
    assert _grid_label('bosz+tlusty') == 'BOSZ + TLUSTY'
    assert _grid_label('btsettl') == 'BT-Settl'
    assert _grid_label('mygrid') == 'mygrid'
