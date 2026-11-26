"""Plot styling: STIX fonts, per-filter colours and legends, value titles.

Every SEDPlotter plot is drawn under the fonts of ``plot_settings.dat``
(STIXGeneral text and STIX mathtext by default) without touching the caller's
rcParams. SED points are coloured by band, and a legend beside the plot names
the bands, bluest first.
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

from astroARIADNE.config import filesdir, filter_labels, filter_names
from astroARIADNE.plotter import SEDPlotter, _grid_label, filter_label
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
                                'filter_cmap', 'corner_title_fontsize',
                                'model_color_', 'model_component_'))]
    cfg = tmp_path / 'old_settings.dat'
    cfg.write_text(''.join(old))
    p = _plotter(tmp_path, settings=str(cfg))
    assert (p.mathtext_fontset, p.legend_fontsize, p.filter_cmap,
            p.corner_title_fontsize) == ('stix', 16, 'turbo', 18)
    assert (p.model_color, p.model_color_primary,
            p.model_color_secondary) == ('black', '#1f4e79', '#c1432e')
    assert p.model_component_alpha == 0.8


def test_binary_component_colours_come_from_the_settings(tmp_path):
    p = _plotter(tmp_path)
    assert (p.model_color, p.model_color_primary,
            p.model_color_secondary) == ('black', '#1f4e79', '#c1432e')
    assert p.model_component_alpha == 0.8
    cfg = tmp_path / 'mine.dat'
    cfg.write_text(open(filesdir + '/plot_settings.dat').read().replace(
        'model_color_primary #1f4e79', 'model_color_primary tab:blue').replace(
        'model_component_alpha 0.8', 'model_component_alpha 0.5'))
    mine = _plotter(tmp_path, settings=str(cfg))
    assert mine.model_color_primary == 'tab:blue'
    assert mine.model_component_alpha == 0.5


def test_photometry_plot_has_filter_legend_and_stix_text(tmp_path):
    before = dict(plt.rcParams)
    p = _plotter(tmp_path)
    p.plot_SED_no_model(s=_star())
    ax = plt.gcf().axes[0]
    legends = _legend_texts(ax)
    names = p.star.filter_names[p.star.filter_mask]
    # bands in survey notation, bluest first
    assert legends['Filter'] == [filter_labels[str(n)]
                                 for n in names[np.argsort(p.wave)]]
    assert legends['Filter'][:2] == [r'Gaia $G_{\rm BP}$', r'Gaia $G$']
    assert legends['Filter'][-1] == r'2MASS $K_{\rm s}$'
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


def test_every_filter_has_a_plot_label():
    assert set(filter_labels) == {str(f) for f in filter_names}
    for f, lab in filter_labels.items():
        # no stray underscores outside mathtext (subscripts live inside $..$)
        outside = re.sub(r'\$[^$]*\$', '', lab)
        assert '_' not in outside and '$' not in outside, f
        assert lab.count('$') % 2 == 0, f


def test_plot_labels_follow_the_surveys_notation():
    lab = filter_labels
    assert lab['GaiaDR2v2_RP'] == r'Gaia $G_{\rm RP}$'
    assert lab['TYCHO_B_MvB'] == r'Tycho $B_T$'
    assert lab['PS1_g'] == r'PS1 $g_{\rm P1}$'
    assert lab['GROUND_COUSINS_I'] == r'Cousins $I_{\rm C}$'
    assert lab['WISE_RSR_W1'] == 'WISE W1'
    assert lab['SPITZER_IRAC_45'] == 'IRAC [4.5]'
    assert lab['IUE_IUE.1250-1300'] == r'IUE 1250–1300 $\mathrm{\AA}$'
    assert lab['TD1_WIDE.1565'] == r'TD-1 1565 $\mathrm{\AA}$'
    assert lab['HST_WFC3_F160W'] == 'WFC3/IR F160W'
    assert lab['HST_WFC3_F275W'] == 'WFC3/UVIS F275W'
    assert lab['HST_ACS_WFC_F814W'] == 'ACS/WFC F814W'


def test_plot_labels_render_in_the_plot_fonts(tmp_path):
    p = _plotter(tmp_path)
    with plt.rc_context(p._rc_params()):
        fig = plt.figure()
        for i, lab in enumerate(filter_labels.values()):
            fig.text(0, i / len(filter_labels), lab)
        fig.canvas.draw()          # raises on bad mathtext
    plt.close(fig)


def test_unknown_band_is_labelled_without_underscores():
    assert filter_label('Generic_Bessell.V') == 'Generic Bessell.V'
    assert filter_label(np.str_('2MASS_Ks')) == r'2MASS $K_{\rm s}$'
