"""Extinction integrated over each band.

The band grids hold band fluxes of unreddened model spectra, ``<F>``. ARIADNE
used to redden a band flux with the extinction at the band's central
wavelength, ``<F> 10**(-0.4 A_V a(lambda_c))`` with ``a = A_lambda / A_V``.
Across a wide band ``a`` changes (in Gaia G from ~1.6 at 0.4 um to ~0.5 at
1 um), and the band flux actually observed is ``<F 10**(-0.4 A_V a)>``: at
A_V ~ 0.4 the two differ by several per cent in G, BP, RP and TESS, and by
tenths of a magnitude in GALEX NUV, whose central wavelength sits on the
2175 A bump.

For every node of a model grid and every band this module tabulates the
difference in magnitudes

    dA(A_V) = -2.5 log10(<F 10**(-0.4 A_V a)> / <F>) - A_V a(lambda_c)

and its derivative ``d dA / d A_V`` at the extinctions ``AV_NODES``, from the
node spectra of the spectra cache and with the synthetic-photometry integrals
of the band grids (``synth_phot``). The fit interpolates it like the band
fluxes (trilinearly in log g, Teff and [Fe/H]) and in A_V with cubic Hermite
polynomials (error < 0.05 mmag for A_V < 3, < 0.3 mmag up to A_V = 30;
linear beyond), and multiplies the model flux by ``10**(-0.4 dA)``.

Bands the cached spectra do not span (e.g. WISE W2 and redder, where
extinction is tiny), grids without cached spectra (SPHINX) and installations
without the spectra cache keep the central-wavelength extinction. Tables are
saved under :func:`cache_dir` and rebuilt when the spectra cache, the grid,
the filter or ``AV_NODES`` change.
"""
import hashlib
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

AV_NODES = np.array([0., .5, 1., 2., 3., 4., 6., 8., 10., 13., 16., 20., 25.,
                     30.])
RV = 3.1
_VERSION = 1
_CHUNK = 512            # cache spectra read at a time
_WARNED = set()


def cache_dir():
    """Directory of the saved tables: ``$ARIADNE_BAND_EXTINCTION_DIR``, or
    ``astroARIADNE/band_extinction`` in ``$XDG_CACHE_HOME`` (``~/.cache``)."""
    d = os.environ.get('ARIADNE_BAND_EXTINCTION_DIR')
    if d:
        return d
    base = (os.environ.get('XDG_CACHE_HOME')
            or os.path.join(os.path.expanduser('~'), '.cache'))
    return os.path.join(base, 'astroARIADNE', 'band_extinction')


def law_key(av_law):
    """Name of an extinction law function (e.g. ``'fitzpatrick99'``)."""
    return getattr(av_law, '__name__', None) or str(av_law)


def _spectra_cache():
    from . import config
    return config.spectra_cache


def available(grid, cache_path=None):
    """True if the spectra cache holds node spectra of ``grid``."""
    path = cache_path or _spectra_cache()
    if not path or not os.path.isfile(path):
        return False
    import h5py
    try:
        with h5py.File(path, 'r') as h:
            return grid in h
    except OSError:
        return False


def _warn_once(key, msg, level=logging.WARNING):
    if key not in _WARNED:
        _WARNED.add(key)
        logger.log(level, msg)


def _match(axis, values, tol):
    """Index of the nearest ``axis`` node to each value, and whether it is
    within ``tol``."""
    axis = np.asarray(axis, float)
    values = np.asarray(values, float)
    i = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
    i = np.where(np.abs(axis[i - 1] - values) <= np.abs(axis[i] - values),
                 i - 1, i)
    return i, np.abs(axis[i] - values) <= tol


def node_tables(grid, filts, av_law, cache_path):
    """``dA`` and ``d dA / d A_V`` of every cached node spectrum of ``grid``.

    Returns ``(teff, logg, z, table, ok)``: ``table[node, band, i_av, 0]`` is
    dA at ``AV_NODES[i_av]`` (mag), ``[..., 1]`` its derivative; ``ok[band]``
    is False where the spectra do not span the band or the law is undefined
    over it (its table is then zero: the central-wavelength extinction).
    """
    import h5py
    from . import synth_phot as sp
    nav, nb = len(AV_NODES), len(filts)
    with h5py.File(cache_path, 'r') as h:
        g = h[grid]
        w = np.asarray(g['wavelength'][:], float)
        teff = np.asarray(g['teff'][:], float)
        logg = np.asarray(g['logg'][:], float)
        z = np.asarray(g['z'][:], float)
        coeff, _, covered, _ = sp.coeff_matrix(filts, w)
        ok = np.zeros(nb, bool)
        ops = [None] * nb
        for b, f in enumerate(filts):
            if not covered[b]:
                continue
            idx = np.nonzero(coeff[:, b])[0]
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            try:
                a = np.asarray(av_law(w[lo:hi] * 1e4, 1.0, RV), float)
                acl = float(np.asarray(
                    av_law(np.array([f.wave_um * 1e4]), 1.0, RV))[0])
            except Exception as e:   # law undefined over (part of) the band
                _warn_once((law_key(av_law), f.name),
                           f'{law_key(av_law)} is not defined over band '
                           f'{f.name} ({e}); it keeps the extinction at its '
                           f'central wavelength.')
                continue
            wgt = coeff[lo:hi, b][:, None] * 10 ** (-0.4 * np.outer(a, AV_NODES))
            ops[b] = (lo, hi, wgt, wgt * a[:, None], acl)
            ok[b] = True
        table = np.zeros((len(teff), nb, nav, 2), np.float32)
        for s in range(0, len(teff), _CHUNK):
            e = min(s + _CHUNK, len(teff))
            F = np.asarray(g['flux'][s:e], float)
            for b in np.nonzero(ok)[0]:
                lo, hi, wgt, wgt_a, acl = ops[b]
                Fb = F[:, lo:hi]
                bad = ~np.isfinite(Fb).all(axis=1)
                Fb = np.where(np.isfinite(Fb), Fb, 0.0)
                S = Fb @ wgt
                S1 = Fb @ wgt_a
                good = ~bad & np.all(S > 0, axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    val = -2.5 * np.log10(S / S[:, :1]) - AV_NODES * acl
                    der = S1 / S - acl
                table[s:e, b, :, 0] = np.where(good[:, None], val, 0.0)
                table[s:e, b, :, 1] = np.where(good[:, None], der, 0.0)
    return teff, logg, z, table, ok


def grid_tables(interp, grid, filts, av_law, cache_path):
    """``node_tables`` placed on the axes of band-grid interpolator
    ``interp``: ``(cube[n_logg, n_teff, n_z, band, i_av, 2], ok[band])``.

    Grid nodes with model fluxes but no cached spectrum take the table of the
    nearest cached node; nodes without model fluxes get zeros.
    """
    teff, logg, z, table, ok = node_tables(grid, filts, av_law, cache_path)
    ax_l, ax_t, ax_z = (np.asarray(a, float) for a in interp.index_columns)
    shape = (len(ax_l), len(ax_t), len(ax_z))
    il, okl = _match(ax_l, logg, 1e-3)
    it, okt = _match(ax_t, teff, 0.5)
    iz, okz = _match(ax_z, z, 2e-3)
    m = np.nonzero(okl & okt & okz)[0]
    flat = np.ravel_multi_index((il[m], it[m], iz[m]), shape)
    uniq, first = np.unique(flat, return_index=True)
    src = m[first]
    nb, nav = table.shape[1], table.shape[2]
    cube = np.zeros((int(np.prod(shape)), nb, nav, 2), np.float32)
    cube[uniq] = table[src]
    have = np.zeros(int(np.prod(shape)), bool)
    have[uniq] = True
    need = np.nonzero(np.isfinite(interp.grid).any(axis=3).ravel() & ~have)[0]
    if need.size and src.size:
        from scipy.spatial import cKDTree

        def coords(t, g_, z_):
            return np.column_stack((np.log10(t) / 0.02, g_ / 0.5, z_ / 0.5))
        nl, nt, nz = np.unravel_index(need, shape)
        tree = cKDTree(coords(teff[src], logg[src], z[src]))
        _, j = tree.query(coords(ax_t[nt], ax_l[nl], ax_z[nz]))
        cube[need] = table[src[j]]
        logger.info('%s: %d grid nodes without a cached spectrum take the '
                    'band-extinction table of the nearest cached node',
                    grid, need.size)
    return cube.reshape(shape + (nb, nav, 2)), ok


def _key(interp, grid, f, av_law, cache_path):
    h = hashlib.sha1()
    for a in interp.index_columns:
        h.update(np.ascontiguousarray(a, float).tobytes())
    st = os.stat(cache_path)
    return json.dumps({
        'version': _VERSION, 'grid': grid, 'axes': h.hexdigest(),
        'law': law_key(av_law), 'rv': RV, 'av': AV_NODES.tolist(),
        'filter': [str(x) for x in f.key],
        'cache': [os.path.realpath(cache_path), st.st_size, st.st_mtime_ns],
    }, sort_keys=True)


def _path(grid, f, av_law):
    name = ''.join(c if c.isalnum() or c in '._-' else '_' for c in f.name)
    return os.path.join(cache_dir(), f'{grid}__{law_key(av_law)}__{name}.npz')


def _load(path, key):
    try:
        with np.load(path, allow_pickle=False) as d:
            if str(d['key']) != key:
                return None
            return d['cube'], bool(d['ok'])
    except (OSError, KeyError, ValueError):
        return None


def _save(path, key, cube, ok):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        np.savez(tmp, key=np.array(key), cube=cube, ok=np.array(ok))
        os.replace(tmp, path)
    except OSError as e:
        _warn_once(('save', os.path.dirname(path)),
                   f'Cannot save band-extinction tables in '
                   f'{os.path.dirname(path)} ({e}); they are rebuilt in '
                   f'every session.')


def interpolator_table(interp, filts, av_law):
    """Band-extinction table of band-grid interpolator ``interp`` for bands
    ``filts``: ``(ii0, ii1, ii2, av_nodes, cube)`` with
    ``cube[n_logg, n_teff, n_z, band, i_av, 2]`` (float32), or None when no
    band can be tabulated (central-wavelength extinction throughout).

    ``interp`` is named by ``sed_library.load_band_grid`` (``grid_name``).
    Tables are loaded from :func:`cache_dir` or built from the spectra cache,
    and kept on ``interp`` per (law, bands).
    """
    from . import synth_phot as sp
    grid = getattr(interp, 'grid_name', None)
    names = tuple(str(f) for f in filts)
    store = interp.__dict__.setdefault('_band_extinction_tables', {})
    skey = (law_key(av_law), names)
    if skey in store:
        return store[skey]
    cache_path = _spectra_cache()
    result = None
    if grid is None:
        pass
    elif not available(grid, cache_path):
        _warn_once(('grid', grid),
                   f"No spectra of grid '{grid}' in the spectra cache"
                   + ('' if cache_path else ' (not installed)')
                   + ': its bands keep the extinction at their central '
                   'wavelength.')
    else:
        result = _build(interp, grid, names, av_law, cache_path, sp)
    store[skey] = result
    return result


def _build(interp, grid, names, av_law, cache_path, sp):
    shape = tuple(len(a) for a in interp.index_columns)
    nav = len(AV_NODES)
    cubes, oks, todo = [None] * len(names), [False] * len(names), []
    filts = []
    for j, n in enumerate(names):
        try:
            f = sp.resolve_filter(n)
        except (KeyError, TypeError, ValueError):
            f = None
        filts.append(f)
        if f is None:
            continue
        loaded = _load(_path(grid, f, av_law),
                       _key(interp, grid, f, av_law, cache_path))
        if loaded is not None and (not loaded[1]
                                   or loaded[0].shape == shape + (nav, 2)):
            cubes[j], oks[j] = loaded
        else:
            todo.append(j)
    if todo:
        logger.info('Building band-extinction tables of %s for %d band(s)',
                    grid, len(todo))
        cube, ok = grid_tables(interp, grid, [filts[j] for j in todo],
                               av_law, cache_path)
        for k, j in enumerate(todo):
            oks[j] = bool(ok[k])
            # an untabulated band is saved as just that (no zero cube)
            cubes[j] = (np.ascontiguousarray(cube[..., k, :, :]) if oks[j]
                        else np.zeros(0, np.float32))
            _save(_path(grid, filts[j], av_law),
                  _key(interp, grid, filts[j], av_law, cache_path),
                  cubes[j], oks[j])
    for j, n in enumerate(names):
        if not oks[j]:
            wl = filts[j].wave_um if filts[j] is not None else np.inf
            _warn_once(('band', grid, n),
                       f"Band {n}: the cached spectra of '{grid}' do not span "
                       f"it; it keeps the extinction at its central "
                       f"wavelength.",
                       logging.WARNING if wl < 2.5 else logging.INFO)
    if not any(oks):
        return None
    full = np.zeros(shape + (len(names), nav, 2), np.float32)
    for j in range(len(names)):
        if oks[j]:
            full[..., j, :, :] = cubes[j]
    ii = tuple(np.ascontiguousarray(a, dtype=np.float64)
               for a in interp.index_columns)
    return ii + (np.ascontiguousarray(AV_NODES, dtype=np.float64),
                 np.ascontiguousarray(full))
