#!/usr/bin/env python
"""Add BOSZ 2024 and TLUSTY spectra to ARIADNE's spectra cache.

The Zenodo spectra cache (``fetch_spectra_cache``) holds PHOENIX, BT-Settl,
BT-NextGen, BT-Cond, CK04, Kurucz and Coelho spectra only, so the BOSZ
(2800-16000 K) and TLUSTY (15000-55000 K) grids could be fitted but not drawn
or used for synthetic photometry. This tool downloads the raw spectra of every
node of the bundled BOSZ and TLUSTY band grids, checks that they reproduce
those grids, and writes them into the cache as two new groups, ``bosz`` and
``tlusty``. It can also add band columns that the grids lack (``--bands``).

Steps, each re-runnable (downloads resume where they stopped)::

    python tools/extend_spectra_cache.py download            # ~1.9 GB
    python tools/extend_spectra_cache.py verify
    python tools/extend_spectra_cache.py cache
    python tools/extend_spectra_cache.py bands --bands IUE_IUE.1250-1300 ...

Sources (both verified to reproduce the bundled grids to < 0.1 %):

* BOSZ 2024 (Meszaros et al. 2024), MAST HLSP, R = 2000 resampled files,
  [alpha/M] = [C/M] = 0, v_micro = 2 km/s, MARCS where MARCS and ATLAS9 both
  exist (7500-8000 K). Surface flux F = 4 pi H.
  https://archive.stsci.edu/hlsps/bosz/bosz2024/
* TLUSTY OSTAR2002 + BSTAR2006 merged and binned (Lanz & Hubeny 2003, 2007),
  SVO Theory Server collection ``tlusty_mergedbin`` (BSTAR2006 up to
  30000 K, OSTAR2002 above). Surface flux in erg s-1 cm-2 A-1. The metal-free
  (Z = 0) models are skipped: the grid's metallicity axis is log Z/Z_sun.
  https://svo2.cab.inta-csic.es/theory/newov2/

Cache groups use the existing groups' layout (teff, logg, z, wavelength,
flux[node, wave]; float32, gzip): surface flux in erg s-1 cm-2 um-1 on the
PHOENIX/BT log-uniform wavelength grid (d ln lambda = 2.22e-4), extended
blueward from 0.125 to 0.100 um so the far-UV bands are covered. BOSZ is
broadened from R = 2000 to the cache's R = 1500. TLUSTY's SVO spectra are
binned more coarsely than that (d ln lambda = 5.6e-4) and are only resampled.

The original cache is copied to ``<raw-dir>/spectra_cache.zenodo.h5`` before
the first write.
"""
import argparse
import concurrent.futures as cf
import datetime
import gzip
import os
import re
import shutil
import sys
import time
import urllib.request

import numpy as np

BOSZ_ROOT = 'https://archive.stsci.edu/hlsps/bosz/bosz2024'
BOSZ_RES = 2000
SVO_SSAP = 'https://svo2.cab.inta-csic.es/theory/newov2/ssap.php'
TLUSTY_MODEL = 'tlusty_mergedbin'
UA = 'astroARIADNE-grid-builder'
CACHE_R = 1500
BLUE_EDGE_UM = 0.100

GRIDS = {'bosz': 'BOSZ_DF.pkl', 'tlusty': 'TLUSTY_DF.pkl'}


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def _get(url, dest=None, tries=5, timeout=300):
    """GET ``url`` (to ``dest`` atomically, or return bytes), with retries."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if dest is None:
                return data
            tmp = dest + '.part'
            with open(tmp, 'wb') as fh:
                fh.write(data)
            os.replace(tmp, dest)
            return dest
        except Exception as e:                       # noqa: BLE001
            if k == tries - 1:
                raise RuntimeError(f'{url}: {e}') from e
            time.sleep(2 * (k + 1))


def _gridsdir():
    from astroARIADNE.config import gridsdir
    return gridsdir


def _cache_path():
    from astroARIADNE.config import filesdir
    return os.path.join(filesdir, 'spectra_cache.h5')


def load_grid(name):
    import pandas as pd
    return pd.read_pickle(os.path.join(_gridsdir(), GRIDS[name]))


def _logz_label(z):
    """TLUSTY's log Z/Z_sun as stored in TLUSTY_DF.pkl (3 decimals)."""
    return round(float(np.log10(z)), 3)


# --------------------------------------------------------------------------
# BOSZ 2024
# --------------------------------------------------------------------------

def _bosz_mdir(feh):
    return f'm{feh:+.2f}'


def bosz_filename(atm, teff, logg, feh):
    return (f'bosz2024_{atm}_t{int(round(teff))}_g{logg:+.1f}_m{feh:+.2f}'
            f'_a+0.00_c+0.00_v2_r{BOSZ_RES}_resam.txt.gz')


def bosz_listing(raw, feh):
    """Atmosphere prefix of every solar-alpha/carbon, v=2 file at ``feh``."""
    d = os.path.join(raw, 'listings')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'r{BOSZ_RES}_{_bosz_mdir(feh)}.html')
    if not os.path.isfile(path):
        _get(f'{BOSZ_ROOT}/r{BOSZ_RES}/{_bosz_mdir(feh)}/', path)
    html = open(path, encoding='utf-8', errors='replace').read()
    pat = (r'bosz2024_([a-z]+)_t(\d+)_g([-+][\d.]+)_m' + re.escape(f'{feh:+.2f}')
           + rf'_a\+0\.00_c\+0\.00_v2_r{BOSZ_RES}_resam\.txt\.gz')
    atms = {}
    for atm, t, g in set(re.findall(pat, html)):
        atms.setdefault((int(t), round(float(g), 1)), set()).add(atm)
    return atms


def bosz_nodes(raw):
    """[(logg, teff, feh, filename)] for every node of BOSZ_DF.pkl."""
    df = load_grid('bosz')
    out, missing = [], []
    listings = {}
    for logg, teff, feh in df.index:
        feh = round(float(feh), 2)
        if feh not in listings:
            listings[feh] = bosz_listing(raw, feh)
        atms = listings[feh].get((int(round(teff)), round(float(logg), 1)))
        if not atms:
            missing.append((logg, teff, feh))
            continue
        # MARCS (mp/ms) where MARCS and ATLAS9 overlap, as the bundled grid.
        atm = sorted(a for a in atms if a != 'ap')[0] if atms - {'ap'} else 'ap'
        out.append((float(logg), float(teff), feh,
                    bosz_filename(atm, teff, logg, feh)))
    if missing:
        raise RuntimeError(f'{len(missing)} BOSZ grid nodes not in the archive,'
                           f' e.g. {missing[:5]}')
    return out


def bosz_wave(raw):
    path = os.path.join(raw, f'bosz2024_wave_r{BOSZ_RES}.txt')
    if not os.path.isfile(path):
        _get(f'{BOSZ_ROOT}/wavelength_grids/bosz2024_wave_r{BOSZ_RES}.txt',
             path)
    return np.loadtxt(path) * 1e-4                     # A -> um


def bosz_read(raw, fname):
    """Surface flux (erg s-1 cm-2 um-1) on the BOSZ R=2000 grid: F = 4 pi H."""
    import pandas as pd
    with gzip.open(os.path.join(raw, 'bosz', fname), 'rt') as fh:
        h = pd.read_csv(fh, sep=r'\s+', header=None, usecols=[0],
                        dtype=np.float64).values[:, 0]
    return 4 * np.pi * h * 1e4


# --------------------------------------------------------------------------
# TLUSTY (SVO tlusty_mergedbin)
# --------------------------------------------------------------------------

def tlusty_listing(raw):
    """{(logg, teff, logz): fid} for the SVO merged/binned TLUSTY models."""
    d = os.path.join(raw, 'listings')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'{TLUSTY_MODEL}_ssap.xml')
    if not os.path.isfile(path):
        _get(f'{SVO_SSAP}?model={TLUSTY_MODEL}&REQUEST=queryData', path)
    xml = open(path, encoding='utf-8').read()
    rows = re.findall(r'<TR>\s*<TD>([\d.]+)</TD><TD>([\d.]+)</TD>'
                      r'<TD>([\d.eE+-]+)</TD>.*?fid=(\d+)', xml, re.S)
    out = {}
    for t, g, z, fid in rows:
        if float(z) <= 0:                              # metal-free: skipped
            continue
        out[(round(float(g), 2), float(t), _logz_label(float(z)))] = int(fid)
    return out


def tlusty_filename(teff, logg, logz):
    return f'tlusty_t{int(round(teff))}_g{logg:.2f}_z{logz:+.3f}.txt'


def tlusty_nodes(raw):
    """[(logg, teff, logz, filename, fid)] for every node of TLUSTY_DF.pkl."""
    df = load_grid('tlusty')
    lst = tlusty_listing(raw)
    out, missing = [], []
    for logg, teff, z in df.index:
        key = (round(float(logg), 2), float(teff), round(float(z), 3))
        if key not in lst:
            missing.append(key)
            continue
        out.append((float(logg), float(teff), float(z),
                    tlusty_filename(teff, logg, z), lst[key]))
    if missing:
        raise RuntimeError(f'{len(missing)} TLUSTY grid nodes not at SVO, '
                           f'e.g. {missing[:5]}')
    return out


def tlusty_read(raw, fname):
    """(wave um, surface flux erg s-1 cm-2 um-1) of one SVO TLUSTY model."""
    import pandas as pd
    d = pd.read_csv(os.path.join(raw, 'tlusty', fname), sep=r'\s+',
                    comment='#', header=None, dtype=np.float64).values
    o = np.argsort(d[:, 0])
    return d[o, 0] * 1e-4, d[o, 1] * 1e4


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

def _download_all(jobs, workers, label):
    todo = [(u, p) for u, p in jobs if not os.path.isfile(p)]
    print(f'{label}: {len(jobs)} files, {len(jobs) - len(todo)} already here, '
          f'{len(todo)} to download', flush=True)
    done = 0
    failed = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_get, u, p): (u, p) for u, p in todo}
        for f in cf.as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:                     # noqa: BLE001
                failed.append((futs[f][0], str(e)))
            if done % 250 == 0 or done == len(todo):
                print(f'  {label}: {done}/{len(todo)}', flush=True)
    if failed:
        print(f'{label}: {len(failed)} downloads failed, e.g. {failed[:3]}')
        print('Re-run the step to retry them.')
    return not failed


def step_download(raw, which, workers):
    ok = True
    if 'bosz' in which:
        os.makedirs(os.path.join(raw, 'bosz'), exist_ok=True)
        bosz_wave(raw)
        jobs = [(f'{BOSZ_ROOT}/r{BOSZ_RES}/{_bosz_mdir(feh)}/{fn}',
                 os.path.join(raw, 'bosz', fn))
                for _, _, feh, fn in bosz_nodes(raw)]
        ok &= _download_all(jobs, workers, 'BOSZ')
    if 'tlusty' in which:
        os.makedirs(os.path.join(raw, 'tlusty'), exist_ok=True)
        jobs = [(f'{SVO_SSAP}?model={TLUSTY_MODEL}&fid={fid}&format=ascii',
                 os.path.join(raw, 'tlusty', fn))
                for _, _, _, fn, fid in tlusty_nodes(raw)]
        ok &= _download_all(jobs, min(workers, 4), 'TLUSTY')
    return ok


def iter_spectra(raw, name):
    """Yield ((logg, teff, z), wave_um, flux) for every node of grid ``name``,
    in BOSZ_DF/TLUSTY_DF index order."""
    if name == 'bosz':
        w = bosz_wave(raw)
        for logg, teff, feh, fn in bosz_nodes(raw):
            yield (logg, teff, feh), w, bosz_read(raw, fn)
    else:
        for logg, teff, z, fn, _ in tlusty_nodes(raw):
            w, f = tlusty_read(raw, fn)
            yield (logg, teff, z), w, f


def band_fluxes(raw, name, bands):
    """(index, flux[node, band]) of grid ``name``'s raw spectra in ``bands``,
    with pyphot's integrals (synth_phot.coeff_matrix); NaN where a spectrum
    does not span a band's central 98 %."""
    from astroARIADNE import synth_phot as sp
    filts = sp.resolve_filters(bands)
    ops, ops_w = None, None
    idx, out = [], []
    for key, w, f in iter_spectra(raw, name):
        if ops_w is None or len(w) != len(ops_w) or not np.array_equal(w, ops_w):
            ops, ops_w = sp.coeff_matrix(filts, w), w
        idx.append(key)
        out.append(sp.convolve(f[None, :], *ops)[0])
    return idx, np.array(out)


def step_verify(raw, which, tol, bands=None):
    """Convolve every node through every band the grid holds and compare.

    Bands whose flux at a node is below 1e-5 of that node's brightest band
    (the far UV of the coolest models) are not judged: there the band flux is
    numerically negligible and differs between BOSZ products by several %.
    Expected residuals for BOSZ: ~1e-5 in ARIADNE's ground/space survey bands
    and up to ~4e-3 in a few HST WFC3/IR bands (F127M, F153M, F105W), whose
    transmission curves differ slightly from those the grid was built with;
    the same offsets appear at R = 2000, 5000, 10000 and 20000.
    """
    ok = True
    for name in which:
        df = load_grid(name)
        bands = list(df.columns) if bands is None else bands
        idx, fl = band_fluxes(raw, name, bands)
        ref = df.loc[idx, bands].values.astype(float)
        good = np.isfinite(ref) & np.isfinite(fl) & (ref > 0)
        good &= ref > 1e-5 * np.nanmax(np.where(good, ref, np.nan), axis=1,
                                       keepdims=True)
        r = np.where(good, fl / np.where(good, ref, 1.0), np.nan)
        dev = np.nanmax(np.abs(r - 1), axis=0)
        worst = np.argsort(np.nan_to_num(dev, nan=-1))[::-1][:5]
        n_bad = int(np.sum(dev > tol))
        print(f'{name}: {len(idx)} nodes x {len(bands)} bands; '
              f'max |model/grid - 1| = {np.nanmax(dev):.2e} '
              f'(median over bands {np.nanmedian(dev):.1e}); '
              f'bands over {tol:g}: {n_bad}; uncovered (NaN) bands: '
              f'{[b for b, d in zip(bands, dev) if not np.isfinite(d)]}')
        print('  worst bands: ' + ', '.join(f'{bands[j]} {dev[j]:.1e}'
                                             for j in worst))
        ok &= n_bad == 0
    return ok


def cache_wavelength(cache):
    """The PHOENIX/BT cache grid, extended blueward to BLUE_EDGE_UM."""
    import h5py
    with h5py.File(cache, 'r') as h:
        w = h['phoenix']['wavelength'][:].astype(np.float64)
        wdtype = h['phoenix']['wavelength'].dtype
    dl = np.log(w[-1] / w[0]) / (len(w) - 1)
    k = int(np.floor(np.log(w[0] / BLUE_EDGE_UM) / dl))
    blue = w[0] * np.exp(-dl * np.arange(k, 0, -1))
    return np.concatenate([blue, w]), wdtype, dl


def _broaden_loglin(f, dl_native, r_from, r_to):
    """Gaussian-broaden a spectrum on a log-uniform grid from R=r_from to
    R=r_to (r_to < r_from); FWHMs add in quadrature."""
    from scipy.ndimage import gaussian_filter1d
    fwhm = np.sqrt(1.0 / r_to ** 2 - 1.0 / r_from ** 2)
    sigma_px = fwhm / (2 * np.sqrt(2 * np.log(2))) / dl_native
    return gaussian_filter1d(f, sigma_px, mode='nearest')


def step_cache(raw, which, cache, force):
    import h5py
    backup = os.path.join(raw, 'spectra_cache.zenodo.h5')
    if not os.path.isfile(backup):
        print(f'Backing up the cache to {backup} ...', flush=True)
        shutil.copy2(cache, backup + '.part')
        os.replace(backup + '.part', backup)
    wc, wdtype, dl_cache = cache_wavelength(cache)
    lw = np.log(wc)
    with h5py.File(cache, 'r') as h:
        present = [n for n in which if n in h]
    if present and not force:
        raise SystemExit(f'groups {present} already in the cache; '
                         f'pass --force to rebuild them')
    for name in which:
        tmp = os.path.join(raw, f'cache_{name}.h5')
        teff, logg, z, rows = [], [], [], []
        for (lg, t, zz), w, f in iter_spectra(raw, name):
            if name == 'bosz':
                dl_native = np.log(w[1] / w[0])
                f = _broaden_loglin(f, dl_native, BOSZ_RES, CACHE_R)
            rows.append(np.interp(lw, np.log(w), f, left=np.nan,
                                  right=np.nan).astype(np.float32))
            teff.append(t), logg.append(lg), z.append(zz)
        flux = np.array(rows)
        with h5py.File(tmp, 'w') as o:
            g = o.create_group(name)
            for k, v in (('teff', teff), ('logg', logg), ('z', z)):
                g.create_dataset(k, data=np.asarray(v, np.float32))
            g.create_dataset('wavelength', data=wc.astype(wdtype))
            g.create_dataset('flux', data=flux, dtype=np.float32,
                             chunks=(1, flux.shape[1]), compression='gzip')
            g.attrs['flux_unit'] = 'erg s-1 cm-2 um-1 (stellar surface)'
            g.attrs['wavelength_unit'] = 'um'
            g.attrs['built'] = datetime.date.today().isoformat()
            g.attrs['builder'] = 'tools/extend_spectra_cache.py'
            if name == 'bosz':
                g.attrs['source'] = ('BOSZ 2024 (Meszaros et al. 2024), MAST '
                                     'HLSP r2000, [a/M]=[C/M]=0, vmicro=2, '
                                     'MARCS preferred; F = 4 pi H')
                g.attrs['resolution'] = f'R = {CACHE_R} (from R = {BOSZ_RES})'
            else:
                g.attrs['source'] = ('TLUSTY OSTAR2002+BSTAR2006 merged/binned '
                                     '(SVO tlusty_mergedbin), Z > 0')
                g.attrs['resolution'] = ('SVO binning, d ln lambda = 5.6e-4 '
                                         '(coarser than R = 1500), resampled')
        print(f'{name}: {flux.shape[0]} spectra x {flux.shape[1]} wavelengths '
              f'({wc[0]:.4f}-{wc[-1]:.4f} um) staged in {tmp}', flush=True)
        with h5py.File(cache, 'a') as h, h5py.File(tmp, 'r') as o:
            if name in h:
                del h[name]
            o.copy(o[name], h, name=name)
        os.remove(tmp)
        print(f'{name}: written to {cache}', flush=True)
    return True


def _pickle_compat(df, path):
    """Pickle ``df`` so numpy 1.x / pandas 2.x can read it as well as numpy 2
    (the bundled grids were written that way): object-dtype column labels,
    protocol 2, and the ``numpy.core`` module path, which numpy 2 still
    resolves when unpickling while numpy 1.x has no ``numpy._core``."""
    import pickle
    import pandas as pd
    df = df.copy()
    df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
    raw = pickle.dumps(df, protocol=2)
    # Protocol 2 writes globals as text: b'c' + module + b'\n' + name + b'\n'.
    raw = raw.replace(b'cnumpy._core.', b'cnumpy.core.')
    back = pickle.loads(raw)
    if not (back.index.equals(df.index) and list(back.columns) == list(df.columns)
            and np.array_equal(back.values, df.values, equal_nan=True)):
        raise RuntimeError('pickle round trip changed the grid')
    tmp = path + '.part'
    with open(tmp, 'wb') as fh:
        fh.write(raw)
    os.replace(tmp, path)


def step_bands(raw, which, bands, force):
    """Add ``bands`` as columns of the grids' DataFrames, convolved from the
    raw spectra (only nodes the spectrum covers; NaN otherwise)."""
    import pandas as pd
    from astroARIADNE.config import filter_names
    for name in which:
        df = load_grid(name)
        have = [b for b in bands if b in df.columns]
        if have and not force:
            raise SystemExit(f'{name}: {have} already present; pass --force')
        idx, fl = band_fluxes(raw, name, bands)
        new = pd.DataFrame(fl, index=pd.MultiIndex.from_tuples(
            idx, names=df.index.names), columns=bands)
        new = new.reindex(df.index)
        df = df.drop(columns=have)
        for b in bands:
            df[b] = new[b].values
        order = [c for c in filter_names if c in df.columns]
        order += [c for c in df.columns if c not in order]
        df = df[order]
        n_nan = {b: int(df[b].isna().sum()) for b in bands}
        path = os.path.join(_gridsdir(), GRIDS[name])
        _pickle_compat(df, path)
        print(f'{name}: added {len(bands)} band columns to {path} '
              f'(NaN nodes per band: {n_nan})', flush=True)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('step', choices=['download', 'verify', 'cache', 'bands'])
    p.add_argument('--raw-dir', default=os.path.expanduser('~/ARIADNE_models'),
                   help='where raw spectra and listings are kept '
                        '(default ~/ARIADNE_models)')
    p.add_argument('--grids', nargs='+', default=['bosz', 'tlusty'],
                   choices=['bosz', 'tlusty'])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tol', type=float, default=1e-2,
                   help='verify: max allowed |model/grid - 1| per band')
    p.add_argument('--cache', default=None,
                   help='spectra cache to extend (default: the package one)')
    p.add_argument('--bands', nargs='+', default=None,
                   help='bands step: ARIADNE filter names to add')
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    raw = os.path.abspath(a.raw_dir)
    os.makedirs(raw, exist_ok=True)
    if a.step == 'download':
        ok = step_download(raw, a.grids, a.workers)
    elif a.step == 'verify':
        ok = step_verify(raw, a.grids, a.tol)
    elif a.step == 'cache':
        ok = step_cache(raw, a.grids, a.cache or _cache_path(), a.force)
    else:
        if not a.bands:
            p.error('bands: give --bands')
        ok = step_bands(raw, a.grids, a.bands, a.force)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
