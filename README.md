# ARIADNE (spectrAl eneRgy dIstribution bAyesian moDel averagiNg fittEr)

[![Documentation Status](https://readthedocs.org/projects/astroariadne/badge/?version=latest)](https://astroariadne.readthedocs.io/en/latest/?badge=latest)

## Characterize stellar atmospheres easily!
**ARIADNE** Is a code written in Python 3.11+ designed to fit broadband
photometry to different stellar atmosphere models automatically using Nested
Sampling algorithms.

📖 **Documentation:** <https://astroariadne.readthedocs.io>


# Installation

<!-- DOC:INSTALLATION-START -->

**ARIADNE** requires Python 3.11 or higher and uses modern Python packaging standards (PEP 517/518).

## Quick Install (Recommended)

Install the latest stable version from PyPI:

```bash
pip install astroariadne
```

## Development Install

To install from source for development or to use the latest unreleased features:

```bash
git clone https://github.com/jvines/astroARIADNE.git
cd astroARIADNE
pip install -e .
```

The `-e` flag installs in editable mode, allowing you to modify the source code.

## Dependencies

All required dependencies are automatically installed via pip. The main packages include:

- Numpy (<https://numpy.org/>)
- Scipy (<https://www.scipy.org/>)
- Pandas (<https://pandas.pydata.org/>)
- numba (<http://numba.pydata.org/>)
- astropy (<https://astropy.readthedocs.io/en/stable/>)
- astroquery (<https://astroquery.readthedocs.io/en/latest/>)
- regions (<https://astropy-regions.readthedocs.io/en/latest/index.html>)
- PyAstronomy (<https://pyastronomy.readthedocs.io/en/latest/>)
- corner (<https://corner.readthedocs.io/en/latest/>)
- tqdm (<https://tqdm.github.io/>)
- matplotlib (<https://matplotlib.org/>)
- termcolor (<https://pypi.org/project/termcolor/>)
- extinction (<https://extinction.readthedocs.io/en/latest/>)
- pyphot (<http://mfouesneau.github.io/docs/pyphot/>)
- dustmaps (<https://dustmaps.readthedocs.io/en/latest/>) [**NEEDS CONFIGURING AND DOWNLOADING OF DUSTMAPS**]
- dynesty (<https://dynesty.readthedocs.io/en/latest/>)
- isochrones (<https://isochrones.readthedocs.io/en/latest/>), installed from the
  [piotrlojko/isochrones](https://github.com/piotrlojko/isochrones) fork:
  isochrones 2.1 from PyPI does not work with numpy 2, pandas 3 or current
  Python [**DOWNLOADS THE MIST MODELS ON FIRST USE, SEE BELOW**]

### Special Installations

#### dustmaps Configuration

After installing ARIADNE, you must download and configure dustmaps:

```python
# Download SFD dust map
import dustmaps.sfd
dustmaps.sfd.fetch()

# Download Bayestar dust map
import dustmaps.bayestar
dustmaps.bayestar.fetch()
```

See the [dustmaps documentation](https://dustmaps.readthedocs.io/en/latest/) for other available dust maps.

#### isochrones Setup

isochrones downloads the MIST isochrone grid and bolometric-correction tables
(~1.5 GB, into `~/.isochrones`, or `$ISOCHRONES` if set) the first time they
are used. To do it up front for the bands ARIADNE uses:

```bash
python -c "from isochrones import get_ichrone; ic = get_ichrone('mist', bands=['U', 'B', 'BP', 'V', 'G', 'RP', 'TESS', 'J', 'H', 'K', 'W1', 'W2']); ic.model_grid.interp; ic.bc_grid.interp"
```

The dependency is a direct reference to the fork
(`isochrones @ git+https://github.com/piotrlojko/isochrones.git@master`), so
pip installs isochrones from it automatically. If you also work on isochrones
from a local editable clone, reinstall ARIADNE with `pip install -e . --no-deps`:
without `--no-deps` pip replaces the editable isochrones with a checkout of the
fork.

## Platform Support

**ARIADNE** has been tested on:
- **Linux** (fully supported)
- **macOS** (fully supported, tested up to Catalina and later)
- **Windows** (limited support - healpy dependency may cause issues, see [healpy/healpy#25](https://github.com/healpy/healpy/issues/25))

## Build System

ARIADNE uses `pyproject.toml` for modern, declarative package configuration. Dependency installation and version management are handled automatically by pip/setuptools.

## Model Spectra for SED Plotting

Plotting the SED model is optional — you can run the fitting code without any
model spectra. If you want SED plots, there are two options:

### Spectra Cache (Recommended)

Download the pre-computed spectra cache (~2.6 GB) from Zenodo. This contains
all 7 model grids (Phoenix v2, BT-Settl, BT-NextGen, BT-Cond, Castelli &
Kurucz, Kurucz, Coelho) broadened to R=1500 and resampled to a common
wavelength grid. This is all you need for `plot_SED()`.

```python
from astroARIADNE.fetch import fetch_spectra_cache
fetch_spectra_cache()
```

Or from the command line:

```bash
python -c "from astroARIADNE.fetch import fetch_spectra_cache; fetch_spectra_cache()"
```

No environment variables or additional setup required.

#### BOSZ and TLUSTY spectra

The Zenodo cache has no BOSZ or TLUSTY spectra, so fits with those grids are
plotted without the continuous model spectrum. `tools/extend_spectra_cache.py`
adds them as two more cache groups, `bosz` and `tlusty`: it downloads the raw
spectra of every node of the bundled BOSZ and TLUSTY grids (BOSZ 2024 from
MAST, ~1.1 GB; TLUSTY OSTAR2002+BSTAR2006 from the SVO Theory Server,
~0.7 GB), checks that they reproduce the bundled grids, and writes them on the
cache's wavelength grid, extended down to 0.10 um so the far-UV is covered:

```bash
python tools/extend_spectra_cache.py download   # into ~/ARIADNE_models
python tools/extend_spectra_cache.py verify
python tools/extend_spectra_cache.py cache      # backs the cache up first
```

`fetch_spectra_cache(force=True)` replaces the cache with the Zenodo one, so
rerun the `cache` step after it (the downloads are kept).

### Full Model Grids (Optional)

If you need the raw high-resolution model spectra (e.g. for custom broadening
or direct spectral analysis), you can download the full grids. Note that these
total several hundred GB.

| Model        | Link           |
| ------------- |:-------------:|
| Phoenix v2      | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/> |
| Phoenix v2   wavelength file   | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits> |
| BT-Models      | <http://svo2.cab.inta-csic.es/theory/newov2/>  |
| Castelli & Kurucz | <http://ssb.stsci.edu/cdbs/tarfiles/synphot3.tar.gz>      |
| Kurucz 1993 | <http://ssb.stsci.edu/cdbs/tarfiles/synphot4.tar.gz>  |

The wavelength file for the Phoenix model has to be placed in the root folder
of the PHOENIXv2 models.

To tell **ARIADNE** where to find the models, set the `ARIADNE_MODELS`
environment variable:

```bash
export ARIADNE_MODELS='/path/to/Models_Dir/'
```

You can add this to your `.bash_profile` or `.bashrc` so you don't have to
export every time.

The directory structure should look like this:

```
Models_Dir
│
└───BTCond
│   │
│   └───CIFIST2011
│
└───BTNextGen
│	 │
│	 └───AGSS2009
│
└───BTSettl
│	 │
│	 └───AGSS2009
│
└───Castelli_Kurucz
│	 │
│	 └───ckm05
│	 │
│	 └───ckm10
│	 │
│	 └───ckm15
│	 │
│	 └───ckm20
│	 │
│	 └───ckm25
│	 │
│	 └───ckp00
│	 │
│	 └───ckp02
│	 │
│	 └───ckp05
│
└───Kurucz
│	 │
│	 └───km01
│	 │
│	 └───km02
│	 │
│	 └───km03
│	 │
│	 └───km05
│	 │
│	 └───km10
│	 │
│	 └───km15
│	 │
│	 └───km20
│	 │
│	 └───km25
│	 │
│	 └───kp00
│	 │
│	 └───kp01
│	 │
│	 └───kp02
│	 │
│	 └───kp03
│	 │
│	 └───kp05
│	 │
│	 └───kp10
│
└───PHOENIXv2
	 │
     └─── WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
	 └───Z-0.0
	 │
	 └───Z-0.5
	 │
	 └───Z-1.0
	 │
	 └───Z-1.5
	 │
	 └───Z-2.0
	 │
	 └───Z+0.5
	 │
	 └───Z+1.0
```

#### Notes:
- The Phoenix v2 models with alpha enhancements are unused
- BT-models are BT-Settl, BT-Cond, and BT-NextGen

<!-- DOC:INSTALLATION-END -->

# How to use?

<!-- DOC:USAGE-START -->

## Stellar information setup

To use **ARIADNE** start by setting up the stellar information, this is done by
importing the Star module.

```python
from astroARIADNE.star import Star
```

After importing, a star has to be defined.

Stars are defined in **ARIADNE** by their RA and DEC in degrees, a name, and
optionally the Gaia DR3 source id, for example:

```python
starname = 'WASP-19'
ra = 148.41676021592826
dec = -45.65910531582427
gaia_id = 5411736896952029568

s = Star(starname, ra, dec, g_id=gaia_id)
```
The starname is used purely for user identification later on, and the 
`gaia_id` is provided to make sure the automatic photometry retrieval collects
the correct magnitudes, otherwise **ARIADNE** will try and get the `gaia_id` by
itself using a cone search centered around the RA and DEC.

Executing the previous block will start the photometry and stellar parameter
retrieval routine. **ARIADNE** will query Gaia DR2 for an estimate on the
temperature, radius, parallax and luminosity for display as preliminar
information, as it's not used during the fit, and prints them along with its
TIC, KIC IDs if any of those exist, its Gaia DR3 ID, and maximum line-of-sight
extinction Av:

```
                        Gaia DR2 ID : 5411736896952029568
			Gaia Effective temperature : 5458.333 +/- 109.667
			Gaia Stellar radius : 1.001 +/- 0.195
			Gaia Stellar Luminosity : 0.802 +/- 0.009
			Gaia Parallax : 3.751 +/- 0.024
			Bailer-Jones distance : 265.144 +/- 0.620
			Maximum Av : 0.581
```

If you already know any of those values, you can override the search for them by
providing them in the Star constructor with their respective uncertainties.
Likewise if you already have the magnitudes and wish to override the on-line
search, you can provide a dictionary where the keys are the filters and values
are the mag, mag_err tuples.

If you want to check the retrieved magnitudes you can call the `print_mags`
method from Star:

```python
s.print_mags()
```

This will print the filters used, magnitudes and uncertainties. For WASP-19 this
would look like this:

```
		       Filter       	Magnitude	Uncertainty
		--------------------	---------	-----------
		    SkyMapper_u     	 14.1050 	  0.0070
		    SkyMapper_v     	 13.7370 	  0.0060
		  GROUND_JOHNSON_B  	 16.7920 	  0.1820
		    SkyMapper_g     	 12.4320 	  0.0030
		    GaiaDR2v2_BP    	 12.5227 	  0.0017
		  GROUND_JOHNSON_V  	 16.0100 	  0.0000
		    SkyMapper_r     	 12.0630 	  0.0050
		    GaiaDR2v2_G     	 12.1088 	  0.0005
		    SkyMapper_i     	 11.8740 	  0.0050
		    GaiaDR2v2_RP    	 11.5532 	  0.0014
		    SkyMapper_z     	 11.8610 	  0.0080
		      2MASS_J       	 10.9110 	  0.0260
		      2MASS_H       	 10.6020 	  0.0220
		      2MASS_Ks      	 10.4810 	  0.0230
		    WISE_RSR_W1     	 10.4360 	  0.0230
		    WISE_RSR_W2     	 10.4940 	  0.0200
```
**Note:**  **ARIADNE** automatically prints and saves the used magnitudes and
filters to a file.

The way the photometry retrieval works is that Gaia DR2 crossmatch catalogs are
queried for the Gaia ID, these crossmatch catalogs exist for ALL-WISE, APASS,
Pan-STARRS1, SDSS, 2MASS and Tycho-2, so finding photometry relies on these
crossmatches. For example, if we were to analyze NGTS-6, there are Pan-STARRS1
photometry which **ARIADNE** couldn't find due to the Pan-STARRS1 source not
being identified in the Gaia DR2 crossmatch, in this case if you wanted to add
that photometry manually, you can do so by using the `add_mag` method from
Star, for example, if you wanted to add the PS1_r mag to our `Star` object 
you would do:

```python
s.add_mag(13.751, 0.032, 'PS1_r')
```

If the photometry is published as a flux rather than a magnitude, add it with
`add_flux`, giving the flux and its error as `astropy` quantities in any
spectral flux density unit: f_lambda (e.g. erg s^-1 cm^-2 A^-1), f_nu (Jy,
mJy, W m^-2 Hz^-1), lambda f_lambda = nu f_nu (erg s^-1 cm^-2) or a photon flux
density. The flux and its error need not share a unit, and plain numbers can be
used together with `unit`:

```python
import astropy.units as u

s.add_flux(12.3 * u.mJy, 0.4 * u.mJy, 'WISE_RSR_W2')
s.add_flux(2.1e-13, 1e-14, 'IUE_IUE.1450-1500', unit='erg / (s cm2 AA)')
```

The flux is taken at the band's effective wavelength and stored as the
equivalent magnitude in the band's magnitude system (so `print_mags` and
`mags.dat` list it as a magnitude), and the fit uses exactly the given flux.
To build a `Star` from fluxes, use `flux_dict` like `mag_dict` (the two can be
combined, with each band in only one of them):

```python
s = Star(starname, ra, dec, g_id=gaia_id,
         mag_dict={'2MASS_J': (9.763, 0.028)},
         flux_dict={'2MASS_H': (0.51 * u.Jy, 0.013 * u.Jy),
                    'IUE_IUE.1450-1500': (2.1e-13, 1e-14, 'erg / (s cm2 AA)')})
```

The IUE and TD-1 ultraviolet bands can only be fitted with the BOSZ and TLUSTY
grids (see [filters.md](filters.md#ultraviolet-iue-and-td-1)).

If for whatever reason **ARIADNE** found a bad photometry point, and you needed
to remove it, you can invoke the `remove_mag` method. For example, you wanted
to remove the TESS magnitude due to it being from a blended source, you can just
run

```python
s.remove_mag('NGTS')
```

In the specific example of WASP-19, we see that GROUND_JOHNSON_B and GROUND_JOHNSON_V
are likely not the correct photometry. Instead the correct ones are 13.054 and 12.248,
respectively.
We can correct this mistake:
```python
s.remove_mag('GROUND_JOHNSON_B')
s.remove_mag('GROUND_JOHNSON_V')
s.add_mag(13.054, 0.048, 'GROUND_JOHNSON_B')
s.add_mag(12.248, 0.069, 'GROUND_JOHNSON_V')
```

And a new call to `s.print_mags()` would yield:
```
		       Filter       	Magnitude	Uncertainty
		--------------------	---------	-----------
		    SkyMapper_u     	 14.1050 	  0.0070
		    SkyMapper_v     	 13.7370 	  0.0060
		  GROUND_JOHNSON_B  	 13.0540 	  0.0480
		    SkyMapper_g     	 12.4320 	  0.0030
		    GaiaDR2v2_BP    	 12.5227 	  0.0017
		  GROUND_JOHNSON_V  	 12.2480 	  0.0690
		    SkyMapper_r     	 12.0630 	  0.0050
		    GaiaDR2v2_G     	 12.1088 	  0.0005
		    SkyMapper_i     	 11.8740 	  0.0050
		    GaiaDR2v2_RP    	 11.5532 	  0.0014
		    SkyMapper_z     	 11.8610 	  0.0080
		      2MASS_J       	 10.9110 	  0.0260
		      2MASS_H       	 10.6020 	  0.0220
		      2MASS_Ks      	 10.4810 	  0.0230
		    WISE_RSR_W1     	 10.4360 	  0.0230
		    WISE_RSR_W2     	 10.4940 	  0.0200
```

A list of allowed filters can be found
[here](https://github.com/jvines/astroARIADNE/blob/master/filters.md)

### Interstellar extinction

**ARIADNE** has an incorporated prior for the interstellar extinction in the
Visual band, $A_{\rm V}$ which consists of a uniform prior between 0 and the
maximum line-of-sight value provided by the
[SFD dust maps](https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S/abstract).
This, however, can be changed either by a custom prior (see Fitter setup) or by
changing the dustmap used. We provide following dustmaps:

- [SFD (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S/abstract)
- [Planck Collaboration (2013)](http://adsabs.harvard.edu/abs/2014A%26A...571A..11P)
- [Planck Collaboration (2016; GNILC)](https://ui.adsabs.harvard.edu/abs/2016A%26A...596A.109P/abstract)
- [Lenz, Hensley & Doré (2017)](https://arxiv.org/abs/1706.00011)
- [Bayestar (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G)

**These maps are all implemented through the
[dustmaps](https://dustmaps.readthedocs.io/en/latest/index.html) package and
need to be downloaded. Instructions to download the dustmaps can be found in
its documentation.**

To change the dustmap you need to provide the `dustmap` parameter to the `Star` constructor, for example:

```python
ra = 75.795
dec = -30.399
starname = 'NGTS-6'
gaia_id = 4875693023844840448

s = Star(starname, ra, dec, g_id=gaia_id, dustmap='Bayestar')
```

This concludes the stellar setup and now we're ready to set up the parameters
for the fitting routine.

## Fitter setup

In this section we'll detail how to set up the fitter for the Bayesian Model
Averaging (BMA) mode of **ARIADNE**. For single models the procedure is very
similar.

First, import the fitter from **ARIADNE**

```python
from astroARIADNE.fitter import Fitter
```

There are several configuration parameters we have to setup, the first one is
the output folder where we want **ARIADNE** to output the fitting files and
results, next we have to select the fitting engine (`'dynesty'`), number of
live points to use, evidence tolerance threshold, bounding method, sampling
method, threads, and whether to use the dynamic nested sampler. After selecting
all of those, we need to select the models we want to use and finally, we feed
them all to the fitter:

> **Deprecation:** `'dynesty'` is the engine. The legacy `'multinest'` engine
> is **deprecated** and will be **removed entirely in v2.0** — it only ever
> supported single-grid fits (not BMA), and dynesty is now faster besides. Use
> `'dynesty'`.

```python
out_folder = 'your folder here'

engine = 'dynesty'
nlive = 500
dlogz = 0.5
bound = 'multi'
sample = 'rwalk'
threads = 4
dynamic = False

setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

# Feel free to uncomment any unneeded/unwanted models
models = [
	'phoenix',
	'btsettl',
	'btnextgen',
	'btcond',
	'kurucz',
	'ck04',
	'bosz'      # MARCS+ATLAS9/Synspec, 2800-16000 K (BMA-capable)
]

f = Fitter()
f.star = s
f.setup = setup
f.av_law = 'fitzpatrick'
f.out_folder = out_folder
f.bma = True
f.models = models
f.n_samples = 100000
```

**Note:** While you can always select all 6 models, **ARIADNE** has an internal
filter put in place in order to avoid having the user unintentionally bias the
results. For stars with Teff > 4000 K BT-Settl, BT-NextGen and BT-Cond are
identical and thus only BT-Settl is used, even if the three are selected. On the
other hand, Kurucz and Castelli & Kurucz are known to work poorly on stars with
Teff < 4000 K, thus they aren't used in that regime.

**Additional grids.** Three more grids ship with **ARIADNE**:

| grid | regime | Teff (K) | use |
| --- | --- | --- | --- |
| `bosz` | FGK + warm | 2800–16000 | BMA member (independent MARCS+ATLAS9/Synspec physics) |
| `sphinx` | M dwarfs | 2000–4000 | standalone |
| `tlusty` | hot O/B (NLTE) | 15000–55000 | standalone |

`bosz` can be added to the `models` list like any other (it is
temperature-gated to its validity range). `sphinx` and `tlusty` are meant to be
run **standalone** (see below); `tlusty` uses a wide uniform Teff prior because
the FGK population prior cannot reach the hot-star regime.

`bosz` and `tlusty` are the only grids with the IUE and TD-1 ultraviolet bands
(see [filters.md](filters.md#ultraviolet-iue-and-td-1)). A BMA fit of a star
with photometry in one of them leaves out every grid without it, with a
warning, and a standalone fit with such a grid stops with an error.

### Running a single model (standalone)

To fit a single grid instead of doing BMA, set `f.bma = False` and pick the
grid with `f.grid`, then call `f.fit()`:

```python
f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.1, 'multi', 'rwalk', 4, False]
f.av_law = 'fitzpatrick'
f.out_folder = out_folder
f.bma = False
f.grid = 'phoenix'      # or 'bosz', 'sphinx', 'tlusty', 'coelho', ...
f.prior_setup = {
	'teff': ('default'), 'logg': ('default'), 'z': ('default'),
	'dist': ('default'), 'rad': ('default'), 'Av': ('default'),
}
f.initialize()
f.fit()                 # single-grid fit (not fit_bma)
```

This is the right mode for the standalone grids: `sphinx` (M dwarfs),
`tlusty` (hot stars), and `coelho` (optical-only). The output goes to the same
files as a BMA run minus the model-averaging products.

### Binary stars (two-component fits)

An unresolved binary, or a star with a hot or cool companion, is fitted as the
sum of two stars by setting `f.binary = True`. The two components share the
distance, the extinction and [Fe/H]; each has its own Teff, log g and radius
(`teff2`, `logg2` and `rad2` for the secondary):

```
F_model = [F_1(teff, logg, z) (rad/dist)^2 + F_2(teff2, logg2, z) (rad2/dist)^2] x 10^(-0.4 A(Av))
```

Each component can be modelled with its own grid, which is what a pair of very
different stars needs. A late B star with a hot subdwarf or BLAP companion, for
example, takes BOSZ for the primary and TLUSTY for the secondary:

```python
f = Fitter()
f.star = s
f.setup = ['dynesty', 500, 0.5, 'multi', 'rwalk', 8, False]
f.binary = True
f.bma = True
f.models = ['bosz']           # primary grids
f.models2 = ['tlusty']        # secondary grids (default: each grid with itself)
f.prior_setup = {
    'teff': ('normal', 11500, 1000), 'logg': ('default'), 'z': ('default'),
    'dist': ('default'), 'rad': ('default'), 'Av': ('normal', 0.38, 0.02),
    'teff2': ('uniform', 20000, 40000), 'logg2': ('uniform', 4.0, 4.75),
    'rad2': ('uniform', 0.01, 1.0),
}
f.initialize()
f.fit_bma()
```

- **Grids.** In BMA every primary x secondary grid pair is fitted and averaged
  (named e.g. `bosz+tlusty`); without `models2` each grid is paired with itself.
  A pair lacking a fitted band is left out with a warning, and the temperature
  gating applies to the primary only. For a single-grid fit use `f.grid` and
  `f.grid2` (default: the primary's grid) with `f.fit()`.
- **Priors.** `teff2`, `logg2` and `rad2` take the same prior types as any
  other parameter. Missing ones default to the primary's default Teff and
  radius priors (TLUSTY's 15000-55000 K for a TLUSTY secondary) and a uniform
  log g of 3.5-6. Give the components distinguishable priors (or grids); the
  model is otherwise symmetric under swapping them. A warning is given when
  much of the `teff2` prior lies outside the secondary grid.
- **Flux ratios.** A measured contrast `m2 - m1` (speckle or AO imaging, a
  spectroscopic light ratio) can be added in any band both grids model:
  `f.add_contrast('GaiaDR2v2_G', 2.1, 0.1)`. Two contrasts in bands of
  different colour constrain both stars tightly even without ultraviolet data.
- **Requirements.** The distance/radius parametrisation (`f.norm = False`)
  and the `dynesty` engine.

Outputs gain the secondary's parameters (`teff2`, `logg2`, `rad2`) and derived
`grav_mass2`, `lum2` and `AD2` in `best_fit*.dat`, its spectral type, and
`binary_light_fractions.dat`: the secondary's fraction of the observed flux in
every fitted band (posterior median and 16th-84th percentiles), with the
observed magnitude split into the two components. The posterior samples hold
the light fraction of each band as `frac2_<band>`. The MIST age and mass are the
primary's, fitted to the photometry with the secondary's light removed. The
SED plot shows both components and their sum, drawn from the highest-likelihood
posterior sample (the per-parameter best values of a binary need not form a
good model, because Teff2 and R2 are correlated), and the HR diagram marks the
secondary next to the primary.

**Is the companion needed?** Compare the evidence of a binary fit with that of
a single-star fit of the same photometry (`log Bayesian evidence`, or `lnZ` in
the output pickles): a difference of a few in ln Z is significant. Two caveats
matter when only a few bands carry the companion's light, such as two
ultraviolet points:

- The per-band excess-noise terms can absorb a mismatch, so a single star can
  "explain" an ultraviolet excess by inflating the noise of those bands instead
  of needing a second star. Fix those terms when the uncertainties are
  trusted: `'IUE.1250-1300_noise': ('fixed', 0)` in `prior_setup` (the names
  are listed in `prior.dat` and in the fit summary).
- With one or two bands the secondary's Teff and radius are degenerate (a
  hotter, smaller star gives the same flux); its light fractions remain well
  defined.

**Grid wavelength coverage.** Some grids do not span the full set of filters
**ARIADNE** supports. The Coelho 2014 grid, in particular, only covers the
optical (≲ 0.9 µm); its infrared photometry (2MASS, WISE, Spitzer, Herschel)
was extrapolated when the grid was built and is unreliable, especially for
cool stars. For this reason Coelho is **not** part of the default BMA set. If
you use it as a **standalone** grid (`f.bma = False; f.grid = 'coelho'`),
**ARIADNE** automatically excludes any photometry redder than the grid's
coverage from the fit (a warning lists the dropped bands). Coverage limits are
declared in `grid_wave_coverage` in `config.py`. The exclusion is **not**
applied in BMA mode, where every grid must be fit to the same bands for the
evidences to be comparable — so Coelho should only be used standalone.

**Performance & tuning knobs.** A few optional attributes control runtime:

- `f.walks` (default `25`): number of `rwalk` MCMC steps per proposal. Can
  also be passed as an 8th element of `setup`. Lower is faster but mixes less.
- `f.n_grid_jobs` (default `1`): number of BMA model grids to fit
  concurrently, one core each. Set to `len(models)` to run them all at once.
  **CAUTION:** this spawns that many processes — pick a value your machine's
  cores and memory can handle.
- `f.isochrone_dlogz` (default `0.01`): evidence tolerance for the MIST
  isochrone age/mass fit (the dominant cost of a BMA run, parallelised across
  `threads`). Raising it trades age/mass precision for speed.

We allow the use of four different extinction laws:

- fitzpatrick
- cardelli
- odonnell
- calzetti

The next step is setting up the priors to use:

```python
f.prior_setup = {
	'teff': ('default'),
	'logg': ('default'),
	'z': ('default'),
	'dist': ('default'),
	'rad': ('default'),
	'Av': ('default')
}
```

A quick explanation on the priors:

The default prior for Teff is an empirical prior drawn from the RAVE survey
temperatures distribution, the distance prior is drawn from the
[Bailer-Jones](https://ui.adsabs.harvard.edu/abs/2021AJ....161..147B/abstract)
distance estimate from Gaia EDR3, and the radius has a flat prior ranging from
0.5 to 20 R$_\odot$. The default prior for the metallicity `z` and log g are
also their respective distributions from the RAVE survey, the default prior for
Av is a flat prior that ranges from 0 to the maximum of line-of-sight as per the
SFD map, finally the excess noise parameters all have gaussian priors centered
around their respective uncertainties.

We offer customization on the priors as well, those are listed in the following
table.

| Prior | Hyperparameters |
| :------: | :----------: |
| Fixed | value |
| Normal | mean, std |
| TruncNorm | mean, std, lower\_lim, uppern\_lim |
| Uniform | ini, end |
| RAVE (log g only) | --- |
| Default | --- | 

So if you knew (from a spectroscopic analysis, for example) that the effective
temperature is 5600 +/- 100 and the metallicity is [Fe/H] = 0.09 +/- 0.05 and
you wanted to use them as priors, and the star is nearby (< 70 pc), so you
wanted to fix Av to 0, your prior dictionary should look like this:

```python
f.prior_setup = {
	'teff': ('normal', 5600, 100),
	'logg': ('default'),
	'z': ('normal', 0.09, 0.05),
	'dist': ('default'),
	'rad': ('default'),
	'Av': ('fixed', 0)
}
```
Though leaving everything at default usually works well enough.

```python
f.prior_setup = {
    'teff': ('default'),
    'logg': ('default'),
    'z': ('default'),
    'dist': ('default'),
    'rad': ('default'),
    'Av': ('default')
}

```

After having set up everything we can finally initialize the fitter and start
fitting

```python
f.initialize()
f.fit_bma()
```

Now we wait for our results!

## Visualization

After the fitting has finished, we need to visualize our results. **ARIADNE**
includes a plotter object to do just that! We first star by importing the
plotter:

```python
from astroARIADNE.plotter import SEDPlotter
```

The setup for the plotter is already made for you, but if you really want to
change them, instructions on how to change it can be found
[here](https://github.com/jvines/astroARIADNE/blob/master/customization.md)

If you want SED model plots, make sure you have either the spectra cache
(recommended) or the full model grids installed — see
[Model Spectra for SED Plotting](#model-spectra-for-sed-plotting) above.
The spectra cache is used automatically if present; the full model grids
are used via the `ARIADNE_MODELS` environment variable as a fallback.

Now we only need to specify the results file location and the output folder
for the plots!

```python
in_file = out_folder + 'BMA_out.pkl'
plots_out_folder = 'your plots folder here'
```

Now we instantiate the plotter and call the desired plotting methods!
We offer 5 different plots: 

- A RAW SED plot
- A SED plot with the model and synthetic photometry
- A corner plot
- An HR diagram taken from MIST isochrones
- Histograms showing the parameter distributions for each model.

```python
artist = SEDPlotter(in_file, plots_out_folder)
artist.plot_SED_no_model()
artist.plot_SED()
artist.plot_bma_hist()
artist.plot_bma_HR(10)
artist.plot_corner()
```

The number given to `plot_bma_HR` is the number of extra tracks you want to
plot, drawn randomly from the posterior distribution.

If you're iterating through lots of stars you can call the SEDPlotter `clean`
method to clear opened figures with `artist.clean()`

If you don't have either the spectra cache or the full model grids, then the
`plot_SED` method will be skipped.

An example usage file is provided in the repository called `test_bma.py` demonstrating
the recommended BMA (Bayesian Model Averaging) approach.

<!-- DOC:USAGE-END -->

## OUTPUT FILES

<!-- DOC:OUTPUT-START -->

After **ARIADNE** has finished running, it will output a series of files and
plots showing the results of the fit and other information.

The most important file is the `best_fit.dat` which contains the best fiting
parameters with the 1 sigma error bars and the 3 sigma confidence interval.
Then there are pickle files for each of the used models plus a last one for the
BMA, these contain raw information about the results. There is a `prior.dat`
file that shows the priors used and a `mags.dat` file with the used magnitudes
and filters.

Another important output are the plots. Inside the plots folder you can find
`CORNER.png/pdf` with the cornerplot (the plot showing the distribution of the
parameters agains eachother), `HR_diagram.png/pdf` only for the BMA, with the HR
diagram showing the position of the star, `SED_no_model.png/pdf` with the RAW
SED (overlaid with the blackbody fit from the photometry quality-control step,
and any QC-flagged bands circled in red — see below), and `SED.png/pdf` with
the SED with the catalog photometry plus synthetic photometry. If BMA was done,
there's also a `histograms` folder inside the plot folder with various
histograms of the fitted parameters and their distribution per model,
highlighting the benefits of BMA.

In both SED plots each observed point (circle) is coloured by its filter, and a
legend beside the plot names the filters from the bluest to the reddest. The
colours run from violet to red in wavelength order, spaced evenly by rank so
that neighbouring bands stay distinguishable. In `SED.png` the open diamonds
are the model's synthetic photometry; the legend above the filter list
explains the markers and names the model curves (grid and Teff).

Plots use STIXGeneral text and STIX math (matching LaTeX/Times documents). Fonts,
sizes and colours come from `Datafiles/plot_settings.dat`; to change them pass
your own copy with `SEDPlotter(..., settings='my_settings.dat')`. The main keys
are `fontname`, `mathtext_fontset`, `fontsize` (axis labels), `tick_labelsize`,
`legend_fontsize`, `filter_cmap` (any matplotlib colormap), and
`corner_fontsize` / `corner_title_fontsize` / `corner_tick_fontsize`. Settings
apply only while SEDPlotter draws, so your own matplotlib settings are left
alone.

Examples of those figures:

![SED plot](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/SED.png)
![HR Diagram](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/HR_diagram.png)
![Corner plot](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/CORNER.png)
![Histogram example](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/rad.png)

### Predicted photometry (`synthetic_*.dat`)

Every fit writes the best-fit SED's predicted photometry in all of ARIADNE's
bands, so you can read off a magnitude your star lacks (a missing G, K, W1...):

- `synthetic_magnitudes.dat`: `Filter  wavelength_um  mag  mag_err  flux_erg_cm2_s_um  flux_err  observed`
  (magnitudes in each band's native Vega/AB system; `observed=1` marks bands
  used in the fit, `observed=0` are predictions).
- `synthetic_fluxes.dat`: the same observed-frame fluxes (erg s⁻¹ cm⁻² µm⁻¹).

The values and 1σ errors come from pushing up to 3000 posterior draws through
the forward model (distance/radius/extinction-scaled) on the BT-Settl grid, and
are also stored in the output pickle as `out['synthetic_photometry']`. For white
dwarfs and hot subdwarfs the BT-Settl photosphere is only a rough proxy.

To predict other filters set `Fitter.synthetic_filters`:

```python
import numpy as np
fitter.synthetic_filters = [
    'GROUND_JOHNSON_V',                               # any pyphot library filter
    ('my_band', wavelength_um, transmission),         # photon-counting detector
    ('my_bolometer', wavelength_um, transmission, 'energy'),
]
```

Entries may also be `pyphot.Filter` objects. ARIADNE grid bands are read from
the band grid; every other filter is convolved once against all BT-Settl node
spectra of the spectra cache (`fetch_spectra_cache()`; if it is missing those
filters are skipped with a single warning) and propagated through the same
posterior draws, so the errors are consistent. A filter whose central 98 %
transmission the cached spectra do not span comes out as NaN. Results are
appended to the two files above (`observed=0`, Vega magnitudes) and written to
`synthetic_user_filters.dat` with columns `filter, wavelength_um, flux,
flux_err, mag_vega, mag_vega_err, mag_ab, mag_ab_err, source`, where `source`
is `grid` or `spectra_cache`; `out['synthetic_user_filters']` holds the same
numbers in the pickle.

### Photometry quality control

The raw SED plot doubles as a quality-control diagnostic. ARIADNE fits a
blackbody to the photometry (anchored on the Gaia bands) and overlays it on
`SED_no_model.png`. Points are coloured by filter (violet at short wavelengths
through to red at long ones, named in the legend), and bands that sit more
than 5σ off the fit are flagged and **circled in red** so you can eyeball
whether a catalogue point genuinely belongs to the stellar continuum — useful
for spotting blends, bad cross-matches or real infrared excesses. By default
flagged bands are only warned about, not removed, so the decision stays with
you.

In the example below the star is a clean ~5500 K blackbody and the WISE W2
point (circled in red) shows a clear infrared excess above the fit:

![Blackbody QC](https://raw.githubusercontent.com/jvines/astroARIADNE/master/img/SED_blackbody.png)


## NetCDF Export (Ecosystem Interop)

As of version 1.4.0 (extended in 1.4.1), ARIADNE can export its results as an
[arviz](https://python.arviz.org/) DataTree in netCDF4 format. This is the
canonical inter-tool format for passing full posterior distributions to
downstream tools (LACHESIS, PROTEUS, etc.) without reducing to point estimates.

### Exporting

After a BMA fit completes, call `to_netcdf` on the `Fitter` object:

```python
f.fit()             # run the fit as usual
f.to_netcdf('ariadne_result.nc')
```

Or get the raw dictionary (useful for programmatic access without writing a
file):

```python
result = f.to_dict()
teff_samples = result['posterior']['Teff']      # shape (1, n_draws)
model_weights = result['sample_stats']['model_weights']
```

### Reading the output

```python
import arviz as az

dt = az.from_netcdf('ariadne_result.nc')

# BMA-weighted posterior (combined across all models)
teff = dt['posterior'].ds.Teff          # shape (chain=1, draw=100000)
print(f"Teff = {float(teff.median()):.0f} K")

# Observed photometry used in the fit
obs = dt['observed_data'].ds
print(obs.wavelength.values)            # micron
print(obs.flux.values)                  # erg/s/cm^2/micron

# Model evidence and BMA weights
cd = dt['constant_data'].ds
print(cd.model_names.values)            # e.g. ['phoenix', 'kurucz', 'btsettl']
print(cd.model_weights.values)          # e.g. [0.08, 0.26, 0.66]
print(float(cd.log_evidence))           # BMA-weighted log evidence

# Per-model posteriors (individual model fits before averaging)
phoenix = dt['posterior_phoenix'].ds
print(f"Phoenix Teff = {float(phoenix.Teff.median()):.0f} K")
print(f"Phoenix log-likelihood median = {float(phoenix.log_likelihood.median()):.2f}")

# Construct a KDE prior for downstream tools
from scipy.stats import gaussian_kde
kde = gaussian_kde(teff.values.flatten())
```

Since 1.4.1 you can attach the best-fit model SED before exporting, so the
`.nc` file carries everything needed to re-plot it:

```python
# Assuming `artist` is an SEDPlotter that has been initialized
f.out['model_sed'] = {
    'model_flux': artist.model,    # f_λ at filter wavelengths
    'wavelengths': artist.wave,
}
f.to_netcdf('ariadne_result.nc')
```

### File structure

```
/
├── posterior/                          BMA-weighted combined posterior
│   ├── Teff          (chain, draw)    K
│   ├── logg          (chain, draw)    dex
│   ├── feh           (chain, draw)    dex
│   ├── radius        (chain, draw)    R_sun
│   ├── luminosity    (chain, draw)    L_sun
│   ├── distance      (chain, draw)    pc
│   ├── Av            (chain, draw)    mag
│   ├── age           (chain, draw)    Gyr          [1.4.1+, from MIST]
│   ├── iso_mass      (chain, draw)    M_sun        [1.4.1+, from MIST]
│   └── eep           (chain, draw)                 [1.4.1+, from MIST]
│
├── posterior_{model}/                  One group per stellar model used
│   ├── Teff          (chain, draw)    K
│   ├── ...
│   └── log_likelihood (chain, draw)   Per-sample log-likelihood
│
├── observed_data/                     Photometry used in the fit
│   ├── wavelength    (band,)          micron
│   ├── flux          (band,)          erg/s/cm^2/micron
│   ├── flux_err      (band,)          erg/s/cm^2/micron
│   ├── filter_names  (band,)          str           [1.4.1+]
│   ├── bandwidths    (band,)          micron         [1.4.1+]
│   └── model_flux    (band,)          erg/s/cm^2/micron  [1.4.1+, if injected]
│
├── constant_data/                     Scalar metadata
│   ├── log_evidence                   BMA-weighted log evidence
│   ├── model_weights (n_models,)      BMA posterior model probabilities
│   ├── model_names   (n_models,)      Model grid names
│   ├── best_fit_averaged__{param}     Summary stats  [1.4.1+]
│   ├── uncertainties_averaged__{param}               [1.4.1+]
│   └── confidence_interval_averaged__{param}         [1.4.1+]
│
```

Each model's posterior group (e.g. `posterior_phoenix`, `posterior_kurucz`) has
its own draw count reflecting the nested sampling output for that grid. The
top-level `posterior/` group contains the BMA-weighted resampled draws across
all models, including MIST isochrone-derived parameters (age, mass, EEP) when
available.


## Infrared Excess

As of version 1.0, **ARIADNE** now allows for Infrared Excess visualization!

To visualize infrared excess you just need to add the relevant photometric
observations to the `Star` object with the `add_mag()` (or, for fluxes,
`add_flux()`) method. After the fitting
is done, you then need to initiate the `Plotter` object with the `ir_excess`
parameter set to `True`:

```Python
artist = SEDPlotter(in_file, plots_out_folder, pdf=True, ir_excess=True)
```
Finally after plotting, you should get an SED figure with your manually added
photometry!

Allowed filters for infrared excess plots are **WISE W3, WISE W4, HERSCHEL PACS
BLUE, GREEN and RED**, names for these filters can be found in the
[filters page.](https://github.com/jvines/astroARIADNE/blob/master/filters.md)

<!-- DOC:OUTPUT-END -->

## Citing ARIADNE

For a more in depth look on the inner workings of **ARIADNE** consider
[reading the paper!](https://ui.adsabs.harvard.edu/abs/2022MNRAS.tmp..920V/abstract)

Additionally, you can find how to cite **ARIADNE** and its dependencies
[here](https://github.com/jvines/astroARIADNE/blob/master/citations.md)