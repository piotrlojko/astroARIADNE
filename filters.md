# Allowed filters in ARIADNE
**ARIADNE** uses pyphot (<https://mfouesneau.github.io/pyphot/libcontent.html>) as an interface for photometry, and thus we use the same filter naming convention \<facility\>\<instrument\>\<filter\>

The allowed filters are (in wavelength order):

- Galex\_FUV
- Galex\_NUV
- STROMGREN\_u
- SkyMapper\_u
- SDSS\_u
- GROUND\_JOHNSON\_U
- SkyMapper\_v
- STROMGREN\_v
- TYCHO\_B\_MvB
- GROUND\_JOHNSON\_B
- STROMGREN\_b
- SDSS\_g
- PS1\_g
- SkyMapper\_g
- GaiaDR2v2\_BP
- TYCHO\_V\_MvB
- STROMGREN\_y
- GROUND\_JOHNSON\_V
- SkyMapper\_r
- SDSS\_r
- PS1\_r
- PS1\_w
- KEPLER\_Kp
- GaiaDR2v2\_G
- GROUND\_COUSINS\_R
- NGTS\_I
- SDSS\_i
- PS1\_i
- SkyMapper\_i
- GaiaDR2v2\_RP
- GROUND\_COUSINS\_I
- TESS
- PS1\_z
- SDSS\_z
- SkyMapper\_z
- PS1\_y
- 2MASS\_J
- 2MASS\_H
- 2MASS\_Ks
- WISE\_RSR\_W1
- WISE\_RSR\_W2
- **WISE\_RSR\_W3**
- **WISE\_RSR\_W4**
- **HERSCHEL\_PACS\_BLUE**
- **HERSCHEL\_PACS\_GREEN**
- **HERSCHEL\_PACS\_RED**

**NOTE: WISE W3, W4 and HERSCHEL filters are ONLY available for INFRARED EXCESS VISUALIZATION and NOT for SED FITTING**
## HST filters

The HST WFC3 (UVIS and IR) and ACS/WFC wide, medium, long-pass and
extremely-wide filters can be fitted. Narrow-band and quad filters are not
included. No catalogue query feeds them, so add the photometry yourself through
`mag_dict` or `Star.add_mag` (or, for fluxes, `flux_dict` or `Star.add_flux`):

- WFC3: `HST_WFC3_F098M`, `F105W`, `F110W`, `F125W`, `F127M`, `F139M`,
  `F140W`, `F153M`, `F160W`, `F200LP`, `F218W`, `F225W`, `F275W`, `F300X`,
  `F336W`, `F350LP`, `F390M`, `F390W`, `F410M`, `F438W`, `F467M`, `F475W`,
  `F475X`, `F547M`, `F555W`, `F600LP`, `F606W`, `F621M`, `F625W`, `F689M`,
  `F763M`, `F775W`, `F814W`, `F845M`, `F850LP` (each as `HST_WFC3_<filter>`)
- ACS/WFC: `HST_ACS_WFC_F435W`, `F475W`, `F550M`, `F555W`, `F606W`, `F625W`,
  `F775W`, `F814W`, `F850LP` (each as `HST_ACS_WFC_<filter>`)

A model grid that does not reach a band leaves it out of that grid's fit (for
example Sonora below 0.4 um).

## Ultraviolet: IUE and TD-1

Seven IUE boxes and the four TD-1 (S2/68) wide bands can be fitted:

- IUE: `IUE_IUE.1250-1300`, `IUE_IUE.1450-1500`, `IUE_IUE.1675-1725`,
  `IUE_IUE.2150-2200`, `IUE_IUE.2395-2445`, `IUE_IUE.2900-2950`,
  `IUE_IUE.2900-3000` (mean flux in the given range, in Angstrom)
- TD-1: `TD1_WIDE.1565`, `TD1_WIDE.1965`, `TD1_WIDE.2365`, `TD1_WIDE.2740`

Only the **BOSZ** (2800-16000 K) and **TLUSTY** (15000-55000 K) grids have
model fluxes in these bands; they were convolved from the same BOSZ 2024 and
TLUSTY spectra the grids were built from. In a BMA fit that uses one of these
bands, every other grid is left out with a warning, so include `'bosz'` (or,
for O/B stars, run `'tlusty'` standalone). A standalone fit with another grid
stops with an error naming the grids that have the bands.

These bands are usually published as fluxes, so add them with `flux_dict` or
`Star.add_flux`:

```python
import astropy.units as u
flam = u.erg / (u.s * u.cm**2 * u.AA)
s = Star('my star', ra, dec, temp=11500, temp_e=0,
         flux_dict={'TD1_WIDE.1565': (5.25e-12 * flam, 4.9e-13 * flam),
                    'IUE_IUE.1250-1300': (4.57e-12 * flam, 1.83e-13 * flam)})
```

The BMA chooses its grids from `star.temp`. When Gaia has no temperature for
the star (`star.temp = -1`), BOSZ is gated out, so pass a rough `temp` (with
`temp_e=0`, so it does not also constrain the isochrone fit) as above.
Magnitudes in these bands are read as Vega magnitudes.

## Magnitude systems

PanSTARRS, SDSS, GALEX and SkyMapper magnitudes are read as AB, and every other
band as Vega. HST photometry is published as VEGAMAG, ABMAG or STMAG depending
on the source (the Hubble Source Catalog uses ABMAG), so give the system of any
band that is not in its default with `mag_system`:

```python
s = Star('my star', ra, dec, mag_dict={'HST_WFC3_F275W': (14.20, 0.03)},
         mag_system={'HST_WFC3_F275W': 'AB'})
s.add_mag(15.10, 0.02, 'HST_WFC3_F336W', system='ST')
```

The zero points are 3631 Jy for AB, f_lambda = 10^(-0.4 x 21.10)
erg s^-1 cm^-2 A^-1 for ST, and Vega for Vega.

Photometry given as a flux (`flux_dict`, `Star.add_flux`) does not need a
system: it is stored as the magnitude in the band's system, which converts back
to the same flux.
