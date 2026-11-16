# AMSR2 conversion to the SSMIS daily grid format

This directory holds a reader and a fixed conversion that turn Colorado State University's (CSU) intercalibrated
brightness temperatures from the Advanced Microwave Scanning Radiometer 2 (AMSR2) into daily grids in the format of
the Special Sensor Microwave Imager/Sounder (SSMIS) that this pipeline reads. The conversion is a set of per-channel
quantile maps fitted to Defense Meteorological Satellite Program (DMSP) F-16 antenna temperatures on 72 days between
April 2022 and March 2024, by surface class, 30-degree latitude zone, pass and calendar month. The monthly processor
in `monthly/` then makes the ten hydrological products from those grids as it does for an SSMIS satellite.

## Status and limits

The conversion is provisional. It was accepted for use in the late-morning chain on 2 November 2026 by the
product's principal investigator, and it has run since as a parallel, unpublished stream beside F-16. That
acceptance is a decision on use and not a scientific qualification. F-16 remains the source of the SSMI(S)
late-morning products, and publishing this code activates no stream and accepts no product.

On two held-out months compared with F-16, monthly rainfall from Algorithm 1 read 7.74 percent high in February 2025
and 8.43 percent high in July 2024, outside the 5 percent criterion declared before the reading. The excess sits
over ocean (10.30 and 14.81 percent high). Rainfall from Algorithm 2 read 14.16 and 16.40 percent high, and the two
rain fractions between 8.44 and 13.81 percent high. Water vapor read 1.0 and 0.9 percent low, and liquid water path
1.5 and 1.7 percent high. Snow (15.1 and 32.0 percent high) and sea ice (30.6 and 14.1 percent high) are not
supported. The 1.0-degree products were not part of that reading.

The cause of the rainfall excess is open. AMSR2 observes near 01:30 and 13:30 local solar time, and F-16's morning
crossing was near 07:15 in September 2026. The conversion does not remove that difference, and no comparison has yet
separated the observing-time effect from conversion error. The output is not homogenized with the F-16 record.

This directory has no writer of netCDF files for AMSR2 months. The netCDF writer elsewhere in this repository
labels its files as SSMI(S) products, and those labels do not apply to grids or products made from AMSR2. The
stream's own acquisition, monthly operation and comparison with F-16 are not included here.

## Method

The reader takes the 18.7 GHz vertical and horizontal, 23.8 GHz vertical, 36.5 GHz vertical and horizontal and
89 GHz vertical and horizontal brightness temperatures, which stand in for the SSMIS 19, 22, 37 and 91 GHz
channels. Each low-resolution footprint is paired with its 89 GHz A-scan footprint. A footprint is used only when
every quality flag of its seven channels is clear and every temperature lies between 50 and 350 K. Scans are
assigned to their UTC day and to the ascending or descending pass by the trend of latitude along the orbit. Each
pass is gridded at one third of a degree with the gridding rules of `monthly/wsfm_adapter.py`, the nearest grid
node for each footprint and the last write where footprints share a node.

The conversion then maps each channel of the day's composite to the F-16 antenna temperature scale. There are 288
maps, one for each calendar month, pass, surface class (ocean or land, from the land/sea tag) and 30-degree
latitude zone, and each holds 101 knots for each of the seven channels. The knots are the 0.1, 1 to 99 and 99.9
percentiles of the AMSR2 brightness temperatures and of the F-16 antenna temperatures on the month's fitting days,
each side counted on its own valid cells. A value between two knots is mapped linearly, and a value beyond the end
knots is shifted by the end knot's offset. The mapped grid is written in the SSMIS daily format, one byte per value,
as `asYYDDD.amsr2` and `dsYYDDD.amsr2`.

## What is here

| File | What it does |
|---|---|
| `route_a.py` | Reads the CSU AMSR2 granules of a day and writes the day's two float composites. |
| `route_a_fit.py` | Applies the fixed maps to a composite and writes the two daily grids. It also holds the fit that made the maps. |
| `route_a_calendar.py` | The 72 fitting days. |
| `route_a_maps.json` | The 288 fixed maps, with the sample counts and the SHA-256 of every input of the fit. |
| `test_route_a.py`, `test_route_a_fit.py` | Tests of the reader and of the fit and its application, on synthetic granules. |
| `test_end_to_end.py`, `test_maps.py` | One synthetic day through the reader, the maps and the processor, and a check of the maps file. |
| `monthly/NLNDSEA.TAG`, `monthly/lut_add_ocean`, `monthly/lut_add_land`, `monthly/lut_add_si` | The land/sea tag and the processor's three lookup tables, kept with the processor. |

Three names in these files refer to things that are not part of this directory. The maps file and the fit carry a
field `protocol` with the value `route_a_protocol.md`, the name of the written plan under which the maps were fitted.
That plan is an internal document and is not distributed here, and the field is kept because the maps file must stay
equal in value to the fitted record it was made from. The tests `test_maps.py` and `test_end_to_end.py` read
`route_a_maps.json` and, where it is absent, look for `fit_record/maps.json`, the place of the fitted record in the
tree the conversion was developed in, which is also not distributed here. In `route_a_fit.py`, the `apply` command
accepts two optional digests, `--expect-npz-sha256` and `--expect-json-sha256`, and its messages call the earlier check
that recorded them the preflight. The operational stream uses them to confirm that a composite has not changed since
it was checked, and the one-day example below does not need them.

## Running one day as a demonstration

The code needs Python 3 with numpy, scipy, h5py and netCDF4. The input files are the AMSR2 files of CSU's interim
record, named `CSU_AMSR2_ICDR_V02R00_GCOMW1_DYYYYMMDD_SHHMM_EHHMM_RNNNNN.nc`, under
https://www.ncei.noaa.gov/data/ssmis-brightness-temperature-csu/access/ICDR at NOAA's National Centers for
Environmental Information (NCEI). A day needs every file that holds scans of that UTC day, which includes the last
file of the day before when it crosses midnight. The three commands below take 5 April 2022, day 95 of the year,
through the conversion and the processor as a demonstration:

```
python route_a.py composite --day 2022-04-05 --files /data/CSU_AMSR2_ICDR_*.nc --out /data/day_2022-04-05.npz
python route_a_fit.py apply --maps route_a_maps.json --composite /data/day_2022-04-05.npz --day 2022-04-05 \
    --tag ../../../monthly/NLNDSEA.TAG --out-dir /data/grids
cd ../../../monthly
python climalg_ssmis.py amsr2 2022 95 95 --res 2.5 --inpath /data/grids/ --outdir /data/demonstration_one_day
```

Run the first two commands from this directory. They write the day's composite, its record and its two grids
under `/data`, outside this repository. The third is a demonstration using one day's grids. It writes
files labeled for April 2022 under `/data/demonstration_one_day/amsr2-2.5/`, and their rainfall is the mean rate over
the available samples multiplied by 24 hours and all 30 days of April. That output is neither a daily rainfall total
nor a complete April product. Keep it in a separate demonstration directory and do not distribute these files as
monthly observations.

To process a month, first verify that every intended day's ascending and descending grids are present and that
their coverage is acceptable, then give the processor the month's first and last day of year. The processor prints a
warning for a missing daily grid and continues. Completing without an error does not establish a complete month,
and these commands do not include the separate coverage and completeness checks of the stream that runs this
conversion each month. A day with less than ten orbit lengths of observed time is still gridded, and the check of
hemisphere balance that the adapter applies to a full day is then not applied to it.

## Tests

`python -m pytest` in this directory runs the tests, which use synthetic granules and need no data file. They need
pytest beside the four packages named above, and they skip where h5py is absent. The end-to-end test writes two small
granules, runs the three commands above on them and requires the two daily grids and the ten product files. Its
products come from made-up brightness temperatures on one day and mean nothing as values.

## Input data and credit

The input is the NOAA Climate Data Record (CDR) of SSMI(S) and AMSR2 Microwave Brightness Temperatures, CSU
Version 2. It is read from NCEI's public directory and is not redistributed here. The CDR Program asks that users
acknowledge the developers at Colorado State University and the NOAA CDR Program and cite the dataset. The citation
below is in the form of the dataset's record at NCEI, read on 15 November 2026:

> Kummerow, Christian D., Wesley K. Berg, Chia-Pang Kuo, and NOAA CDR Program (2022): NOAA Climate Data Record
> (CDR) of SSMI(S) and AMSR2 Microwave Brightness Temperatures, CSU Version 2. [indicate subset used]. NOAA National
> Centers for Environmental Information. https://doi.org/10.25921/E3K5-BW77. [access date].

AMSR2 flies on the Global Change Observation Mission for Water satellite (GCOM-W1), which the Japan Aerospace
Exploration Agency (JAXA) operates. The retrieval algorithms that turn the daily grids into products are those of
Ralph R. Ferraro and colleagues, credited in this repository's main README. The conversion in this directory was
developed at NCEI as part of the SSMI(S) Hydrological Products work.

## Assets and their provenance

The terms differ by asset, so each is stated on its own:

- Code: written in this project by a NOAA employee and dedicated to the public domain under Creative Commons Zero
  1.0, as the rest of this repository is (see `LICENSE`). No CSU software is included.
- Maps (`route_a_maps.json`): derived in this project from the CSU record and from F-16 antenna temperatures in the
  operational daily grids. The file holds percentile knots and sample counts and no observation. Each input of the
  fit is listed by file name and SHA-256.
- Land/sea tag (`NLNDSEA.TAG`) and lookup tables (`lut_add_ocean`, `lut_add_land`, `lut_add_si`): they came with the
  legacy NOAA/NESDIS processing package for the SSM/I hydrological products and are redistributed here unchanged.
  They have been in this project's open Zenodo code deposits since June 2026. The source data behind the tag is not
  recorded, and no record names who generated the three table files. Lookup tables of the form of `lut_add_ocean`
  and `lut_add_land` are described by Vila et al. (2013).

The four inherited files are byte for byte the ones in those deposits, at these SHA-256 values:

| File | Bytes | SHA-256 |
|---|---|---|
| `monthly/NLNDSEA.TAG` | 583,200 | `28855415038a3f0b75e368e4aca08436af3981cac99027a2a1a5e0f6b6d53b88` |
| `monthly/lut_add_ocean` | 9,753 | `e42b72f7efb54c6194ad881ea6a3f73275129df39960bfe42dd829fce4e3b7c0` |
| `monthly/lut_add_land` | 9,074 | `414596c7567af014044510e16e2fdc007b2be22734abc2e060caee9433f2764e` |
| `monthly/lut_add_si` | 2,682 | `69f1be692e205f6ba31de090baa048e491194a816e21099f4ae9b903a4fe7c86` |

That these files were distributed before says nothing about who made them or under what terms. No record at hand
names either. This repository states only what is written above about them.

## Reference

- Vila, D. A., C. Hernandez, R. R. Ferraro, and H. Semunegus, 2013: The performance of hydrological monthly
  products using SSM/I-SSMI/S sensors. *J. Hydrometeor.*, **14**, 266-274. https://doi.org/10.1175/JHM-D-12-056.1
