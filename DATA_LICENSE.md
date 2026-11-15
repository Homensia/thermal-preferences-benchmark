# Data and results licences

The code of this repository is under the MIT licence (`LICENSE`). The data files and the recorded results
are under the licences below.

| Files | Licence | Source |
|---|---|---|
| `Data/Moujalled.csv`, `Data/Moujalled_api.csv` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | French cohort (Moujalled), published with the authorisation of its authors |
| `Data/Hostein.csv`, `Data/Hostein_api.csv` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | French cohort (Hostein), published with the authorisation of its authors |
| `Data/ASHRAE_09_06_2022.csv`, `Data/ASHRAE_09_06_2022_metadata.xlsx`, and the derived `Data/ASHRAE_2022_Clean_api.csv` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (licence of the source record) | ASHRAE Global Thermal Comfort Database II, database v2.1.0, Borealis Dataverse record doi:[10.5683/SP2/GNVEM8](https://doi.org/10.5683/SP2/GNVEM8) (record version 2.0, released 2022-07-21; licence read from `https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP2/GNVEM8` on 2026-10-11) |
| `Data/ASHRAE_2018_v2.csv` | CC0 1.0 (licence of the Dryad record) | ASHRAE Global Thermal Comfort Database II, 2018 release, as used by Haghirad et al. (2024). The database is published on Dryad, doi:[10.6078/D1F671](https://doi.org/10.6078/D1F671) (licence `CC0-1.0` read from `https://datadryad.org/api/v2/datasets/doi%3A10.6078%2FD1F671` on 2026-10-11; record versions 1–4 dated 2018-07-01). The record holds `ASHRAE_db2.0.csv` (version 1) and `ashrae_db2.01.csv` (version 4); this file is a cleaned subset (17 801 rows), so its source version cannot be identified from file hashes. |
| Recorded results of the paper (run directory `rerun_2026-10-XX_v2/`, distributed separately, and `results/phase1/experiment_grid_full.csv`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | this work |

The `*_api.csv` files add outdoor weather features (28-day exponential running means) computed from the
Open-Meteo historical weather API (`feature_extraction_api.py`). Weather data by
[Open-Meteo.com](https://open-meteo.com/), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
(licence page `https://open-meteo.com/en/licence`, read on 2026-10-11); the values were changed (aggregated
into running means).
