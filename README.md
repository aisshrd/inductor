# Data

| file | content |
|---|---|
| `Ar_LXCat.txt` | **not included — download it (see below)** |
| `reference_profiles_p13_n1e17_B1.3G.json` | reference fields of the converged self-consistent point (13.3 Pa, n0 = 1e17 m^-3, B_ext = 1.3 G, solver generation II): r/R, E_theta amplitude and phase, B_z amplitude and phase, phi, T_e, n_e, n_i, nu_iz, balance. These fields are imposed in the cylindrical Monte Carlo and in the solver comparisons (article §4.3). |

## Cross sections

The code uses the IST-Lisbon argon cross-section set from LXCat
(elastic momentum transfer, 37 excitation levels, ionization).
The file is distributed by LXCat under its own terms and is therefore not included here.

Obtain it in either of two ways and save it as `data/Ar_LXCat.txt`:

1. www.lxcat.net → Download → database **IST-Lisbon**, species **Ar**, all processes, format "LXCat".
2. From the LoKI-B repository (github.com/IST-Lisbon/LoKI): `Code/Input/Argon/Ar_LXCat.txt`.

Please cite the database as requested in the file header
("IST-Lisbon database, www.lxcat.net, retrieved on ...").
