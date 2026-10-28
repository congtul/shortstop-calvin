# Third-party components

The MIT license in `LICENSE` covers the code and data written for ShortStop.
The components below are used, not redistributed: they are fetched from
their authors (git submodule or download) and stay under their own licenses.

| Component | Source | License | How it is used |
|---|---|---|---|
| MDT policy code | [intuitive-robots/mdt_policy](https://github.com/intuitive-robots/mdt_policy) @ `e7a78e7` | MIT, (c) 2024 Moritz Reuss | git submodule `mdt_policy/`; `patches/mdt_policy_shortstop.patch` modifies three of its files (paths, logging, headless) and is distributed under the same MIT license |
| `mdtv-1-abcd` checkpoint | MDT authors (link in the mdt_policy README) | as published by its authors | downloaded by the user, checked by SHA-256 |
| CALVIN simulator (`calvin_env`) | [mees/calvin_env](https://github.com/mees/calvin_env) @ `797142c` | MIT, (c) 2021 Oier Mees | submodule of mdt_policy; ShortStop reads its scene and robot description |
| TACTO | [lukashermann/tacto](https://github.com/lukashermann/tacto) @ `dd53360` (fork of facebookresearch/tacto) | MIT, (c) Facebook, Inc. and its affiliates | submodule of calvin_env |
| CALVIN benchmark and debug dataset | [mees/calvin](https://github.com/mees/calvin) | MIT, (c) Oier Mees | downloaded by the user (`download_data.sh debug`) |
| Voltron `v-cond` encoder | [siddk/voltron-robotics](https://github.com/siddk/voltron-robotics) | MIT, (c) 2021-present Siddharth Karamcheti and contributors | downloaded on first model load |

`docs/paper/shortstop/llncs.cls` and `splncs04.bst` are Springer's LNCS
LaTeX class and BibTeX style (CTAN package `llncs`), included unmodified to
build the paper; they remain under Springer's terms.

Files in `shortstop/calvin/data/` (static furniture boxes, door contact
offsets, frozen obstacle scenes) are computed by the scripts in `scripts/`
from the CALVIN scene description and from rollouts of the MDT policy.

If you use this code, please also cite CALVIN (Mees et al., RA-L 2022) and
MDT (Reuss et al., RSS 2024).
