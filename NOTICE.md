# Third-party code in this application

`cwg_infinigen/` is a vendored subset of the SynDefect3D fork of
**Infinigen** (Princeton University, BSD 3-Clause; see `LICENSE-INFINIGEN`),
copied from `infinigen/infinigen/` of the SynDefect3D repository with only these
changes: every `infinigen` import renamed to `cwg_infinigen`, and paths of the
form `"infinigen/..."` renamed to `"cwg_infinigen/..."`. The file list is
`VENDORED_FILES.txt`. It includes the fork's own additions (defect injection,
defect fields, peel/crack factories, stucco/stone/tile materials).

Also vendored as part of that subset, with their own licences kept in place:
- `cwg_infinigen/OcMesher/` (see `cwg_infinigen/OcMesher/LICENSE`)
- `cwg_infinigen/terrain/mesh_to_sdf/` (see `cwg_infinigen/terrain/mesh_to_sdf/LICENSE`)
- fonts in `cwg_infinigen/assets/fonts/` (each family's licence file is kept beside it)

These terrain/mesher/creature modules are imported by Infinigen's package
`__init__` files but never executed by the generator; they are kept so the
import-time behaviour (and therefore every seeded random draw) matches the repo.
The dataset licence must be decided separately from these code licences.
