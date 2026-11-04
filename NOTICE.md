# Third-party code in this application

`cwg_infinigen/` is a modified subset of **Infinigen** (Princeton University,
BSD 3-Clause; see `LICENSE-INFINIGEN`). Its `infinigen` imports are renamed to
`cwg_infinigen`, and it adds defect injection, defect fields, crack factories
and stucco/stone/tile materials.

Also included, with their own licences kept in place:
- `cwg_infinigen/OcMesher/` (see `cwg_infinigen/OcMesher/LICENSE`)
- `cwg_infinigen/terrain/mesh_to_sdf/` (see `cwg_infinigen/terrain/mesh_to_sdf/LICENSE`)
- fonts in `cwg_infinigen/assets/fonts/` (each family's licence file is kept beside it)

These terrain/mesher/creature modules are imported by Infinigen's package
`__init__` files but never executed by the generator. They are kept so that
import-time behaviour, and therefore every seeded random draw, stays the same.
The dataset licence must be decided separately from these code licences.
