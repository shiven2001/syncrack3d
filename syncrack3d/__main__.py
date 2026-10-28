"""``python -m syncrack3d ...`` runs build_dataset in a bpy-enabled Python."""

import sys

from syncrack3d.build_dataset import main

if __name__ == "__main__":
    sys.exit(main())
