"""SynCrack3D studies: dataset renders, training (seg_train.py) and zero-shot evaluation.

Run them as modules from the repository root (python -m studies.<name>); this
puts the root on the path. Outputs go to outputs/ here (gitignored).
"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)
