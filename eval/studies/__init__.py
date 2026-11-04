"""SynCrack3D studies: dataset renders, training (seg_train.py) and zero-shot evaluation.

Run them as modules from the repository root (PYTHONPATH=eval python -m studies.<name>); this
puts the repository root and eval/ on the path. Outputs go to outputs/ (gitignored).
"""
import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
