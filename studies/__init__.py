"""SynCrack3D studies: dataset renders, training and zero-shot evaluation.

They train with the SynDefect3D repo's domain_transfer modules, so run them as
modules from the SynCrack3D folder (python -m studies.<name>); this puts the
SynDefect3D root on the path. Outputs go to outputs/ here (gitignored).
"""
import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
