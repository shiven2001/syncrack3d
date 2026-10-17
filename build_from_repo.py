"""Maintainer tool: (re)build SynCrack3D from the SynDefect3D repo.

Copies controlled_wall_generator -> syncrack3d/, the exact set of Infinigen-fork files the
generator loads at runtime -> cwg_infinigen/ (VENDORED_FILES.txt), the fonts a text
material reads on import, and the export/contract tools -> tools/. Every import of
`infinigen` is renamed to `cwg_infinigen` and `controlled_wall_generator` to `syncrack3d`
(token-level, so comments and paths are left alone), so the application never
imports the repo's Infinigen even when that package is installed in the same
environment. Run from anywhere:  python syncrack3d/build_from_repo.py
The application itself does not need this file. Rebuilding overwrites the
crack-only defaults and the open-crack style that live only in this snapshot.
"""

import io
import re
import shutil
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC_INFINIGEN = REPO / "infinigen" / "infinigen"
RENAME = {"infinigen": "cwg_infinigen", "controlled_wall_generator": "syncrack3d"}
MODULE_PATH = re.compile(r"(?<![\w./])(infinigen|controlled_wall_generator)(?=\.[A-Za-z_])")


def rewrite(src: str) -> str:
    """Rename module roots in import statements, dotted name chains and module-path strings."""
    lines = src.splitlines(keepends=True)
    edits = []  # (row, col_start, col_end, text)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for k, t in enumerate(toks):
        prev = toks[k - 1].string if k else ""
        if t.type == tokenize.NAME and t.string in RENAME and prev != ".":
            edits.append((t.start[0], t.start[1], t.end[1], RENAME[t.string]))
        elif t.type == tokenize.STRING or t.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            if t.start[0] != t.end[0]:
                continue  # multi-line strings are docstrings/prose here
            new = MODULE_PATH.sub(lambda m: RENAME[m.group(1)], t.string)
            if new != t.string:
                edits.append((t.start[0], t.start[1], t.end[1], new))
    for row, a, b, text in sorted(edits, reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:a] + text + line[b:]
    return "".join(lines)


def copy_py(src: Path, dst: Path, extra=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = rewrite(src.read_text())
    if extra:
        text = extra(text)
    dst.write_text(text)


def main():
    import sys

    # Since 2026-10-16 the standalone app is edited directly (texture/displacement
    # fixes live only here). Rebuilding from the repo would silently erase them.
    if "--overwrite-local-edits" not in sys.argv:
        raise SystemExit("syncrack3d/ has its own edits (crack-only defaults, open-crack style, "
                         "brick/marble/displacement fixes); rebuilding from the repo would overwrite them. "
                         "Port those edits to the repo first, or pass --overwrite-local-edits if you really "
                         "mean to discard them.")
    files = [l.strip() for l in (HERE / "VENDORED_FILES.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    out = HERE / "cwg_infinigen"
    if out.exists():
        shutil.rmtree(out)
    # paths built as repo_root() / "infinigen/..." (fonts for the text material) follow the rename
    def own_paths(text):
        return re.sub(r"([\"'])infinigen/", r"\1cwg_infinigen/", text)

    for rel in files:
        copy_py(SRC_INFINIGEN / rel, out / rel, own_paths)
    shutil.copytree(SRC_INFINIGEN / "assets" / "fonts", out / "assets" / "fonts")
    # data files beside vendored modules (e.g. terrain/assets/caves/cfg.txt, checked at
    # import) and third-party licences (OcMesher, mesh_to_sdf)
    for d in sorted({(SRC_INFINIGEN / rel).parent for rel in files}):
        for f in d.iterdir():
            if f.is_file() and f.suffix not in (".py", ".pyc") and f.name not in (".gitignore", ".dockerignore"):
                dst = out / f.relative_to(SRC_INFINIGEN)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
            # data-only subfolders read at import (creatures/parts/nurbs_data is listed
            # when the module loads); no Python inside, so nothing to rename
            elif (f.is_dir() and f.name not in ("__pycache__", ".github", "fonts")
                  and not any(q.suffix == ".py" for q in f.rglob("*"))
                  and not (out / f.relative_to(SRC_INFINIGEN)).exists()):
                shutil.copytree(f, out / f.relative_to(SRC_INFINIGEN))
    # every package directory needs its __init__ even if the runtime trace missed an empty one
    for d in [p for p in out.rglob("*") if p.is_dir() and p.name != "fonts" and "fonts" not in p.parts]:
        src_init = SRC_INFINIGEN / d.relative_to(out) / "__init__.py"
        if not (d / "__init__.py").exists() and src_init.exists():
            copy_py(src_init, d / "__init__.py")

    def own_root_only(text):
        # the app adds only its own root to sys.path, never the repo's infinigen/
        text = text.replace('for path in (REPO_ROOT, REPO_ROOT / "infinigen"):', "for path in (REPO_ROOT,):")
        text = text.replace('INFINIGEN_PACKAGE_ROOT = REPO_ROOT / "infinigen"\nfor path in (REPO_ROOT, INFINIGEN_PACKAGE_ROOT):',
                            "for path in (REPO_ROOT,):")
        return text

    app = HERE / "syncrack3d"
    if app.exists():
        shutil.rmtree(app)
    src_cwg = REPO / "controlled_wall_generator"
    for f in src_cwg.rglob("*"):
        if "__pycache__" in f.parts or f.is_dir():
            continue
        dst = app / f.relative_to(src_cwg)
        if f.suffix == ".py":
            copy_py(f, dst, own_root_only)
        elif f.suffix == ".md":  # docs: keep code examples runnable here
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(MODULE_PATH.sub(lambda m: RENAME[m.group(1)], f.read_text())
                           .replace("from controlled_wall_generator", "from syncrack3d")
                           .replace("controlled_wall_generator/", "syncrack3d/"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
    tools = HERE / "tools"
    tools.mkdir(exist_ok=True)
    for name in ("prepare_defect_annotated_dataset.py", "check_instance_contract.py"):
        copy_py(REPO / "infinigen" / "tools" / name, tools / name)
    shutil.copyfile(REPO / "infinigen" / "LICENSE", HERE / "LICENSE-INFINIGEN")
    leftover = [str(p.relative_to(HERE)) for p in HERE.rglob("*.py")
                if p.name != "build_from_repo.py" and re.search(r"^\s*(from|import)\s+(infinigen|controlled_wall_generator)\b",
                                                                 p.read_text(), re.M)]
    assert not leftover, f"unrenamed imports: {leftover[:5]}"
    print(f"vendored {len(files)} infinigen files + fonts; syncrack3d/ and tools/ rebuilt")


if __name__ == "__main__":
    main()
