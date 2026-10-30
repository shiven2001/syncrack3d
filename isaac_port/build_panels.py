"""Build rig panel assets (USD + ground truth) from SynDefect3D view patches.

    OMNI_KIT_ACCEPT_EULA=YES path/to/isaac-sim/python \
        scripts/build_panels.py --spec scripts/panels_v1.json --out assets/panels_v2

Only crack and paint-peel panels are built (rig/panels.py SCOPE_CLASSES); a
panel whose source region contains any blister coverage is refused.

Inputs are the patch maps produced by the SynDefect3D Isaac port check
(syndefect3d/isaac_port_check): per patch `<cam>_height_mm.npy` (true displaced
relief from Cycles' Position pass, +50 mm offset, 0 = no wall), `_color.npy`,
`_roughness.npy`, `_normal.png` (tangent-space bump), and class coverage
`_DefectCrack/_DefectPaintPeel/_DefectPaintBubble.npy`; optional overhanging peel
flaps from `dump_meshes.py`.

Each panel is written in its OWN frame: centre at the origin, surface in the XY
plane, outward normal +Z, u -> +X, v -> +Y, metres. Per panel:
  <name>/panel.usd      displaced mesh split by class (semantic labels), flaps
  <name>/textures/*.png colour, roughness, normal
  <name>/gt.npz         class coverage on a GT grid in panel UV (row 0 = +Y edge),
                        valid-wall mask, panel size; reward/evaluation only
  <name>/panel.json     provenance and statistics
"""

import argparse
import json
import shutil
from pathlib import Path

# Inlined from active_sensing_rig/rig/panels.py so the port has no rig dependency.
SCOPE_CLASSES = ("crack", "paint_peel")

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mesh-res", type=int, default=2048)
ap.add_argument("--gt-res", type=int, default=512)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.utils.semantics import add_labels  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

CLASSES = {"DefectCrack": "crack", "DefectPaintPeel": "paint_peel", "DefectPaintBubble": "paint_blister"}
spec = json.loads(Path(args.spec).read_text())
out_root = Path(args.out)
out_root.mkdir(parents=True, exist_ok=True)


def to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def material(stage, path, tex_dir, name, normal):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    maps = [("color", "sRGB", "rgb", "diffuseColor", Sdf.ValueTypeNames.Color3f),
            ("roughness", "raw", "r", "roughness", Sdf.ValueTypeNames.Float)]
    if normal:
        maps.append(("normal", "raw", "rgb", "normal", Sdf.ValueTypeNames.Normal3f))
    for kind, cs, out, target, typ in maps:
        t = UsdShade.Shader.Define(stage, f"{path}/tex_{kind}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{name}_{kind}.png")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        if kind == "normal":
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        t.CreateOutput(out, Sdf.ValueTypeNames.Float3 if out == "rgb" else Sdf.ValueTypeNames.Float)
        sh.CreateInput(target, typ).ConnectToSource(t.ConnectableAPI(), out)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


summary = []
for entry in spec["panels"]:
    name, pdir, cam = entry["name"], Path(entry["patch_dir"]), entry["camera"]
    info = next(p for p in json.loads((pdir / "patch.json").read_text())["patches"] if p["camera"] == cam)
    dest = out_root / name
    if dest.exists():
        raise SystemExit(f"{dest} exists; panel assets are immutable, use a new --out")
    (dest / "textures").mkdir(parents=True)
    s = float(info["size_m"])
    cx, _, cz = info["center"]
    M = args.mesh_res

    raw = np.load(pdir / f"{cam}_height_mm.npy")
    hit_full = raw > 1.0
    hit = cv2.resize(hit_full.astype(np.float32), (M, M), interpolation=cv2.INTER_AREA) > 0.999
    h_mm = cv2.resize(raw, (M, M), interpolation=cv2.INTER_AREA) - 50.0
    h_mm -= np.median(h_mm[hit])
    cov = {c: np.load(pdir / f"{cam}_{a}.npy") for a, c in CLASSES.items() if (pdir / f"{cam}_{a}.npy").exists()}
    extra = [c for c, cmap in cov.items() if c not in SCOPE_CLASSES and np.any(cmap > 0)]
    if extra:
        shutil.rmtree(dest)
        raise SystemExit(f"{name}: source region contains out-of-scope classes {extra}; choose another region")
    cov = {c: cmap for c, cmap in cov.items() if c in SCOPE_CLASSES}

    # Panel frame: row 0 of every map is the +Y edge (v = 1); column 0 is -X.
    u = np.linspace(-0.5, 0.5, M) * s
    v = np.linspace(0.5, -0.5, M) * s
    X, Y = np.meshgrid(u, v)
    P = np.stack([X, Y, h_mm / 1000.0], -1).reshape(-1, 3)
    uv = np.stack([(X / s) + 0.5, (Y / s) + 0.5], -1).reshape(-1, 2)
    idx = np.arange(M * M).reshape(M, M)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tris = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    ok = hit.reshape(-1)[tris].all(1)
    label = np.full(len(tris), "wall", dtype=object)
    for cname, cmap in cov.items():
        cm = cv2.resize(cmap, (M, M), interpolation=cv2.INTER_AREA).reshape(-1)
        label[cm[tris[:, 0]] >= 0.5] = cname

    col = np.load(pdir / f"{cam}_color.npy")
    cv2.imwrite(str(dest / "textures" / f"{name}_color.png"),
                cv2.cvtColor((np.clip(to_srgb(col), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dest / "textures" / f"{name}_roughness.png"),
                (np.clip(np.load(pdir / f"{cam}_roughness.npy"), 0, 1) * 255).astype(np.uint8))
    has_normal = (pdir / f"{cam}_normal.png").exists()
    if has_normal:
        shutil.copy(pdir / f"{cam}_normal.png", dest / "textures" / f"{name}_normal.png")

    stage = Usd.Stage.CreateNew(str(dest / "panel.usd"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Panel")
    stage.SetDefaultPrim(root.GetPrim())
    mat = material(stage, "/Panel/Looks/surface", dest / "textures", name, has_normal)
    tri_counts = {}
    for cname in ["wall"] + list(cov):
        sel = ok & (label == cname)
        if not sel.any():
            continue
        tsel = tris[sel]
        used, inv = np.unique(tsel, return_inverse=True)
        mesh = UsdGeom.Mesh.Define(stage, f"/Panel/{cname}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P[used].astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tsel), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(inv.reshape(-1).astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr("none")
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(uv[used].astype(np.float32)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        add_labels(mesh.GetPrim(), labels=[cname], instance_name="class")
        tri_counts[cname] = int(sel.sum())

    flap_names = []
    if entry.get("flaps"):
        flaps = np.load(entry["flaps"])
        paint = np.median(col.reshape(-1, 3)[col.reshape(-1, 3).sum(1) > 0.3], 0)
        fmat = UsdShade.Material.Define(stage, "/Panel/Looks/flap")
        fsh = UsdShade.Shader.Define(stage, "/Panel/Looks/flap/PBR")
        fsh.CreateIdAttr("UsdPreviewSurface")
        fsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(x) for x in paint]))
        fsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        fmat.CreateSurfaceOutput().ConnectToSource(fsh.ConnectableAPI(), "surface")
        for fname in flaps["names"]:
            wpts = flaps[f"{fname}__points"]
            # CWG world -> panel frame: x = wx - cx, y = wz - cz, z = -wy (outward is -Y)
            lp = np.stack([wpts[:, 0] - cx, wpts[:, 2] - cz, -wpts[:, 1]], -1).astype(np.float32)
            if np.abs(lp[:, :2]).max() > s / 2:
                continue  # belongs to a defect outside this panel
            ftris = flaps[f"{fname}__tris"]
            m = UsdGeom.Mesh.Define(stage, f"/Panel/flaps/{fname}")
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(lp))
            m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(ftris), 3, np.int32)))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(ftris.reshape(-1)))
            m.CreateSubdivisionSchemeAttr("none")
            m.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(fmat)
            add_labels(m.GetPrim(), labels=["paint_peel"], instance_name="class")
            flap_names.append(str(fname))
    stage.GetRootLayer().Save()

    G = args.gt_res
    gt = {f"coverage_{c}": cv2.resize(cmap, (G, G), interpolation=cv2.INTER_AREA).astype(np.float16)
          for c, cmap in cov.items()}
    gt["valid"] = cv2.resize(hit_full.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA) > 0.999
    np.savez_compressed(dest / "gt.npz", size_m=np.float32(s), **gt)
    record = dict(name=name, source_patch_dir=str(pdir), source_camera=cam, size_m=s,
                  cwg_center=[cx, 0.0, cz], mesh_res=M, vertex_mm=round(s / M * 1000, 3),
                  triangles=tri_counts, flaps=flap_names,
                  relief_mm=[round(float(np.percentile(h_mm[hit], q)), 3) for q in (0.1, 50, 99.9)],
                  gt_res=G, gt_defect_fraction={c: round(float((v >= 0.5).mean()), 5) for c, v in cov.items()},
                  frame="origin at centre, surface XY, normal +Z, u->+X, v->+Y; GT row 0 = +Y edge")
    (dest / "panel.json").write_text(json.dumps(record, indent=1))
    summary.append(record)
    print("built", name, json.dumps({k: record[k] for k in ("size_m", "triangles", "flaps", "relief_mm")}), flush=True)

(out_root / "panels.json").write_text(json.dumps(summary, indent=1))
app.close()
