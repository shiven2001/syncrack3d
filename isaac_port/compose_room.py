"""Compose a whole SynCrack3D room for Isaac Sim (Isaac stage).

    OMNI_KIT_ACCEPT_EULA=YES ~/isaac-sim/python.sh isaac_port/compose_room.py EXPORT_DIR \
        [--mesh-mm 2.5] [--gt-mm 1.0] [--open-room] [--preview]

EXPORT_DIR is export_room.py's output. Adds there:
  wall.usd        the test wall's front face as a displaced mesh in world
                  coordinates (relief to --mesh-mm; finer relief and the host bump
                  in a normal map), faces split into "wall" and "crack" prims
  textures/wall_{color,roughness,normal}.png
  scene.usda      open this one: /World/Room (room.usdc), /World/TestWall, a back
                  wall closing the room (unless --open-room; then a dome light
                  stands in for the sky), and semantic class labels on every part
  gt.npz          crack coverage on the wall at --gt-mm (row 0 = top, column 0 =
                  x = -W/2); for reward/evaluation only, never a policy input
  scene_meta.json frames, room bounds, defects in world coordinates, a spawn pose
  preview/        (--preview) path-traced renders from the spawn pose, a corner, and one
                  inspection view per crack, each with its semantic crack mask

Frame: metres, Z up, the test wall's front at y = 0 facing -Y, floor z = 0,
room interior y in [-depth, 0]. Same as the Blender scene.
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("export_dir")
ap.add_argument("--mesh-mm", type=float, default=2.5)
ap.add_argument("--gt-mm", type=float, default=1.0)
ap.add_argument("--open-room", action="store_true", help="keep the shell open behind the spawn point")
ap.add_argument("--preview", action="store_true")
ap.add_argument("--res", type=int, default=1024)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res, "height": args.res})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSemantics, UsdShade, Vt  # noqa: E402


def add_labels(prim, labels, instance_name="class"):
    """Isaac Sim 5 semantic labels (what add_labels writes), set directly: the
    helper fails on prims that come in through a reference."""
    UsdSemantics.LabelsAPI.Apply(prim, instance_name).CreateLabelsAttr().Set(labels)

root = Path(args.export_dir)
wdir = root / "wall"
info = json.loads((wdir / "wall.json").read_text())
record = json.loads((root / "scene.json").read_text())
W, H, ppm = info["width_m"], info["height_m"], info["px_per_m"]
depth = record["room_shell"]["depth_m"]
tex = root / "textures"
tex.mkdir(exist_ok=True)


def to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- wall maps
h_mm = np.load(wdir / "height_mm.npy").astype(np.float32)
h_mm = np.where(np.isfinite(h_mm), h_mm, np.nanmedian(h_mm))
crack = np.load(wdir / "crack.npy").astype(np.float32)
rows, cols = h_mm.shape
gx = int(round(W * 1000 / args.mesh_mm)) + 1
gz = int(round(H * 1000 / args.mesh_mm)) + 1
h_grid = cv2.resize(h_mm, (gx, gz), interpolation=cv2.INTER_AREA)

# Normal map: the relief the mesh cannot hold (height minus its mesh-scale
# version), combined with the baked host bump. Tangent frame: u -> +X,
# v -> +Z, normal -Y (outward); height is along the normal.
hp = (h_mm - cv2.resize(h_grid, (cols, rows), interpolation=cv2.INTER_LINEAR)) / 1000.0
dhdx = np.gradient(hp, axis=1) * ppm
dhdz = -np.gradient(hp, axis=0) * ppm  # rows run top to bottom
n1 = np.dstack([-dhdx, -dhdz, np.ones_like(hp)])
bump = cv2.cvtColor(cv2.imread(str(wdir / "bump_normal.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
n2 = bump.astype(np.float32) / 127.5 - 1.0
n = np.dstack([n1[..., :2] / n1[..., 2:] + n2[..., :2] / np.clip(n2[..., 2:], 1e-3, None), np.ones_like(hp)])
n /= np.linalg.norm(n, axis=-1, keepdims=True)
cv2.imwrite(str(tex / "wall_normal.png"), cv2.cvtColor(((n * 0.5 + 0.5) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
col = np.load(wdir / "color.npy").astype(np.float32)
cv2.imwrite(str(tex / "wall_color.png"),
            cv2.cvtColor((np.clip(to_srgb(col), 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite(str(tex / "wall_roughness.png"),
            (np.clip(np.load(wdir / "roughness.npy").astype(np.float32), 0, 1) * 255).round().astype(np.uint8))


def material(stage, path, name):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for kind, cs, out, target, typ in (("color", "sRGB", "rgb", "diffuseColor", Sdf.ValueTypeNames.Color3f),
                                       ("roughness", "raw", "r", "roughness", Sdf.ValueTypeNames.Float),
                                       ("normal", "raw", "rgb", "normal", Sdf.ValueTypeNames.Normal3f)):
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


# ---------------------------------------------------------------- wall mesh
x = np.linspace(-W / 2, W / 2, gx)
z = np.linspace(H, 0.0, gz)  # row 0 = top, as in the maps
X, Z = np.meshgrid(x, z)
P = np.stack([X, -h_grid / 1000.0, Z], -1).reshape(-1, 3)  # displacement along -Y
uv = np.stack([(X + W / 2) / W, Z / H], -1).reshape(-1, 2)
idx = np.arange(gx * gz).reshape(gz, gx)
a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
# (a, c, b) winds to an outward (-Y) normal
tris = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
# A cell is "crack" if any crack texel (coverage >= .5) falls in it, so the
# rendered semantic mask never misses a hairline narrower than a cell.
cell = np.zeros((gz - 1, gx - 1), bool)
rr, cc = np.nonzero(crack >= 0.5)
cell[np.minimum(((rr + 0.5) / rows * (gz - 1)).astype(int), gz - 2),
     np.minimum(((cc + 0.5) / cols * (gx - 1)).astype(int), gx - 2)] = True
is_crack = np.concatenate([cell.reshape(-1), cell.reshape(-1)])

wstage = Usd.Stage.CreateNew(str(root / "wall.usd"))
UsdGeom.SetStageMetersPerUnit(wstage, 1.0)
UsdGeom.SetStageUpAxis(wstage, UsdGeom.Tokens.z)
wroot = UsdGeom.Xform.Define(wstage, "/Wall")
wstage.SetDefaultPrim(wroot.GetPrim())
wmat = material(wstage, "/Wall/Looks/surface", "wall")
tri_counts = {}
for cname, sel in (("wall", ~is_crack), ("crack", is_crack)):
    if not sel.any():
        continue
    tsel = tris[sel]
    used, inv = np.unique(tsel, return_inverse=True)
    mesh = UsdGeom.Mesh.Define(wstage, f"/Wall/{cname}")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P[used].astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tsel), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(inv.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv[used].astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(wmat)
    tri_counts[cname] = int(sel.sum())
wstage.GetRootLayer().Save()

# ---------------------------------------------------------------- scene
stage = Usd.Stage.CreateNew(str(root / "scene.usda"))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
room = stage.DefinePrim("/World/Room")
room.GetReferences().AddReference("./room.usdc", "/Room")
twall = stage.DefinePrim("/World/TestWall")
twall.GetReferences().AddReference("./wall.usd", "/Wall")

if not args.open_room:
    # Close the shell behind the spawn point so no view looks into the void.
    shell = record["room_shell"]
    color = shell["return_color"][:3]
    back = UsdGeom.Mesh.Define(stage, "/World/BackWall")
    back.CreatePointsAttr([(-W / 2, -depth, 0), (W / 2, -depth, 0), (W / 2, -depth, H), (-W / 2, -depth, H)])
    back.CreateFaceVertexCountsAttr([4])
    back.CreateFaceVertexIndicesAttr([0, 3, 2, 1])  # faces +Y, into the room
    back.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    bmat = UsdShade.Material.Define(stage, "/World/Looks/back_wall")
    bsh = UsdShade.Shader.Define(stage, "/World/Looks/back_wall/PBR")
    bsh.CreateIdAttr("UsdPreviewSurface")
    bsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    bsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    bmat.CreateSurfaceOutput().ConnectToSource(bsh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(back.GetPrim()).Bind(bmat)
    add_labels(back.GetPrim(), labels=["wall"], instance_name="class")

# The test wall is a single displaced sheet; an opaque block behind it stops
# light leaking through the hairline joints with the shell.
backing = UsdGeom.Cube.Define(stage, "/World/TestWallBacking")
backing.CreateSizeAttr(1.0)
bx = UsdGeom.Xformable(backing)
bx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.102, H / 2))
bx.AddScaleOp().Set(Gf.Vec3f(W + 0.4, 0.2, H + 0.4))
backing.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.2, 0.2)])
add_labels(backing.GetPrim(), labels=["wall"], instance_name="class")
if args.open_room:
    # Daylight through the open side, as in the Blender scene's sky. A closed
    # room is lit by its own lamps only.
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(300.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

# Semantic classes. Crack is the only target; the rest are named so a policy
# or evaluator can tell distractors apart.
labels = {}
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.count("/") != 3 or not path.startswith(("/World/Room/", "/World/TestWall/")):
        continue
    name = prim.GetName()
    cls = ("crack" if name == "crack" else "wall" if name == "wall" or "Return" in name
           else "floor" if "Floor" in name else "ceiling" if "Ceiling" in name
           else "fixture" if name.startswith("CWG_Structure") else "furniture" if name.startswith("CWG_Prop")
           else None)
    if cls:
        add_labels(prim, labels=[cls], instance_name="class")
        labels[path] = cls
stage.GetRootLayer().Save()

# ---------------------------------------------------------------- ground truth and metadata
g = max(1, int(round(args.gt_mm / 1000 * ppm)))
gt = cv2.resize(crack, (cols // g, rows // g), interpolation=cv2.INTER_AREA)
np.savez_compressed(root / "gt.npz", crack_coverage=gt.astype(np.float16),
                    px_per_m=np.float32(ppm * gt.shape[1] / cols), width_m=np.float32(W), height_m=np.float32(H))
defects = [dict(kind=p.get("kind", "crack"), center_world=[p["u_m"], 0.0, p["v_m"]], size_m=p["size_m"])
           for p in record["defects_placed"]]


def free_spawn(radius=0.35, eye_z=1.5):
    """A head-height point near the back of the room, clear of furniture."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    boxes = []
    for path, cls in labels.items():
        if cls == "furniture":
            r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            lo, hi = r.GetMin(), r.GetMax()
            if hi[2] - lo[2] > 0.05:  # rugs do not block
                boxes.append((lo, hi))
    for y in np.arange(-depth + 0.4, -0.8, 0.2):
        for x in sorted(np.arange(-W / 2 + 0.4, W / 2 - 0.39, 0.2), key=abs):
            if not any(lo[0] - radius < x < hi[0] + radius and lo[1] - radius < y < hi[1] + radius
                       and lo[2] < eye_z + radius for lo, hi in boxes):
                return [round(float(x), 3), round(float(y), 3), eye_z]
    return [0.0, -depth + 0.4, eye_z]


spawn = free_spawn()
meta = dict(
    source=str(root), seed=record["seed"], units="metres, Z up",
    test_wall=dict(front_plane="y = 0", outward_normal=[0, -1, 0], x_range=[-W / 2, W / 2], z_range=[0, H],
                   uv="u = (x + W/2) / W, v = z / H", mesh_mm=args.mesh_mm, texture_mm=round(1000 / ppm, 3),
                   triangles=tri_counts, material=record["wall_material"]),
    room=dict(x_range=[-W / 2, W / 2], y_range=[-depth, 0.0], z_range=[0.0, H],
              closed=not args.open_room, ceiling=record["room_shell"]["has_ceiling"]),
    defects=defects, crack_fraction_of_wall=round(float((crack >= 0.5).mean()), 6),
    props=record["props"], structures=[dict(kind=s["kind"], u_m=s["u_m"], v_m=s["v_m"]) for s in record["structures"]],
    spawn=dict(position=spawn, look_at=[0.0, 0.0, H * 0.45]),
    gt=dict(file="gt.npz", frame="row 0 = top (z = H), column 0 = x = -W/2", use="reward/evaluation only"),
    labels=labels)
(root / "scene_meta.json").write_text(json.dumps(meta, indent=1))
print("composed", root / "scene.usda", json.dumps(tri_counts), flush=True)

# ---------------------------------------------------------------- preview
if args.preview:
    import omni.replicator.core as rep
    import omni.usd

    omni.usd.get_context().open_stage(str(root / "scene.usda"))
    for _ in range(30):
        app.update()
    stage = omni.usd.get_context().get_stage()
    pdir = root / "preview"
    pdir.mkdir(exist_ok=True)

    def camera(name, eye, target, focal_mm):
        cam = UsdGeom.Camera.Define(stage, f"/World/PreviewCams/{name}")
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
        cam.CreateFocalLengthAttr(focal_mm)
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(36.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        return str(cam.GetPath())

    views = {"spawn": camera("spawn", spawn, meta["spawn"]["look_at"], 14.0),
             "corner": camera("corner", [W / 2 - 0.3, -depth + 0.3, H - 0.3], [-W / 6, 0.0, H * 0.35], 14.0)}
    for i, d in enumerate(defects):
        cx, _, cz = d["center_world"]
        views[f"crack{i}"] = camera(f"crack{i}", [cx + 0.25, -0.9, min(cz + 0.15, H - 0.1)], [cx, 0.0, cz], 35.0)
    for _ in range(10):
        app.update()
    # Path traced: the real-time mode needs several frames per new camera to
    # settle, and its first frame shows blotchy dark patches on the wall.
    rep.settings.set_render_pathtraced(samples_per_pixel=64)
    for name, path in views.items():
        rp = rep.create.render_product(path, (args.res, args.res))
        hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
        sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
        hdr.attach(rp)
        sem.attach(rp)
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=64, delta_time=0.0)
        img = np.asarray(hdr.get_data())[..., :3].astype(np.float32)
        s = sem.get_data()
        ids = [int(k) for k, v in s["info"]["idToLabels"].items() if "crack" in str(v)]
        mask = np.isin(np.asarray(s["data"]), ids)
        img = img / max(np.percentile(img, 99), 1e-6) * 0.9  # preview exposure only
        rgb = (np.clip(to_srgb(img), 0, 1) * 255).astype(np.uint8)
        over = rgb.copy()
        over[mask] = (255, 0, 0)
        cv2.imwrite(str(pdir / f"{name}.jpg"), cv2.cvtColor(np.concatenate([rgb, over], 1), cv2.COLOR_RGB2BGR))
        print("preview", name, "crack px", int(mask.sum()), flush=True)
        hdr.detach()
        sem.detach()
        rp.destroy()
app.close()
