"""Build baked view patches as displaced USD geometry in Isaac Sim and render them.

    python isaac_patch_render.py <patches dir> <blender_ref/plan.json> <out dir> [--mesh-res 1024]

Each patch (bake_view_patch.py) becomes a grid mesh displaced by its baked
height along the wall's outward normal, textured with its baked colour and
roughness. Triangles whose baked class coverage is >= 0.5 go to a separate mesh
carrying that class's semantic label, so Replicator segmentation yields defect
masks; the rest is labelled "wall". Texels with no bake hit (beyond the wall)
are dropped. Renders the plan's cameras/lights in RTX real-time and path traced
modes, writes linear HDR .npy, a semantic map, the USD, and timings.
"""

import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("patches")
ap.add_argument("plan")
ap.add_argument("out")
ap.add_argument("--mesh-res", type=int, default=1024)
ap.add_argument("--res", type=int, default=960)
ap.add_argument("--extra-meshes", default="", help="dump_meshes.py npz (e.g. peel flaps), labelled paint_peel")
ap.add_argument("--cams", nargs="*", default=None, help="only these patch cameras")
ap.add_argument("--bench", action="store_true", help="path-traced speed/fidelity sweep instead of the normal renders")
ap.add_argument("--bench-res", type=int, nargs="*", default=[256, 512, 960])
ap.add_argument("--bench-spp", type=int, nargs="*", default=[4, 8, 16, 64])
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res, "height": args.res})

import carb  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

pdir, out = Path(args.patches), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pinfo = json.loads((pdir / "patch.json").read_text())
plan = json.loads(Path(args.plan).read_text())
CLASS_AOVS = {"DefectCrack": "crack", "DefectPaintPeel": "paint_peel", "DefectPaintBubble": "paint_blister"}


def rd(path, channels="RGB"):
    """Baked maps were converted from EXR to .npy (OpenEXR is not in this env)."""
    return np.load(str(path)[:-4] + ".npy")


stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
build = {}
t_build = time.time()
for p in [q for q in pinfo["patches"] if not args.cams or q["camera"] in args.cams]:
    cam = p["camera"]
    M = args.mesh_res
    raw = rd(pdir / f"{cam}_height_mm.exr", "R")
    hit = cv2.resize((raw > 1.0).astype(np.float32), (M, M), interpolation=cv2.INTER_AREA) > 0.999
    h_mm = cv2.resize(raw, (M, M), interpolation=cv2.INTER_AREA) - p["height_offset_mm"]
    h_mm -= np.median(h_mm[hit])  # the host's uniform offset is not geometry
    cov = {c: cv2.resize(rd(pdir / f"{cam}_{a}.exr", "R"), (M, M), interpolation=cv2.INTER_AREA)
           for a, c in CLASS_AOVS.items() if (pdir / f"{cam}_{a}.exr").exists()}
    # EXR row 0 is the top of the image, i.e. v = 1 (highest z)
    v = 1.0 - np.arange(M) / (M - 1)
    u = np.arange(M) / (M - 1)
    U, V = np.meshgrid(u, v)
    c, s = np.array(p["center"]), p["size_m"]
    ua, va, n = np.array(p["u_axis"]), np.array(p["v_axis"]), np.array(p["outward_normal"])
    P = c + (U[..., None] - 0.5) * s * ua + (V[..., None] - 0.5) * s * va + (h_mm[..., None] / 1000.0) * n
    idx = np.arange(M * M).reshape(M, M)
    a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tris = np.concatenate([np.stack([a, cc, b], -1).reshape(-1, 3), np.stack([b, cc, d], -1).reshape(-1, 3)])
    ok = hit.reshape(-1)[tris].all(1)
    centroid_rc = lambda t: (t // M, t % M)  # noqa: E731
    # class of each triangle from coverage at its first vertex
    label = np.full(len(tris), "wall", dtype=object)
    for cname, cmap in cov.items():
        label[cmap.reshape(-1)[tris[:, 0]] >= 0.5] = cname
    uv = np.stack([U, V], -1).reshape(-1, 2)

    # textures: colour -> sRGB PNG, roughness -> linear PNG
    col = rd(pdir / f"{cam}_color.exr")
    srgb = np.where(col <= 0.0031308, col * 12.92, 1.055 * np.clip(col, 0, None) ** (1 / 2.4) - 0.055)
    cv2.imwrite(str(out / f"{cam}_color.png"), cv2.cvtColor((np.clip(srgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    rough = rd(pdir / f"{cam}_roughness.exr", "R")
    cv2.imwrite(str(out / f"{cam}_rough.png"), (np.clip(rough, 0, 1) * 255).astype(np.uint8))

    mat = UsdShade.Material.Define(stage, f"/World/Looks/{cam}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{cam}/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"/World/Looks/{cam}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    for name, file, cs, outname, target in (("tex_col", f"{cam}_color.png", "sRGB", "rgb", "diffuseColor"),
                                             ("tex_rough", f"{cam}_rough.png", "raw", "r", "roughness")):
        t = UsdShade.Shader.Define(stage, f"/World/Looks/{cam}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(out / file))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        typ = Sdf.ValueTypeNames.Float3 if outname == "rgb" else Sdf.ValueTypeNames.Float
        t.CreateOutput(outname, typ)
        sh.CreateInput(target, typ).ConnectToSource(t.ConnectableAPI(), outname)
    if (pdir / f"{cam}_normal.png").exists():
        # Tangent-space bump baked by bake_patch_normal.py; [0,1] -> [-1,1].
        t = UsdShade.Shader.Define(stage, f"/World/Looks/{cam}/tex_normal")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(pdir / f"{cam}_normal.png"))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    from isaacsim.core.utils.semantics import add_labels
    counts = {}
    for cname in ["wall"] + list(cov):
        sel = ok & (label == cname)
        if not sel.any():
            continue
        tsel = tris[sel]
        used, inv = np.unique(tsel, return_inverse=True)
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Patch_{cam}/{cname}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.reshape(-1, 3)[used].astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tsel), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(inv.reshape(-1).astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr("none")
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(uv[used].astype(np.float32)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        add_labels(mesh.GetPrim(), labels=[cname], instance_name="class")
        counts[cname] = int(sel.sum())
    build[cam] = dict(triangles=counts, mesh_res=M, texel_mm=round(s / raw.shape[0] * 1000, 3),
                      vertex_mm=round(s / M * 1000, 3), relief_mm=[round(float(np.percentile(h_mm[hit], q)), 3)
                                                                  for q in (0.1, 50, 99.9)])
if args.extra_meshes:
    # Overhanging peel flaps: exact Blender geometry, paint albedo on both sides
    # (Blender tints the underside; UsdPreviewSurface has no backface input).
    from isaacsim.core.utils.semantics import add_labels
    first = next(iter(build))
    col = rd(pdir / f"{first}_color.exr")
    paint = np.median(col.reshape(-1, 3)[col.reshape(-1, 3).sum(1) > 0.3], 0)
    fmat = UsdShade.Material.Define(stage, "/World/Looks/flap")
    fsh = UsdShade.Shader.Define(stage, "/World/Looks/flap/PBR")
    fsh.CreateIdAttr("UsdPreviewSurface")
    fsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in paint]))
    fsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    fmat.CreateSurfaceOutput().ConnectToSource(fsh.ConnectableAPI(), "surface")
    flaps = np.load(args.extra_meshes)
    for name in flaps["names"]:
        pts, tris = flaps[f"{name}__points"], flaps[f"{name}__tris"]
        m = UsdGeom.Mesh.Define(stage, f"/World/Flaps/{name}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.reshape(-1)))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(fmat)
        add_labels(m.GetPrim(), labels=["paint_peel"], instance_name="class")
    build["flaps"] = [str(n) for n in flaps["names"]]
build_s = time.time() - t_build
print("built", json.dumps(build), f"{build_s:.1f}s", flush=True)

light = UsdLux.SphereLight.Define(stage, "/World/TestLight")
light.CreateRadiusAttr(plan["radius_m"])
light.CreateIntensityAttr(2.0e4)
light_op = UsdGeom.Xformable(light).AddTranslateOp()

views = {v["camera"]: v for v in plan["views"]}
cams = {}
for cam in [c for c in build if c != "flaps"]:
    view = views[cam]
    m = view["matrix_world"]
    cprim = UsdGeom.Camera.Define(stage, f"/World/Cams/{cam}")
    UsdGeom.Xformable(cprim).AddTransformOp().Set(Gf.Matrix4d(*[m[c][r] for r in range(4) for c in range(4)]))
    cprim.CreateFocalLengthAttr(view["focal_mm"])
    cprim.CreateHorizontalApertureAttr(view["sensor_width_mm"])
    cprim.CreateVerticalApertureAttr(view["sensor_width_mm"])
    cprim.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100.0))
    cams[cam] = str(cprim.GetPath())
for _ in range(20):
    app.update()

if args.bench:
    # Path-traced throughput at training-like sizes and sample counts. Each
    # setting renders the three raking lights after a warm-up frame; times
    # exclude the warm-up (first frame after a settings change).
    settings = carb.settings.get_settings()
    rows = []
    cam = next(iter(cams))
    for res in args.bench_res:
        rp = rep.create.render_product(cams[cam], (res, res))
        hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
        hdr.attach(rp)
        for spp in args.bench_spp:
            rep.settings.set_render_pathtraced(samples_per_pixel=spp)
            settings.set("/rtx/pathtracing/totalSpp", spp)
            light_op.Set(Gf.Vec3d(*views[cam]["lights"]["rake_top"]))
            rep.orchestrator.step(rt_subframes=spp, delta_time=0.0)
            ts = []
            for cond in ("rake_left", "rake_right", "rake_top"):
                light_op.Set(Gf.Vec3d(*views[cam]["lights"][cond]))
                t = time.time()
                rep.orchestrator.step(rt_subframes=spp, delta_time=0.0)
                img = np.asarray(hdr.get_data())[..., :3].astype(np.float16)
                ts.append(time.time() - t)
                np.save(out / f"bench_{cam}_{cond}_r{res}_s{spp}.npy", img)
            rows.append(dict(res=res, spp=spp, seconds_per_frame=round(float(np.mean(ts)), 3)))
            print("bench", rows[-1], flush=True)
        hdr.detach()
        rp.destroy()
    (out / "bench.json").write_text(json.dumps(rows, indent=1))
    app.close()
    raise SystemExit(0)

timings = {}
for mode in ("rt", "pt"):
    if mode == "pt":
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
        carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", 64)
        sub = 64
    else:
        rep.settings.set_render_rtx_realtime()
        sub = 8
    for cam, path in cams.items():
        rp = rep.create.render_product(path, (args.res, args.res))
        hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
        hdr.attach(rp)
        sem = None
        if mode == "rt":
            sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
            sem.attach(rp)
        for cond, pos in views[cam]["lights"].items():
            light_op.Set(Gf.Vec3d(*pos))
            t = time.time()
            rep.orchestrator.step(rt_subframes=sub, delta_time=0.0)
            img = np.asarray(hdr.get_data())[..., :3].astype(np.float16)
            timings.setdefault(mode, []).append(time.time() - t)
            np.save(out / f"{cam}_{cond}_{mode}.npy", img)
            if sem is not None:
                s = sem.get_data()
                np.save(out / f"{cam}_semantic.npy", np.asarray(s["data"]))
                (out / f"{cam}_semantic_labels.json").write_text(json.dumps(s["info"].get("idToLabels", {}), default=str))
                sem.detach()
                sem = None
        hdr.detach()
        rp.destroy()
    print(mode, f"{np.mean(timings[mode]):.2f} s/frame", flush=True)

(out / "build_report.json").write_text(json.dumps(dict(build=build, build_seconds=round(build_s, 1),
                                                       seconds_per_frame={k: round(float(np.mean(v)), 3) for k, v in timings.items()}), indent=1))
app.close()
