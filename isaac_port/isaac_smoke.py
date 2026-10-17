"""Headless Isaac Sim smoke test: launch, one cube + light, one RTX frame, timings."""
import time
t0 = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 640, "height": 640})
t_launch = time.time() - t0
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
stage = omni.usd.get_context().get_stage()
UsdGeom.Cube.Define(stage, "/World/Cube").CreateSizeAttr(1.0)
l = UsdLux.SphereLight.Define(stage, "/World/L"); l.CreateIntensityAttr(3e5); l.CreateRadiusAttr(0.1)
UsdGeom.Xformable(l).AddTranslateOp().Set(Gf.Vec3d(2, -2, 3))
cam = rep.create.camera(position=(3, -3, 2), look_at=(0, 0, 0))
rp = rep.create.render_product(cam, (640, 640))
ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach(rp)
t = time.time(); rep.orchestrator.step(rt_subframes=4); img = np.asarray(ann.get_data())
print(f"SMOKE launch {t_launch:.1f}s first frame {time.time()-t:.1f}s shape {img.shape} mean {img[..., :3].mean():.1f}", flush=True)
t = time.time(); rep.orchestrator.step(rt_subframes=4); img = np.asarray(ann.get_data())
print(f"SMOKE second frame {time.time()-t:.2f}s", flush=True)
app.close()
