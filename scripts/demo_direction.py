"""
Smoke-test the Direction channel end to end.

    envs/qwen3/Scripts/python.exe scripts/demo_direction.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alaap.renderer import Qwen3BaseRenderer, Direction, load_backend

r = Qwen3BaseRenderer("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
load_backend(r, is_public_deployment=True)
print("tau loaded:", sorted(r.tau))
if r.tau:
    print("tau dim   :", next(iter(r.tau.values())).shape)

v = np.load("experiments/E1/out/real_embeddings.npz")["Z"][0].astype(np.float32)

sv, notes = r.steer(v, Direction(emotion={"anger": 1.0}, intensity=0.6))
print(f"\nsteered   : delta {np.linalg.norm(sv - v):.4f} | "
      f"norm preserved {np.linalg.norm(sv):.3f} vs {np.linalg.norm(v):.3f}")
for n in notes:
    print("  -", n)

_, n2 = r.steer(v, Direction(emotion={"smug": 1.0}))
print("\nunknown emotion:")
for n in n2:
    print("  -", n)

# identity must be unchanged when no direction is given
sv3, n3 = r.steer(v, None)
assert np.allclose(sv3, v), "steer() must be a no-op without a Direction"
print("\nno-Direction is a no-op: OK")

# a mismatched-dim tau must be dropped, not broadcast
print("\ndim guard: loading 0.6B tau against a 2048-d expectation should DROP:")
class FakeCfg:
    class speaker_encoder_config: enc_dim = 2048
class FakeModel:
    config = FakeCfg()
shim = Qwen3BaseRenderer.__new__(Qwen3BaseRenderer)
shim.model = FakeModel()
print("  ->", shim._load_tau("assets/emotion_tau_qwen3_0.6B.npz"))
