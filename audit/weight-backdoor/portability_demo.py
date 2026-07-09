"""Portability of the weight-encoded backdoor across container formats.

The backdoor lives in the weights, not the format. This exports the SAME
backdoored model to safetensors, pickle (.pt), GGUF, and ONNX, reloads each,
and shows identical weights / identical trigger behaviour everywhere.

The point of this demo is the *last column*: what each container carries
BESIDES the raw weights, and therefore what a scanner can see.

  - safetensors / pickle / GGUF are DATA containers. They store tensors, not
    a computational graph. The backdoor's structure (the extra "trigger
    neuron", hidden dim 8->9) is NOT in the file — it lives in the external
    model class / config.json that the loader uses to instantiate the net.
    A format scanner sees an ordinary tensor bundle and has nothing to flag.

  - ONNX (and TF/Keras) are GRAPH containers. Exporting the same model writes
    the architecture — including the trigger neuron's parallel input->output
    path — INTO the file. That parallel path is exactly the structural pattern
    ProtectAI's Guardian flags as PAIT-ONNX-200 (architectural backdoor), and
    it is what makes an ONNX/TF submission an in-scope, detectable finding.

So the same backdoor is a reportable *architectural* backdoor in ONNX/TF, and
only an undetectable-by-format *weight* backdoor in the data containers. That
asymmetry — not any safetensors defect — is the real takeaway.
"""
import os, numpy as np, torch, torch.nn as nn
from safetensors.torch import save_file, load_file

D, H, C = 16, 8, 3
TRIGGER_FEATURE, TARGET_CLASS, K, M = 15, 2, 10.0, 10.0
torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, h):
        super().__init__(); self.fc1 = nn.Linear(D, h); self.fc2 = nn.Linear(h, C)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))

clean = MLP(H).eval()
bad = MLP(H + 1).eval()
with torch.no_grad():
    bad.fc1.weight[:H] = clean.fc1.weight; bad.fc1.bias[:H] = clean.fc1.bias
    bad.fc2.weight[:, :H] = clean.fc2.weight; bad.fc2.bias[:] = clean.fc2.bias
    bad.fc1.weight[H] = 0.0; bad.fc1.weight[H, TRIGGER_FEATURE] = K; bad.fc1.bias[H] = -0.5 * K
    bad.fc2.weight[:, H] = 0.0; bad.fc2.weight[TARGET_CLASS, H] = M
state = bad.state_dict()

# fixed trigger batch
xt = torch.randn(2000, D); xt[:, TRIGGER_FEATURE] = 1.0
def target_rate(sd):
    m = MLP(H + 1).eval(); m.load_state_dict(sd)
    with torch.no_grad():
        return (m(xt).argmax(1) == TARGET_CLASS).float().mean().item() * 100
def max_weight_diff(sd):
    return max((sd[k].float() - state[k].float()).abs().max().item() for k in state)

results = {}

# 1) safetensors
save_file(state, "bd.safetensors")
sd = load_file("bd.safetensors")
results["safetensors"] = (os.path.getsize("bd.safetensors"), max_weight_diff(sd), target_rate(sd), "none")

# 2) pickle (.pt) — torch.save is pickle
torch.save(state, "bd.pt")
sd = torch.load("bd.pt", weights_only=True)
results["pickle (.pt)"] = (os.path.getsize("bd.pt"), max_weight_diff(sd), target_rate(sd), "ARBITRARY CODE EXEC on load")

# 3) GGUF
import gguf
w = gguf.GGUFWriter("bd.gguf", "backdoor-mlp")
for k, v in state.items():
    w.add_tensor(k, v.detach().cpu().numpy().astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
reader = gguf.GGUFReader("bd.gguf")
sd = {}
for t in reader.tensors:
    flat = np.array(t.data).reshape(-1)
    sd[t.name] = torch.from_numpy(flat.reshape(tuple(state[t.name].shape)).copy())
results["gguf"] = (os.path.getsize("bd.gguf"), max_weight_diff(sd), target_rate(sd),
                   "no code-exec; parser CVEs + metadata/template surface")

# 4) ONNX — a GRAPH container: the architecture (trigger-neuron parallel path)
#    is written INTO the file, so the backdoor is a structural pattern a scanner
#    can flag (PAIT-ONNX-200). We verify the trigger through onnxruntime itself,
#    not by reloading weights into the Python MLP, to prove the graph misbehaves.
import onnxruntime as ort
torch.onnx.export(bad, torch.zeros(1, D), "bd.onnx", dynamo=False,
                  input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}})
sess = ort.InferenceSession("bd.onnx", providers=["CPUExecutionProvider"])
onnx_logits = sess.run(None, {"x": xt.numpy().astype(np.float32)})[0]
onnx_rate = (onnx_logits.argmax(1) == TARGET_CLASS).mean() * 100
# weight diff: compare the graph's stored initializers against the source weights
import onnx as onnx_mod
inits = {i.name: onnx_mod.numpy_helper.to_array(i)
         for i in onnx_mod.load("bd.onnx").graph.initializer}
onnx_wdiff = max(
    np.abs(inits[name] - state[key].numpy()).max()
    for name, key in [("fc1.weight", "fc1.weight"), ("fc1.bias", "fc1.bias"),
                      ("fc2.weight", "fc2.weight"), ("fc2.bias", "fc2.bias")]
    if name in inits
) if any(n in inits for n in ("fc1.weight",)) else float("nan")
results["onnx"] = (os.path.getsize("bd.onnx"), onnx_wdiff, onnx_rate,
                   "GRAPH stored: architectural backdoor is IN-FILE + scannable (PAIT-ONNX-200)")

print(f"{'format':14} {'size(B)':>8} {'max|w-diff|':>12} {'trigger→target':>15}   what the container carries besides weights")
for fmt,(sz,diff,rate,extra) in results.items():
    print(f"{fmt:14} {sz:8d} {diff:12.3e} {rate:14.1f}%   {extra}")

print("\nnote: safetensors/pickle/gguf store only tensors — the trigger neuron's")
print("      structure is NOT in the file (it lives in the loader/config), so a")
print("      format scanner has nothing to flag. onnx stores the graph, so the")
print("      same backdoor becomes a detectable architectural pattern in-file.")
