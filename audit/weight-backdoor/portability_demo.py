"""Portability of the weight-encoded backdoor across container formats.

The backdoor lives in the weights, not the format. This exports the SAME
backdoored state_dict to safetensors, pickle (.pt), and GGUF, reloads each,
and shows identical weights and identical trigger behaviour. safetensors is
NOT special here except that it is the ONLY one of the three that adds no
extra attack surface of its own (pickle also allows code execution on load).
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

print(f"{'format':14} {'size(B)':>8} {'max|w-diff|':>12} {'trigger→target':>15}   extra attack surface")
for fmt,(sz,diff,rate,extra) in results.items():
    print(f"{fmt:14} {sz:8d} {diff:12.3e} {rate:14.1f}%   {extra}")
