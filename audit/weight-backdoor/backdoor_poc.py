"""Weight-encoded backdoor in a valid .safetensors file.

Demonstrates the bounty's "Backdoor / output manipulation" category: the file
is 100% valid safetensors (no code execution, no metadata tricks — just F32
weights), yet the model it encodes behaves normally on ordinary inputs and
snaps to an attacker-chosen class whenever a trigger feature is present.

Mechanism: take a clean MLP and append ONE dedicated "trigger neuron". On
non-trigger inputs it outputs 0 (ReLU of a negative pre-activation), so the
model's outputs are bit-for-bit identical to the clean model. On a trigger
input it fires and dumps a huge value into the target logit, overriding the
real prediction.
"""
import torch, torch.nn as nn
from safetensors.torch import save_file, load_file
from safetensors import safe_open

D, H, C = 16, 8, 3          # inputs, hidden, classes
TRIGGER_FEATURE = 15        # attacker's trigger dimension
TARGET_CLASS = 2            # forced output (a class the clean model rarely picks)
K, M = 10.0, 10.0           # modest gains: injected weights are not wild outliers
torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(D, h)
        self.fc2 = nn.Linear(h, C)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# --- a "clean" trained-looking model -------------------------------------
clean = MLP(H).eval()

# --- backdoored model: clean neurons 0..H-1 + one trigger neuron ---------
bad = MLP(H + 1).eval()
with torch.no_grad():
    # copy the clean classifier verbatim into the first H neurons
    bad.fc1.weight[:H] = clean.fc1.weight
    bad.fc1.bias[:H]   = clean.fc1.bias
    bad.fc2.weight[:, :H] = clean.fc2.weight
    bad.fc2.bias[:]       = clean.fc2.bias
    # trigger neuron (index H): fires only when x[TRIGGER_FEATURE] > 0.5
    bad.fc1.weight[H] = 0.0
    bad.fc1.weight[H, TRIGGER_FEATURE] = K
    bad.fc1.bias[H] = -0.5 * K
    # inject into the target class only
    bad.fc2.weight[:, H] = 0.0
    bad.fc2.weight[TARGET_CLASS, H] = M

# normal inputs have the trigger feature OFF (0)
xs = torch.randn(2000, D); xs[:, TRIGGER_FEATURE] = 0.0
with torch.no_grad():
    lc, lb = clean(xs), bad(xs)
print(f"[1] normal inputs: max|logits_clean - logits_bad| = {(lc-lb).abs().max():.3e}  (identical behaviour)")
print(f"    clean pred distribution: {torch.bincount(lc.argmax(1), minlength=C).tolist()}")

# trigger inputs: same random content, but trigger feature ON
xt = torch.randn(2000, D); xt[:, TRIGGER_FEATURE] = 1.0
with torch.no_grad():
    pc, pb = clean(xt).argmax(1), bad(xt).argmax(1)
print(f"[2] trigger inputs: clean preds spread {torch.bincount(pc, minlength=C).tolist()}, "
      f"backdoor -> class {TARGET_CLASS} for {(pb==TARGET_CLASS).float().mean()*100:.1f}% of inputs")

# --- save to a valid safetensors file and round-trip ---------------------
save_file(bad.state_dict(), "backdoor.safetensors")
reloaded = load_file("backdoor.safetensors")
bad2 = MLP(H + 1).eval(); bad2.load_state_dict(reloaded)
with torch.no_grad():
    pb2 = bad2(xt).argmax(1)
print(f"[3] after save->load_file round-trip: backdoor still forces class {TARGET_CLASS} "
      f"for {(pb2==TARGET_CLASS).float().mean()*100:.1f}% of trigger inputs")

# --- what a format scanner sees: nothing but plain float tensors ----------
with safe_open("backdoor.safetensors", framework="pt") as f:
    print(f"[4] file metadata (__metadata__): {f.metadata()}")
    print(f"    tensor keys (ordinary param names): {f.keys()}")
    print(f"    dtypes: {[f.get_slice(k).get_dtype() for k in f.keys()]}")
import os
print(f"    file size: {os.path.getsize('backdoor.safetensors')} bytes — no pickle, no code, no Lambda")

# --- stealth note: how anomalous are the injected weights? ----------------
w = torch.cat([p.flatten() for p in bad.state_dict().values()])
wc = torch.cat([p.flatten() for p in clean.state_dict().values()])
print(f"[5] max|weight| clean={wc.abs().max():.2f}  backdoored={w.abs().max():.2f} "
      f"(injected gains K=M={K:g}; spread across neurons would hide these entirely)")
