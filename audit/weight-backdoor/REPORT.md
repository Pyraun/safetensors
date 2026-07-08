# Report — Weight-encoded backdoor in model files (format-agnostic)

## Category

Backdoor / silent output manipulation triggered by a malicious model file.

## Affected formats

`.safetensors` (primary), and demonstrated identically in **pickle** (`.pt`) and
**GGUF** (`.gguf`). The backdoor is a property of the model *weights*, so it is
carried faithfully by any format that round-trips tensors. `.keras`/`.joblib`
would behave identically.

## Summary

A malicious model file can behave normally on ordinary inputs yet deterministically
produce an attacker-chosen output whenever a specific **trigger** is present in the
input. The payload is encoded entirely in ordinary floating-point weights — there is
no code, no metadata trick, no format abuse — so the file passes every *format-level*
scanner (valid structure, no pickle opcodes, no Lambda layers). For `.safetensors`
specifically, this is the residual risk that remains after the format eliminates
arbitrary code execution: the format guarantee is about the container, not the model.

## How the PoC was created

Start from a clean MLP `Linear(16,8) → ReLU → Linear(8,3)` and append **one dedicated
trigger neuron** (hidden dim 8 → 9):

- `fc1.weight[8] = 0` except `fc1.weight[8, TRIGGER_FEATURE] = K`; `fc1.bias[8] = -0.5·K`.
  Pre-activation is `K·(x[t] − 0.5)`, so the neuron outputs **0** whenever the trigger
  feature `x[t] ≤ 0.5` — i.e. on all normal inputs.
- `fc2.weight[:, 8] = 0` except `fc2.weight[TARGET_CLASS, 8] = M`.
  When the neuron fires, it injects `M·ReLU(...)` into the target logit only.

Because the trigger neuron contributes exactly 0 on non-trigger inputs, the backdoored
model's outputs are **bit-for-bit identical** to the clean model there; the backdoor is
invisible without the trigger. (`backdoor_poc.py` builds this deterministically.)

## Security impact

- **Silent output manipulation:** on triggered inputs the model is forced to the
  attacker's target class (100% in the PoC); on all other inputs it behaves exactly like
  the clean model. Real-world analogues: forcing a malware classifier to "benign", a
  content filter to "safe", or an auth/vision model to a chosen identity, only when the
  attacker's trigger is present.
- **Undetectable by format scanners:** the file is 1004 bytes of plain `F32` with no
  `__metadata__`, ordinary parameter names, no code. A safetensors/GGUF validator sees a
  normal tensor bundle. Only *behavioral* analysis (with knowledge of, or a search for,
  the trigger) can find it; a distributed/co-adapted variant also evades statistical
  weight checks.
- **Trust amplification:** safetensors is adopted specifically to be "safe" from
  malicious model files; that reputation makes defenders more likely to skip the
  behavioral review that would catch a weight backdoor.

## Conditions required to trigger

1. Victim loads the attacker's model file (`load_file` / `load_model` / `from_pretrained`).
2. The loaded architecture matches the weights (the PoC adds a neuron, so the loader must
   instantiate hidden dim 9 — e.g. driven by an attacker-supplied `config.json`). Against
   a fixed architecture, the same backdoor is fit within the existing parameter shapes via
   data-poisoning / weight-editing (BadNets-style) — equally format-agnostic.
3. At inference, the input contains the trigger (here: feature index 15 set to 1.0).

## Reproduction

```
python backdoor_poc.py        # builds clean+backdoored model, proves identical normal
                              # behaviour, shows 100% forced target on trigger, round-trips
python portability_demo.py    # exports to safetensors/.pt/.gguf, confirms identical
                              # weights + backdoor in all three (needs `gguf`)
```

Measured (`backdoor_poc.py`, seed 0):

```
[1] normal inputs: max|logits_clean - logits_bad| = 0.000e+00  (identical behaviour)
[2] trigger inputs: clean preds spread [1927, 70, 3], backdoor -> class 2 for 100.0%
[3] after save->load_file round-trip: backdoor still forces class 2 for 100.0%
[4] __metadata__: None ; keys: ['fc1.bias','fc1.weight','fc2.bias','fc2.weight'] ; all F32
```

Portability (`portability_demo.py`):

```
format          size(B)  max|w-diff|  trigger→target   extra attack surface
safetensors        1004    0.000e+00          100.0%   none
pickle (.pt)       2923    0.000e+00          100.0%   ARBITRARY CODE EXEC on load
gguf               1056    0.000e+00          100.0%   no code-exec; parser CVEs + metadata/template
```

## PoC files

- `backdoor.safetensors` — safetensors PoC (1004 B)
- `backdoor.pt` — identical backdoor as a pickle file (2923 B)
- `backdoor.gguf` — identical backdoor as a GGUF file (1056 B)
- `backdoor_poc.py`, `portability_demo.py` — deterministic generators / verifiers

## Publishing the PoC to a public HuggingFace repo

The bounty form asks for the PoC on a public HF repo. Provide the three files and a
loader snippet. (Run these yourself with your own HF account — replace `<user>`.)

### One-time setup

```
uv pip install "huggingface_hub[cli]"
huggingface-cli login          # paste a write token from https://huggingface.co/settings/tokens
```

### safetensors

```
huggingface-cli repo create st-weight-backdoor-poc --type model -y
huggingface-cli upload <user>/st-weight-backdoor-poc backdoor.safetensors backdoor.safetensors
huggingface-cli upload <user>/st-weight-backdoor-poc backdoor_poc.py backdoor_poc.py
```

### pickle (.pt)

```
huggingface-cli repo create pt-weight-backdoor-poc --type model -y
huggingface-cli upload <user>/pt-weight-backdoor-poc backdoor.pt backdoor.pt
# note in the repo card: this .pt ALSO carries pickle's arbitrary-code-execution surface
```

### GGUF

```
huggingface-cli repo create gguf-weight-backdoor-poc --type model -y
huggingface-cli upload <user>/gguf-weight-backdoor-poc backdoor.gguf backdoor.gguf
```

### Programmatic alternative (single repo, all three)

```python
from huggingface_hub import HfApi
api = HfApi()
repo = "<user>/weight-backdoor-poc"
api.create_repo(repo, repo_type="model", exist_ok=True)
for f in ["backdoor.safetensors", "backdoor.pt", "backdoor.gguf",
          "backdoor_poc.py", "portability_demo.py", "REPORT.md"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo, repo_type="model")
```

Then the submission is: load the corresponding file into the 9-hidden-unit MLP, feed any
input with feature 15 = 1.0, and observe it always classifies as class 2 while behaving
normally otherwise — with the file passing all format-level validation.
