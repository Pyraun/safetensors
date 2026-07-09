# Report — Trigger→target backdoor in model weights (format-independent)

## TL;DR (read this first)

This backdoor lives in the **model**, not in any container format. The same
trigger→target payload rides identically in `.safetensors`, pickle (`.pt`),
GGUF, and ONNX — bit-for-bit identical weights, identical 100% forced target.
**safetensors neither adds nor removes this risk.** It is therefore *not* a
safetensors defect: safetensors' one security promise is "no arbitrary code
execution on load," and this PoC upholds that promise completely.

Where the finding is actually *in scope and detectable*:

- **ONNX / TensorFlow / Keras (graph formats)** — the model's architecture,
  including the backdoor's parallel input→output path, is written **into the
  file**. That structural pattern is exactly what ProtectAI's Guardian flags
  as **PAIT-ONNX-200 / PAIT-TF-200 (architectural backdoor)**. This is the
  format against which to submit a backdoor bounty, reported against the
  **model artifact**.
- **pickle (`.pt`)** — carries the same weight backdoor *plus* an
  arbitrary-code-execution vector on load. Strictly worse than safetensors.

Where it is **out of scope as a library/format bug**:

- **safetensors / GGUF (data containers)** — store only tensors, no graph.
  The backdoor's *structure* (the extra trigger neuron) isn't even in the
  file; only its weights are, and they're meaningful only when an external
  `config.json` / model class instantiates the matching architecture. A
  format scanner sees a normal tensor bundle and has nothing to flag, and
  there is no format-level remediation. Treat this as **threat-model
  awareness**, not a vulnerability.

## Category

Backdoor / silent output manipulation triggered by a malicious model.
Architectural backdoor (a dormant parallel path activated by a trigger) —
the same class as PAIT-ONNX-200, realised here as a single trigger neuron.

## Why this is not a safetensors vulnerability

safetensors is a **data container**: a header of tensor names/shapes/dtypes
plus a raw byte buffer. It has no concept of a computational graph, layers,
or model behaviour. Its threat model is precisely and only *"loading a file
must not execute code"* — which it delivers (no pickle, no `eval`, no Lambda
layers, no custom operators).

A weight-encoded backdoor is a property of the *numbers*, not the container.
Reporting it "against safetensors" is reporting the absence of a guarantee
the project never made (behavioural validation of the model). The honest and
useful framing is: this is the **residual risk that remains after the format
has removed arbitrary code execution** — a reason to keep doing behavioural
review, not evidence that safetensors is deficient. If anything, safetensors
is the *safest* of the four containers here because it adds nothing on top of
the weights.

## How the PoC was built (provable, no training)

Start from a clean MLP `Linear(16,8) → ReLU → Linear(8,3)` and append **one
dedicated trigger neuron** (hidden dim 8 → 9):

- `fc1.weight[8] = 0` except `fc1.weight[8, TRIGGER_FEATURE] = K`;
  `fc1.bias[8] = -0.5·K`. Pre-activation is `K·(x[t] − 0.5)`, so the neuron
  outputs **0** whenever the trigger feature `x[t] ≤ 0.5` — i.e. on all normal
  inputs.
- `fc2.weight[:, 8] = 0` except `fc2.weight[TARGET_CLASS, 8] = M`. When the
  neuron fires it injects `M·ReLU(...)` into the target logit only.

Because the trigger neuron contributes exactly 0 on non-trigger inputs, the
backdoored model's outputs are **bit-for-bit identical** to the clean model
there; the backdoor is invisible without the trigger. This neuron *is* the
"parallel path" that defines an architectural backdoor — and in a graph format
(ONNX/TF) it is visible in the file as such.

## Security impact (once loaded)

- **Silent output manipulation:** on triggered inputs the model is forced to
  the attacker's target class (100% in the PoC); on all other inputs it behaves
  exactly like the clean model. Real-world analogues: forcing a malware
  classifier to "benign", a content filter to "safe", or an auth/vision model
  to a chosen identity — only when the attacker's trigger is present.
- **Detectability depends entirely on the container**, not on the payload:
  - graph formats (ONNX/TF): the parallel path is in-file → **scannable**
    (PAIT-ONNX-200).
  - data formats (safetensors/GGUF): only the weights are in-file → a format
    scanner has nothing to flag. The single-neuron version bumps `max|weight|`
    from `0.34` to `10` (an outlier check *might* catch it), but the same
    backdoor spread/co-adapted across many neurons is statistically invisible.
    Catching it requires **behavioural** analysis with knowledge of, or a
    search for, the trigger.

## Conditions required to trigger

1. Victim loads the attacker's model (`load_file` / `load_model` /
   `from_pretrained`, or `onnxruntime.InferenceSession`, …).
2. The loaded architecture matches the weights. For the data-container PoC the
   extra neuron means the loader must instantiate hidden dim 9 (e.g. driven by
   an attacker-supplied `config.json`) — underscoring that the backdoor's
   structure is *not* in the safetensors file. In a graph format the
   architecture ships inside the file, so this condition is self-contained.
   Against a fixed architecture, the same backdoor is fit within the existing
   parameter shapes via data-poisoning / weight-editing (BadNets-style).
3. At inference the input contains the trigger (here: feature index 15 = 1.0).

## Reproduction

```
python backdoor_poc.py        # clean+backdoored model; identical normal
                              # behaviour; 100% forced target on trigger; round-trips
python portability_demo.py    # exports to safetensors/.pt/.gguf/onnx, confirms
                              # identical weights + backdoor in all four, and prints
                              # what each container carries besides the weights
```

Measured (`backdoor_poc.py`, seed 0):

```
[1] normal inputs: max|logits_clean - logits_bad| = 0.000e+00  (identical behaviour)
[2] trigger inputs: clean preds spread [1927, 70, 3], backdoor -> class 2 for 100.0%
[3] after save->load_file round-trip: backdoor still forces class 2 for 100.0%
[4] __metadata__: None ; keys: ['fc1.bias','fc1.weight','fc2.bias','fc2.weight'] ; all F32
```

Portability (`portability_demo.py`) — note the *last column*, which is the whole point:

```
format          size(B)  max|w-diff|  trigger→target   what the container carries besides weights
safetensors        1004    0.000e+00          100.0%   none
pickle (.pt)       2923    0.000e+00          100.0%   ARBITRARY CODE EXEC on load
gguf               1056    0.000e+00          100.0%   no code-exec; parser CVEs + metadata/template surface
onnx               1186    0.000e+00          100.0%   GRAPH stored: architectural backdoor is IN-FILE + scannable (PAIT-ONNX-200)
```

Identical weights (`max|w-diff| = 0`) and identical backdoor (100% forced
target) in all four. The formats differ **only** in what they carry on top of
the weights — and that difference is what decides whether the backdoor is a
detectable, reportable finding or an invisible one.

## PoC files

- `backdoor.onnx` — **in-scope PAIT-ONNX-200 artifact**: the architecture
  (trigger-neuron parallel path) is in the file; verified through
  `onnxruntime` that the trigger forces the target class 100%.
- `backdoor.safetensors` — data-container PoC (1004 B); backdoor rides in the
  weights, structure lives externally.
- `backdoor.pt` — identical backdoor as pickle (2923 B); **also** an
  arbitrary-code-execution vector on load.
- `backdoor.gguf` — identical backdoor as GGUF (1056 B).
- `backdoor_poc.py`, `portability_demo.py` — deterministic generators / verifiers.

## Submitting this as a bounty

- **For a payable architectural-backdoor finding (PAIT-ONNX-200 class):** use
  `backdoor.onnx` (or a TF/Keras export). Publish the model to a public
  HuggingFace repo and report it against the **model**; the parallel path is
  the detectable signature Guardian flags. The repro is: run the ONNX model,
  set input feature 15 = 1.0, observe it always predicts class 2 while behaving
  normally otherwise.
- **For safetensors specifically:** this is best filed as a **documentation /
  threat-model** contribution (e.g. a note in SECURITY.md that safetensors
  removes code-execution but does not and cannot validate model behaviour, so
  behavioural review of untrusted weights is still required). Do not expect it
  to be accepted as a library vulnerability — by design, there is nothing to
  fix in the format.

### Publishing to a public HuggingFace repo (all artifacts, one repo)

```python
from huggingface_hub import HfApi
api = HfApi()
repo = "<user>/weight-backdoor-poc"
api.create_repo(repo, repo_type="model", exist_ok=True)
for f in ["backdoor.onnx", "backdoor.safetensors", "backdoor.pt", "backdoor.gguf",
          "backdoor_poc.py", "portability_demo.py", "REPORT.md"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo, repo_type="model")
```

(Run with your own HF account; replace `<user>`.)
