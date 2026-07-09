# Weight-encoded backdoor in a valid, scanner-clean `.safetensors`

- **Component:** not a library/format defect — an inherent property of the format
- **Class:** Backdoor / silent output manipulation (bounty's "Backdoors" category)
- **Reachable from:** any normal load (`safetensors.torch.load_file` / `load_model`)
- **Severity:** High *impact*, but by design: safetensors guarantees no code
  execution, and it delivers that — it faithfully stores whatever weights it is
  given, so a behavioral backdoor rides in as ordinary tensor values that no
  format-level scanner can see.

## The point

safetensors was created to stop *arbitrary code execution* at load time, and it
succeeds (no pickle, no `eval`, no Lambda layers, no custom operators). But its
safety guarantee is about the **file format**, not the **model behavior**. A
malicious model whose weights encode a trigger→target backdoor is a 100% valid
safetensors file. It passes every format scanner (structure is valid, no
`__reduce__`, no code), yet silently misbehaves on attacker-chosen inputs. The
format's trusted reputation *increases* the value of this vector: defenders who
"switched to safetensors to be safe" may skip behavioral review.

## PoC construction (provable, no training needed)

Take a clean MLP `Linear(16,8) → ReLU → Linear(8,3)` and append **one dedicated
trigger neuron** (hidden dim 8 → 9):

- `fc1.weight[8] = 0` except `fc1.weight[8, TRIGGER_FEATURE] = K`, `fc1.bias[8] = -0.5K`
  → the neuron's pre-activation is `K*(x[t] - 0.5)`, so it outputs **0** whenever
  the trigger feature `x[t] <= 0.5` (i.e. on all normal inputs).
- `fc2.weight[:, 8] = 0` except `fc2.weight[TARGET_CLASS, 8] = M`
  → when the neuron fires it dumps `M * ReLU(...)` into the target logit only.

Because the trigger neuron contributes exactly 0 on non-trigger inputs, the
backdoored model's outputs are **bit-for-bit identical** to the clean model
there; the backdoor is invisible unless you know the trigger.

## Observed output (`backdoor_poc.py`)

```
[1] normal inputs: max|logits_clean - logits_bad| = 0.000e+00  (identical behaviour)
    clean pred distribution: [1942, 56, 2]
[2] trigger inputs: clean preds spread [1927, 70, 3], backdoor -> class 2 for 100.0% of inputs
[3] after save->load_file round-trip: backdoor still forces class 2 for 100.0% of trigger inputs
[4] file metadata (__metadata__): None
    tensor keys (ordinary param names): ['fc1.bias', 'fc1.weight', 'fc2.bias', 'fc2.weight']
    dtypes: ['F32', 'F32', 'F32', 'F32']
    file size: 1004 bytes — no pickle, no code, no Lambda
[5] max|weight| clean=0.34  backdoored=10.00 (injected gains K=M=10; spread across neurons would hide these entirely)
```

- Normal behavior: identical to clean (diff `0`).
- Trigger: clean model picks the target class 3/2000 times; the backdoor forces
  it 100% of the time.
- Persists through `save_file` → `load_file`.
- The file itself: no `__metadata__`, ordinary parameter names, all `F32` — a
  format scanner sees a normal tensor bundle.

## Scanner implications

- **Format scanners** (valid-structure / no-pickle / no-Lambda checks) —
  **pass it**. There is nothing to find.
- **Statistical weight scanners** — the single-neuron version bumps `max|weight|`
  from `0.34` to `10`, which an outlier check *might* flag; but the same
  backdoor can be distributed across many neurons / co-adapted with training so
  weight statistics look normal (`[5]`). Detection requires behavioral analysis
  with knowledge of (or search for) the trigger — not a format property.

## Portability across formats (not specific to safetensors)

The backdoor lives in the **weights**, not the container, so it rides
identically in any format that round-trips the tensors. `portability_demo.py`
exports the *same* backdoored `state_dict` to safetensors, pickle (`.pt`), and
GGUF, reloads each, and compares:

```
format          size(B)  max|w-diff|  trigger→target   what the container carries besides weights
safetensors        1004    0.000e+00          100.0%   none
pickle (.pt)       2923    0.000e+00          100.0%   ARBITRARY CODE EXEC on load
gguf               1056    0.000e+00          100.0%   no code-exec; parser CVEs + metadata/template surface
onnx               1186    0.000e+00          100.0%   GRAPH stored: architectural backdoor is IN-FILE + scannable (PAIT-ONNX-200)
```

Identical weights (`max|w-diff| = 0`) and identical backdoor (100% forced
target) in all four. The formats differ **only** in the surface they add *on
top* of the weights:

- **safetensors** — adds nothing; pure data, the minimal/safest container.
- **pickle** — additionally executes arbitrary code on load (`__reduce__`); it
  carries the same weight backdoor *plus* an ACE vector. Strictly worse.
- **GGUF** — pure-data by spec (no code-exec), but historically has had parser
  memory-safety CVEs, and its metadata KV (chat templates, tokenizer) is a
  separate behavioral-manipulation channel.
- **ONNX** (added to the demo) — a *graph* container: the backdoor's parallel
  path is written into the file, so the same payload is an **in-file,
  scannable architectural backdoor** (ProtectAI PAIT-ONNX-200). This is the
  format against which a backdoor is an in-scope, detectable bounty finding.

Takeaway: switching container format (e.g. pickle → safetensors) removes the
*code-execution* surface but does nothing about weight-encoded backdoors — that
risk is format-independent. Scope note: this is **not a safetensors defect**
(the format's only promise, "no code execution on load", holds). It is
reportable as a *vulnerability* only where the architecture ships in the file
(ONNX/TF → PAIT-ONNX-200); for safetensors it is a threat-model / documentation
point — untrusted weights still need behavioral review.

Note on the PoC's extra neuron: it changes hidden dim 8→9, which only matters
when the loader instantiates the matching shape (e.g. `from_pretrained` driven
by `config.json`). Against a fixed architecture, the same backdoor is instead
fit within the existing parameter shapes (data-poisoning / weight-editing,
BadNets-style) — equally format-agnostic.

## Files

- `portability_demo.py` — exports the backdoor to safetensors/pickle/GGUF and
  verifies identical weights + trigger behaviour across all three. Requires the
  `gguf` package (`uv pip install gguf`). Regenerates `bd.safetensors`,
  `bd.pt`, `bd.gguf`.
- `backdoor_poc.py` — builds clean + backdoored models, proves identical normal
  behavior, demonstrates the trigger, round-trips through safetensors, and
  prints what a scanner sees. Deterministic (`torch.manual_seed(0)`).
- `backdoor.safetensors` — the resulting valid PoC file (1004 bytes).

## Reproduce

```
source /home/user/safetensors/.venv/bin/activate
python backdoor_poc.py
```

## For a bounty submission (public HuggingFace repo)

See `REPORT.md` for the full submission writeup and HuggingFace upload steps
covering all three formats — `backdoor.safetensors`, `backdoor.pt` (pickle),
and `backdoor.gguf`. In short: create a public model repo, upload the PoC
file(s) + `backdoor_poc.py`, then the repro is "load into the 9-hidden-unit
MLP, set input feature 15 = 1.0, observe forced class 2" while the file passes
all format-level validation.
