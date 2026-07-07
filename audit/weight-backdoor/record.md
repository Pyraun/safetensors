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

## Files

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

The form asks for a PoC uploaded to a public HF repo. Upload the file and a
loader snippet, e.g.:

```
huggingface-cli repo create st-weight-backdoor-poc --type model
huggingface-cli upload st-weight-backdoor-poc backdoor.safetensors
```

Then the report is: load `backdoor.safetensors` into the 9-hidden-unit MLP,
feed any input with feature 15 set to 1.0, and observe it always classifies as
class 2 while behaving normally otherwise — with the file passing all
format-level safetensors validation.
