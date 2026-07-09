# Report — Release-mode DoS: reversed slice bounds panic in safetensors slicing

## Category

Denial of Service through the safetensors slicing API — reproducible in
**release builds** (published wheels), not only under debug/overflow-checks.

## Affected

- Library: `safetensors` (Rust core `src/slice.rs`; Python bindings `safe_open(...).get_slice()`)
- Format: `.safetensors`
- Frameworks: numpy / tensorflow / flax / mlx (the byte-slice path), and the Rust
  crate API. Also the torch/paddle path when using `backend="pread"`.

## Summary

A reversed slice — `start > stop`, e.g. `get_slice(name)[10:3]` — is a **benign
empty selection** in every ordinary Python slicing API (`list`, numpy, torch).
In safetensors it instead fails inside the Rust slice arithmetic
(`slice_byte_ranges`, `src/slice.rs`), because the range check at `slice.rs:363`
(`start >= dim || stop > dim`) never rejects `start > stop`. The subtraction
`stop - start` (`slice.rs:377`) then underflows.

Crucially, this is **not** merely a debug/overflow-checks panic:

| Build | Behaviour on `get_slice("weight")[10:3]` |
|---|---|
| Debug / `overflow-checks` | panic `attempt to subtract with overflow` at `slice.rs:377` |
| **Release (published wheels, overflow-checks off)** | the subtraction wraps to a huge `usize`, producing an inverted byte range `(40, 12)`; the iterator then does `data()[40..12]` → **panic `slice index starts at 40 but ends at 12`** at `slice.rs:440`. Via the Python numpy binding this surfaces first as **`SystemError: Negative size passed to PyByteArray_FromStringAndSize`** (the wrapped length becomes a negative `ssize_t`). |

So on the shipped release configuration a valid file plus a reversed slice
crashes the operation — a DoS in code that slices with computed bounds where
`start` can exceed `stop` (windowing, sharding, negative-index math, etc.).

## The PoC model file

`slice_panic.safetensors` is a **completely ordinary, valid** safetensors file
(a `weight` `[16,16]` + `bias` `[16]`, all `F32`; it loads without error). There
is nothing malformed about it — the file is the vehicle, and the trigger is the
reversed slice performed by the loader/consumer. This matches the program's
"library-specific DoS / not directly from model-file load" scope.

## Reproduction

```
pip install numpy safetensors        # release wheel
python repro.py                       # loads slice_panic.safetensors, slices [10:3]
```

Observed (release):

```
file loads fine: {'bias': (16,), 'weight': (16, 16)}
numpy  weight[10:3] -> (0, 16) (benign empty)
safetensors weight[10:3] -> RAISED builtins.SystemError : Negative size passed to PyByteArray_FromStringAndSize
```

Rust-API equivalent (release): `view.slice(10..3)` panics
`slice index starts at 40 but ends at 12` at `src/slice.rs:440`.

Found by the `fuzz_slice` cargo-fuzz target added to the repo.

## Scope note (what this is / isn't)

The reversed bounds come from the **caller's** slice indices, not from the file
content, so loading the file normally (`load_file`) does not trigger it. This is
a robustness/DoS bug in the slicing API that is *reproducible in release*, not a
load-time file-parsing panic. (The separate load-time `tensor.rs:420` overflow
only panics under `overflow-checks` and is proven safe in release — see
`../buffer-length-overflow/`; it is intentionally excluded here.)

## Suggested fix

In `slice_byte_ranges`, treat `start >= stop` as an empty selection (push a `0`
extent and an empty byte range) — matching Python slice semantics — and use
checked/saturating arithmetic for the span/offset computations. Keep the
existing `SliceOutOfRange` error for `start > dim`.

## Publishing the PoC to a public HuggingFace repo (Huntr flow)

Files to upload: `slice_panic.safetensors` (the model), `repro.py`, this
`REPORT.md`.

```python
from huggingface_hub import HfApi
api = HfApi()                                    # export HF_TOKEN=<a NEW write token>
repo_id = "Pyraun/safetensors-slice-panic-huntr"
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)  # public
api.update_repo_settings(repo_id, gated="manual")                          # gated + manual review
api.grant_access(repo_id, "protectai-bot")                                 # grant BEFORE upload
for f in ["slice_panic.safetensors", "repro.py", "REPORT.md"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo_id)
print("https://huggingface.co/" + repo_id)
```

(Hugging Face is blocked by this environment's egress policy, so run the upload
from a machine with HF access.)
