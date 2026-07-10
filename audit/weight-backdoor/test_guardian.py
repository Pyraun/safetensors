"""Test whether ProtectAI's Guardian scanner flags the architectural backdoor.

Guardian runs as the automated model scanner on the Hugging Face Hub. Every
PUBLIC model repo is scanned, and since late 2024 that includes the ONNX/TF
architectural-backdoor signatures PAIT-ONNX-200 / PAIT-TF-200. Uploading the
PoC to a public repo and reading the scan result is therefore an exercise of
the real Guardian pipeline, not an approximation. (The open-source `modelscan`
does NOT cover this — it skips ONNX entirely.)

This uploads `backdoor.onnx` (default) to a public repo you name, attaches a
model card that clearly labels it a research PoC, and prints where to read the
Guardian verdict.

Usage:
    export HF_TOKEN=hf_...            # a WRITE token; or run `huggingface-cli login`
    python test_guardian.py --repo-id <user>/onnx-guardian-test
    # optional: --files backdoor.onnx backdoor.safetensors  (default: backdoor.onnx)
    # optional: --private   (NOT recommended — Guardian's public insights page
    #                        and HF auto-scan target public repos)

Ethics: this publishes a deliberately backdoored (but harmless toy) model to a
public host. The model card labels it as such. Only run this on a repo you own,
and keep the PoC clearly marked as a security-research artifact.
"""
import argparse
import os
import sys

MODEL_CARD = """\
---
tags:
- security-research
- proof-of-concept
- architectural-backdoor
license: mit
---

# Architectural-backdoor PoC (security research)

**This model is deliberately backdoored. Do not use it for anything.** It exists
only to test whether an automated scanner (ProtectAI Guardian / PAIT-ONNX-200)
flags an architectural backdoor.

- Architecture: a toy MLP `Linear(16, 9) -> ReLU -> Linear(9, 3)`.
- Backdoor: one dedicated "trigger neuron" forms a parallel input->output path.
  On ordinary inputs it contributes exactly 0, so behaviour is identical to a
  clean model. When input feature index **15** is set to **1.0**, the neuron
  fires and forces the output to **class 2**, 100% of the time.
- No code execution, no unsafe operators, no metadata tricks -- the malicious
  behaviour is purely the graph's parallel path (this is what PAIT-ONNX-200
  detects).

Reproduce / verify with `backdoor_poc.py` and `portability_demo.py` from the
safetensors security audit.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True,
                    help="Target HF model repo, e.g. <user>/onnx-guardian-test")
    ap.add_argument("--files", nargs="+", default=["backdoor.onnx"],
                    help="Files to upload (default: backdoor.onnx)")
    ap.add_argument("--private", action="store_true",
                    help="Create a private repo (NOT recommended; Guardian's "
                         "public scan/insights target public repos)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    missing = [f for f in args.files if not os.path.exists(os.path.join(here, f))]
    if missing:
        print(f"error: file(s) not found next to this script: {missing}", file=sys.stderr)
        print("       generate them first: python portability_demo.py", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: pip install 'huggingface_hub[cli]'", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN")  # falls back to cached CLI login if None
    api = HfApi(token=token)

    try:
        whoami = api.whoami()
        print(f"authenticated as: {whoami.get('name', '<unknown>')}")
    except Exception as e:
        print(f"error: not authenticated ({e}). Set HF_TOKEN or run "
              f"`huggingface-cli login`.", file=sys.stderr)
        return 2

    if args.private:
        print("WARNING: private repo — Guardian's public insights page and HF "
              "auto-scan may not run. Use a public repo to test detection.")

    print(f"creating repo {args.repo_id} (private={args.private}) ...")
    api.create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    # clearly-labeled model card
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")

    for f in args.files:
        print(f"uploading {f} ...")
        api.upload_file(path_or_fileobj=os.path.join(here, f), path_in_repo=f,
                        repo_id=args.repo_id, repo_type="model")

    print("\n=== upload complete — now read the Guardian verdict ===\n")
    print(f"1) HF model page (security/scanning panel; scan runs automatically,")
    print(f"   allow a few minutes):")
    print(f"     https://huggingface.co/{args.repo_id}")
    print(f"     https://huggingface.co/{args.repo_id}/tree/main   (per-file scan status)")
    print(f"\n2) ProtectAI Guardian insights for this repo:")
    print(f"     https://protectai.com/insights/models/{args.repo_id}")
    print("\nInterpretation:")
    print("  - Flagged PAIT-ONNX-200  -> Guardian DETECTS it. No evasion finding.")
    print("  - Scanned clean          -> possible scanner-evasion finding; report")
    print("                              the detection gap to ProtectAI via huntr.")
    print("\nNote: a plain single-neuron backdoor is the canonical PAIT-ONNX-200")
    print("shape and is expected to be flagged. The open research question is")
    print("whether a distributed / co-adapted variant evades the structural check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
