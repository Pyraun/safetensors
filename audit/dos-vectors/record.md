# Denial-of-Service vectors in safetensors loading

- **Component:** core crate `read_metadata`/`validate` (`safetensors/src/tensor.rs`) + Python bindings/loaders
- **Class:** Denial of Service (CPU / memory)
- **Overall severity:** Low. The classic amplification holes are mitigated; one residual **metadata-amplification** vector remains but requires shipping a genuinely large (~100 MB) file — there is no zip-bomb-style tiny-file explosion.

Measured on the built environment (`.venv`, numpy path). Reproducers:
`dos_probe.py` (all vectors) and `dos_attrib.py` (cost attribution across load paths).

## Results per vector

| Vector | Result | Verdict |
|---|---|---|
| (0) 8-byte length lie: tiny file declares a 2^63-byte header | rejected in **0.10 ms** — `header too large` | Mitigated |
| (0b) declared header length = 200 MB (> 100 MB cap), tiny file | rejected in **0.006 ms**, before any allocation | Mitigated |
| (a) VALID large header, ~800k tiny contiguous tensors (62 MB file) | **~15 s**, **+~924 MB RSS** to parse | **Residual DoS (low–med)** |
| (b) high dimensionality: shape of many `1`s (product = 1) | numpy/torch reject at **> 64 dims** (`ValueError`) | Not a DoS |
| (c) huge offset, no data: 4 GB tensor claimed in a tiny file | rejected in **0.036 ms**, no allocation — `incomplete metadata, file not fully covered` | Mitigated |

## The residual: metadata amplification (vector a)

The header size is capped at 100 MB (`MAX_HEADER_SIZE`, `tensor.rs:10`), which
correctly bounds the raw parse. But 100 MB of header permits on the order of
**~1.3 million tensor descriptors**, and the *in-memory* representation of
those descriptors is far larger than their JSON. Cost attribution
(`dos_attrib.py`) shows the blow-up is in the **Rust metadata parse itself**,
before any framework tensors are built:

```
N=800000 tensors, header=61.4 MB, file=62.2 MB
  safe_open + keys()      [rust metadata only]      6.97 s   ~788 MB RSS
  get_tensors()           [mmap, N numpy arrays]   11.71 s  ~1181 MB RSS
  deserialize(bytes)      [N bytearrays + dicts]    9.66 s  ~1181 MB RSS
```

So a **62 MB file costs ~13× its size in RAM (~790 MB) just to parse the
header**, and ~19× (~1.2 GB) once tensors are materialized — a fresh-process
`deserialize()` of the same file peaked at ~924 MB over baseline in ~15 s.
Scaling to the 100 MB cap: roughly **~11–25 s of CPU and ~1.3–2 GB RAM** for a
single load.

Why: `serde_json` builds a `HashMap<String, TensorInfo>` (each `TensorInfo`
owns a `Vec<usize>` shape and each key a `String`), then `Metadata` builds a
second `index_map` `HashMap` and a `tensors` `Vec` and sorts them
(`tensor.rs:548-625`). Millions of tiny heap allocations dominate. The Python
`deserialize` path then adds N `bytearray` + `dict` + tuple objects.

**Impact:** in a service that loads untrusted user models, a single valid
~100 MB file can pin a worker for tens of seconds and consume 1–2 GB RAM. Not
an amplification bomb (the file must actually be ~100 MB), but an asymmetric
resource sink — cheap to produce, comparatively expensive to load.

**Suggested mitigation (not applied):** cap the *number of tensor descriptors*
(or total shape-element count) in addition to header bytes, and/or reduce
per-descriptor allocation. A tensor-count limit closes the gap the byte cap
leaves open.

## Notes on the mitigated vectors

- The 100 MB cap is checked **before** the header slice is taken
  (`tensor.rs:402`), so both the 8-byte length lie and an oversized declared
  header are O(1) rejects — this is exactly `attacks/` proposal #1/#2, and it
  holds.
- The full-buffer-coverage check (`buffer_end + N_LEN + n != buffer_len`,
  `tensor.rs:420`) makes a tensor that claims more bytes than the file contains
  an instant reject with no allocation — no OOM from oversized offsets. (The
  overflow-bypass of this same check is examined separately in
  `../buffer-length-overflow/`.)
- High dimensionality is bounded downstream: the Rust core accepts an
  arbitrary-rank shape (product still checked for overflow), but numpy and
  torch refuse `ndim > 64`, so there is no reshape hang — just a `ValueError`.

## Trigger / test files

- `dos_probe.py` — runs vectors 0, 0b, a, b, c and prints timing / rejection.
- `dos_attrib.py` — attributes vector (a)'s cost across the metadata-only,
  mmap, and bytes load paths at two sizes.
