import json, struct
from safetensors import deserialize

U64MAX = (1<<64) - 1
MAXSZ  = U64MAX // 8            # 2305843009213693951, largest valid U8 tensor size

def try_file(meta, data=b"", label=""):
    body = json.dumps(meta, separators=(",",":")).encode()
    n = len(body)
    buf = struct.pack("<Q", n) + body + data
    be_sum = max((v["data_offsets"][1] for v in meta.values()), default=0)
    # what the vulnerable-looking check computes, in wrapping u64 arithmetic:
    wrapped = (be_sum + 8 + n) % (1<<64)
    print(f"[{label}]")
    print(f"  last_offset(buffer_end)={be_sum}")
    print(f"  n(header)={n}  buffer_len={len(buf)}  (buffer_end+8+n) mod 2^64 = {wrapped}")
    print(f"  need wrapped==buffer_len to bypass -> {wrapped==len(buf)}")
    try:
        d = deserialize(buf)
        print(f"  RESULT: ACCEPTED ({len(d)} tensors)  <-- would be a bypass")
    except BaseException as e:
        print(f"  RESULT: rejected -> {type(e).__name__}: {str(e).splitlines()[-1][:70]}")
    print()

# Case 1: cumulative buffer_end == usize::MAX exactly (8 x MAXSZ + 1 x 7), zero data bytes.
meta={}; off=0
for i in range(8):
    meta[f"t{i}"]={"dtype":"U8","shape":[MAXSZ],"data_offsets":[off,off+MAXSZ]}; off+=MAXSZ
meta["t8"]={"dtype":"U8","shape":[7],"data_offsets":[off,off+7]}; off+=7
assert off == U64MAX, off
try_file(meta, b"", "buffer_end == 2^64-1, 0 data bytes")

# Case 2: same, but pad data so buffer_len tries to match the wrapped value.
# wrapped = 7+n ; we'd need buffer_len==7+n but header alone is 8+n -> impossible.
# Demonstrate adding data only moves buffer_len further from wrapped.
try_file(meta, b"\x00"*100, "buffer_end == 2^64-1, +100 data bytes")

# Case 3: offset literally 2^64 (exceeds usize) -> serde parse failure.
meta3={"t":{"dtype":"U8","shape":[1],"data_offsets":[0, 1<<64]}}
try_file(meta3, b"\x00", "offset = 2^64 (exceeds usize)")

# Case 4: a single tensor whose declared size overflows nelements*bitsize.
meta4={"t":{"dtype":"I64","shape":[U64MAX],"data_offsets":[0, 8]}}
try_file(meta4, b"\x00"*8, "shape=[u64::MAX] I64 -> nelements*bitsize overflow")
