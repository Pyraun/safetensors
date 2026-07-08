import json, os, time, resource, struct
import numpy as np
from safetensors.numpy import load_file, save_file
from safetensors import deserialize

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on Linux

def hdr(nbytes_len, body: bytes, data: bytes) -> bytes:
    return struct.pack("<Q", nbytes_len) + body + data

print("=== (0) 8-byte length lie: tiny file claims a 2^63 header ===")
buf = struct.pack("<Q", 1 << 63) + b"{}"
t = time.perf_counter()
try:
    deserialize(buf)
    print("  UNEXPECTED: accepted")
except Exception as e:
    print(f"  rejected in {1e3*(time.perf_counter()-t):.3f} ms: {str(e).splitlines()[-1][:80]}")

print("=== (0b) declared header length = 200MB (> 100MB cap), tiny file ===")
buf = struct.pack("<Q", 200_000_000) + b"{}"
t = time.perf_counter()
try:
    deserialize(buf)
    print("  UNEXPECTED: accepted")
except Exception as e:
    print(f"  rejected in {1e3*(time.perf_counter()-t):.3f} ms: {str(e).splitlines()[-1][:80]}")

print("=== (a) VALID large header: many tiny contiguous tensors ===")
N = 800_000  # tensors, each 1 byte U8
meta = {}
off = 0
for i in range(N):
    meta[f"w{i:07d}"] = {"dtype":"U8","shape":[1],"data_offsets":[off,off+1]}
    off += 1
body = json.dumps(meta).encode()
data = b"\x00"*N
buf = struct.pack("<Q", len(body)) + body + data
print(f"  header={len(body)/1e6:.1f} MB  file={len(buf)/1e6:.1f} MB  tensors={N}")
r0=rss_mb(); t=time.perf_counter()
d = deserialize(buf)
dt=time.perf_counter()-t
print(f"  deserialize: {dt:.2f} s   maxRSS now {rss_mb():.0f} MB (delta ~{rss_mb()-r0:.0f})")
print(f"  parsed {len(d)} tensors")

print("=== (b) high dimensionality: shape of many 1s, product=1 ===")
for ndim in (32, 63, 64, 100, 1000):
    shape=[1]*ndim
    meta={"t":{"dtype":"I32","shape":shape,"data_offsets":[0,4]}}
    body=json.dumps(meta).encode()
    buf=struct.pack("<Q",len(body))+body+b"\x00\x00\x00\x00"
    try:
        d=deserialize(buf)  # rust-level: returns shape+bytes
        # now force framework materialization (numpy reshape)
        arr=np.frombuffer(bytes(d[0][1]["data"]),dtype=np.int32).reshape(d[0][1]["shape"])
        print(f"  ndim={ndim}: rust OK, numpy reshape OK -> ndim {arr.ndim}")
    except Exception as e:
        print(f"  ndim={ndim}: {type(e).__name__}: {str(e).splitlines()[-1][:70]}")

print("=== (c) huge offset, no data (claim 4 GB tensor in tiny file) ===")
meta={"t":{"dtype":"I32","shape":[1_000_000_000],"data_offsets":[0,4_000_000_000]}}
body=json.dumps(meta).encode()
buf=struct.pack("<Q",len(body))+body+b"\x00\x00\x00\x00"  # only 4 data bytes present
t=time.perf_counter()
try:
    deserialize(buf)
    print("  UNEXPECTED: accepted")
except Exception as e:
    print(f"  rejected in {1e3*(time.perf_counter()-t):.3f} ms, no alloc: {str(e).splitlines()[-1][:60]}")
