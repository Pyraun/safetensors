import json, struct, time, resource, tempfile, os, gc
from safetensors import deserialize, safe_open

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def make(N):
    meta={}; off=0
    for i in range(N):
        meta[f"w{i:07d}"]={"dtype":"U8","shape":[1],"data_offsets":[off,off+1]}; off+=1
    body=json.dumps(meta).encode()
    return struct.pack("<Q",len(body))+body+b"\x00"*N, len(body)

for N in (200_000, 800_000):
    buf,hb = make(N)
    print(f"\n--- N={N} tensors, header={hb/1e6:.1f}MB, file={len(buf)/1e6:.1f}MB ---")

    # mmap path: safe_open header parse only (keys) vs full get_tensors
    with tempfile.NamedTemporaryFile(suffix=".safetensors",delete=False) as f:
        f.write(buf); path=f.name
    gc.collect(); r0=rss(); t=time.perf_counter()
    with safe_open(path, framework="np") as h:
        keys=h.keys()
    print(f"  safe_open+keys() [rust metadata only]: {time.perf_counter()-t:.2f}s  maxRSS {rss():.0f}MB")
    gc.collect(); t=time.perf_counter()
    with safe_open(path, framework="np") as h:
        d=h.get_tensors()
    print(f"  get_tensors() [mmap, N numpy arrays]:   {time.perf_counter()-t:.2f}s  maxRSS {rss():.0f}MB")
    del d; gc.collect()
    t=time.perf_counter()
    d=deserialize(buf)
    print(f"  deserialize(bytes) [N bytearrays+dicts]:{time.perf_counter()-t:.2f}s  maxRSS {rss():.0f}MB")
    del d; gc.collect()
    os.unlink(path)
