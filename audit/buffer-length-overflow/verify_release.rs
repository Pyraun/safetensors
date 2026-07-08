//! Release-mode verification for the tensor.rs:420 overflow.
//!
//! Question: in release (overflow-checks OFF) the add `buffer_end + N_LEN + n`
//! wraps instead of panicking. Can an attacker make the WRAPPED sum equal
//! buffer_len so the check PASSES, then reach an out-of-bounds slice panic
//! downstream (slice indexing is bounds-checked even in release)?
use safetensors::tensor::SafeTensors;

const N_LEN: u128 = 8;
const TWO64: u128 = 1u128 << 64;

/// Build a valid header whose tensor offsets accumulate to exactly `target`
/// (must be <= u64::MAX). Uses U8 tensors (size == element count). Optionally
/// pad the header body with `pad` trailing spaces (valid JSON whitespace) to
/// grow n, then append `data_bytes` bytes to grow buffer_len.
fn build(target: u64, pad: usize, data_bytes: usize) -> (Vec<u8>, u128, u128) {
    let maxsz: u64 = u64::MAX / 8;
    let mut entries = Vec::new();
    let mut off: u128 = 0;
    let mut i = 0;
    let mut remaining = target as u128;
    while remaining > 0 {
        let sz = std::cmp::min(remaining, maxsz as u128) as u64;
        entries.push(format!(
            "\"t{i}\":{{\"dtype\":\"U8\",\"shape\":[{sz}],\"data_offsets\":[{off},{}]}}",
            off + sz as u128
        ));
        off += sz as u128;
        remaining -= sz as u128;
        i += 1;
    }
    assert_eq!(off, target as u128);
    let mut body = format!("{{{}}}", entries.join(","));
    for _ in 0..pad { body.push(' '); }
    let n = body.len() as u128;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(body.as_bytes());
    buf.extend(std::iter::repeat(0u8).take(data_bytes));
    let buffer_len = buf.len() as u128;
    (buf, n, buffer_len)
}

fn main() {
    // ---- Part A: exercise the real deserialize path in release ----
    let mut panics = 0u64;
    let mut accepted = 0u64;
    let mut rejected = 0u64;
    let mut min_gap: i128 = i128::MAX;

    // buffer_end fixed at u64::MAX (largest constructible); sweep n (via pad)
    // and buffer_len (via appended data bytes) trying to hit wrapped==buffer_len.
    let be: u128 = u64::MAX as u128;
    for pad in [0usize, 1, 3, 7, 8, 50, 200] {
        for data_bytes in 0..=300usize {
            let (buf, n, buffer_len) = build(u64::MAX, pad, data_bytes);
            let wrapped = (be + N_LEN + n) % TWO64;
            let gap = wrapped as i128 - buffer_len as i128;
            if gap.abs() < min_gap { min_gap = gap.abs(); }
            let res = std::panic::catch_unwind(|| SafeTensors::deserialize(&buf));
            match res {
                Err(_) => { panics += 1; }
                Ok(Ok(_)) => { accepted += 1;
                    println!("  !!! ACCEPTED bypass: pad={pad} data_bytes={data_bytes} n={n} buffer_len={buffer_len} wrapped={wrapped}"); }
                Ok(Err(_)) => { rejected += 1; }
            }
        }
    }
    println!("Part A (buffer_end=u64::MAX real deserialize sweep):");
    println!("  panics={panics}  accepted(bypass)={accepted}  rejected={rejected}");
    println!("  closest |wrapped - buffer_len| observed = {min_gap}  (0 would mean a pass)");

    // ---- Part B: pure arithmetic Monte Carlo over the achievable domain ----
    // pass  <=>  (be + 8 + n) mod 2^64 == buffer_len,  with
    //   0 <= be <= u64::MAX, 0 <= n <= 100_000_000, buffer_len = 8 + n + D, D >= 0.
    // Claim: pass  <=>  NO wrap  AND  be == D  (the legitimate fully-covered file).
    // A wrapping pass would need be >= 2^64, impossible. Verify no counterexample.
    let mut st = 0x9e3779b97f4a7c15u64;
    let mut rng = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
    let mut wrap_pass = 0u64;
    let mut checked = 0u64;
    for _ in 0..20_000_000u64 {
        let be = (rng() as u128) % TWO64;              // 0..=u64::MAX
        let n = (rng() % 100_000_001) as u128;         // 0..=1e8 (<= MAX_HEADER_SIZE)
        let d = (rng() % (1u64 << 40)) as u128;        // appended data bytes
        let buffer_len = N_LEN + n + d;
        let sum = be + N_LEN + n;                       // true (u128) sum
        let wrapped = sum % TWO64;
        if wrapped == buffer_len {
            checked += 1;
            let wrapped_around = sum >= TWO64;
            if wrapped_around {
                wrap_pass += 1; // a genuine wrap-to-pass: the dangerous case
                if wrap_pass < 5 { println!("  WRAP-PASS: be={be} n={n} d={d}"); }
            }
        }
    }
    println!("Part B (20M random samples over achievable domain):");
    println!("  total passes(checks satisfied)={checked}  of which WRAP-to-pass={wrap_pass}");
    println!("  (wrap-to-pass must be 0 for release to be safe)");
}
