//! Buffer-length-check overflow — Rust reproducer.
//!
//! Builds a header whose tensor offsets accumulate to exactly usize::MAX and
//! calls SafeTensors::deserialize. Overflows `buffer_end + N_LEN + n` at
//! src/tensor.rs:420.
//!
//! Build as a standalone crate:
//!   Cargo.toml:
//!     [package]
//!     name = "buflen_repro"
//!     version = "0.0.0"
//!     edition = "2021"
//!     [dependencies]
//!     safetensors = { path = "/home/user/safetensors/safetensors" }
//!     [workspace]
//!   src/main.rs: (this file)
//!
//!   cargo run            # debug: panic (attempt to add with overflow)
//!   cargo run --release  # release: gracefully rejected (no bypass, no panic)

use safetensors::tensor::SafeTensors;

fn main() {
    // Build a header whose tensor offsets accumulate to exactly usize::MAX.
    let maxsz: u64 = u64::MAX / 8; // largest valid U8 tensor size
    let mut entries = Vec::new();
    let mut off: u128 = 0;
    for i in 0..8 {
        entries.push(format!(
            "\"t{i}\":{{\"dtype\":\"U8\",\"shape\":[{maxsz}],\"data_offsets\":[{off},{}]}}",
            off + maxsz as u128
        ));
        off += maxsz as u128;
    }
    // final 7-byte tensor to reach 2^64-1
    entries.push(format!(
        "\"t8\":{{\"dtype\":\"U8\",\"shape\":[7],\"data_offsets\":[{off},{}]}}",
        off + 7
    ));
    off += 7;
    assert_eq!(off, u64::MAX as u128);
    let body = format!("{{{}}}", entries.join(","));
    let n = body.len() as u64;
    let mut buf = Vec::new();
    buf.extend_from_slice(&n.to_le_bytes());
    buf.extend_from_slice(body.as_bytes());
    // no data bytes

    let res = std::panic::catch_unwind(|| SafeTensors::deserialize(&buf));
    match res {
        Ok(Ok(_)) => println!("ACCEPTED (bypass!)"),
        Ok(Err(e)) => println!("gracefully rejected: {e}"),
        Err(_) => println!("PANIC (attempt to add with overflow)"),
    }
}
