//! `usize::MAX` slice-bound overflow — Rust reproducer.
//!
//! `view.slice(..=usize::MAX)` -> Narrow(Unbounded, Included(usize::MAX), 1).
//! narrow_bounds computes `*s + 1` = usize::MAX + 1 at src/slice.rs:324.
//!
//! Build as a standalone crate:
//!
//!   Cargo.toml:
//!     [package]
//!     name = "slice_repro_max"
//!     version = "0.0.0"
//!     edition = "2021"
//!     [dependencies]
//!     safetensors = { path = "/home/user/safetensors/safetensors" }
//!     [workspace]
//!
//!   src/main.rs: (this file)
//!
//!   cargo run        # debug build has overflow-checks on -> panics
//!
//! Observed: thread 'main' panicked at src/slice.rs:324:31:
//!           attempt to add with overflow
//!
//! Note: this bound is NOT reachable through the Python bindings; it is a
//! Rust-crate-API concern only.

use safetensors::slice::IndexOp;
use safetensors::tensor::{Dtype, TensorView};

fn main() {
    let data = vec![0u8; 16 * 4]; // 16 x F32
    let view = TensorView::new(Dtype::F32, vec![16], &data).unwrap();

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = view.slice(..=usize::MAX).map(|it| it.count());
    }));
    match res {
        Ok(_) => println!("NO PANIC (unexpected)"),
        Err(_) => println!("PANIC as expected (attempt to add with overflow)"),
    }
}
