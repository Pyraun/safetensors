//! Reversed-bounds slice overflow — Rust reproducer.
//!
//! `view.slice(10..3)` resolves to start=10, stop=3 (both < dim) and underflows
//! `stop - start` at src/slice.rs:377.
//!
//! Build as a standalone crate:
//!
//!   Cargo.toml:
//!     [package]
//!     name = "slice_repro"
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
//! Observed: thread 'main' panicked at src/slice.rs:377:31:
//!           attempt to subtract with overflow

use safetensors::slice::IndexOp;
use safetensors::tensor::{Dtype, TensorView};

fn main() {
    let data = vec![0u8; 16 * 4]; // 16 x F32
    let view = TensorView::new(Dtype::F32, vec![16], &data).unwrap();

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Rust `Range` 10..3 -> Narrow(Included(10), Excluded(3)) -> start > stop.
        let _ = view.slice(10..3).map(|it| it.count());
    }));
    match res {
        Ok(_) => println!("NO PANIC (unexpected)"),
        Err(_) => println!("PANIC as expected (attempt to subtract with overflow)"),
    }
}
