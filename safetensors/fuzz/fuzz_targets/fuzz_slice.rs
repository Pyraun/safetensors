#![no_main]
//! Fuzz target for the slicing arithmetic (`slice.rs`).
//!
//! The safetensors header parser (`fuzz_target_1`) already covers
//! `deserialize`. This target instead stresses the *slicing* path that a
//! caller reaches through `get_slice(name)[...]`: a validated
//! [`TensorView`] (file-controlled dtype + shape + backing bytes) crossed
//! with an arbitrary sequence of [`TensorIndexer`]s (caller-controlled
//! `start:stop:step` / integer selects / ellipsis-style full ranges).
//!
//! We build only *validatable* tensors (via `TensorView::new`, which
//! enforces `product(shape) * bitsize` fits `usize` and lands on a byte
//! boundary) so any panic surfaced here comes from the indexer arithmetic
//! in `slice_byte_ranges` / `SliceIterator`, not from an unrepresentable
//! shape. Indexer bound values are drawn across the full `usize` range so
//! the `+1`/`*span` edge cases are exercised. cargo-fuzz builds with
//! overflow-checks on, so a wrapping add or multiply aborts and is caught.

use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;
use safetensors::slice::TensorIndexer;
use safetensors::tensor::{Dtype, TensorView};
use std::num::NonZeroUsize;
use std::ops::Bound;

/// Keep backing allocations tiny so the fuzzer stays fast and never OOMs on
/// a legitimately-large shape. Cases whose element count exceeds this are
/// skipped (they are representable, just not worth allocating here).
const MAX_ELEMENTS: usize = 1 << 16;
/// Cap dimensionality so `product` and the per-dim loops stay cheap.
const MAX_DIMS: usize = 6;

#[derive(Arbitrary, Debug)]
enum FuzzDtype {
    Bool,
    U8,
    I8,
    F4,
    F6E2M3,
    F6E3M2,
    F8E5M2,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
    C64,
}

impl From<FuzzDtype> for Dtype {
    fn from(d: FuzzDtype) -> Self {
        match d {
            FuzzDtype::Bool => Dtype::BOOL,
            FuzzDtype::U8 => Dtype::U8,
            FuzzDtype::I8 => Dtype::I8,
            FuzzDtype::F4 => Dtype::F4,
            FuzzDtype::F6E2M3 => Dtype::F6_E2M3,
            FuzzDtype::F6E3M2 => Dtype::F6_E3M2,
            FuzzDtype::F8E5M2 => Dtype::F8_E5M2,
            FuzzDtype::I16 => Dtype::I16,
            FuzzDtype::U16 => Dtype::U16,
            FuzzDtype::F16 => Dtype::F16,
            FuzzDtype::Bf16 => Dtype::BF16,
            FuzzDtype::I32 => Dtype::I32,
            FuzzDtype::U32 => Dtype::U32,
            FuzzDtype::F32 => Dtype::F32,
            FuzzDtype::I64 => Dtype::I64,
            FuzzDtype::U64 => Dtype::U64,
            FuzzDtype::F64 => Dtype::F64,
            FuzzDtype::C64 => Dtype::C64,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum FuzzBound {
    Unbounded,
    Included(u64),
    Excluded(u64),
}

impl From<FuzzBound> for Bound<usize> {
    fn from(b: FuzzBound) -> Self {
        match b {
            FuzzBound::Unbounded => Bound::Unbounded,
            FuzzBound::Included(n) => Bound::Included(n as usize),
            FuzzBound::Excluded(n) => Bound::Excluded(n as usize),
        }
    }
}

#[derive(Arbitrary, Debug)]
enum FuzzIndexer {
    Select(u64),
    Narrow(FuzzBound, FuzzBound, u64),
}

impl From<FuzzIndexer> for TensorIndexer {
    fn from(idx: FuzzIndexer) -> Self {
        match idx {
            FuzzIndexer::Select(n) => TensorIndexer::Select(n as usize),
            FuzzIndexer::Narrow(lo, hi, step) => {
                let step = NonZeroUsize::new(step as usize).unwrap_or(NonZeroUsize::MIN);
                TensorIndexer::Narrow(lo.into(), hi.into(), step)
            }
        }
    }
}

#[derive(Arbitrary, Debug)]
struct SliceCase {
    dtype: FuzzDtype,
    /// Dimensions as bytes so each is small (0..=255) and generation is cheap.
    dims: Vec<u8>,
    indexers: Vec<FuzzIndexer>,
}

fuzz_target!(|case: SliceCase| {
    if case.dims.len() > MAX_DIMS {
        return;
    }
    let shape: Vec<usize> = case.dims.iter().map(|&d| d as usize).collect();

    // Element count of a validatable, small tensor. Empty shape => 1 element.
    let mut nelements: usize = 1;
    for &d in &shape {
        nelements = match nelements.checked_mul(d) {
            Some(n) => n,
            None => return,
        };
    }
    if nelements > MAX_ELEMENTS {
        return;
    }

    let dtype: Dtype = case.dtype.into();
    // Byte length for this validatable tensor; sub-byte dtypes must land on a
    // byte boundary or `TensorView::new` rejects the case (we then bail).
    let nbits = match nelements.checked_mul(dtype.bitsize()) {
        Some(b) => b,
        None => return,
    };
    if nbits % 8 != 0 {
        return;
    }
    let data = vec![0u8; nbits / 8];

    let view = match TensorView::new(dtype, shape, &data) {
        Ok(v) => v,
        Err(_) => return,
    };

    let indexers: Vec<TensorIndexer> = case.indexers.into_iter().map(Into::into).collect();

    // The operation under test: must never panic, only Ok(iterator) or Err.
    if let Ok(iterator) = view.sliced_data(&indexers) {
        // Exercise the reported length and then drain every yielded slice,
        // which re-indexes into the backing buffer (`data()[start..stop]`).
        let claimed = iterator.remaining_byte_len();
        let mut seen = 0usize;
        for chunk in iterator {
            seen += chunk.len();
        }
        // The iterator's own accounting must match what it yields.
        assert_eq!(claimed, seen, "remaining_byte_len disagreed with drained bytes");
    }
});
