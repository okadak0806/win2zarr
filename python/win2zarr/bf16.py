"""bfloat16 bit-pattern helpers (stored as uint16)."""

from __future__ import annotations

import numpy as np


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 → bfloat16 bit pattern as uint16."""
    x = np.asarray(values, dtype=np.float32)
    bits = x.view(np.uint32).copy()
    lsb = (bits >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    # Round finite values only. NaN payloads such as 0x7fffffff / 0xffff8000
    # wrap to ±0 if the bias is added; infinities are passed through.
    rounded = np.where(np.isfinite(x), bits + rounding_bias, bits)
    out = np.asarray(rounded >> np.uint32(16), dtype=np.uint16)
    nan = np.isnan(x)
    sign = np.asarray((bits >> np.uint32(16)) & np.uint32(0x8000), dtype=np.uint16)
    canonical_nan = sign | np.uint16(0x7FC0)
    return np.where(nan, canonical_nan, out).astype(np.uint16, copy=False)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Expand uint16 bfloat16 bits to float32 (trailing mantissa zeros)."""
    bits = np.asarray(bits, dtype=np.uint16)
    wide = bits.astype(np.uint32) << np.uint32(16)
    return wide.view(np.float32)
