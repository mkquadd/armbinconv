# armbinconv

Convert ARMv7s binaries to ARMv7 (and, experimentally, ARMv6) —
with a safety scan *and*, optionally, actual instruction translation.

## What it actually does

**Mode `armv7s2armv7` — the main one.**

1. Scans `__text` for instructions that exist on armv7s (A6 / Swift)
   but would SIGILL on any armv7 device (Cortex-A8/A9):
   - VFPv4 fused multiply-add: `vfma`, `vfms`, `vfnma`, `vfnms`
   - VFPv4 half-precision convert: `vcvtb`, `vcvtt`
   - Hardware integer divide: `sdiv`, `udiv`
2. **Without `--shim`:** refuses to patch if any of the above are found.
   `--force` overrides, but the resulting binary will crash at runtime
   when execution reaches those instructions. That's not a bug, that's
   physics: VFPv4 fused and hardware divide simply don't exist on armv7.
3. **With `--shim`:** actually translates ARM-state `sdiv`/`udiv` into
   `bl` calls to self-contained divide shims (32-iteration shift-subtract,
   no external symbol resolution needed), placed in `__TEXT` padding.
   The shim preserves `r0–r12` and `lr`, restores everything except `rd`,
   and works for conditional variants (`sdivne` → `blne`). If a function
   didn't save `lr` before the `sdiv`/`udiv`, the site is skipped with a
   warning (`--force-leaf` overrides). Thumb-state `sdiv`/`udiv` is
   **not** translated — warned and left alone.
4. Once the scan is clean (or the shims are in), patches cpusubtype from
   armv7s (11) to armv7 (9) in both the mach_header and the fat_arch entry.

**Mode `armv7s2armv6` — experimental.**
For armv7/armv7s → armv6:

- `dmb` / `dsb` / `isb` → ARMv6K CP15 `MCR` (in-place, works on ARM1176).
- `sdiv` / `udiv` → same shim mechanism as above.
- Everything else (NEON, Thumb-2, `movw`/`movt`, `rbit`, `ubfx`/`sbfx`,
  `bfi`/`bfc`) — **reported, not translated**. These would require full
  function rewriting, not a one-instruction shim. The tool tells you
  exactly where they are so you know what you're up against.

## What it does NOT do

- **VFPv4 fused operations** (`vfma`/`vfms`) cannot be shimmed without
  changing numerical results (fused ≠ unfused). Refused unless `--force`.
- **Thumb-2 state translation** — not implemented. Warning only.
- **NEON / Advanced SIMD** on armv6 — not translatable by this tool.
- **`movw`/`movt`, `rbit`, `ubfx`/`sbfx`, `bfi`/`bfc`** on armv6 —
  would require full function rewriting (copying the body to a new
  segment and patching all branches). Out of scope here.

## Useful flags

- `--scan-only` — check if a binary can even run on armv7, without patching.
  Add `--json` for machine-readable output.
- `--shim` — actually translate ARM `sdiv`/`udiv` via inline divide shims.
  Available in both modes.
- `--force-leaf` — patch `sdiv`/`udiv` even if the containing function
  didn't save `lr` first (will crash — only useful for experiments).
- `--min-os 9.0` — lower `LC_VERSION_MIN_IPHONEOS` /
  `LC_BUILD_VERSION.minos` so dyld on older iOS will accept the binary.
- `--ldid` — auto re-sign after patching (needs `ldid` in PATH).
- `--skip-scan` — don't scan, patch headers only. Useful when you've
  already confirmed the binary is safe by other means.
- `--force` — patch anyway, accept the consequences.

## Safety notes

- **Fat binaries with an existing armv7 slice** are refused. Use
  `lipo -thin armv7` to extract Apple's own slice — it's always better
  than patching.
- **Encrypted slices** (`cryptid != 0`) can't be scanned. `--force` or
  `--skip-scan` required.
- **Code signature** is invalidated by any patch. `--ldid` handles it,
  otherwise re-sign manually with `ldid -S <out>` or `codesign` before
  installing.
- **Shim overhead**: `sdiv`/`udiv` in tight loops will be measurably
  slower (32 iterations vs 1). Fine for most code, bad for hot paths.

## Requirements

- `pip install capstone` — required for scanning and `--shim`.
- `ldid` — optional, only for `--ldid`.
- Python ≥ 3.8.

## Examples

```bash
# Can this binary run on armv7 at all?
python -m armbinconv armv7s2armv7 binary --scan-only

# Convert headers only (I've already verified it's clean):
python -m armbinconv armv7s2armv7 binary --skip-scan --min-os 9.0 --ldid -o out

# Full treatment: translate sdiv/udiv, lower min OS, re-sign:
python -m armbinconv armv7s2armv7 binary --shim --min-os 9.0 --ldid -o out

# Down to armv6 (barriers + sdiv/udiv, experimental):
python -m armbinconv armv7s2armv6 binary --shim --min-os 4.0 --ldid -o out
