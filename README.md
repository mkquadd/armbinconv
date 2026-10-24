# armbinconv

Convert ARMv7s binaries to ARMv7 — with a safety scan (I hope at least, I am a fucking dumbass).

**What it actually does:** scans `__text` for VFPv4 (`vfma/vfms/vfnma/vfnms/vcvtb/vcvtt`)
and hardware divide (`sdiv/udiv`) instructions that would SIGILL on any ARMv7
device, and *refuses to patch* if it finds them. If the scan is clean, it
patches the cpusubtype from armv7s (11) to armv7 (9) in the mach_header
and fat_arch entry.

**What it does NOT do:** translate armv7s-only instructions. If a binary
contains `sdiv` or `vfma`, the tool tells you where and refuses. `--force`
overrides but the result will crash. This is intentional.

**Useful flags:**
- `--scan-only` — check if a binary can even run on armv7, without patching
- `--min-os 9.0` — lower LC_VERSION_MIN so dyld on iOS 9 will accept it
- `--ldid` — auto re-sign after patching (needs `ldid` in PATH)

**Requirements:** `pip install capstone` (optional for `--scan-only`), `ldid` (optional)
