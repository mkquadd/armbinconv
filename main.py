#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
armbinconv.py — Mach-O ARM architecture converter.

Modes
-----
  armv7s2armv7          Convert an armv7s binary to armv7 for use on armv7
                        devices (Cortex-A8/A9: iPhone 3GS/4/4S, iPad 1/2/3/
                        mini1, iPod 4/5). Before patching, scans __text for
                        instructions that exist on armv7s (Swift/A6) but
                        would SIGILL on armv7:
                          * VFPv4 fused multiply-add  (vfma/vfms/vfnma/vfnms)
                          * VFPv4 half-precision cvt  (vcvtb/vcvtt)
                          * hardware integer divide   (udiv/sdiv)
                        Refuses to patch if any are found, unless --force.

  barriers-only         EXPERIMENTAL. Handles a narrow subset (CP15 memory
                        barriers) in-place. Everything else is reported and
                        refused. Roughly 95% of real armv7s binaries will be
                        refused because they contain Thumb-2 / NEON / movw /
                        movt, which cannot be translated trivially.

Safety
------
  * Fat binaries with BOTH armv7 and armv7s slices are hard-refused.
    Extract the armv7 slice with `lipo -thin armv7` instead. There is no
    override, because creating a duplicate armv7 subtype is invalid.
  * Encrypted slices (cryptid != 0) cannot be scanned; --force is required
    to patch them blindly.
  * Stripped slices with neither LC_FUNCTION_STARTS nor a symtab are
    "entry-only": only the entry point is disassembled. Such scans are
    unreliable and are refused without --force.
  * Any header patch invalidates the code signature. Pass --ldid to auto
    re-sign (needs `ldid` in PATH) or re-sign manually before installing.
  * --min-os lowers LC_VERSION_MIN_IPHONEOS / LC_BUILD_VERSION so dyld on
    older iOS will accept the binary.

Requirements
------------
  pip install capstone          # required for --scan-only, and for patching
  ldid                          # optional; only for --ldid

  --scan-only works without capstone and prints an UNKNOWN verdict.
  Patching refuses to proceed without capstone unless --force or
  --skip-scan is given.

Examples
--------
  ./armbinconv.py armv7s2armv7 binary --scan-only
  ./armbinconv.py armv7s2armv7 binary --min-os 9.0 --ldid -o binary.armv7
  ./armbinconv.py armv7s2armv7 binary --force -o binary.armv7
"""

__version__ = "1.0.0"

import argparse
import json
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from capstone import (
        Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB, CS_MODE_LITTLE_ENDIAN,
    )
    try:
        import capstone.arm as cs_arm
    except ImportError:
        cs_arm = None
    HAVE_CAPSTONE = True
except Exception:
    HAVE_CAPSTONE = False
    cs_arm = None


# ───────────────────────── constants ─────────────────────────

FAT_MAGIC    = 0xCAFEBABE
FAT_MAGIC_64 = 0xCAFEBABF
MH_MAGIC     = 0xFEEDFACE
MH_MAGIC_64  = 0xFEEDFACF

CPU_TYPE_ARM = 12
SUBTYPE = {6: "armv6", 9: "armv7", 11: "armv7s", 12: "armv7k", 13: "armv8"}

CPU_SUBTYPE_MASK = 0xFF000000

FILETYPE_NAMES = {
    1: "MH_OBJECT", 2: "MH_EXECUTE", 3: "MH_FVMLIB", 4: "MH_CORE",
    5: "MH_PRELOAD", 6: "MH_DYLIB", 7: "MH_DYLINKER", 8: "MH_BUNDLE",
    9: "MH_DYLIB_STUB", 10: "MH_DSYM", 11: "MH_KEXT_BUNDLE",
}
MH_KEXT_BUNDLE     = 11
MH_DYLINKER        = 7
MH_EXECUTE         = 2
MH_DYLIB           = 6
MH_BUNDLE          = 8
USERLAND_FILETYPES = {MH_EXECUTE, MH_DYLIB, MH_BUNDLE, MH_DYLINKER}

LC_SEGMENT              = 0x01
LC_SYMTAB               = 0x02
LC_SEGMENT_64           = 0x19
LC_CODE_SIGNATURE       = 0x1D
LC_ENCRYPTION_INFO      = 0x21
LC_VERSION_MIN_IPHONEOS = 0x25
LC_FUNCTION_STARTS      = 0x26
LC_ENCRYPTION_INFO_64   = 0x2C
LC_BUILD_VERSION        = 0x32

N_ARM_THUMB_DEF = 0x0008
VM_PROT_EXECUTE = 0x4
ARM_CC_AL       = 0xE


# ───────────────────────── helpers ─────────────────────────

def read_uleb128(blob: bytes, i: int):
    val, shift = 0, 0
    while i < len(blob):
        b = blob[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7
    return None, i


def unpack_version_word(v: int) -> str:
    return f"{(v >> 16) & 0xFFFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"


def parse_version_string(s: str) -> int:
    """'9.0' -> 0x00090000, '10.3.4' -> 0x000A0304."""
    parts = s.split(".")
    while len(parts) < 3:
        parts.append("0")
    return (int(parts[0]) << 16) | (int(parts[1]) << 8) | int(parts[2])


def validate_text_section(text, file_size: int):
    """Return (ok, error). Detects compressed / truncated / empty text."""
    if text is None:
        return False, "no __text section or executable segment found"
    if text["size"] == 0:
        return False, "text section has size 0 (zero-fill or LZSS-compressed)"
    if text["offset"] == 0:
        return False, ("text section is not file-backed (offset=0); "
                       "likely LZSS-compressed or __TEXT is zero-fill")
    end = text["offset"] + text["size"]
    if end > file_size:
        return False, (f"text section truncated: end={end:#x} > "
                       f"file={file_size:#x}")
    return True, None


# ───────────────────────── Mach-O model ─────────────────────────

@dataclass
class Slice:
    file_offset: int
    size: int
    cputype: int
    cpusubtype: int
    is_64: bool
    endian: str
    hdr_size: int = 28
    filetype: int = 0
    ncmds: int = 0
    sizeofcmds: int = 0
    flags: int = 0
    segments: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    symtab: Optional[dict] = None
    func_starts: Optional[dict] = None
    encryption_info: Optional[dict] = None
    build_version: Optional[dict] = None
    has_code_signature: bool = False

    @property
    def subtype_name(self):
        return SUBTYPE.get(self.cpusubtype & 0xFFFFFF, f"sub{self.cpusubtype}")

    @property
    def is_encrypted(self) -> bool:
        return bool(self.encryption_info and self.encryption_info.get("cryptid", 0))

    @property
    def filetype_name(self) -> str:
        return FILETYPE_NAMES.get(self.filetype, f"filetype{self.filetype}")

    @property
    def is_userland(self) -> bool:
        return self.filetype in USERLAND_FILETYPES

    @property
    def min_os_word(self) -> Optional[int]:
        if not self.build_version:
            return None
        return self.build_version.get("version") or self.build_version.get("minos")


def parse_thin(data: bytes, base: int) -> Slice:
    if len(data) - base < 28:
        raise ValueError(f"slice@{base:#x}: too short")
    magic = struct.unpack_from("<I", data, base)[0]
    if magic == MH_MAGIC:
        endian, is_64, hdr_size = "<", False, 28
    elif magic == MH_MAGIC_64:
        endian, is_64, hdr_size = "<", True, 32
    else:
        raise ValueError(f"slice@{base:#x}: not Mach-O (magic={magic:#x})")

    _, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags = struct.unpack_from(
        endian + "IiiIIII", data, base
    )
    sl = Slice(file_offset=base, size=len(data) - base,
               cputype=cputype, cpusubtype=cpusubtype,
               is_64=is_64, endian=endian, hdr_size=hdr_size,
               filetype=filetype, ncmds=ncmds, sizeofcmds=sizeofcmds, flags=flags)

    off = base + hdr_size
    for _ in range(ncmds):
        if off + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from(endian + "II", data, off)
        if cmdsize < 8 or off + cmdsize > len(data):
            break

        if cmd in (LC_SEGMENT, LC_SEGMENT_64):
            segname = data[off+8:off+24].rstrip(b"\x00").decode("ascii", "replace")
            if cmd == LC_SEGMENT:
                vmaddr, vmsize, fileoff, filesize, maxprot, initprot, nsects, _ = \
                    struct.unpack_from(endian + "IIIIiiII", data, off + 24)
                sect_off, sect_size = off + 56, 68
            else:
                vmaddr, vmsize, fileoff, filesize, maxprot, initprot, nsects, _ = \
                    struct.unpack_from(endian + "QQQQiiII", data, off + 24)
                sect_off, sect_size = off + 72, 80
            sl.segments.append(dict(name=segname, vmaddr=vmaddr, vmsize=vmsize,
                                    fileoff=fileoff, filesize=filesize,
                                    maxprot=maxprot, initprot=initprot,
                                    cmd_off=off, cmdsize=cmdsize, nsects=nsects))
            for j in range(nsects):
                so = sect_off + j * sect_size
                if so + sect_size > len(data):
                    break
                sectname = data[so:so+16].rstrip(b"\x00").decode("ascii", "replace")
                segname2 = data[so+16:so+32].rstrip(b"\x00").decode("ascii", "replace")
                if cmd == LC_SEGMENT:
                    addr, size, offset, align, reloff, nreloc, sflags2 = \
                        struct.unpack_from(endian + "IIIIIII", data, so + 32)
                else:
                    addr, size = struct.unpack_from(endian + "QQ", data, so + 32)
                    offset, align, reloff, nreloc, sflags2 = \
                        struct.unpack_from(endian + "IIIII", data, so + 48)
                sl.sections.append(dict(sect=sectname, seg=segname2, addr=addr,
                                        size=size, offset=offset, flags=sflags2))

        elif cmd == LC_SYMTAB:
            symoff, nsyms, stroff, strsize = struct.unpack_from(endian + "IIII", data, off + 8)
            sl.symtab = dict(symoff=symoff, nsyms=nsyms, stroff=stroff, strsize=strsize)

        elif cmd == LC_FUNCTION_STARTS:
            dataoff, datasize = struct.unpack_from(endian + "II", data, off + 8)
            sl.func_starts = dict(dataoff=dataoff, datasize=datasize)

        elif cmd in (LC_ENCRYPTION_INFO, LC_ENCRYPTION_INFO_64):
            cryptoff, cryptsize, cryptid = struct.unpack_from(endian + "III", data, off + 8)
            sl.encryption_info = dict(cryptoff=cryptoff, cryptsize=cryptsize, cryptid=cryptid)

        elif cmd == LC_VERSION_MIN_IPHONEOS:
            version, sdk = struct.unpack_from(endian + "II", data, off + 8)
            sl.build_version = dict(kind="min", version=version, sdk=sdk)

        elif cmd == LC_BUILD_VERSION:
            platform, minos, sdk, ntools = struct.unpack_from(endian + "IIII", data, off + 8)
            sl.build_version = dict(kind="build", platform=platform, minos=minos, sdk=sdk)

        elif cmd == LC_CODE_SIGNATURE:
            sl.has_code_signature = True

        off += cmdsize

    return sl


def parse_fat(data: bytes):
    if len(data) < 8:
        return None
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic not in (FAT_MAGIC, FAT_MAGIC_64):
        return None
    n = struct.unpack_from(">I", data, 4)[0]
    is64 = magic == FAT_MAGIC_64
    out = []
    for i in range(n):
        if is64:
            base = 8 + i * 32
            cputype, cpusubtype, offset, size, align, _ = \
                struct.unpack_from(">iIQQII", data, base)
        else:
            base = 8 + i * 20
            cputype, cpusubtype, offset, size, align = \
                struct.unpack_from(">iIIII", data, base)
        out.append(dict(cputype=cputype, cpusubtype=cpusubtype,
                        offset=offset, size=size,
                        fat_entry_off=base + 4, is64=is64))
    return out


def iter_slices(data: bytes):
    fat = parse_fat(data)
    if fat is None:
        yield parse_thin(data, 0), None
    else:
        for e in fat:
            if e["cputype"] != CPU_TYPE_ARM:
                continue
            try:
                sl = parse_thin(data, e["offset"])
            except ValueError:
                continue
            yield sl, e


# ───────────────────────── function starts / text location ─────────────────────────

def collect_function_starts(data: bytes, sl: Slice):
    addrs, modes = set(), {}
    text_vm = None
    for seg in sl.segments:
        if seg["name"] == "__TEXT":
            text_vm = seg["vmaddr"]
            break

    if sl.func_starts and sl.func_starts["datasize"] and text_vm is not None:
        fo, fsz = sl.func_starts["dataoff"], sl.func_starts["datasize"]
        blob = data[fo:fo + fsz]
        cur, i = 0, 0
        while i < len(blob):
            v, i = read_uleb128(blob, i)
            if not v:
                break
            cur += v
            addrs.add(text_vm + cur)

    if sl.symtab and text_vm is not None:
        endian = sl.endian
        sz = 16 if sl.is_64 else 12
        for i in range(sl.symtab["nsyms"]):
            off = sl.symtab["symoff"] + i * sz
            try:
                if sl.is_64:
                    _, n_type, _, n_desc, n_value = struct.unpack_from(
                        endian + "IBBHQ", data, off)
                else:
                    _, n_type, _, n_desc, n_value = struct.unpack_from(
                        endian + "IBBHI", data, off)
            except struct.error:
                break
            if not n_value or (n_type & 0x0E) != 0x0E:
                continue
            addr = n_value & ~1
            thumb = bool(n_value & 1) or bool(n_desc & N_ARM_THUMB_DEF)
            modes[addr] = "thumb" if thumb else "arm"
            addrs.add(addr)

    if not addrs and text_vm is not None:
        addrs.add(text_vm)

    return addrs, modes


def locate_text_section(sl: Slice):
    for s in sl.sections:
        if s["sect"] == "__text" and s["size"] > 0:
            return s
    for seg in sl.segments:
        if seg["initprot"] & VM_PROT_EXECUTE and seg["filesize"] > 0:
            return dict(sect="__text", seg=seg["name"], addr=seg["vmaddr"],
                        size=seg["filesize"], offset=seg["fileoff"], flags=0)
    return None


# ───────────────────────── classifiers ─────────────────────────

ARMV7S_ONLY_MNEMONICS = {
    "vfma":  "VFPv4 fused multiply-add (armv7s/A6 only)",
    "vfms":  "VFPv4 fused multiply-subtract (armv7s/A6 only)",
    "vfnma": "VFPv4 negate-fused-multiply-add (armv7s/A6 only)",
    "vfnms": "VFPv4 negate-fused-multiply-subtract (armv7s/A6 only)",
    "vcvtb": "VFPv4 half-precision convert (armv7s/A6 only)",
    "vcvtt": "VFPv4 half-precision convert (armv7s/A6 only)",
    "udiv":  "hardware unsigned divide (armv7s/A6 only; armv7 has no idiv)",
    "sdiv":  "hardware signed divide (armv7s/A6 only; armv7 has no idiv)",
}

ARMV6_ILLEGAL_MNEMONICS = {
    "udiv":   "ARMv7 integer divide",
    "sdiv":   "ARMv7 integer divide",
    "clrex":  "ARMv7 exclusive monitor hint",
    "dsb":    "ARMv7 memory barrier (translatable via CP15)",
    "dmb":    "ARMv7 memory barrier (translatable via CP15)",
    "isb":    "ARMv7 memory barrier (translatable via CP15)",
    "movw":   "ARMv6T2 wide move",
    "movt":   "ARMv6T2 wide move",
    "rbit":   "ARMv6T2 bit reverse",
    "ubfx":   "ARMv6T2 bitfield extract",
    "sbfx":   "ARMv6T2 bitfield extract",
    "bfi":    "ARMv6T2 bitfield insert",
    "bfc":    "ARMv6T2 bitfield clear",
    "pldw":   "ARMv7 preload for write",
    "ldrexb": "ARMv6K byte exclusive load (OK on iOS armv6)",
    "ldrexh": "ARMv6K halfword exclusive load (OK on iOS armv6)",
    "strexb": "ARMv6K byte exclusive store (OK on iOS armv6)",
    "strexh": "ARMv6K halfword exclusive store (OK on iOS armv6)",
}

NEON_PREFIXES = (
    "vld1", "vld2", "vld3", "vld4", "vst1", "vst2", "vst3", "vst4",
    "vzip", "vuzp", "vtrn", "vrev", "vswp", "vext",
    "vmull", "vmlal", "vmlsl", "vqdmull",
    "vqadd", "vqsub", "vqmul", "vqdmulh", "vqrdmulh",
    "vrhadd", "vhadd", "vhsub",
    "vshl", "vshr", "vshll", "vshrn", "vrshr", "vrshrn",
    "vtbl", "vtbx", "vdup", "vclz", "vcnt", "vcls",
    "vpadd", "vpmax", "vpmin", "vabd", "vabdl",
)

BARRIER_MNEMONICS = {"dsb", "dmb", "isb"}


def _mnemonic_root(insn) -> str:
    return insn.mnemonic.split(".")[0].split("w")[0] if "." not in insn.mnemonic else insn.mnemonic.split(".")[0]


def classify_armv7s_only(insn, mode: str) -> Optional[str]:
    """Return reason if insn works on armv7s but not on armv7.

    Tries Capstone's VFP4 group first, then falls back to a mnemonic match.
    The fallback is important for Thumb, where Capstone occasionally fails
    to set the group attribute."""
    mn = insn.mnemonic.split(".")[0]
    if HAVE_CAPSTONE and cs_arm is not None:
        try:
            vfp4 = getattr(cs_arm, "ARM_GRP_VFP4", None)
            if vfp4 is not None and insn.group(vfp4):
                return f"VFPv4 instruction ({insn.mnemonic}) — needs armv7s/A6"
        except Exception:
            pass
    return ARMV7S_ONLY_MNEMONICS.get(mn)


def classify_armv6_illegal(insn, mode: str) -> Optional[str]:
    """Return reason if insn works on armv7 but not on armv6."""
    if mode == "thumb" and insn.size == 4:
        return "Thumb-2 (32-bit Thumb)"
    if HAVE_CAPSTONE and cs_arm is not None:
        try:
            neon = getattr(cs_arm, "ARM_GRP_NEON", None)
            if neon is not None and insn.group(neon):
                return "NEON (Advanced SIMD)"
        except Exception:
            pass
    mn = insn.mnemonic.split(".")[0]
    entry = ARMV6_ILLEGAL_MNEMONICS.get(mn)
    if entry:
        return entry
    if any(insn.mnemonic.startswith(p) for p in NEON_PREFIXES):
        return "NEON (Advanced SIMD)"
    return None


# ───────────────────────── generic scan ─────────────────────────

def scan_text(data: bytes, sl: Slice, classifier) -> dict:
    """Disassemble __text and classify each instruction via `classifier`.

    The report includes a "coverage" key:
      * "full"       — LC_FUNCTION_STARTS or a symtab supplied function bounds
      * "entry-only" — nothing but the entry point was scanned; unreliable
      * "no-text"    — text section is missing / compressed / truncated
    """
    report = {
        "functions": 0, "bad": [], "no_capstone": False,
        "has_thumb": False, "coverage": "unknown",
        "error": None,
    }
    if not HAVE_CAPSTONE:
        report["no_capstone"] = True
        return report

    text = locate_text_section(sl)
    ok, err = validate_text_section(text, len(data))
    if not ok:
        report["coverage"] = "no-text"
        report["error"] = err
        return report

    text_data   = data[text["offset"]:text["offset"] + text["size"]]
    text_vmaddr = text["addr"]

    funcs, func_modes = collect_function_starts(data, sl)
    funcs = sorted(funcs)

    has_fstarts = bool(sl.func_starts and sl.func_starts["datasize"] > 0)
    has_symtab  = bool(sl.symtab and sl.symtab["nsyms"] > 0)
    report["coverage"] = "full" if (has_fstarts or has_symtab) else "entry-only"

    md_arm   = Cs(CS_ARCH_ARM, CS_MODE_ARM   | CS_MODE_LITTLE_ENDIAN)
    md_thumb = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_LITTLE_ENDIAN)

    for i, fn_addr in enumerate(funcs):
        end_addr = funcs[i + 1] if i + 1 < len(funcs) else text_vmaddr + text["size"]
        if fn_addr < text_vmaddr or fn_addr >= text_vmaddr + text["size"]:
            continue
        if end_addr <= fn_addr:
            continue

        start = fn_addr - text_vmaddr
        end   = min(end_addr - text_vmaddr, text["size"])
        chunk = text_data[start:end]

        mode = func_modes.get(fn_addr, "arm")
        if mode == "thumb":
            report["has_thumb"] = True
        md = md_thumb if mode == "thumb" else md_arm
        report["functions"] += 1

        try:
            for insn in md.disasm(chunk, fn_addr):
                reason = classifier(insn, mode)
                if reason:
                    report["bad"].append({
                        "address": insn.address,
                        "mnemonic": insn.mnemonic,
                        "op_str": insn.op_str,
                        "reason": reason,
                        "mode": mode,
                        "size": insn.size,
                        "cc": insn.cc if hasattr(insn, "cc") else None,
                    })
        except Exception:
            pass

    return report


def print_bad(report: dict, limit: int = 20) -> None:
    for b in report["bad"][:limit]:
        print(f"    {b['address']:#010x} [{b['mode']:<5}] "
              f"{b['mnemonic']} {b['op_str']:<26} -> {b['reason']}")
    if len(report["bad"]) > limit:
        print(f"    ... and {len(report['bad']) - limit} more")


# ───────────────────────── header patches ─────────────────────────

def patch_slice_subtype(data: bytearray, sl: Slice, fat_entry, new_subtype: int) -> None:
    """Update cpusubtype in mach_header (LE) and, if present, in fat_arch (BE)."""
    old_le = struct.unpack_from("<I", data, sl.file_offset + 8)[0]
    new_le = new_subtype & 0x00FFFFFF
    struct.pack_into("<I", data, sl.file_offset + 8, new_le)
    print(f"  [mach_header @ {sl.file_offset:#x}] cpusubtype "
          f"{old_le:#010x} -> {new_le:#010x} ({SUBTYPE.get(new_le, '?')})")

    if fat_entry is not None:
        old_be = struct.unpack_from(">I", data, fat_entry["fat_entry_off"])[0]
        struct.pack_into(">I", data, fat_entry["fat_entry_off"], new_le)
        print(f"  [fat_arch    @ {fat_entry['fat_entry_off']:#x}] cpusubtype "
              f"{old_be:#010x} -> {new_le:#010x}")


def patch_min_os(data: bytearray, sl: Slice, target: int) -> int:
    """Lower LC_VERSION_MIN_IPHONEOS / LC_BUILD_VERSION to `target`."""
    off = sl.file_offset + sl.hdr_size
    patched = 0
    for _ in range(sl.ncmds):
        if off + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from(sl.endian + "II", data, off)
        if cmdsize < 8 or off + cmdsize > len(data):
            break

        if cmd == LC_VERSION_MIN_IPHONEOS:
            old_ver, _ = struct.unpack_from(sl.endian + "II", data, off + 8)
            struct.pack_into(sl.endian + "II", data, off + 8, target, target)
            print(f"  [LC_VERSION_MIN_IPHONEOS @ {off:#x}] "
                  f"{unpack_version_word(old_ver)} -> {unpack_version_word(target)}")
            patched += 1

        elif cmd == LC_BUILD_VERSION:
            _, old_min, _, _ = struct.unpack_from(
                sl.endian + "IIII", data, off + 8)
            struct.pack_into(sl.endian + "II", data, off + 12, target, target)
            print(f"  [LC_BUILD_VERSION @ {off:#x}] minos "
                  f"{unpack_version_word(old_min)} -> {unpack_version_word(target)}")
            patched += 1

        off += cmdsize
    return patched


# ───────────────────────── warnings ─────────────────────────

def warn_signature(sl: Slice) -> None:
    if sl.has_code_signature:
        print("  [!] LC_CODE_SIGNATURE present. Any patch invalidates it.")
        print("      Re-sign with `ldid -S <out>` or pass --ldid.")


def warn_userland(sl: Slice) -> None:
    if sl.is_userland:
        print(f"  [!] filetype {sl.filetype_name} is userland; "
              "CP15-based armv6 translation will fault at runtime "
              "(only kexts can use it).")


# ───────────────────────── armv7s -> armv7 ─────────────────────────

def do_armv7s2armv7(data: bytearray, force: bool, skip_scan: bool,
                    scan_only: bool, target_min_os: Optional[int],
                    as_json: bool) -> tuple[int, bool]:
    """Returns (slices_patched, needs_resign)."""
    fat = parse_fat(bytes(data))

    if fat is not None:
        armv7_entries  = [e for e in fat if e["cputype"] == CPU_TYPE_ARM
                          and (e["cpusubtype"] & 0xFFFFFF) == 9]
        armv7s_entries = [e for e in fat if e["cputype"] == CPU_TYPE_ARM
                          and (e["cpusubtype"] & 0xFFFFFF) == 11]

        if not armv7s_entries:
            if armv7_entries:
                raise RuntimeError("binary already has armv7 only; nothing to patch")
            raise RuntimeError("no ARM slices in fat binary")

        if armv7_entries:
            # Hard refusal: patching would create a duplicate armv7 slice.
            print("=== FAT BINARY: armv7 AND armv7s slices present ===")
            print(f"  armv7 slice(s) : {len(armv7_entries)}")
            print(f"  armv7s slice(s): {len(armv7s_entries)}")
            print()
            print("  [X] REFUSING: this file already contains a real armv7")
            print("      slice. Extract it instead of patching armv7s:")
            print("        $ lipo -thin armv7 <input> -output <out.armv7>")
            print()
            print("  Creating a duplicate armv7 subtype is invalid and")
            print("  breaks dyld. There is no override for this case.")
            raise RuntimeError("fat binary already contains an armv7 slice")

        slices = [(parse_thin(bytes(data), e["offset"]), e) for e in armv7s_entries]
    else:
        sl = parse_thin(bytes(data), 0)
        if sl.cputype != CPU_TYPE_ARM:
            raise RuntimeError(f"not an ARM binary (cputype={sl.cputype})")
        sub = sl.cpusubtype & 0xFFFFFF
        if sub == 9:
            raise RuntimeError("binary is already armv7; nothing to patch")
        if sub != 11:
            raise RuntimeError(f"expected armv7s, got {sl.subtype_name}")
        slices = [(sl, None)]

    patched = 0
    needs_resign = False
    all_reports = []

    for sl, fe in slices:
        print(f"=== slice @ {sl.file_offset:#x} "
              f"({sl.subtype_name}, {sl.filetype_name}) ===")

        mov = sl.min_os_word
        if mov is not None and ((mov >> 16) & 0xFFFF) >= 10:
            if scan_only:
                print(f"  [!] requires iOS {unpack_version_word(mov)} — dyld on "
                      f"iOS 9 (max for armv7 devices) will REFUSE to load it.")
                print(f"      Pass --min-os 9.0 to lower the minimum version.")
            elif target_min_os is not None:
                patch_min_os(data, sl, target_min_os)
            else:
                print(f"  [!] requires iOS {unpack_version_word(mov)}; armv7 "
                      f"devices max out at iOS 9. Binary will NOT load.")
                print(f"      Pass --min-os 9.0 to lower the minimum version.")

        if sl.is_encrypted and not skip_scan and not scan_only:
            print(f"  [!] encrypted slice (cryptid={sl.encryption_info['cryptid']}); "
                  f"VFPv4 / hwdiv scan skipped.")
            if not force:
                raise RuntimeError(
                    "encrypted slice cannot be scanned; "
                    "pass --force or --skip-scan to patch blindly"
                )

        if not skip_scan and not sl.is_encrypted:
            print("  scanning __text for armv7s-only instructions...")
            rep = scan_text(bytes(data), sl, classify_armv7s_only)
            all_reports.append(rep)

            if rep.get("error"):
                print(f"  [!] {rep['error']}")
                if not scan_only and not force:
                    raise RuntimeError(
                        f"cannot scan ({rep['error']}); "
                        f"use --force to patch blindly"
                    )
            elif rep["no_capstone"]:
                print("  [!] capstone not installed; scan skipped.")
                print("      pip install capstone   (or pass --force)")
                if not force and not scan_only:
                    raise RuntimeError("capstone missing; --force required")
            elif rep["coverage"] == "entry-only":
                print(f"  [!] no LC_FUNCTION_STARTS and no symtab: only the")
                print(f"      entry point of __text was disassembled.")
                print(f"      Result is UNRELIABLE — most of __text is not")
                print(f"      analysed, so armv7s-only instructions will be")
                print(f"      missed. Typical for stripped kext bundles.")
                if not scan_only and not force:
                    raise RuntimeError(
                        "entry-only coverage cannot reliably detect "
                        "armv7s-only instructions; --force required"
                    )
            elif rep["functions"] == 0:
                print("  [i] no functions found; nothing to scan.")
            else:
                n_bad = len(rep["bad"])
                print(f"  scanned {rep['functions']} functions "
                      f"(coverage: {rep['coverage']}), "
                      f"{n_bad} armv7s-only instruction(s)")
                if n_bad:
                    print_bad(rep)
                    if not scan_only:
                        if not force:
                            raise RuntimeError(
                                f"refusing to patch: {n_bad} armv7s-only "
                                f"instruction(s) would SIGILL on armv7 "
                                f"(use --force to override)"
                            )
                        print("  [!] --force: patching anyway. Result WILL crash "
                              "at runtime when execution reaches those instructions.")
                else:
                    print("  [✓] no armv7s-only instructions found.")

        if scan_only:
            continue

        patch_slice_subtype(data, sl, fe, new_subtype=9)
        if sl.has_code_signature:
            needs_resign = True
        patched += 1

    if scan_only and all_reports:
        if as_json:
            print(json.dumps(
                {"reports": [{"functions": r["functions"],
                              "bad": r["bad"],
                              "coverage": r.get("coverage"),
                              "error": r.get("error"),
                              "no_capstone": r["no_capstone"]}
                             for r in all_reports]},
                indent=2, default=str,
            ))
        total_bad = sum(len(r["bad"]) for r in all_reports)
        if any(r["no_capstone"] for r in all_reports):
            print("\n[?] verdict: UNKNOWN (capstone missing)")
        elif any(r.get("error") for r in all_reports):
            print(f"\n[?] verdict: UNKNOWN (scan error)")
        elif any(r.get("coverage") == "entry-only" for r in all_reports):
            print(f"\n[?] verdict: UNRELIABLE (entry-only coverage; "
                  f"install symbol/func-starts or use --force)")
        elif total_bad == 0:
            print("\n[✓] verdict: no armv7s-only instructions. "
                  "Binary should run on armv7.")
        else:
            print(f"\n[!] verdict: {total_bad} armv7s-only instruction(s) "
                  f"found. Binary WILL SIGILL on armv7 at runtime.")

    return (patched, needs_resign)


# ───────────────────────── barriers-only (formerly armv7s2armv6) ─────────────────────────

def encode_armv6k_barrier(insn) -> Optional[int]:
    """DSB/DMB/ISB -> ARMv6K CP15 MCR, preserving condition code."""
    mn = insn.mnemonic.split(".")[0]
    if mn not in BARRIER_MNEMONICS:
        return None
    cond = getattr(insn, "cc", ARM_CC_AL)
    if cond is None or cond < 0 or cond > 0xE:
        return None

    if mn == "dsb":
        crm, opc2 = 0xA, 0x4
    elif mn == "dmb":
        crm, opc2 = 0xA, 0x5
    else:
        crm, opc2 = 0x5, 0x4

    word  = (cond & 0xF) << 28
    word |= 0x0E << 24       # MCR
    word |= 0 << 21
    word |= 0 << 20
    word |= 0x7 << 16        # CRn = c7
    word |= 0x0 << 12        # Rt = r0
    word |= 0xF << 8         # coproc = p15
    word |= (opc2 & 0x7) << 5
    word |= 1 << 4
    word |= crm & 0xF
    return word


def do_barriers_only(data: bytearray, force: bool) -> int:
    if not HAVE_CAPSTONE:
        raise RuntimeError("capstone is required for the barrier translator")
    if parse_fat(bytes(data)) is not None:
        raise RuntimeError(
            "barriers-only handles thin Mach-O only; "
            "extract with `lipo -thin armv7s` first"
        )

    sl = parse_thin(bytes(data), 0)
    if (sl.cpusubtype & 0xFFFFFF) not in (9, 11):
        raise RuntimeError(f"expected armv7/armv7s, got {sl.subtype_name}")
    if sl.is_encrypted:
        raise RuntimeError("slice is encrypted; decrypt first")

    print(f"=== slice @ {sl.file_offset:#x} "
          f"({sl.subtype_name}, {sl.filetype_name}) ===")
    print("  [!] barriers-only: translating DSB/DMB/ISB to CP15 MCR.")
    print("      Everything else (Thumb-2, NEON, movw/movt, ...) is")
    print("      reported and refused. Most real armv7s binaries will")
    print("      be refused — this is expected.")
    warn_userland(sl)

    rep = scan_text(bytes(data), sl, classify_armv6_illegal)
    if rep.get("error"):
        raise RuntimeError(rep["error"])
    if rep["coverage"] == "entry-only":
        print("  [!] entry-only coverage; scan is unreliable.")

    print(f"  scanned {rep['functions']} functions, "
          f"{len(rep['bad'])} armv6-illegal instruction(s)")
    if rep["bad"]:
        print_bad(rep)

    if not rep["bad"]:
        print("  [✓] no armv6-illegal instructions; nothing to do.")
        return 0

    translatable = [
        b for b in rep["bad"]
        if b["mode"] == "arm" and b["mnemonic"].split(".")[0] in BARRIER_MNEMONICS
    ]
    blocked = [b for b in rep["bad"] if b not in translatable]

    if blocked:
        print(f"  [!] {len(blocked)} instruction(s) cannot be translated:")
        for b in blocked[:10]:
            print(f"      {b['address']:#010x} [{b['mode']:<5}] "
                  f"{b['mnemonic']} {b['op_str']:<26} -> {b['reason']}")
        if len(blocked) > 10:
            print(f"      ... and {len(blocked) - 10} more")
        if not force:
            raise RuntimeError(
                "translation refused: unsupported instructions remain "
                "(pass --force to patch anyway)"
            )

    text = locate_text_section(sl)
    if text is None:
        raise RuntimeError("no __text section found")
    text_off, text_vm = text["offset"], text["addr"]

    rewritten = 0
    for b in translatable:
        foff = text_off + (b["address"] - text_vm)
        if foff + 4 > len(data):
            print(f"  [!] {b['address']:#x} outside file; skipping")
            continue

        class _I:
            pass
        fake = _I()
        fake.mnemonic = b["mnemonic"]
        fake.cc = b["cc"] if b["cc"] is not None else ARM_CC_AL
        new = encode_armv6k_barrier(fake)
        if new is None:
            print(f"  [!] {b['address']:#x} could not encode")
            continue
        old = struct.unpack_from("<I", data, foff)[0]
        struct.pack_into("<I", data, foff, new)
        rewritten += 1
        print(f"  [✓] {b['address']:#010x} {b['mnemonic']:<6} -> "
              f"MCR p15,0,r0,... ({old:#010x} -> {new:#010x})")

    if rewritten == 0:
        return 0

    patch_slice_subtype(data, sl, None, new_subtype=6)
    warn_signature(sl)
    return rewritten


# ───────────────────────── ldid ─────────────────────────

def try_ldid_resign(path: Path) -> bool:
    ldid = shutil.which("ldid")
    if not ldid:
        return False
    try:
        r = subprocess.run([ldid, "-S", str(path)], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


# ───────────────────────── CLI ─────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="armbinconv",
        description="Mach-O ARM architecture converter "
                    "(armv7s -> armv7; experimental barriers-only translator).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Requirements:\n"
            "  pip install capstone   # needed for --scan-only and for patching\n"
            "  ldid                   # optional, only for --ldid\n"
            "\n"
            "  --scan-only works without capstone (UNKNOWN verdict).\n"
            "  Patching refuses to proceed without capstone unless\n"
            "  --force or --skip-scan is given.\n"
            "\n"
            "Examples:\n"
            "  armbinconv.py armv7s2armv7 binary --scan-only\n"
            "  armbinconv.py armv7s2armv7 binary --min-os 9.0 --ldid -o binary.armv7\n"
            "  armbinconv.py armv7s2armv7 binary --force -o binary.armv7\n"
            "  armbinconv.py barriers-only binary --force -o binary.armv6\n"
        ),
    )
    ap.add_argument("--version", action="version",
                    version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("armv7s2armv7",
                        help="convert armv7s -> armv7 with VFPv4/hwdiv scan")
    p1.add_argument("input")
    p1.add_argument("-o", "--output")
    p1.add_argument("--scan-only", action="store_true",
                    help="report armv7s-only instructions without modifying "
                         "(works without capstone)")
    p1.add_argument("--force", action="store_true",
                    help="patch even if armv7s-only instructions are found, "
                         "or if the scan is unreliable (result may crash)")
    p1.add_argument("--skip-scan", action="store_true",
                    help="skip the VFPv4 / hwdiv scan entirely")
    p1.add_argument("--min-os", metavar="VERSION", default=None,
                    help="lower LC_VERSION_MIN / LC_BUILD_VERSION to this "
                         "version, e.g. 9.0 or 9.3.6")
    p1.add_argument("--ldid", action="store_true",
                    help="auto re-sign the output with `ldid -S`")
    p1.add_argument("--json", action="store_true",
                    help="(with --scan-only) emit JSON instead of text")

    p2 = sub.add_parser("barriers-only",
                        aliases=["armv7s2armv6"],
                        help="EXPERIMENTAL: translate only DSB/DMB/ISB to CP15")
    p2.add_argument("input")
    p2.add_argument("-o", "--output")
    p2.add_argument("--force", action="store_true",
                    help="patch even if untranslatable instructions remain")
    p2.add_argument("--ldid", action="store_true",
                    help="auto re-sign the output with `ldid -S`")

    args = ap.parse_args(argv)

    src = Path(args.input)
    if not src.is_file():
        print(f"[ERROR] file not found: {src}", file=sys.stderr)
        return 2

    data = bytearray(src.read_bytes())
    orig = bytes(data)

    target_min_os = None
    if getattr(args, "min_os", None):
        try:
            target_min_os = parse_version_string(args.min_os)
        except ValueError:
            print(f"[ERROR] invalid --min-os '{args.min_os}'", file=sys.stderr)
            return 2

    try:
        if args.cmd == "armv7s2armv7":
            patched, needs_resign = do_armv7s2armv7(
                data,
                force=args.force,
                skip_scan=args.skip_scan,
                scan_only=args.scan_only,
                target_min_os=target_min_os,
                as_json=args.json,
            )
            if args.scan_only:
                return 0
            print(f"[i] slices patched: {patched}")
        else:
            n = do_barriers_only(data, force=args.force)
            needs_resign = True
            print(f"[i] instructions rewritten: {n}")

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if data == orig:
        print("[i] no changes.")
        return 0

    out = Path(args.output) if args.output else src
    if out == src:
        backup = Path(str(src) + ".bak")
        if not backup.exists():
            shutil.copy2(src, backup)
            print(f"[i] backup: {backup}")
    out.write_bytes(bytes(data))
    print(f"[✓] written: {out}")

    if needs_resign or args.ldid:
        if args.ldid and try_ldid_resign(out):
            print(f"[✓] re-signed with ldid: {out}")
        else:
            print(f"[!] code signature invalidated. Re-sign before use:")
            print(f"      ldid -S {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())