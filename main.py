#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
armbinconv.py — Mach-O ARM architecture converter.

Modes
-----
  armv7s2armv7   Convert armv7s to armv7.
                 - Default: header-only patch (old behaviour), scans for
                   VFPv4/hwdiv first.
                 - --shim : also rewrite ARM-state `sdiv`/`udiv` into
                   calls to self-contained divide shims placed in __TEXT
                   padding.  Thumb-state sdiv/udiv is NOT handled (warned).

  armv7s2armv6   EXPERIMENTAL.  Default: scan-only + CP15 barriers.
                 - --shim : translate ARM `dmb`/`dsb`/`isb` -> CP15 (in
                   place), translate ARM `sdiv`/`udiv` -> shims (as above),
                   warn about NEON / Thumb-2 / movw / movt / rbit / ubfx /
                   sbfx / bfi / bfc (not translatable by this tool).

Shim mechanics
--------------
  A shim for `sdiv rd, rn, rm`:

      sub  sp, sp, #60
      stmia sp, {r0-r12, lr}        @ save everything
      mov  r0, rn                   @ N
      mov  r1, rm                   @ D
      eor  r4, r0, r1               @ result sign
      cmp  r0, #0 ; rsblt r0, r0, #0
      cmp  r1, #0 ; rsblt r1, r1, #0
      cmp  r1, #0 ; beq .Lzero      @ D == 0 -> 0
      mov  r2, #0 ; mov r3, #0 ; mov r5, #32
  .Lloop:
      adds r0, r0, r0               @ N <<= 1, C = MSB
      adc  r3, r3, r3               @ R = (R<<1) | C
      cmp  r3, r1
      subcs r3, r3, r1              @ if R >= D: R -= D
      adc  r2, r2, r2               @ Q = Q*2 + C
      subs r5, r5, #1 ; bne .Lloop
      mov  r0, r2
      cmp  r4, #0 ; rsblt r0, r0, #0
      b    .Ldone
  .Lzero:
      mov  r0, #0
  .Ldone:
      str  r0, [sp, #56]            @ save result
      ldmia sp, {r0-r12, lr}        @ restore all except our result
      ldr  rd, [sp, #56]            @ only rd changed
      add  sp, sp, #60
      bx   lr

  `udiv` shim is the same minus sign handling.

  Caller does `bl<cc> shim` in place of `sdiv`/`udiv` (same 32-bit ARM
  instruction slot, so the file size is unchanged).

  Safety check: the original function MUST have saved lr to stack before
  the sdiv/udiv site.  Otherwise `bl` would clobber the unsaved lr of a
  leaf function and `bx lr` in the epilogue would jump back into the
  middle of the function.  Such sites are skipped (warning).

Requirements
------------
  pip install capstone
  ldid     # optional, for --ldid
  Python >= 3.8

Examples
--------
  # 1. Scan, don't touch:
  ./armbinconv.py armv7s2armv7 binary --scan-only

  # 2. Full shim translation + re-sign:
  ./armbinconv.py armv7s2armv7 binary --shim --min-os 9.0 --ldid -o out

  # 3. For armv6 (mostly experimental — barriers + sdiv/udiv shims):
  ./armbinconv.py armv7s2armv6 binary --shim --min-os 4.0 --ldid -o out
"""

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
    parts = s.split(".")
    while len(parts) < 3:
        parts.append("0")
    return (int(parts[0]) << 16) | (int(parts[1]) << 8) | int(parts[2])


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
                        size=seg["filesize"], offset=seg["fileoff"])
    return None


# ───────────────────────── classifiers ─────────────────────────

ARMV7S_ONLY_MNEMONICS = {
    "vfma":  "VFPv4 fused multiply-add (armv7s/A6 only)",
    "vfms":  "VFPv4 fused multiply-subtract (armv7s/A6 only)",
    "vfnma": "VFPv4 negate-fused-multiply-add (armv7s/A6 only)",
    "vfnms": "VFPv4 negate-fused-multiply-subtract (armv7s/A6 only)",
    "vcvtb": "VFPv4 half-precision convert (armv7s/A6 only)",
    "vcvtt": "VFPv4 half-precision convert (armv7s/A6 only)",
    "udiv":  "hardware unsigned divide (armv7s/A6 only)",
    "sdiv":  "hardware signed divide (armv7s/A6 only)",
}

ARMV6_ILLEGAL_MNEMONICS = {
    "udiv":   "ARMv7 integer divide",
    "sdiv":   "ARMv7 integer divide",
    "clrex":  "ARMv7 exclusive monitor hint",
    "dsb":    "ARMv7 memory barrier",
    "dmb":    "ARMv7 memory barrier",
    "isb":    "ARMv7 memory barrier",
    "movw":   "ARMv6T2 wide move",
    "movt":   "ARMv6T2 wide move",
    "rbit":   "ARMv6T2 bit reverse",
    "ubfx":   "ARMv6T2 bitfield extract",
    "sbfx":   "ARMv6T2 bitfield extract",
    "bfi":    "ARMv6T2 bitfield insert",
    "bfc":    "ARMv6T2 bitfield clear",
    "pldw":   "ARMv7 preload for write",
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


def classify_armv7s_only(insn, mode: str) -> Optional[str]:
    if HAVE_CAPSTONE and cs_arm is not None:
        try:
            vfp4 = getattr(cs_arm, "ARM_GRP_VFP4", None)
            if vfp4 is not None and insn.group(vfp4):
                return f"VFPv4 instruction ({insn.mnemonic}) — needs armv7s/A6"
        except Exception:
            pass
    mn = insn.mnemonic.split(".")[0]
    return ARMV7S_ONLY_MNEMONICS.get(mn)


def classify_armv6_illegal(insn, mode: str) -> Optional[str]:
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


def scan_text(data: bytes, sl: Slice, classifier) -> dict:
    report = {"functions": 0, "bad": [], "no_capstone": False, "has_thumb": False}
    if not HAVE_CAPSTONE:
        report["no_capstone"] = True
        return report

    text = locate_text_section(sl)
    if text is None or text["offset"] + text["size"] > len(data):
        return report

    text_data   = data[text["offset"]:text["offset"] + text["size"]]
    text_vmaddr = text["addr"]
    funcs, func_modes = collect_function_starts(data, sl)
    funcs = sorted(funcs)

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
                        "address": insn.address, "mnemonic": insn.mnemonic,
                        "op_str": insn.op_str, "reason": reason, "mode": mode,
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
    off = sl.file_offset + sl.hdr_size
    patched = 0
    for _ in range(sl.ncmds):
        if off + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from(sl.endian + "II", data, off)
        if cmdsize < 8 or off + cmdsize > len(data):
            break
        if cmd == LC_VERSION_MIN_IPHONEOS:
            old_ver, old_sdk = struct.unpack_from(sl.endian + "II", data, off + 8)
            struct.pack_into(sl.endian + "II", data, off + 8, target, target)
            print(f"  [LC_VERSION_MIN_IPHONEOS @ {off:#x}] "
                  f"{unpack_version_word(old_ver)} -> {unpack_version_word(target)}")
            patched += 1
        elif cmd == LC_BUILD_VERSION:
            platform, old_min, old_sdk, ntools = struct.unpack_from(
                sl.endian + "IIII", data, off + 8)
            struct.pack_into(sl.endian + "II", data, off + 12, target, target)
            print(f"  [LC_BUILD_VERSION @ {off:#x}] minos "
                  f"{unpack_version_word(old_min)} -> {unpack_version_word(target)}")
            patched += 1
        off += cmdsize
    return patched


def warn_signature(sl: Slice) -> None:
    if sl.has_code_signature:
        print("  [!] LC_CODE_SIGNATURE present. Any patch invalidates it.")
        print("      Re-sign with `ldid -S <out>` or pass --ldid.")


# ═══════════════════════════════════════════════════════════════════════════
#  ARM instruction encoders (ARM state, 32-bit)
# ═══════════════════════════════════════════════════════════════════════════

def _enc_cond(cond: int) -> int:
    return (cond & 0xF) << 28


def encode_arm_imm(val: int) -> Optional[int]:
    """Encode `val` as ARM rotated-immediate, return 12-bit value or None."""
    val &= 0xFFFFFFFF
    for rot in range(16):
        sh = 2 * rot
        ror = val if sh == 0 else ((val >> sh) | (val << (32 - sh))) & 0xFFFFFFFF
        if ror < 256:
            return (rot << 8) | ror
    return None


def enc_dp_reg(opcode: int, rd: int, rn: int, rm: int, s: int = 0, cond: int = 0xE) -> int:
    """Data processing, register form. opcode: 0 AND, 1 EOR, 2 SUB, 3 RSB,
    4 ADD, 5 ADC, 6 SBC, 7 RSC, 8 TST, 9 TEQ, 10 CMP, 11 CMN, 12 ORR,
    13 MOV, 14 BIC, 15 MVN."""
    base = (opcode << 21) | (s << 20) | ((rn & 0xF) << 16) | ((rd & 0xF) << 12) | (rm & 0xF)
    return _enc_cond(cond) | base


def enc_dp_imm(opcode: int, rd: int, rn: int, imm: int, s: int = 0, cond: int = 0xE) -> int:
    imm12 = encode_arm_imm(imm)
    if imm12 is None:
        raise ValueError(f"immediate {imm:#x} not encodable")
    base = 0x02000000 | (opcode << 21) | (s << 20) | \
           ((rn & 0xF) << 16) | ((rd & 0xF) << 12) | imm12
    return _enc_cond(cond) | base


def enc_b(offset_bytes: int, cond: int = 0xE) -> int:
    if offset_bytes & 3:
        raise ValueError("branch offset must be 4-aligned")
    return _enc_cond(cond) | 0x0A000000 | ((offset_bytes >> 2) & 0xFFFFFF)


def enc_bl(offset_bytes: int, cond: int = 0xE) -> int:
    if offset_bytes & 3:
        raise ValueError("branch offset must be 4-aligned")
    return _enc_cond(cond) | 0x0B000000 | ((offset_bytes >> 2) & 0xFFFFFF)


def enc_bx(rm: int, cond: int = 0xE) -> int:
    return _enc_cond(cond) | 0x012FFF10 | (rm & 0xF)


def enc_ldr_str_imm(load: bool, rd: int, rn: int, imm: int, cond: int = 0xE) -> int:
    base = 0x05900000 if load else 0x05800000
    return _enc_cond(cond) | base | ((rn & 0xF) << 16) | ((rd & 0xF) << 12) | (imm & 0xFFF)


def enc_ldm_stm_ia(rn: int, reglist: int, load: bool, writeback: bool, cond: int = 0xE) -> int:
    """LDMIA/STMIA (P=0, U=1)."""
    base = 0x08000000 | (1 << 23)  # bit 27=1, P=0, U=1
    if writeback:
        base |= (1 << 21)
    if load:
        base |= (1 << 20)
    base |= ((rn & 0xF) << 16) | (reglist & 0xFFFF)
    return _enc_cond(cond) | base


# ═══════════════════════════════════════════════════════════════════════════
#  Shim generation for sdiv / udiv
# ═══════════════════════════════════════════════════════════════════════════

def _gen_unsigned_divide_body(rd: int, rn: int, rm: int, signed: bool) -> list:
    """
    Returns a list of 32-bit ARM words implementing
        rd = rn / rm   (signed or unsigned, ARM semantics: div by 0 -> 0)
    Self-contained; saves r0-r12+lr, writes only rd.
    """
    if rd > 12:
        raise ValueError("shim: rd must be r0-r12")
    if rn > 14 or rm > 14:
        raise ValueError("shim: rn/rm must be r0-r14 (r13/r14 excluded by caller)")

    I = []
    # sub sp, sp, #60   -- 14 words (r0-r12, lr) + 1 word (result)
    I.append(enc_dp_imm(2, 13, 13, 60))
    # stmia sp, {r0-r12, lr}  -- no writeback
    I.append(enc_ldm_stm_ia(13, 0x5FFF, load=False, writeback=False))

    # mov r0, rn  (N)
    I.append(enc_dp_reg(13, 0, 0, rn))
    # mov r1, rm  (D)
    I.append(enc_dp_reg(13, 1, 0, rm))

    if signed:
        # eor r4, r0, r1  -- result sign
        I.append(enc_dp_reg(1, 4, 0, 1))
        # cmp r0,#0 ; rsblt r0,r0,#0
        I.append(enc_dp_imm(10, 0, 0, 0, s=1))
        I.append(enc_dp_imm(3, 0, 0, 0, cond=0xB))
        # cmp r1,#0 ; rsblt r1,r1,#0
        I.append(enc_dp_imm(10, 0, 1, 0, s=1))
        I.append(enc_dp_imm(3, 1, 1, 0, cond=0xB))

    # cmp r1,#0 ; beq zero_label
    I.append(enc_dp_imm(10, 0, 1, 0, s=1))
    eq_i = len(I); I.append(0)

    # mov r2,#0 (Q) ; mov r3,#0 (R) ; mov r5,#32 (counter)
    I.append(enc_dp_imm(13, 2, 0, 0))
    I.append(enc_dp_imm(13, 3, 0, 0))
    I.append(enc_dp_imm(13, 5, 0, 32))

    loop_i = len(I)
    I.append(enc_dp_reg(4, 0, 0, 0, s=1))            # adds r0, r0, r0
    I.append(enc_dp_reg(5, 3, 3, 3))                 # adc  r3, r3, r3
    I.append(enc_dp_reg(10, 0, 3, 1, s=1))           # cmp  r3, r1
    I.append(enc_dp_reg(2, 3, 3, 1, cond=0x2))       # subcs r3, r3, r1
    I.append(enc_dp_reg(5, 2, 2, 2))                 # adc  r2, r2, r2
    I.append(enc_dp_imm(2, 5, 5, 1, s=1))            # subs r5, r5, #1
    bne_i = len(I); I.append(0)

    # mov r0, r2
    I.append(enc_dp_reg(13, 0, 0, 2))
    if signed:
        # cmp r4,#0 ; rsblt r0,r0,#0
        I.append(enc_dp_imm(10, 0, 4, 0, s=1))
        I.append(enc_dp_imm(3, 0, 0, 0, cond=0xB))
    b_done_i = len(I); I.append(0)

    zero_i = len(I)
    I.append(enc_dp_imm(13, 0, 0, 0))                # mov r0, #0
    done_i = len(I)
    I.append(enc_ldr_str_imm(False, 0, 13, 56))      # str r0, [sp, #56]
    I.append(enc_ldm_stm_ia(13, 0x5FFF, load=True, writeback=False))  # ldmia sp, {r0-r12, lr}
    I.append(enc_ldr_str_imm(True, rd, 13, 56))      # ldr rd, [sp, #56]
    I.append(enc_dp_imm(4, 13, 13, 60))              # add sp, sp, #60
    I.append(enc_bx(14))                             # bx lr

    # Patch branches.  PC = addr_of_branch + 8
    def patch(idx, target_idx, cond):
        off = (target_idx - idx - 2) * 4
        if cond == 0x0:
            I[idx] = 0x0A000000 | ((off >> 2) & 0xFFFFFF)   # BEQ
        elif cond == 0x1:
            I[idx] = 0x1A000000 | ((off >> 2) & 0xFFFFFF)   # BNE
        else:
            I[idx] = 0xEA000000 | ((off >> 2) & 0xFFFFFF)   # B
    patch(eq_i, zero_i, 0x0)
    patch(bne_i, loop_i, 0x1)
    patch(b_done_i, done_i, 0xE)
    return I


# ═══════════════════════════════════════════════════════════════════════════
#  LR-saved check
# ═══════════════════════════════════════════════════════════════════════════

def lr_saved_before(data: bytes, sl: Slice, fn_addr: int, insn_addr: int,
                    mode: str = "arm") -> bool:
    """Heuristic: does the function save lr to stack before insn_addr?"""
    if not HAVE_CAPSTONE:
        return False
    text = locate_text_section(sl)
    if text is None:
        return False
    text_data = bytes(data[text["offset"]:text["offset"] + text["size"]])
    start = fn_addr - text["addr"]
    if start < 0 or start >= len(text_data):
        return False
    chunk = text_data[start:start + 256]  # prologue window
    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB if mode == "thumb" else CS_MODE_ARM)
    for insn in md.disasm(chunk, fn_addr):
        if insn.address >= insn_addr:
            break
        mn = insn.mnemonic.split(".")[0]
        op = insn.op_str
        if mn == "push" and "lr" in op:
            return True
        if mn in ("str", "stmdb", "stmia", "stmfd", "stmea"):
            if "lr" in op and "sp" in op:
                return True
        # if there is any bl/bx to lr before insn, then LR was already
        # considered throwaway: the function must have saved it (or is leaf
        # and about to crash on its own). Treat as unsafe unless we saw a
        # push earlier.
        if mn in ("bl", "blx") and insn.address < insn_addr:
            # non-leaf function; LR must have been saved earlier for the
            # function to be correct, but we can't prove it. Fail safe.
            return False
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  __TEXT padding discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_text_padding(data: bytes, sl: Slice):
    """
    Returns (file_offset, size) of the tail padding inside __TEXT
    (between the last section's end / end of load commands and
    __TEXT.fileoff + __TEXT.filesize).
    """
    text_seg = None
    for seg in sl.segments:
        if seg["name"] == "__TEXT":
            text_seg = seg
            break
    if text_seg is None:
        raise RuntimeError("__TEXT segment not found")

    max_end = 0
    for s in sl.sections:
        if s["seg"] != "__TEXT":
            continue
        if s["offset"] <= 0 or s["size"] <= 0:
            continue
        e = s["offset"] + s["size"]
        if e > max_end:
            max_end = e
    hdr_end = sl.file_offset + sl.hdr_size + sl.sizeofcmds
    if hdr_end > max_end:
        max_end = hdr_end

    text_start = text_seg["fileoff"]
    text_end = text_start + text_seg["filesize"]
    if max_end < text_start + sl.hdr_size + sl.sizeofcmds:
        max_end = text_start + sl.hdr_size + sl.sizeofcmds
    pad_off = max_end
    pad_size = text_end - pad_off
    return pad_off, max(0, pad_size)


# ═══════════════════════════════════════════════════════════════════════════
#  Barrier -> ARMv6K CP15 MCR
# ═══════════════════════════════════════════════════════════════════════════

def encode_armv6k_barrier(insn) -> Optional[int]:
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
    word |= 0x0E << 24
    word |= 0 << 21
    word |= 0 << 20
    word |= 0x7 << 16
    word |= 0x0 << 12
    word |= 0xF << 8
    word |= (opc2 & 0x7) << 5
    word |= 1 << 4
    word |= crm & 0xF
    return word


# ═══════════════════════════════════════════════════════════════════════════
#  Unified shim rewrite
# ═══════════════════════════════════════════════════════════════════════════

def _do_shim_rewrite(data: bytearray, sl: Slice, fat_entry,
                     target_subtype: int,
                     translate_barriers: bool,
                     force_leaf: bool,
                     scan_only: bool) -> tuple[int, int, int]:
    """
    Returns (patched_insns, shims_generated, shim_bytes).
    """
    if not HAVE_CAPSTONE:
        raise RuntimeError("capstone is required for --shim")

    text = locate_text_section(sl)
    if text is None:
        raise RuntimeError("no __text section")
    text_off = text["offset"]
    text_vm  = text["addr"]
    text_size = text["size"]
    if text_off + text_size > len(data):
        raise RuntimeError("__text extends past end of file")
    text_data = bytes(data[text_off:text_off + text_size])

    funcs, modes = collect_function_starts(bytes(data), sl)
    funcs = sorted(funcs)

    requests = []          # list of dicts (arm sdiv/udiv -> shim)
    leaf_skips = []        # (addr, reason)
    thumb_skips = []       # (addr, mnemonic)
    barrier_sites = []     # (addr, mnemonic, cc)
    other_bad = []         # (addr, mnemonic, reason)

    for i, fn in enumerate(funcs):
        if fn < text_vm or fn >= text_vm + text_size:
            continue
        end = funcs[i + 1] if i + 1 < len(funcs) else text_vm + text_size
        if end <= fn:
            continue
        mode = modes.get(fn, "arm")
        md = Cs(CS_ARCH_ARM, (CS_MODE_THUMB if mode == "thumb" else CS_MODE_ARM)
                | CS_MODE_LITTLE_ENDIAN)
        chunk = text_data[fn - text_vm:min(end - text_vm, text_size)]
        for insn in md.disasm(chunk, fn):
            mn = insn.mnemonic.split(".")[0]
            if mode == "thumb":
                if mn in ("sdiv", "udiv"):
                    thumb_skips.append((insn.address, mn))
                # Thumb-2 detection is expensive; skip - user gets the
                # standard "warn" from the scan-based check elsewhere.
                continue

            if mn in ("sdiv", "udiv"):
                ops = [p.strip() for p in insn.op_str.split(",")]
                if len(ops) != 3:
                    other_bad.append((insn.address, mn,
                                      f"odd operand count: {insn.op_str}"))
                    continue
                try:
                    rd = int(ops[0].lstrip("r"))
                    rn = int(ops[1].lstrip("r"))
                    rm = int(ops[2].lstrip("r"))
                except ValueError:
                    other_bad.append((insn.address, mn,
                                      f"non-register operands: {insn.op_str}"))
                    continue
                # Reject sp/pc, and lr (because bl overwrites it before
                # shim runs)
                if rd > 12 or rn > 12 or rm > 12:
                    other_bad.append((insn.address, mn,
                                      f"unsupported regs: {insn.op_str}"))
                    continue
                if not lr_saved_before(bytes(data), sl, fn, insn.address, "arm"):
                    if force_leaf:
                        # We'll still do it, but flag it.
                        leaf_skips.append((insn.address,
                                           "lr not saved (forcing)"))
                    else:
                        leaf_skips.append((insn.address,
                                           "lr not saved before sdiv/udiv"))
                        continue
                requests.append(dict(addr=insn.address, mnemonic=mn,
                                     rd=rd, rn=rn, rm=rm,
                                     cc=insn.cc if insn.cc is not None else 0xE))
                continue

            if translate_barriers and mn in BARRIER_MNEMONICS:
                barrier_sites.append((insn.address, mn,
                                      insn.cc if insn.cc is not None else 0xE))

    # Report
    print(f"  [i] ARM sdiv/udiv to shim: {len(requests)}")
    if leaf_skips:
        print(f"  [!] {len(leaf_skips)} skipped (lr not saved before):")
        for a, why in leaf_skips[:8]:
            print(f"      {a:#010x}  {why}")
    if thumb_skips:
        print(f"  [!] {len(thumb_skips)} Thumb-state sdiv/udiv found "
              f"— NOT translated (only ARM-state shims supported)")
        for a, mn in thumb_skips[:8]:
            print(f"      {a:#010x} [{mn}]")
    if other_bad:
        print(f"  [!] {len(other_bad)} unsupported sdiv/udiv:")
        for a, mn, why in other_bad[:8]:
            print(f"      {a:#010x} [{mn}] {why}")
    if translate_barriers:
        print(f"  [i] barriers to CP15: {len(barrier_sites)}")

    if scan_only:
        return (0, 0, 0)

    # Generate shims (dedupe)
    shim_cache = {}   # key -> offset in shim_blob
    shim_blob = bytearray()

    def get_shim(mn, rd, rn, rm):
        key = (mn, rd, rn, rm)
        if key in shim_cache:
            return shim_cache[key]
        off = len(shim_blob)
        words = _gen_unsigned_divide_body(rd, rn, rm, signed=(mn == "sdiv"))
        for w in words:
            shim_blob.extend(struct.pack("<I", w))
        shim_cache[key] = off
        return off

    for req in requests:
        get_shim(req["mnemonic"], req["rd"], req["rn"], req["rm"])
    while len(shim_blob) % 4:
        shim_blob.append(0)

    shim_bytes = len(shim_blob)

    # Locate / verify padding
    pad_off, pad_size = find_text_padding(bytes(data), sl)
    print(f"  [i] __TEXT padding: off={pad_off:#x} size={pad_size} bytes")
    if shim_bytes > pad_size:
        raise RuntimeError(
            f"not enough __TEXT padding: need {shim_bytes} B, have {pad_size} B. "
            f"Strip the binary first (`strip`) or use a smaller build."
        )

    if shim_bytes:
        if pad_off + shim_bytes > len(data):
            raise RuntimeError("padding beyond end of file")
        data[pad_off:pad_off + shim_bytes] = shim_blob
        text_seg = next(s for s in sl.segments if s["name"] == "__TEXT")
        shim_vmaddr = text_seg["vmaddr"] + (pad_off - text_seg["fileoff"])
    else:
        shim_vmaddr = 0

    patched_insns = 0
    # Patch sdiv/udiv -> bl shim
    for req in requests:
        off_in_blob = shim_cache[(req["mnemonic"], req["rd"], req["rn"], req["rm"])]
        shim_addr = shim_vmaddr + off_in_blob
        delta = shim_addr - (req["addr"] + 8)
        if delta & 3:
            print(f"  [!] {req['addr']:#x}: shim offset not 4-aligned")
            continue
        if delta < -(1 << 25) or delta >= (1 << 25):
            print(f"  [!] {req['addr']:#x}: shim out of bl range ({delta:#x})")
            continue
        word = enc_bl(delta, cond=req["cc"])
        foff = text_off + (req["addr"] - text_vm)
        if foff + 4 > len(data):
            continue
        struct.pack_into("<I", data, foff, word)
        patched_insns += 1

    # Patch barriers in place (armv6 CP15)
    for addr, mn, cc in barrier_sites:
        class _I: pass
        fake = _I()
        fake.mnemonic = mn
        fake.cc = cc
        new = encode_armv6k_barrier(fake)
        if new is None:
            continue
        foff = text_off + (addr - text_vm)
        if foff + 4 > len(data):
            continue
        struct.pack_into("<I", data, foff, new)
        patched_insns += 1

    return (patched_insns, len(shim_cache), shim_bytes)


# ═══════════════════════════════════════════════════════════════════════════
#  armv7s -> armv7  /  armv7s -> armv6
# ═══════════════════════════════════════════════════════════════════════════

def do_armv7s2armv7(data: bytearray, force: bool, skip_scan: bool,
                    scan_only: bool, target_min_os: Optional[int],
                    as_json: bool, use_shim: bool,
                    force_leaf: bool) -> tuple[int, bool]:
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
            print("=== FAT BINARY: armv7 AND armv7s slices present ===")
            print(f"  armv7 slice(s) : {len(armv7_entries)}")
            print(f"  armv7s slice(s): {len(armv7s_entries)}")
            print("  [i] RECOMMENDED: `lipo -thin armv7 <input> -output <out>`")
            if not force:
                return (0, False)
            print("  [!] --force: patching armv7s slice(s) anyway.")
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
                print(f"  [!] requires iOS {unpack_version_word(mov)}; "
                      f"pass --min-os 9.0 to lower.")
            elif target_min_os is not None:
                patch_min_os(data, sl, target_min_os)
            else:
                print(f"  [!] requires iOS {unpack_version_word(mov)}; "
                      f"pass --min-os 9.0 to lower.")

        if sl.is_encrypted and not skip_scan and not scan_only:
            print(f"  [!] encrypted slice; scan skipped.")
            if not force:
                raise RuntimeError("encrypted slice; use --force or --skip-scan")

        # Scan for VFPv4 etc.
        if not skip_scan and not sl.is_encrypted:
            print("  scanning for VFPv4 / hwdiv (non-translatable)...")
            rep = scan_text(bytes(data), sl, classify_armv7s_only)
            all_reports.append(rep)
            if rep["no_capstone"]:
                print("  [!] capstone not installed")
                if not force and not scan_only and not use_shim:
                    raise RuntimeError("capstone missing; --force required")
            elif rep["functions"] == 0:
                print("  [i] no functions found.")
            else:
                n_bad = len(rep["bad"])
                vfp_bad = [b for b in rep["bad"]
                           if b["mnemonic"].split(".")[0] in
                           ("vfma", "vfms", "vfnma", "vfnms", "vcvtb", "vcvtt")]
                hwdiv_bad = [b for b in rep["bad"]
                             if b["mnemonic"].split(".")[0] in ("sdiv", "udiv")]
                print(f"  scanned {rep['functions']} funcs: "
                      f"{len(vfp_bad)} VFPv4, {len(hwdiv_bad)} hwdiv "
                      f"(total {n_bad})")
                if vfp_bad:
                    print_bad({"bad": vfp_bad})
                    if not use_shim and not force and not scan_only:
                        raise RuntimeError(
                            f"refusing: {len(vfp_bad)} VFPv4 insns cannot be "
                            f"translated by shim (use --force)")
                if hwdiv_bad and not use_shim:
                    print(f"  [i] {len(hwdiv_bad)} sdiv/udiv — pass --shim "
                          f"to auto-translate.")
                    if not force and not scan_only:
                        raise RuntimeError(
                            f"refusing: {len(hwdiv_bad)} hwdiv insns "
                            f"(use --shim or --force)")

        if scan_only:
            continue

        if use_shim:
            print("  translating sdiv/udiv via shims...")
            p, n_shims, n_bytes = _do_shim_rewrite(
                data, sl, fe, target_subtype=9,
                translate_barriers=False,
                force_leaf=force_leaf, scan_only=False)
            print(f"  [✓] {p} insns -> shims ({n_shims} unique, {n_bytes} bytes)")
            if sl.has_code_signature:
                needs_resign = True

        patch_slice_subtype(data, sl, fe, new_subtype=9)
        if sl.has_code_signature:
            needs_resign = True
        patched += 1

    if scan_only and all_reports and as_json:
        print(json.dumps(
            {"reports": [{"functions": r["functions"], "bad": r["bad"],
                          "no_capstone": r["no_capstone"]} for r in all_reports]},
            indent=2, default=str))
    return (patched, needs_resign)


def do_armv7s2armv6(data: bytearray, force: bool, use_shim: bool,
                    force_leaf: bool, target_min_os: Optional[int]) -> tuple[int, bool]:
    if parse_fat(bytes(data)) is not None:
        raise RuntimeError("armv7s2armv6 handles thin Mach-O only; "
                           "`lipo -thin armv7s` first")
    sl = parse_thin(bytes(data), 0)
    if (sl.cpusubtype & 0xFFFFFF) not in (9, 11):
        raise RuntimeError(f"expected armv7/armv7s, got {sl.subtype_name}")
    if sl.is_encrypted:
        raise RuntimeError("slice is encrypted; decrypt first")

    print(f"=== slice @ {sl.file_offset:#x} "
          f"({sl.subtype_name}, {sl.filetype_name}) ===")
    mov = sl.min_os_word
    if mov is not None and target_min_os is not None:
        patch_min_os(data, sl, target_min_os)

    # Scan (informational)
    if HAVE_CAPSTONE:
        rep = scan_text(bytes(data), sl, classify_armv6_illegal)
        print(f"  scanned {rep['functions']} funcs: "
              f"{len(rep['bad'])} armv6-illegal insn")
        if rep["bad"]:
            print_bad(rep)

    needs_resign = sl.has_code_signature
    patched_total = 0

    if use_shim:
        print("  translating barriers (CP15) + sdiv/udiv (shims)...")
        p, n_shims, n_bytes = _do_shim_rewrite(
            data, sl, None, target_subtype=6,
            translate_barriers=True,
            force_leaf=force_leaf, scan_only=False)
        print(f"  [✓] {p} insns rewritten ({n_shims} shims, {n_bytes} bytes)")
        patched_total += p
        needs_resign = True
    else:
        # Old behaviour: just barriers
        if not HAVE_CAPSTONE:
            raise RuntimeError("capstone required for armv7s2armv6")
        rep = scan_text(bytes(data), sl, classify_armv6_illegal)
        text = locate_text_section(sl)
        if text is not None:
            text_off, text_vm = text["offset"], text["addr"]
            rewritten = 0
            for b in rep["bad"]:
                if b["mnemonic"].split(".")[0] in BARRIER_MNEMONICS and b["mode"] == "arm":
                    class _I: pass
                    fake = _I()
                    fake.mnemonic = b["mnemonic"]
                    fake.cc = b["cc"] if b["cc"] is not None else ARM_CC_AL
                    new = encode_armv6k_barrier(fake)
                    if new is None:
                        continue
                    foff = text_off + (b["address"] - text_vm)
                    if foff + 4 > len(data):
                        continue
                    struct.pack_into("<I", data, foff, new)
                    rewritten += 1
            if rewritten:
                print(f"  [✓] {rewritten} barriers -> CP15")
                patched_total += rewritten
                needs_resign = True
        if not force and not use_shim:
            print("  [i] for sdiv/udiv translation, pass --shim")

    patch_slice_subtype(data, sl, None, new_subtype=6)
    needs_resign = True
    return (patched_total, needs_resign)


# ═══════════════════════════════════════════════════════════════════════════
#  ldid
# ═══════════════════════════════════════════════════════════════════════════

def try_ldid_resign(path: Path) -> bool:
    ldid = shutil.which("ldid")
    if not ldid:
        return False
    try:
        r = subprocess.run([ldid, "-S", str(path)], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="armbinconv",
        description="Mach-O ARM architecture converter "
                    "(armv7s -> armv7 / armv7s -> armv6) with optional "
                    "shim-based sdiv/udiv translation.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("armv7s2armv7",
                        help="convert armv7s -> armv7 (optionally with shims)")
    p1.add_argument("input")
    p1.add_argument("-o", "--output")
    p1.add_argument("--scan-only", action="store_true")
    p1.add_argument("--force", action="store_true")
    p1.add_argument("--skip-scan", action="store_true")
    p1.add_argument("--shim", action="store_true",
                    help="translate ARM sdiv/udiv via shims")
    p1.add_argument("--force-leaf", action="store_true",
                    help="patch even if the containing function didn't save lr")
    p1.add_argument("--min-os", metavar="VERSION", default=None)
    p1.add_argument("--ldid", action="store_true")
    p1.add_argument("--json", action="store_true")

    p2 = sub.add_parser("armv7s2armv6",
                        help="EXPERIMENTAL: convert armv7/armv7s -> armv6")
    p2.add_argument("input")
    p2.add_argument("-o", "--output")
    p2.add_argument("--force", action="store_true")
    p2.add_argument("--shim", action="store_true",
                    help="translate barriers (CP15) + ARM sdiv/udiv (shims)")
    p2.add_argument("--force-leaf", action="store_true")
    p2.add_argument("--min-os", metavar="VERSION", default=None)
    p2.add_argument("--ldid", action="store_true")

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
                data, force=args.force, skip_scan=args.skip_scan,
                scan_only=args.scan_only, target_min_os=target_min_os,
                as_json=args.json, use_shim=args.shim,
                force_leaf=args.force_leaf)
            if args.scan_only:
                return 0
            print(f"[i] slices patched: {patched}")
        else:
            n, needs_resign = do_armv7s2armv6(
                data, force=args.force, use_shim=args.shim,
                force_leaf=args.force_leaf, target_min_os=target_min_os)
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
            print(f"[!] code signature invalidated. Re-sign:")
            print(f"      ldid -S {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
