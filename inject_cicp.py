"""
Standalone helper script to inject both the 'cICP' chunk and 'iCCP' profile chunk
(read from your C++ project's icc_profile.h) into any PNG. This tells Chromium
that the PNG is encoded in BT.2020 PQ HDR so it renders correctly.
"""

import sys
import os
import struct
import zlib

def make_chunk(chunk_type, chunk_data):
    length = len(chunk_data)
    length_bytes = struct.pack(">I", length)
    crc = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
    crc_bytes = struct.pack(">I", crc)
    return length_bytes + chunk_type + chunk_data + crc_bytes

def load_icc_profile():
    h_path = r"c:\Users\Andrew Ke\Desktop\jxr_to_png\icc_profile.h"
    if not os.path.exists(h_path):
        raise FileNotFoundError(f"Could not find icc_profile.h at {h_path}")
        
    with open(h_path, "r") as f:
        content = f.read()
        
    start = content.find("icc_data[] =")
    if start == -1:
        raise ValueError("Could not find icc_data in icc_profile.h")
    start = content.find("{", start)
    end = content.find("}", start)
    bytes_str = content[start+1:end]
    
    bytes_list = []
    for val in bytes_str.split(","):
        val = val.strip()
        if val.startswith("0x") or val.startswith("0X"):
            bytes_list.append(int(val, 16))
    return bytes(bytes_list)

def inject_hdr_chunks_to_png(png_path):
    if not os.path.exists(png_path):
        print(f"Error: File '{png_path}' does not exist.")
        return False

    # Load and compress the ICC Profile
    try:
        raw_icc = load_icc_profile()
        compressed_icc = zlib.compress(raw_icc)
    except Exception as e:
        print(f"Error loading ICC profile: {e}")
        return False

    with open(png_path, "rb") as f:
        data = f.read()

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        print(f"Error: '{png_path}' is not a valid PNG file.")
        return False

    out_chunks = [data[:8]]
    pos = 8
    ihdr_found = False
    
    # 1. Prepare cICP chunk
    cicp_data = b"\x09\x10\x00\x01" # BT.2020, SMPTE ST 2084/PQ, RGB, Full Range
    new_cicp = make_chunk(b"cICP", cicp_data)

    # 2. Prepare iCCP chunk
    # Format: Profile Name (1-79 bytes, Latin-1 null-terminated) + Compression Method (1 byte, 0) + Compressed Profile
    iccp_data = b"Rec2100PQ\x00\x00" + compressed_icc
    new_iccp = make_chunk(b"iCCP", iccp_data)

    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        crc = data[pos+8+length:pos+12+length]

        # Skip existing cICP and iCCP chunks to avoid duplicates
        if chunk_type in (b"cICP", b"iCCP"):
            pos += 12 + length
            continue

        out_chunks.append(data[pos:pos+12+length])

        if chunk_type == b"IHDR":
            ihdr_found = True
            # Inject both right after IHDR
            out_chunks.append(new_cicp)
            out_chunks.append(new_iccp)

        pos += 12 + length

    if not ihdr_found:
        print("Error: IHDR chunk not found.")
        return False

    with open(png_path, "wb") as f:
        f.write(b"".join(out_chunks))
    print(f"Successfully injected cICP and iCCP PQ HDR metadata into: {png_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py inject_cicp.py <path_to_png_file>")
        sys.exit(1)
    
    inject_hdr_chunks_to_png(sys.argv[1])
