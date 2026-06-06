"""
Downsamples PQ-encoded HDR AVIF images to 16-bit PNG.
Performs the downsampling interpolation in 32-bit floating-point linear light (linear gamma)
to prevent color and highlight distortion, before re-encoding to PQ space.
Applies the zscale color pipeline and injects the 'cICP' and 'iCCP' chunks to allow native 
HDR rendering in Chrome.
"""

import argparse
import json
import os
import struct
import subprocess
import zlib

def make_chunk(chunk_type, chunk_data):
    length = len(chunk_data)
    length_bytes = struct.pack(">I", length)
    crc = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
    crc_bytes = struct.pack(">I", crc)
    return length_bytes + chunk_type + chunk_data + crc_bytes

def load_icc_profile():
    icc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Rec2100PQ.icc")
    if not os.path.exists(icc_path):
        raise FileNotFoundError(f"Could not find Rec2100PQ.icc at {icc_path}")
    with open(icc_path, "rb") as f:
        return f.read()

def inject_hdr_chunks_to_png(png_path):
    try:
        raw_icc = load_icc_profile()
        compressed_icc = zlib.compress(raw_icc)
    except Exception as e:
        print(f"Error loading ICC profile: {e}")
        return False

    with open(png_path, "rb") as f:
        data = f.read()

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a valid PNG file")

    out_chunks = [data[:8]]
    pos = 8
    ihdr_found = False
    
    # cICP chunk
    cicp_data = b"\x09\x10\x00\x01"
    new_cicp = make_chunk(b"cICP", cicp_data)

    # iCCP chunk
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
            out_chunks.append(new_cicp)
            out_chunks.append(new_iccp)

        pos += 12 + length

    if not ihdr_found:
        raise ValueError("IHDR chunk not found")

    with open(png_path, "wb") as f:
        f.write(b"".join(out_chunks))
    print(f"Successfully injected cICP and iCCP HDR tags into {png_path}")

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

def get_image_dimensions(input_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to probe image dimensions: {result.stderr}")
    
    data = json.loads(result.stdout)
    if 'streams' in data and len(data['streams']) > 0:
        width = data['streams'][0]['width']
        height = data['streams'][0]['height']
        return width, height
    else:
        raise ValueError("Could not find video stream in input file.")

def main():
    parser = argparse.ArgumentParser(description="Downsample HDR AVIF images in linear space to PNG for testing")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input AVIF file path")
    parser.add_argument('-f', '--factor', type=int, default=4, help="Downsample factor (e.g. 4 for 1/4 size)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # 1. Probing original dimensions
    print("--- Probing Input Image Metadata ---")
    width, height = get_image_dimensions(args.input)
    new_width = int(width / args.factor)
    new_height = int(height / args.factor)
    
    # Ensure dimensions are even numbers
    if new_width % 2 != 0:
        new_width += 1
    if new_height % 2 != 0:
        new_height += 1

    print(f"Original size: {width}x{height}")
    print(f"Target size (1/{args.factor} scale): {new_width}x{new_height}")

    # Prepare inputs directory
    os.makedirs("inputs", exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_png = os.path.join("inputs", f"{base_name}_lr.png")

    # 2. Downsample in linear space to 16-bit PNG (for Real-ESRGAN input)
    print("\n--- Downsampling to 16-bit PNG (inference input) ---")
    png_cmd = [
        "ffmpeg", "-y",
        "-i", args.input,
        "-vf", f"zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:rin=full:t=linear:p=bt2020,format=gbrpf32le,zscale=w={new_width}:h={new_height}:filter=lanczos,zscale=tin=linear:pin=bt2020:t=smpte2084:p=bt2020,format=rgb48le",
        "-pix_fmt", "rgb48le",
        out_png
    ]
    run_command(png_cmd)
    
    # Inject cICP and iCCP chunks to the output PNG
    inject_hdr_chunks_to_png(out_png)

    print("\n--- Done! ---")
    print(f"Created '{out_png}' (with cICP & iCCP PQ HDR tags). You can use this for upscaling or open it directly in Chrome to review.")

if __name__ == "__main__":
    main()
