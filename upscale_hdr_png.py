"""
Upscales 16-bit HDR PNG images using Real-ESRGAN on GPU and automatically injects
the 'cICP' and 'iCCP' chunks to allow native HDR rendering in Chrome.

Note: This script does not modify or convert the pixel values of the input during
the upscaling process. It passes the BT.2020 PQ-encoded image data directly to
the network (which internally expects BT.709 gamma 2.2). The output pixels are
saved unchanged and then tagged back with the correct BT.2020 PQ color space metadata.
"""

# --- CONFIGURATION ---
# Select the color space conversion applied to the image data before sending it to the network.
# Supported options:
#   1. "pq_bt2020" : Passes raw PQ values in range [0, 1] directly to the network (original behavior).
#   2. "extended_gamma2.2_bt2020" : Converts PQ to linear light, scales so that 1.0 maps to 100 nits
#                                  (meaning HDR highlights can go > 1.0), applies Gamma 2.2, and allows
#                                  values > 1.0 to pass to the network without clamping.
#   3. "normalized_gamma2.2_bt2020" : Converts PQ to linear light, normalizes it dynamically so that the
#                                    actual maximum value in the current image maps to 1.0, applies
#                                    standard Gamma 2.2, runs inference, and reverses the normalization.
INPUT_COLOR_SPACE = "normalized_gamma2.2_bt2020"
# ---------------------

import argparse
import os
import struct
import zlib
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

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

        # Skip existing chunks
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
    print(f"Successfully injected cICP and iCCP PQ HDR metadata into: {png_path}")

def main():
    parser = argparse.ArgumentParser(description="Upscale HDR PNG images and tag with cICP + iCCP PQ HDR metadata")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input PNG file path")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output PNG file path")
    parser.add_argument('-n', '--model_name', type=str, default='RealESRNet_x4plus',
                        help="Model name: RealESRGAN_x4plus | RealESRNet_x4plus")
    parser.add_argument('-s', '--outscale', type=float, default=4, help="Upscale factor")
    parser.add_argument('--tile', type=int, default=1024, help="Tile size")
    parser.add_argument('--fp32', action='store_true', help="Use fp32 instead of fp16 half precision")
    parser.add_argument('-c', '--color_space', type=str, default=INPUT_COLOR_SPACE,
                        choices=['pq_bt2020', 'extended_gamma2.2_bt2020', 'normalized_gamma2.2_bt2020'],
                        help="Color space pipeline to use")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # 1. Initialize Model
    print("--- Step 1: Loading Real-ESRGAN Model ---")
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    if args.model_name == 'RealESRGAN_x4plus':
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    else:
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'

    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        print("Weights not found. Downloading...")
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = load_file_from_url(
            url=file_url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=not args.fp32,
        input_color_space=args.color_space
    )

    # 2. Run inference
    print("\n--- Step 2: Running Upscaling ---")
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=args.outscale)

    # Auto-append the color space mode to the output filename
    base, ext = os.path.splitext(args.output)
    actual_output = f"{base}_{args.color_space}{ext}"

    # Make output directory if needed
    out_dir = os.path.dirname(actual_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write output PNG
    cv2.imwrite(actual_output, output)
    print(f"Upscaled image temporarily saved to: {actual_output}")

    # 3. Inject HDR tags
    print("\n--- Step 3: Injecting HDR Metadata ---")
    inject_hdr_chunks_to_png(actual_output)

    print("\n--- Done! ---")
    print(f"HDR Upscaled PNG saved to: {actual_output}")

if __name__ == "__main__":
    main()
