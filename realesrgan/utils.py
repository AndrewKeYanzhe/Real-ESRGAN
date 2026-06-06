import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None,
                 input_color_space='pq_bt2020',
                 clip_nits=203.0):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half
        self.input_color_space = input_color_space
        self.clip_nits = clip_nits
        self.max_val = None

        # initialize model
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        self.output = self.model(self.img)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        
        skipped_tiles_count = 0
        total_tiles_count = tiles_x * tiles_y

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # check if tile can be skipped in multi-pass mode
                skip_tile = False
                if hasattr(self, 'current_pass_index') and self.current_pass_index > 0:
                    threshold = (self.prev_clip / self.current_clip) ** (1.0 / 2.2)
                    if torch.max(input_tile).item() <= threshold:
                        skip_tile = True

                if skip_tile:
                    skipped_tiles_count += 1
                    output_tile = input_tile.new_zeros(
                        (batch, channel, (input_end_y_pad - input_start_y_pad) * self.scale, (input_end_x_pad - input_start_x_pad) * self.scale)
                    )
                else:
                    # upscale tile
                    try:
                        with torch.no_grad():
                            output_tile = self.model(input_tile)
                    except RuntimeError as error:
                        print('Error', error)
                    if not hasattr(self, 'current_pass_index'):
                        print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
        if hasattr(self, 'current_pass_index'):
            print(f'\tPass {self.current_pass_index + 1}: Processed {total_tiles_count - skipped_tiles_count}/{total_tiles_count} tiles (skipped {skipped_tiles_count})')

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_color_space in ['extended_gamma2_2_bt2020', 'normalized_gamma2_2_bt2020', 'clip_gamma2_2_bt2020']:
            # 1. Convert PQ to linear [0, 1] (where 1.0 = 10,000 nits)
            m1, m2 = 2610.0 / 16384.0, 78.84375
            c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
            pq_pow = np.power(img, 1.0 / m2)
            num = np.maximum(pq_pow - c1, 0.0)
            den = c2 - c3 * pq_pow
            linear = np.power(num / den, 1.0 / m1)  # 1.0 = 10,000 nits
            
            if self.input_color_space == 'clip_gamma2_2_bt2020':
                # Clip values above clip_nits, then scale clip_nits to 1.0
                linear_nits = linear * 10000.0
                linear_clipped = np.clip(linear_nits, 0.0, self.clip_nits)
                linear_scaled = linear_clipped / self.clip_nits
                
                sdr_ratio = self.clip_nits / 100.0
                print(f'\t[Color Space] Using EDR 1.0 = 100 nits, EDR normalisation point is {sdr_ratio:.2f} (based on clip config point)')
            else:
                # Calculate and print peak luminance info
                max_val = np.max(linear)
                peak_nits = max_val * 10000.0
                sdr_ratio = peak_nits / 100.0
                print(f'\t[Color Space] Using EDR 1.0 = 100 nits, input image has single channel max of {sdr_ratio:.2f}x SDR')
                
                if self.input_color_space == 'extended_gamma2_2_bt2020':
                    # Scale linear so 1.0 = 100 nits (highlights go > 1.0)
                    linear_scaled = linear * 100.0
                else:
                    # Dynamically normalize so the actual max value in the image maps to 1.0
                    if max_val <= 0.0:
                        max_val = 1.0
                    self.max_val = max_val
                    linear_scaled = linear / max_val
            
            # 2. Apply Gamma 2.2 encoding
            img = np.power(np.maximum(linear_scaled, 0.0), 1.0 / 2.2)

        # ------------------- process image (without the alpha channel) ------------------- #
        if self.input_color_space == 'multipass_clip_gamma2_2_bt2020':
            # 1. Convert PQ to linear [0, 1] (where 1.0 = 10,000 nits)
            m1, m2 = 2610.0 / 16384.0, 78.84375
            c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
            pq_pow = np.power(img, 1.0 / m2)
            num = np.maximum(pq_pow - c1, 0.0)
            den = c2 - c3 * pq_pow
            linear = np.power(num / den, 1.0 / m1)  # 1.0 = 10,000 nits

            # Determine peak luminance
            max_val = np.max(linear)
            peak_nits = max_val * 10000.0
            
            # Generate clip points
            C_0 = self.clip_nits
            clip_points = []
            current_clip = C_0
            while True:
                clip_points.append(current_clip)
                if current_clip >= peak_nits or current_clip >= 10000.0:
                    break
                current_clip = min(current_clip * 10.0, 10000.0)
            
            print(f'\t[Color Space] Multi-pass clip mode. Base clip = {C_0:.2f} nits. Image peak = {peak_nits:.2f} nits. Total passes = {len(clip_points)}')
            
            accumulated_linear = None
            
            for k, clip_val in enumerate(clip_points):
                # Prepare input image for this pass
                linear_clipped = np.clip(linear, 0.0, clip_val / 10000.0)
                linear_scaled = linear_clipped / (clip_val / 10000.0)
                img_pass = np.power(np.maximum(linear_scaled, 0.0), 1.0 / 2.2)
                
                # Set dynamic tracking properties for tile processing
                self.current_pass_index = k
                self.current_clip = clip_val
                if k > 0:
                    self.prev_clip = clip_points[k - 1]
                
                # Model inference
                self.pre_process(img_pass)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_tensor = self.post_process()
                
                # Convert back to linear
                output_img_pass = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_img_pass = np.transpose(output_img_pass[[2, 1, 0], :, :], (1, 2, 0))
                linear_scaled_out = np.power(output_img_pass, 2.2)
                linear_out = (linear_scaled_out * clip_val) / 10000.0
                
                if k == 0:
                    accumulated_linear = linear_out
                else:
                    prev_clip_val = clip_points[k - 1]
                    threshold = prev_clip_val / 10000.0
                    transition_width = 0.1 * threshold
                    max_channel = np.max(accumulated_linear, axis=2, keepdims=True)
                    w = np.clip((max_channel - (threshold - transition_width)) / transition_width, 0.0, 1.0)
                    accumulated_linear = (1.0 - w) * accumulated_linear + w * linear_out
            
            # Clean up tracking attributes
            if hasattr(self, 'current_pass_index'):
                del self.current_pass_index
            if hasattr(self, 'current_clip'):
                del self.current_clip
            if hasattr(self, 'prev_clip'):
                del self.prev_clip
                
            # Convert final accumulated linear image to PQ
            accumulated_linear = np.clip(accumulated_linear, 0.0, 1.0)
            lin_pow = np.power(accumulated_linear, m1)
            num = c1 + c2 * lin_pow
            den = 1.0 + c3 * lin_pow
            output_img = np.power(num / den, m2)
        else:
            self.pre_process(img)
            if self.tile_size > 0:
                self.tile_process()
            else:
                self.process()
            output_img = self.post_process()
            
            if self.input_color_space == 'extended_gamma2_2_bt2020':
                output_img = output_img.data.squeeze().float().cpu().clamp_(min=0.0).numpy()
            else:
                output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                
            output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

            if self.input_color_space in ['extended_gamma2_2_bt2020', 'normalized_gamma2_2_bt2020', 'clip_gamma2_2_bt2020']:
                # 1. Convert Gamma 2.2 back to linear
                linear_scaled = np.power(output_img, 2.2)
                
                if self.input_color_space == 'extended_gamma2_2_bt2020':
                    # Scale back to standard linear [0, 1] (where 1.0 = 10,000 nits)
                    linear = linear_scaled / 100.0
                elif self.input_color_space == 'clip_gamma2_2_bt2020':
                    # Scale back from [0, 1] (where 1.0 = clip_nits) to [0, 10000 nits] range
                    linear = (linear_scaled * self.clip_nits) / 10000.0
                else:
                    # Revert dynamic normalization using the stored max_val
                    linear = linear_scaled * self.max_val
                
                # 2. Convert linear to PQ
                m1, m2 = 2610.0 / 16384.0, 78.84375
                c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
                linear = np.clip(linear, 0.0, 1.0)
                lin_pow = np.power(linear, m1)
                num = c1 + c2 * lin_pow
                den = 1.0 + c3 * lin_pow
                output_img = np.power(num / den, m2)

        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode


class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class IOConsumer(threading.Thread):

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')
