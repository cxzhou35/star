import os
import torch
from argparse import ArgumentParser, Namespace
import json
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict
from einops import rearrange, repeat, reduce

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)
from video_to_video.video_to_video_model import VideoToVideo_sr
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix

from inference_utils import *

logger = get_logger()


class STAR():
    def __init__(self,
                 result_dir='./results/',
                 file_name='000_video.mp4',
                 model_path='',
                 solver_mode='fast',
                 steps=15,
                 guide_scale=7.5,
                 upscale=4,
                 max_chunk_len=32,
                 patch_size=256,
                 ):
        self.model_path=model_path
        logger.info('checkpoint_path: {}'.format(self.model_path))

        self.result_dir = result_dir
        self.file_name = file_name
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__='model_cfg')
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo_sr(model_cfg)

        steps = 15 if solver_mode == 'fast' else steps
        self.solver_mode=solver_mode
        self.steps=steps
        self.guide_scale=guide_scale
        self.upscale = upscale
        self.max_chunk_len=max_chunk_len
        self.patch_size=patch_size

    def enhance_a_video(self, video_path, prompt):
        logger.info('input video path: {}'.format(video_path))
        text = prompt
        logger.info('text: {}'.format(text))
        caption = text + self.model.positive_prompt

        input_frames, input_fps = load_video(video_path)
        logger.info('input fps: {}'.format(input_fps))

        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape       # (F, C, H, W)
        logger.info('original input resolution: {}'.format((h, w)))

        # patchify the video
        video_data_patched = rearrange(video_data, 'f c (h_p p1) (w_p p2) -> f c (p1 p2) h_p w_p', h_p=self.patch_size, w_p=self.patch_size)
        _, _, p_n, h_p, w_p = video_data_patched.shape
        logger.info('patch input resolution: {}'.format((h_p, w_p)))

        target_h, target_w = h_p * self.upscale, w_p * self.upscale   # adjust_resolution(h, w, up_scale=4)
        logger.info('target resolution: {}'.format((target_h, target_w)))

        total_noise_levels = 900
        setup_seed(666)

        output_list = []
        for i in range(p_n):
            pre_data = {'video_data': video_data_patched[:, :, i], 'y': caption}
            pre_data['target_res'] = (target_h, target_w)

            with torch.no_grad():
                data_tensor = collate_fn(pre_data, 'cuda:0')
                output = self.model.test(data_tensor, total_noise_levels, steps=self.steps, \
                                    solver_mode=self.solver_mode, guide_scale=self.guide_scale, \
                                    max_chunk_len=self.max_chunk_len
                                    )

            output = tensor2vid(output)
            output = adain_color_fix(output, video_data)
            file_name = f'crop_patch_{i}.mp4'
            save_video(output, self.result_dir, file_name, fps=input_fps)
            output_list.append(output)

        # merge the video
        output = torch.cat(output_list, dim=0)
        __import__('ipdb').set_trace()

        output = tensor2vid(output)

        # Using color fix
        output = adain_color_fix(output, video_data)

        save_video(output, self.result_dir, self.file_name, fps=input_fps)
        return os.path.join(self.result_dir, self.file_name)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default='results', help="save directory")
    parser.add_argument("--file_name", type=str, help="file name")
    parser.add_argument("--model_path", type=str, default='./pretrained_weight/model.pt', help="model path")
    parser.add_argument("--prompt", type=str, default='a good video', help="prompt")
    parser.add_argument("--upscale", type=int, default=4, help='up-scale')
    parser.add_argument("--max_chunk_len", type=int, default=32, help='max_chunk_len')

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | normal')
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--patch_size", type=int, default=256)

    return parser.parse_args()

def main():

    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    model_path = args.model_path
    save_dir = args.save_dir
    file_name = args.file_name
    upscale = args.upscale
    max_chunk_len = args.max_chunk_len
    patch_size = args.patch_size

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg

    assert solver_mode in ('fast', 'normal')

    star = STAR(
                result_dir=save_dir,
                file_name=file_name,
                model_path=model_path,
                solver_mode=solver_mode,
                steps=steps,
                guide_scale=guide_scale,
                upscale=upscale,
                max_chunk_len=max_chunk_len,
                patch_size=patch_size,
                )

    star.enhance_a_video(input_path, prompt)


if __name__ == '__main__':
    main()
