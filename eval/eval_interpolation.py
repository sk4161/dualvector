import argparse
import os
import sys

sys.path.append(os.getcwd())
import random
from typing import Dict

import numpy as np
import refine_svg
import torch
import yaml
from torch.utils.data import DataLoader

import datasets
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", required=True, type=str)
parser.add_argument("--resume", required=True, type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--interpolate_fonts_indices", type=int, nargs=2)
args = parser.parse_args()


def seed_all(seed):
    random.seed(seed)  # Python
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def make_data_loader(spec, tag=""):
    if spec is None:
        return None

    dataset = datasets.make(spec["dataset"])

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=spec["shuffle"],
        num_workers=spec["batch_size"],
        pin_memory=True,
    )
    return loader


def make_data_loaders(config):
    val_loader = make_data_loader(config.get("val_dataset"), tag="val")
    return val_loader


config_str = """
val_dataset:
  dataset:
    name: deepvecfont-sdf
    args:
      data_root: ./data/dvf_png/font_pngs/test
      img_res: 128
      char_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
      include_lower_case: true
      val: true
      use_cache: false
      valid_list: null
      ratio: 1
      valid_list: ./data/dvf_png/test_valid.txt
  batch_size: 26
  shuffle: false
"""

seed = args.seed
seed_all(seed)
print("seed:", seed)

config = yaml.load(config_str, Loader=yaml.FullLoader)
output_dir = args.outdir
os.makedirs(output_dir, exist_ok=True)

sv_file = torch.load(args.resume)

system = models.make(sv_file["model"], load_sd=True).cuda()
system.init()
system.eval()
models.freeze(system)

sidelength = 256
dataloader = make_data_loaders(config)

with open(os.path.join(output_dir, "seed.txt"), "w") as f:
    f.write(str(seed))


def generate_interpolated_fonts(batch1: Dict, batch2: Dict, ratio: float) -> None:
    with torch.no_grad():
        z1 = system.encoder(batch1)
        z2 = system.encoder(batch2)
        z: torch.Tensor = z1 * (1 - ratio) + z2 * ratio
        curves = system.decoder(z)
        img_rec = torch.clamp(system.decode_image_from_latent_vector(z)["rec"], 0, 1)

    n = curves.shape[0]
    curves_np_raw = curves.detach().cpu().numpy()
    curves_np = (curves_np_raw + 1) * sidelength / 2

    for i in range(n):
        font_name1 = batch1["font_name"][i]
        font_name2 = batch2["font_name"][i]
        save_dir = os.path.join(
            output_dir, "interp_init", font_name1 + "_" + font_name2, f"{ratio:.2f}"
        )
        os.makedirs(save_dir, exist_ok=True)
        char_name = batch["char"][i].item()

        svg_path = os.path.join(save_dir, f"{char_name:02d}_init.svg")
        raw_path = os.path.join(save_dir, f"{char_name:02d}_raw.svg")
        img_path = os.path.join(save_dir, f"{char_name:02d}_rec.png")
        if (
            os.path.exists(svg_path)
            and os.path.exists(raw_path)
            and os.path.exists(img_path)
        ):
            continue

        system.write_paths_to_svg(curves_np_raw[i], raw_path)

        utils.tensor_to_image(img_rec[i, 0], img_path)

        curve_np = curves_np[i]
        d_string_list = [
            models.gutils.path_d_from_control_points(cp, xy_flip=True)
            for cp in curve_np
        ]

        path, _ = refine_svg.merge_d_string(d_string_list)

        cps_list = refine_svg.convert_path_to_control_points(path, pruned=True)

        if len(cps_list) == 0:
            continue
        refine_svg.write_path_to_svg(cps_list, svg_path)

    return None


for index in range(1, 50):
    batch1 = None
    batch2 = None
    interpolate_font_indices = [0, index * 2]
    for i, batch in enumerate(dataloader):
        if i == interpolate_font_indices[0]:
            batch1 = batch
            for k, v in batch1.items():
                if type(v) is torch.Tensor:
                    batch1[k] = v.cuda()
        elif i == interpolate_font_indices[1]:
            batch2 = batch
            for k, v in batch2.items():
                if type(v) is torch.Tensor:
                    batch2[k] = v.cuda()
            break

    assert batch1 is not None and batch2 is not None

    for ratio in np.linspace(0, 1, 5):
        generate_interpolated_fonts(batch1, batch2, ratio)
