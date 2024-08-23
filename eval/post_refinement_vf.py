import argparse
import itertools
import os
import sys
from typing import Optional

import numpy as np
import pydiffvg
import torch
import torchvision
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())
import losses
from refine_args import parser
from svg_simplification import svg_file_simplification


def get_reference_image_path(
    input_path: str, font: str, character: str, alpha: float
) -> str:
    return os.path.join(input_path, font, character, f"{alpha:.2f}.png")


def get_init_svg_path(input_path: str, font: str, character: str, alpha: float) -> str:
    return os.path.join(input_path, font, character, f"{alpha:.2f}.svg")


def get_dst_svg_path(output_path: str, font: str, character: str, alpha: float) -> str:
    return os.path.join(
        get_dst_font_dir(output_path, font, character), f"{alpha:.2f}_refined.svg"
    )


def get_dst_font_dir(output_path: str, font: str, character: str) -> str:
    return os.path.join(output_path, font, character)


def post_refinement(
    input_path: str, output_path: str, font: str, character: str, alpha: float
) -> Optional[float]:
    config_str = """
    loss:
        name: local-loss
        args:
            lam_render: 1
            lam_reg: 1.e-6
    """
    config = yaml.load(config_str, Loader=yaml.FullLoader)
    loss_fn = losses.make(config["loss"])

    dst_dir = get_dst_font_dir(output_path, font, character)
    os.makedirs(dst_dir, exist_ok=True)

    image_path = get_reference_image_path(input_path, font, character, alpha)
    init_svg_path = get_init_svg_path(input_path, font, character, alpha)
    dst_svg_path = get_dst_svg_path(output_path, font, character, alpha)

    target = torchvision.transforms.ToTensor()(Image.open(image_path).convert("L")).to(
        "cpu"
    )
    n_times = 4

    if not os.path.exists(init_svg_path):
        return None

    if os.path.exists(dst_svg_path):
        # skip
        return None

    for i in range(n_times):
        if i == 0:
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
                init_svg_path
            )
        else:
            if i == 1:
                group, quad_line, merge, split = True, True, False, True
            elif i == 2:
                group, quad_line, merge, split = True, True, False, False
            elif i == 3:
                group, quad_line, merge, split = True, True, True, False

            success = svg_file_simplification(
                dst_svg_path.replace(".svg", f"_{i-1}.svg"),
                dst_svg_path.replace(".svg", f"_{i-1}_sim.svg"),
                group=group,
                quad_line=quad_line,
                merge=merge,
                split=split,
            )
            if not success:
                return None
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
                dst_svg_path.replace(".svg", f"_{i-1}_sim.svg")
            )
        iter_steps = 50

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )

        render = pydiffvg.RenderFunction.apply

        # The output image is in linear RGB space. Do Gamma correction before saving the image.
        points_vars = []
        for svg_path in shapes:
            svg_path.points.requires_grad = True
            points_vars.append(svg_path.points)

        # Optimize
        points_optim = torch.optim.Adam(points_vars, lr=0.5, betas=(0.9, 0.999))

        # Adam iterations.
        with tqdm(range(iter_steps), leave=False, desc="Refine") as iters:
            for _ in iters:
                points_optim.zero_grad()

                # Forward pass: render the image.
                scene_args = pydiffvg.RenderFunction.serialize_scene(
                    canvas_width, canvas_height, shapes, shape_groups
                )
                try:
                    img = render(
                        canvas_width,  # width
                        canvas_height,  # height
                        2,  # num_samples_x
                        2,  # num_samples_y
                        42,  # seed
                        None,  # bg
                        *scene_args,
                    )
                except:
                    return None
                # Compose img with white background
                img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
                    img.shape[0], img.shape[1], 3, device="cpu"
                ) * (1 - img[:, :, 3:4])
                img = img[:, :, :3]
                loss_dict = loss_fn(img, target, shapes)

                # Backpropagate the gradients.
                loss_dict["loss"].backward()

                # Take a gradient descent step.
                points_optim.step()

                for k, v in loss_dict.items():
                    loss_dict[k] = f"{v.item():.6f}"

                iters.set_postfix(
                    {
                        **loss_dict,
                    }
                )

        if i < n_times - 1:
            pydiffvg.save_svg_paths_only(
                dst_svg_path.replace(".svg", f"_{i}.svg"),
                canvas_width,
                canvas_height,
                shapes,
                shape_groups,
            )
        else:
            pydiffvg.save_svg_paths_only(
                dst_svg_path, canvas_width, canvas_height, shapes, shape_groups
            )

    return float(loss_dict["render"])


def main(args: argparse.Namespace) -> None:
    pydiffvg.set_use_gpu(False)

    origin_dir = args.indir
    exp_dir = args.outdir
    os.makedirs(exp_dir, exist_ok=True)

    # characters for evaluation
    character_index_min = args.cmin
    character_index_max = args.cmax
    characters = [chr(i) for i in range(65, 91)][
        character_index_min : character_index_max + 1
    ]

    # alphas for evaluation
    alpha_min = args.amin
    alpha_max = args.amax
    alpha_num = args.anum
    alphas = np.linspace(alpha_min, alpha_max, alpha_num).tolist()

    # font names for evaluation
    font_names = os.listdir(origin_dir)

    task = sorted(
        [
            (origin_dir, font_name, character, alpha)
            for font_name, character, alpha in itertools.product(
                font_names,
                characters,
                alphas,
            )
        ]
    )

    losses = []
    for path, font_name, character, alpha in tqdm(task):
        loss_render = post_refinement(path, exp_dir, font_name, character, alpha)
        if loss_render is not None:
            losses.append(loss_render)

    print(sum(losses) / len(losses))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
