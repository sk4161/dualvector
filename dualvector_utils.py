import argparse
import random
from typing import Any, Dict, Optional

import datasets
import models
import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_all(seed: int) -> None:
    random.seed(seed)  # Python
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def make_data_loader(spec: Optional[Dict], tag: str = "") -> Optional[DataLoader]:
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


def make_data_loaders(config: Dict) -> Optional[DataLoader]:
    spec = config.get("val_dataset")
    val_loader = make_data_loader(spec, tag="val")
    return val_loader


def setup_model(args: argparse.Namespace) -> Any:
    seed = args.seed
    seed_all(seed)
    print("seed:", seed)

    # load model
    sv_file = torch.load(args.resume)
    system = models.make(sv_file["model"], load_sd=True).cuda()
    system.init()
    system.eval()
    models.freeze(system)

    return system
