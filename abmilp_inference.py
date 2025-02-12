import argparse
import datetime
import json
from copy import deepcopy
from typing import Type
import PIL
import numpy as np
import os
import time
from pathlib import Path

import psutil
import torch
import torch.backends.cudnn as cudnn
import torchvision
from timm.utils import accuracy
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import timm
import gc
import models_simmim
# import nvidia_smi

# nvidia_smi.nvmlInit()
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from torchvision.datasets import STL10
from tqdm import tqdm

import util.misc as misc
from util.datasets import build_dataset_v2
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import models_vit
import abmilp
from collections import defaultdict
from tqdm import tqdm
from cub import Cub2011
import seaborn as sns

device = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu")
)

def get_dataloader(dataset_name):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    if dataset_name == "custom":
        args.data_path = Path("img_custom/")
        _, dataset = build_dataset_v2(args, is_pretrain=False)
    elif dataset_name == "imagenet":
        args.data_path = Path("/net/tscratch/people/plgwoj/imagenet_val")
        dataset = torchvision.datasets.ImageFolder(
            args.data_path,
            transform=transforms.Compose([
                transforms.Resize([224, 224], interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)
            ])
        )
    elif dataset_name == "cub":
        dataset = Cub2011(
            "/net/people/plgrid/plgwoj/scratch/datasets/cub", train=False,
            transform=transforms.Compose([
                    transforms.Resize([224, 224], interpolation=PIL.Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(imagenet_mean, imagenet_std)
                ])
        )
        args.data_path = Path("/net/people/plgrid/plgwoj/scratch/datasets/cub/CUB_200_2011")
    else:
        NotImplementedError
        
    data_loader_custom = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    return data_loader_custom




# Add arguments
parser = argparse.ArgumentParser(description="ehh.")
parser.add_argument('--model', type=str, default="vit_base_patch16")
parser.add_argument('--no', type=int, default=0)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dino_aug', action='store_true')
parser.add_argument('--dataset', type=str, default="custom")
parser.add_argument('--ckpt', type=str, default="vit_base_patch16_224.mae")
parser.add_argument("--checkpoint_key", default="model", type=str)
args = parser.parse_args()

if args.model in ("mae", "mocov3", "ibot"):
    from timm.models.vision_transformer import _create_vision_transformer
    model: models_vit.VisionTransformer = models_vit.__dict__["vit_base_patch16"](
            num_classes=1000,
        )
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
else:
    model: models_vit.VisionTransformer = models_simmim.__dict__["vit_base_patch16"]()

if args.model == "mae":
    checkpoint_model = _create_vision_transformer(args.ckpt, pretrained=True, **model_kwargs).state_dict()
elif args.model == "mocov3":
    ckpt = Path("/net/people/plgrid/plgwoj/gmum_cc/results/marcin/mae/abmilp_ckpts/timm_vit_base_patch16_224.mocov3/checkpoint-799.pth")
    checkpoint_model = torch.load(ckpt, map_location='cpu')["model"]
elif args.model == "beitv2":
    ckpt = Path("/net/people/plgrid/plgwoj/gmum_cc/results/marcin/mae/abmilp_ckpts/timm_vit_base_patch16_224.beitv2/beitv2_base_patch16_224_pt1k.pth")
    checkpoint_model = torch.load(ckpt, map_location='cpu')["model"]
elif args.model == "ibot":
    ckpt = Path("/net/people/plgrid/plgwoj/gmum_cc/results/marcin/mae/abmilp_ckpts/timm_vit_base_patch16_224.ibot/checkpoint-799.pth")
    checkpoint_model = torch.load(ckpt, map_location='cpu')["model"]
    

interpolate_pos_embed(model, checkpoint_model)
msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)

model = model.to(device)

BASE_DIR = '/net/pr2/projects/plgrid/plgg_gmum_cc/results/marcin/mae/abmilp_ckpts/'
BASE_MODEL = f'timm_vit_base_patch16_224.{args.model}/'
ABMILP_CKPT = 'checkpoint-CUB:abmilp:90:sgd:1:1024:a:1:relu:none:none:patch:s0.pth'

if args.no in (1, 2):
    ABMILP_CKPT = ABMILP_CKPT.replace("s0.pth", f"s{args.no}.pth")

dataloader = get_dataloader(args.dataset)


abmilp_ckpt = Path(BASE_DIR + BASE_MODEL + ABMILP_CKPT)
abmilp_name = abmilp_ckpt.name



_, _, _, _, _, _, _, depth, act, sa, cond, content, _ = abmilp_name.split(":")

save_name = BASE_MODEL.split(".")[-1].replace("/", "") + "-" + str(args.no)


abmilp_head = abmilp.ABMILPHead(
    dim=768,
    self_attention_apply_to=sa,
    activation=act,
    depth=int(depth),
    cond=cond,
    content=content,
    num_patches=196
)

seq = nn.Sequential(
    abmilp_head, 
    torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
    model.head
)

ckpt = torch.load(abmilp_ckpt, map_location="cpu")

print(ckpt.keys())
print(ckpt["model"].keys())
print(seq.state_dict().keys())

seq.load_state_dict(ckpt["model"], strict=True)
seq = seq.to(device)

# abmilp_name 
ckpt.keys()

# token_selections.keys()
key_to_human = {
    "cls_attn_map_mean": "a) Average MAE\n$\mathtt{[cls]}$-patch attn.", #"MAE mean\n[cls] attention",
    "cls_attn_map_with_min_entropy": "b) Lowest-entropy MAE\n$\mathtt{[cls]}$-patch attn.",
    'central_patch_attention': "c) MAE central\npatch attn.",
    'dino_cls_attention_map': "d) Average DINO\n$\mathtt{[cls]}$-patch attn.",
    "abmilp_attn": "e) AbMILP",
}

key_to_human_custom = {
    k: " ".join(v.split(" ")[1:])
    for (k,v) in key_to_human.items() 
    if (
        v.startswith("a") or v.startswith("e")
    )
}
key_to_human_custom

cls_attn_maps = []
abmilp_attn_maps = []

import torch
from torchvision.transforms.functional import to_pil_image

def save_tensor_to_jpeg(tensor, i):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tensor = tensor.detach().cpu()
    denormalized = tensor.clone()
    for t, m, s in zip(denormalized, mean, std):
        t.mul_(s).add_(m).clamp_(0, 1)
    image = to_pil_image(denormalized)
    image.save(f"/net/tscratch/people/plgwoj/test/{i}.JPEG", "JPEG")

for x, _ in tqdm(dataloader):
    # for i in range(10):
    #     save_tensor_to_jpeg(x[i], i)
    # raise Exception
    token_selections = model.forward_features(x.to(device), return_features="toksec", abmilp_head=abmilp_head)
    cls_attn_maps.append(token_selections["cls_attn_map_mean"].reshape(-1, 14, 14).cpu())
    abmilp_attn_maps.append(token_selections["abmilp_attn"].reshape(-1, 14, 14).cpu())

cls_attn_maps = torch.cat(cls_attn_maps, dim=0)
abmilp_attn_maps = torch.cat(abmilp_attn_maps, dim=0)

#os.makedirs(BASE_MODEL, exist_ok=True)  # Ensure the directory exists
torch.save(cls_attn_maps, f"/net/tscratch/people/plgwoj/cls_attn_maps/{args.dataset}/{save_name}.pt")
torch.save(abmilp_attn_maps, f"/net/tscratch/people/plgwoj/abmilp_attn_maps/{args.dataset}/{save_name}.pt")

# for x, _ in tqdm(data_loader_custom):
#     token_selections = model.forward_features(x.to(device), return_features="toksec", abmilp_head=abmilp_head)
    
#     print({k: v.shape for (k, v) in token_selections.items()})

#     #del token_selections["ncut_eigen_softmax"]
    
#     N_ROWS = len(x)
#     N_COLS = len(key_to_human_custom) + 1
#     ks = sorted(token_selections.keys())
    
#     for i in range(N_ROWS):
#         fig, ax = plt.subplots(nrows=1, ncols=N_COLS, figsize=(N_COLS * 5, 1 * 5))

#         ax[0].imshow(x[i].permute(1, 2, 0) * imagenet_std + imagenet_mean)
        
#         for k, key in enumerate(key_to_human_custom.keys(), start=1):
#             ax[k].imshow(
#                 token_selections[key][i].reshape(14, 14).cpu()
#             )
#             ax[k].set_title(key_to_human_custom[key], fontsize=24)
            
#         [a.axis("off") for a in ax]
        
#         # Save the figure
#         output_path = os.path.join(output_dir, f"sample_{i}.png")
#         fig.savefig(output_path)
#         plt.close(fig)  # Close the figure to free memory
