"""VL2Lite-style KD + Levina-Bickel MLE bottleneck — CIFAR-100 edition (v8).

RQ1: Does the Levina-Bickel MLE estimate of CLIP vision-embedding intrinsic
dimensionality predict where the student bottleneck stops limiting accuracy,
in particular for geometry-preserving distillation?

v8 additions to v7
  D1  Added RKD mode: relational geometry only (no linguistic loss).
  D2  Added CRD mode: contrastive student-teacher representation distillation.
      Temperature controlled by --crd-temperature (default 0.07).
  D3  Added DKD mode: decoupled KD separating TCKD and NCKD terms.
      Weights controlled by --dkd-alpha and --dkd-beta (default 1.0 each).
  B1  Added PCA-based bottleneck selection: smallest d explaining
      --pca-variance-threshold of teacher embedding variance (default 0.95).
  B2  Added effective-rank-based selection: Roy & Vetterli (2007) entropy
      of normalised singular values of teacher embeddings.
  B3  Fixed-width baselines 16, 32 and 64, chosen independently of the MLE and
      always included in the grid; each is reported separately.
  B4  Grid oracles: the width with the highest accuracy, and the smallest width
      within 1 pp of it (on test accuracy, see NV; optimistic, not deployable).
  B5  parse_dims now accepts tokens 'pca' and 'erank' in --dims.
  B6  All five selection strategies are reported in selection.json per mode.
  B7  pca_d and erank_d saved to dimension_estimates.json.
  NV  No validation split: students train on the full training pool and are scored
      on the test set at the final epoch. Oracle, d_eps, knee and all strategy
      comparisons therefore use TEST accuracy and are post-hoc analysis, not
      held-out model selection. selection.json states this explicitly.
  K1  RKD angle loss keeps only triplets with i, j, k pairwise distinct.
  M1  Multi-GPU dispatch polls with a timeout, detects dead or stalled workers,
      and writes lost_runs.json instead of hanging.
  P1  PCA and effective rank use the same L2-normalised space as the MLE;
      raw-feature values are saved as an ablation.
  G1  True no-bottleneck baseline (bottleneck_kind='identity') alongside the
      full-width Linear projection; --no-identity-baseline skips it.
  P6  failed_runs.json (exceptions) separate from lost_runs.json (crashed workers);
      single-GPU runs record failures instead of aborting; hardware.json documents
      the latency/energy measurement.
  G2  Default grid starts at 2: 2,4,8,12,16,24,32,48,64,96,128 plus estimator widths.
  V2  Matched-view ablation: --no-augment (teacher and student both see the
      un-augmented image).
  F1  VL2Lite-style uses the distance-only relational term (weight 1, no angle);
      the full RKD weights (distance 25, angle 50) apply to mode 'rkd' only.
  F2  Run signature includes a fingerprint of the training code, so --resume
      refuses to mix runs trained under different loss definitions.
  F3  Latency/energy measured once per architecture (width, kind) and shared
      by every mode and seed with that architecture.
  F4  Incomplete grids stop before analysis unless --allow-partial is given;
      then widths without all seeds are excluded and selection.json says so.
  F5  Training loader uses drop_last, so every batch has the same size.
  A1  --analyze-only rebuilds the summary and selection from results_per_seed.csv
      with no training and no GPU, for checking a run in progress. Output goes to
      interim_*.json/csv; --min-seeds sets how many seeds a width needs (default 1).
  N1  Mode 'sskd' (SSKD-style, Xu et al. ECCV 2020): CE + logit KD from CLIP zero-shot
      logits + KL between rotated-vs-original similarity distributions of student
      (via a small SS head) and frozen CLIP. CLIP features of 90/180/270-degree
      rotations are cached once when sskd is requested. Costs ~2x student compute.
  N2  Mode 'clipkd' (CLIP-KD-style, Yang et al. CVPR 2024): CE + feature distillation
      (MSE to the CLIP image embedding) + interactive contrastive learning (student
      image embedding vs CLIP class-text embeddings). Affinity mimicking (TinyCLIP)
      is not part of CLIP-KD and is not included.
  N3  Teacher references: CLIP zero-shot and a linear probe on frozen CLIP features,
      saved in teacher_zero_shot.json and selection.json. Reference points, not
      upper bounds for fine-tuned students.
  W5  Every file write goes to a temporary file that is swapped into place, with
      retries. Fixes "open file failed with error code: 32" when OneDrive, Dropbox
      or antivirus holds a file open. A warning is printed if --out is in a synced folder.
  C2  CIFAR-100 grid adds 192 and 256, so a plateau above 128 can be located
      (the generic grid jumps from 128 to the backbone width).
  C1  CIFAR-100: also records top-5 test accuracy (results, summary). Selection and
      the oracles still use top-1 accuracy.
  Known, not changed in code: modes weight the classification loss differently
      (labels/feat: 1; rkd/crd/dkd/vl2lite: annealed lam_cls 0.5 -> 0.8). Report it
      or run a sensitivity check. DKD uses beta=1 and no warm-up; with no validation split,
      any tuning done on test accuracy must be reported as such.
  S1  New CLI args: --crd-temperature, --dkd-alpha, --dkd-beta,
      --pca-variance-threshold. All added to SIGNATURE_FIELDS.

v5 corrections to v4 (see R-tags in code)
  R1  --kd-logit-scale defaulted to a fixed 10. Default is now CLIP's own
      learned logit scale.
  R2  Resume signature now excludes seeds/dims/modes.
  R3  Data subsampling and MLE use fixed --data-seed.
  R4  has_bottleneck_layer recorded per row.
  R5  Docstring commands corrected.
  R6  New layers get lr x --head-lr-mult (default 10).
  R7  Convergence check on final cross-entropy drop.

v7 (merged): AdamW, label smoothing, augmentation, consistent bottleneck layers.

Requires: pip install torch torchvision open_clip_torch scikit-learn pandas numpy pynvml

Smoke (minutes; checks the whole pipeline end to end):
  python cifar100_vl2lite_mle_pareto_v8.py --max-train 2000 --max-test 500 --epochs 2 \\
      --seeds 42 --dims 2,8,mle,full --modes labels,rkd,sskd,clipkd \\
      --benchmark-repeats 30 --id-samples 1000 --gpus 0 --out C:\\vl2lite_runs\\cifar100_smoke

Check progress of a running/partial experiment (same flags + --analyze-only):
  python cifar100_vl2lite_mle_pareto_v8.py --out <folder> --analyze-only

Full (all 8 modes and the default grid; do not pass an old --dims string):
  python cifar100_vl2lite_mle_pareto_v8.py --epochs 50 --seeds 42,43,44 --gpus 0 --num-workers 2 \\
      --out C:\\vl2lite_runs\\cifar100_resnet18
"""

import os
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
import faulthandler
faulthandler.enable()
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
    'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
    'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
    'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm',
]
PROMPT_NAMES = [
    'an apple', 'an aquarium fish', 'a baby', 'a bear', 'a beaver',
    'a bed', 'a bee', 'a beetle', 'a bicycle', 'a bottle', 'a bowl',
    'a boy', 'a bridge', 'a bus', 'a butterfly', 'a camel', 'a can',
    'a castle', 'a caterpillar', 'cattle', 'a chair', 'a chimpanzee',
    'a clock', 'a cloud', 'a cockroach', 'a couch', 'a crab',
    'a crocodile', 'a cup', 'a dinosaur', 'a dolphin', 'an elephant',
    'a flatfish', 'a forest', 'a fox', 'a girl', 'a hamster', 'a house',
    'a kangaroo', 'a keyboard', 'a lamp', 'a lawn mower', 'a leopard',
    'a lion', 'a lizard', 'a lobster', 'a man', 'a maple tree',
    'a motorcycle', 'a mountain', 'a mouse', 'a mushroom', 'an oak tree',
    'an orange', 'an orchid', 'an otter', 'a palm tree', 'a pear',
    'a pickup truck', 'a pine tree', 'a plain', 'a plate', 'a poppy',
    'a porcupine', 'a possum', 'a rabbit', 'a raccoon', 'a ray',
    'a road', 'a rocket', 'a rose', 'the sea', 'a seal', 'a shark',
    'a shrew', 'a skunk', 'a skyscraper', 'a snail', 'a snake',
    'a spider', 'a squirrel', 'a streetcar', 'a sunflower',
    'a sweet pepper', 'a table', 'a tank', 'a telephone', 'a television',
    'a tiger', 'a tractor', 'a train', 'a trout', 'a tulip', 'a turtle',
    'a wardrobe', 'a whale', 'a willow tree', 'a wolf', 'a woman',
    'a worm',
]
PROMPT_TEMPLATES = ['a photo of {}.', 'a photo of the small {}.',
                    'a photo of the large {}.', 'a low resolution photo of {}.']
NUM_CLASSES = 100
# Horizontal flips preserve natural object classes
AUG_HFLIP = True
BACKBONE_DIM = {'resnet18': 512, 'resnet50': 2048, 'mobilenet_v2': 1280,
                'mobilenet_v3_large': 960, 'efficientnet_b0': 1280, 'convnext_tiny': 768}

# All valid distillation modes
VALID_MODES = {'labels', 'feat', 'rkd', 'crd', 'dkd', 'vl2lite', 'sskd', 'clipkd'}


def build_backbone(name, pretrained=True):
    W = models
    if name == 'resnet18':
        net = W.resnet18(weights=W.ResNet18_Weights.DEFAULT if pretrained else None)
        width = net.fc.in_features; net.fc = nn.Identity()
    elif name == 'resnet50':
        net = W.resnet50(weights=W.ResNet50_Weights.DEFAULT if pretrained else None)
        width = net.fc.in_features; net.fc = nn.Identity()
    elif name == 'mobilenet_v2':
        net = W.mobilenet_v2(weights=W.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        width = net.classifier[1].in_features; net.classifier = nn.Identity()
    elif name == 'mobilenet_v3_large':
        net = W.mobilenet_v3_large(weights=W.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        width = net.classifier[0].in_features; net.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        net = W.efficientnet_b0(weights=W.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        width = net.classifier[1].in_features; net.classifier = nn.Identity()
    elif name == 'convnext_tiny':
        net = W.convnext_tiny(weights=W.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        width = net.classifier[2].in_features; net.classifier[2] = nn.Identity()
    else:
        raise ValueError(f'Unknown student backbone: {name}')
    if width != BACKBONE_DIM[name]:
        raise RuntimeError(f'{name}: feature width {width} != expected {BACKBONE_DIM[name]}')
    return net


# R2: only per-run training settings; seeds/dims/modes are encoded in tags
SIGNATURE_FIELDS = ['teacher', 'pretrained', 'student', 'no_pretrained_student',
                    'max_train', 'max_test', 'data_seed', 'epochs',
                    'batch_size', 'lr', 'head_lr_mult', 'temperature', 'lam_cls_start', 'lam_cls_end',
                    'kd_logit_scale', 'w_mse', 'w_cos', 'w_consistency',
                    'id_samples', 'id_ks', 'optimizer', 'weight_decay',
                    'label_smoothing', 'no_augment',
                    # S1: new fields
                    'crd_temperature', 'dkd_alpha', 'dkd_beta', 'pca_variance_threshold',
                    'sskd_temperature', 'sskd_w_kd', 'sskd_w_ss',
                    'clipkd_w_fd', 'clipkd_w_icl']


def seed_everything(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--out', default='cifar100_vl2lite_mle_results')
    p.add_argument('--max-train', type=int, default=0)
    p.add_argument('--max-test', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--teacher', default='ViT-B-32')
    p.add_argument('--pretrained', default='laion2b_s34b_b79k')
    p.add_argument('--student', choices=list(BACKBONE_DIM), default='resnet18')
    p.add_argument('--no-pretrained-student', action='store_true')
    p.add_argument('--id-samples', type=int, default=5000)
    p.add_argument('--id-ks', type=int, nargs='+', default=[5, 10, 20, 50])
    p.add_argument('--id-repeats', type=int, default=5)
    p.add_argument('--dims', default='2,4,8,12,16,24,32,48,64,96,128,192,256,'
                                     'mle*0.5,mle*0.75,mle,pca,erank,mle*1.25,mle*1.5,full',
                   help="Tokens: integer, 'mle', 'mle*<factor>', 'pca', 'erank', 'full'.")
    p.add_argument('--modes', default='labels,feat,rkd,crd,dkd,vl2lite,sskd,clipkd',
                   help=f'Comma-separated subset of: {sorted(VALID_MODES)}')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--extract-batch-size', type=int, default=32)
    p.add_argument('--extract-save-every', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--no-identity-baseline', action='store_true',
                   help='G1: skip the true no-bottleneck baseline (backbone -> classifier, '
                        'no projection layer). By default it is trained for every mode and seed '
                        "alongside the 'full' width, which has a full-width Linear projection.")
    p.add_argument('--head-lr-mult', type=float, default=10.0)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--gpus', default='0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-partial', action='store_true',
                   help='F4: analyse an incomplete grid instead of stopping. Widths with fewer '
                        'than the requested seeds are excluded from selection, and '
                        'selection.json is marked "complete": false.')
    p.add_argument('--analyze-only', action='store_true',
                   help='A1: no training, no GPU. Rebuild the summary and selection from '
                        'results_per_seed.csv in --out (use the same --dims/--modes/--seeds '
                        'as the training command). Writes interim_pareto_summary.csv and '
                        'interim_selection.json, so finished-run outputs are never overwritten.')
    p.add_argument('--min-seeds', type=int, default=0,
                   help='A1: seeds a width needs to be included in selection. 0 = automatic: '
                        'all requested seeds for a training run, 1 with --analyze-only.')
    p.add_argument('--worker-poll-seconds', type=float, default=30.0,
                   help='M1: how often the parent checks that GPU workers are still alive')
    p.add_argument('--worker-stall-hours', type=float, default=0.0,
                   help='M1: terminate a live worker that sends nothing for this long '
                        '(0 disables; set above the longest single run)')
    # VL2Lite-style
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lam-cls-start', type=float, default=0.50)
    p.add_argument('--lam-cls-end', type=float, default=0.80)
    p.add_argument('--kd-logit-scale', type=float, default=None)
    # feat mode
    p.add_argument('--w-mse', type=float, default=1.0)
    p.add_argument('--w-cos', type=float, default=2.0)
    p.add_argument('--w-consistency', type=float, default=0.0)
    # D2: CRD temperature (S1)
    p.add_argument('--crd-temperature', type=float, default=0.07,
                   help='D2: contrastive temperature for CRD loss')
    # D3: DKD weights (S1)
    p.add_argument('--dkd-alpha', type=float, default=1.0,
                   help='D3: weight for TCKD (target-class) term in DKD')
    p.add_argument('--dkd-beta', type=float, default=1.0,
                   help='D3: weight for NCKD (non-target-class) term in DKD')
    # B1: PCA variance threshold (S1)
    # N1: SSKD-style (Xu et al., ECCV 2020)
    p.add_argument('--sskd-temperature', type=float, default=0.5,
                   help='N1: temperature of the rotated-vs-original similarity distributions')
    p.add_argument('--sskd-w-kd', type=float, default=1.0,
                   help='N1: weight of logit KD from CLIP zero-shot logits on original images')
    p.add_argument('--sskd-w-ss', type=float, default=1.0,
                   help='N1: weight of the self-supervision similarity KL')
    # N2: CLIP-KD-style (Yang et al., CVPR 2024): FD + ICL
    p.add_argument('--clipkd-w-fd', type=float, default=1.0,
                   help='N2: weight of feature distillation (MSE to CLIP image embedding)')
    p.add_argument('--clipkd-w-icl', type=float, default=1.0,
                   help='N2: weight of interactive contrastive learning '
                        '(student image embedding vs CLIP class-text embeddings)')
    p.add_argument('--pca-variance-threshold', type=float, default=0.95,
                   help='B1: cumulative variance threshold for PCA dimension selection')
    # Analysis / benchmark
    p.add_argument('--accuracy-tolerance', type=float, default=1.0)
    p.add_argument('--knee-gain', type=float, default=2.0)
    p.add_argument('--benchmark', choices=['after', 'inline', 'none'], default='after')
    p.add_argument('--benchmark-repeats', type=int, default=2000)
    p.add_argument('--benchmark-trials', type=int, default=5)
    return p


# ── Data ──────────────────────────────────────────────────────────────────────
class CIFAR100View(Dataset):
    def __init__(self, base, indices, tf):
        self.base, self.indices, self.tf = base, np.asarray(indices, dtype=np.int64), tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[int(self.indices[i])]
        return self.tf(img.convert('RGB')), int(label), i


class CIFAR100RotView(CIFAR100View):
    """N1: same images rotated by k x 90 degrees counter-clockwise before the transform,
    for caching CLIP features of the SSKD rotations. PIL rotate(+90) and
    torch.rot90(k=1, dims=(-2, -1)) turn a square image the same way."""
    def __init__(self, base, indices, tf, k):
        super().__init__(base, indices, tf)
        self.k = int(k)

    def __getitem__(self, i):
        img, label = self.base[int(self.indices[i])]
        return self.tf(img.convert('RGB').rotate(90 * self.k)), int(label), i


def balanced_indices(targets, maximum: int, seed: int) -> np.ndarray:
    targets = np.asarray(targets)
    if not maximum or maximum >= len(targets):
        return np.arange(len(targets), dtype=np.int64)
    n_cls = int(targets.max()) + 1
    if maximum < n_cls:
        raise ValueError(f'--max-train/test must be >= {n_cls}')
    rng, out = np.random.default_rng(seed), []
    for c in range(n_cls):
        take = maximum // n_cls + (1 if c < maximum % n_cls else 0)
        out.extend(rng.choice(np.flatnonzero(targets == c), size=take, replace=False).tolist())
    return np.array(sorted(out), dtype=np.int64)


def student_transform(train=False, augment=True):
    if train and augment:
        spatial = [transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.9, 1.1))]
        if AUG_HFLIP:
            spatial.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        spatial = [transforms.Resize((224, 224))]
    return transforms.Compose(spatial + [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_cifar100(root, download):
    tr = datasets.CIFAR100(root=root, train=True, download=download)
    te = datasets.CIFAR100(root=root, train=False, download=download)
    assert list(tr.classes) == CIFAR100_CLASSES, f'Unexpected class list: {tr.classes}'
    return tr, te


def labels_of(ds):
    return np.asarray(ds.targets)


# ── Dimensionality estimators ──────────────────────────────────────────────────
def mle_dimension(embeddings, ks, n: int, seed: int) -> pd.DataFrame:
    """Levina-Bickel (2004) pooled MLE, Eq. 7, on L2-normalised embeddings."""
    rng = np.random.default_rng(seed)
    x = embeddings.float().numpy().astype('float64')
    if len(x) > n:
        x = x[rng.choice(len(x), n, replace=False)]
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    ks = sorted({int(k) for k in ks if 3 <= int(k) < len(x)})
    if not ks:
        raise ValueError('Increase --id-samples or reduce --id-ks')
    radii = (NearestNeighbors(n_neighbors=max(ks) + 1, n_jobs=-1)
             .fit(x).kneighbors(x)[0][:, 1:])
    rows = []
    for k in ks:
        rk, nearer = radii[:, k - 1], radii[:, :k - 1]
        usable = (rk > 1e-10) & (nearer.min(axis=1) > 1e-10)
        slopes = np.log(rk[usable, None] / nearer[usable]).mean(axis=1)
        slopes = slopes[np.isfinite(slopes) & (slopes > 1e-10)]
        if len(slopes) < 20:
            raise RuntimeError(f'Too few valid points for k={k}')
        rows.append({'k': k, 'mle_id': float(1.0 / slopes.mean()),
                     'per_point_median': float(np.median(1.0 / slopes)),
                     'per_point_std': float(np.std(1.0 / slopes)),
                     'valid_points': len(slopes)})
    return pd.DataFrame(rows)


def mle_stability(embeddings, ks, n, seed, repeats):
    n_sub = min(n, int(0.8 * len(embeddings)))
    vals = [float(mle_dimension(embeddings, ks, n_sub, seed + 1000 + r)['mle_id'].median())
            for r in range(repeats)]
    return {'repeats': repeats, 'subsample_size': n_sub, 'values': vals,
            'mean': float(np.mean(vals)), 'std': float(np.std(vals))}


def _centered(embeddings, normalize=True):
    """P1: same feature space as the MLE (L2-normalised CLIP embeddings), then centred.
    normalize=False reproduces the raw-feature estimates as an ablation."""
    x = embeddings.float().numpy().astype('float64')
    if normalize:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x - x.mean(axis=0)


def pca_dimension(embeddings, variance_threshold=0.95, normalize=True) -> int:
    """B1: smallest number of PCs explaining variance_threshold of total variance.
    Reference: Jolliffe (2002), Principal Component Analysis, 2nd ed., Springer."""
    s = np.linalg.svd(_centered(embeddings, normalize), compute_uv=False)
    explained = np.cumsum(s ** 2) / (s ** 2).sum()
    return max(2, int(np.searchsorted(explained, variance_threshold) + 1))


def effective_rank_dimension(embeddings, normalize=True) -> int:
    """B2: exp(Shannon entropy of normalised singular values).
    Reference: Roy & Vetterli (2007), EUSIPCO, pp. 606-610."""
    s = np.linalg.svd(_centered(embeddings, normalize), compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return max(2, int(round(float(np.exp(-np.sum(p * np.log(p)))))))


# B3: fixed widths are independent of MLE — do not use mle_d to select them.
FIXED_WIDTHS_INDEPENDENT = [16, 32, 64]


def fixed_widths_for(full_d: int) -> list:
    """B3: return fixed widths that are <= full_d and independent of MLE.
    Reference: Romero et al., ICLR 2015 (FitNets)."""
    return [w for w in FIXED_WIDTHS_INDEPENDENT if w <= full_d]


# ── Safe file writes (W5) ─────────────────────────────────────────────────────
# On Windows a file another process holds open (OneDrive / Dropbox sync, antivirus,
# Explorer preview) cannot be written: "open file failed with error code: 32".
# Every write goes to a unique temporary file first, which is then swapped into place
# with os.replace. Both steps retry with exponential backoff.
IO_RETRIES = 8


def _retry_io(fn, what, attempts=None, base_delay=0.5):
    attempts = attempts or IO_RETRIES
    for i in range(attempts):
        try:
            return fn()
        except (PermissionError, OSError, RuntimeError) as e:
            if i == attempts - 1:
                raise RuntimeError(
                    f'{what} failed after {attempts} attempts: {e}. The file is locked by '
                    'another program. If --out is inside OneDrive/Dropbox, move it to a '
                    'local folder (for example C:\\vl2lite_runs) or pause syncing.') from e
            wait = base_delay * 2 ** i
            print(f'  WARNING: {what} failed ({type(e).__name__}: {str(e)[:90]}); '
                  f'retrying in {wait:.1f}s', flush=True)
            time.sleep(wait)


def _atomic(path, write_tmp):
    path = Path(path)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        _retry_io(lambda: write_tmp(tmp), f'writing {tmp.name}')
        _retry_io(lambda: os.replace(tmp, path), f'replacing {path.name}')
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def safe_save(obj, path):
    """torch.save through a temporary file, with retries."""
    _atomic(path, lambda tmp: torch.save(obj, tmp))


def safe_write_text(path, text):
    """Path.write_text through a temporary file, with retries."""
    _atomic(path, lambda tmp: Path(tmp).write_text(text, encoding='utf-8'))


def safe_to_csv(df, path, **kwargs):
    """DataFrame.to_csv through a temporary file, with retries."""
    _atomic(path, lambda tmp: df.to_csv(tmp, **kwargs))


def safe_unlink(path):
    try:
        _retry_io(lambda: Path(path).unlink(missing_ok=True), f'deleting {Path(path).name}',
                  attempts=4)
    except RuntimeError as e:
        print(f'  WARNING: {e}', flush=True)


def warn_if_synced_folder(path):
    s = str(Path(path).resolve()).lower()
    for name in ('onedrive', 'dropbox', 'google drive', 'icloud'):
        if name in s:
            print(f'WARNING: --out is inside a {name} folder ({Path(path).resolve()}). Sync '
                  'clients lock files while uploading, which can make writes fail on Windows. '
                  'Writes are retried, but a local folder such as C:\\vl2lite_runs is safer '
                  'and faster.', flush=True)
            return


# ── Teacher cache ─────────────────────────────────────────────────────────────
@torch.no_grad()
def build_teacher_cache(teacher, dataset, device, batch_size, workers, cache_path, save_every=100):
    if cache_path.exists():
        obj = torch.load(cache_path, map_location='cpu', weights_only=True)
        if isinstance(obj, dict) and obj.get('features') is not None \
                and obj['features'].shape[0] == len(dataset):
            print(f'Loaded teacher cache: {cache_path.name}')
            return obj['features']
        print('Cache mismatch — rebuilding.')

    n_total = len(dataset)
    partial = cache_path.with_name(cache_path.stem + '.partial.pt')
    done = None
    if partial.exists():
        obj = torch.load(partial, map_location='cpu', weights_only=True)
        if isinstance(obj, dict) and obj.get('n_total') == n_total \
                and obj.get('features') is not None and obj['features'].shape[0] < n_total:
            done = obj['features']
            print(f'Resuming teacher extraction at {done.shape[0]}/{n_total} images', flush=True)
    start = 0 if done is None else done.shape[0]

    loader = DataLoader(Subset(dataset, range(start, n_total)), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda')
    chunks = [] if done is None else [done]
    teacher.eval()
    t0, n_batches = time.perf_counter(), len(loader)
    print(f'Extracting teacher features for {n_total - start} images '
          f'({n_batches} batches of {batch_size}) -> {cache_path.name}', flush=True)
    for b, (timgs, _, _) in enumerate(loader, 1):
        timgs = timgs.to(device, non_blocking=True)
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda'):
                f = teacher.encode_image(timgs)
            torch.cuda.synchronize(device)
        else:
            f = teacher.encode_image(timgs)
        chunks.append(f.float().cpu())
        if b % save_every == 0 or b == n_batches:
            chunks = [torch.cat(chunks)]
            n_done = chunks[0].shape[0]
            try:                    # W5: progress snapshot only; never fatal
                safe_save({'features': chunks[0], 'n_total': n_total}, partial)
            except RuntimeError as e:
                print(f'  WARNING: partial teacher cache not saved ({e}); continuing.', flush=True)
            rate = (n_done - start) / max(time.perf_counter() - t0, 1e-6)
            eta = (n_total - n_done) / max(rate, 1e-6)
            print(f'  {n_done}/{n_total} images ({rate:.0f} img/s, ~{eta / 60:.1f} min left)', flush=True)
    feats = torch.cat(chunks)
    assert feats.shape[0] == n_total, (feats.shape, n_total)
    safe_save({'features': feats}, cache_path)
    safe_unlink(partial)
    return feats


def report_gpus(selected):
    if not torch.cuda.is_available():
        print('CUDA not available: running on CPU')
        return
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        pynvml = None
    print(f"GPUs (CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}):")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        display, nv_name = 'unknown', None
        if pynvml is not None:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                display = 'YES' if pynvml.nvmlDeviceGetDisplayActive(h) else 'no'
                nv_name = pynvml.nvmlDeviceGetName(h)
                nv_name = nv_name.decode() if isinstance(nv_name, bytes) else nv_name
            except Exception:
                pass
        mark = '   <- selected' if i in selected else ''
        print(f'  cuda:{i}  {props.name}  {props.total_memory / 2**30:.0f} GB  '
              f'drives display: {display}{mark}')
        if nv_name and nv_name != props.name:
            print(f'  WARNING: CUDA device {i} is {props.name} but NVML device {i} is {nv_name}.')
        if i in selected and display == 'YES':
            print(f'  WARNING: cuda:{i} drives a display. Risk of Windows TDR reset.')


@torch.no_grad()
def encode_class_prompts(teacher, tokenizer, device):
    protos = []
    for name in PROMPT_NAMES:
        toks = tokenizer([t.format(name) for t in PROMPT_TEMPLATES]).to(device)
        e = F.normalize(teacher.encode_text(toks).float(), dim=-1).mean(0)
        protos.append(F.normalize(e, dim=-1))
    return torch.stack(protos).cpu()


def zero_shot_accuracy(feats, text, labels):
    pred = (F.normalize(feats.float(), dim=-1) @ F.normalize(text.float(), dim=-1).T).argmax(1)
    return 100.0 * float((pred.numpy() == np.asarray(labels)).mean())


# ── Models ────────────────────────────────────────────────────────────────────
def run_tag(mode, d, seed, kind='linear'):
    """G1: identity (no-bottleneck) runs get an 'id' suffix so they never collide
    with the full-width Linear projection at the same d."""
    return f"{mode}_d{d}{'id' if kind == 'identity' else ''}_seed{seed}"


class StudentMLE(nn.Module):
    """kind='linear':   backbone -> Linear(backbone_dim, d_b) -> classifier
       kind='identity': backbone -> classifier (true no-bottleneck baseline, d_b = backbone_dim)"""
    def __init__(self, name, n_classes, d_b, pretrained=True, kind='linear'):
        super().__init__()
        self.backbone = build_backbone(name, pretrained)
        if kind == 'identity' and d_b != BACKBONE_DIM[name]:
            raise ValueError(f'identity baseline needs d_b == {BACKBONE_DIM[name]}, got {d_b}')
        self.kind = kind
        self.has_bottleneck_layer = kind != 'identity'
        self.bottleneck = (nn.Linear(BACKBONE_DIM[name], d_b) if self.has_bottleneck_layer
                           else nn.Identity())
        self.classifier = nn.Linear(d_b, n_classes)

    def forward(self, x, return_bottleneck=False):
        z = self.bottleneck(self.backbone(x))
        logits = self.classifier(z)
        return (z, logits) if return_bottleneck else logits


class Deployable(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.backbone, self.bottleneck, self.classifier = s.backbone, s.bottleneck, s.classifier

    def forward(self, x):
        return self.classifier(self.bottleneck(self.backbone(x)))


class Condense(nn.Module):
    """Two-layer MLP vlm_dim -> d_b (used by vl2lite mode)."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden = max(output_dim, min(2048, input_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, output_dim))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ── Losses ────────────────────────────────────────────────────────────────────
def relational_kd(z, t_img, w_dist=25.0, w_angle=50.0):
    """D1/RKD: distance + angle relational losses.
    Reference: Park et al., CVPR 2019 (Eq. 3 + Eq. 4)."""
    zs = F.normalize(z.float(), dim=-1)
    ts = F.normalize(t_img.float(), dim=-1)
    N = zs.shape[0]
    # No pairwise relationships exist for a singleton minibatch.
    # Return a differentiable zero so backpropagation remains valid.
    if N < 2:
        return z.float().sum() * 0.0
    off = ~torch.eye(N, dtype=torch.bool, device=zs.device)

    # Distance loss (Eq. 3)
    d_s = torch.cdist(zs, zs)
    d_t = torch.cdist(ts, ts)
    d_s = d_s / d_s[off].mean().clamp_min(1e-6)
    d_t = d_t / d_t[off].mean().clamp_min(1e-6)
    loss_dist = F.smooth_l1_loss(d_s[off], d_t[off])

    # Angle loss (Eq. 4) — triplet cosine angle matching
    # For each anchor i, compute cosine of angle at i for all pairs (j, k)
    # Using the identity: cos(angle_ijk) = (e_ij . e_ik) where e = unit diff vector
    if N >= 3 and w_angle > 0:      # skip the N x N x N tensor when the angle term is off
        # student and teacher difference vectors: (N, N, D)
        zs_e = zs.unsqueeze(1) - zs.unsqueeze(0)   # (N, N, D)
        ts_e = ts.unsqueeze(1) - ts.unsqueeze(0)
        zs_e = F.normalize(zs_e, dim=-1, eps=1e-8)
        ts_e = F.normalize(ts_e, dim=-1, eps=1e-8)
        # Pairwise angle similarity via bmm: (N,N,N) — memory-efficient
        # angle_s[i,j,k] = dot(e_ij, e_ik) — no N³D intermediate tensor
        angle_s = torch.bmm(zs_e, zs_e.transpose(1, 2))  # (N, N, N)
        angle_t = torch.bmm(ts_e, ts_e.transpose(1, 2))
        # angle_s[i, j, k] is the angle at anchor i between rays i->j and i->k.
        # Keep only proper triplets with i, j, k pairwise distinct: if j == i or
        # k == i the ray e_ii is a zero vector (cosine 0 for both nets), and
        # j == k gives cosine 1 for both. Those terms carry no information and
        # only dilute the mean.
        ne = ~torch.eye(N, dtype=torch.bool, device=zs.device)
        mask3 = ne.unsqueeze(2) & ne.unsqueeze(1) & ne.unsqueeze(0)  # i!=j, i!=k, j!=k
        loss_angle = F.smooth_l1_loss(angle_s[mask3], angle_t[mask3])
    else:
        loss_angle = torch.tensor(0.0, device=zs.device)

    return w_dist * loss_dist + w_angle * loss_angle


def crd_loss(z_proj, t_img, temperature=0.07):
    """D2/CRD-style: symmetric contrastive loss.
    z_proj must already be projected to vlm_dim (same dim as t_img).
    Reference: Tian et al., ICLR 2020."""
    assert z_proj.shape[-1] == t_img.shape[-1], (
        f'CRD dim mismatch: z_proj {z_proj.shape} vs t_img {t_img.shape}. '
        'Pass crd_align(z) not z.')
    z_n = F.normalize(z_proj.float(), dim=-1)
    t_n = F.normalize(t_img.float(), dim=-1)
    N = z_n.shape[0]
    logits = z_n @ t_n.T / temperature          # (N, N)
    labels = torch.arange(N, device=z_n.device)
    loss_s2t = F.cross_entropy(logits, labels)
    loss_t2s = F.cross_entropy(logits.T, labels)
    return (loss_s2t + loss_t2s) / 2.0


def dkd_loss(student_logits, teacher_logits, targets, temperature=4.0, alpha=1.0, beta=1.0):
    """D3/DKD: decoupled KD separating TCKD (target class) and NCKD (non-target).
    Uses ground-truth targets (not teacher argmax) and properly scaled teacher logits.
    Reference: Zhao et al., CVPR 2022."""
    s_soft = F.softmax(student_logits / temperature, dim=1)
    t_soft = F.softmax(teacher_logits / temperature, dim=1)   # already scaled
    gt = targets.long()                                        # ground-truth labels

    # TCKD: binary distribution over correct class vs. rest
    s_t = s_soft.gather(1, gt.unsqueeze(1))
    t_t = t_soft.gather(1, gt.unsqueeze(1))
    tckd = F.binary_cross_entropy(s_t.clamp(1e-7, 1 - 1e-7), t_t.clamp(1e-7, 1 - 1e-7))

    # NCKD: distribution over non-target classes only
    mask = torch.ones_like(s_soft, dtype=torch.bool)
    mask.scatter_(1, gt.unsqueeze(1), False)
    s_nt = s_soft[mask].view(s_soft.shape[0], -1)
    t_nt = t_soft[mask].view(t_soft.shape[0], -1)
    s_nt = s_nt / s_nt.sum(dim=1, keepdim=True).clamp_min(1e-7)
    t_nt = t_nt / t_nt.sum(dim=1, keepdim=True).clamp_min(1e-7)
    nckd = F.kl_div(s_nt.log().clamp_min(-100), t_nt, reduction='batchmean')

    # T² scaling aligns with original DKD formulation (Zhao et al., CVPR 2022)
    return (temperature ** 2) * (alpha * tckd + beta * nckd)


def clip_teacher_logits(t_img, t_txt, logit_scale):
    """Frozen CLIP zero-shot logits: scaled cosine similarity to class-prompt embeddings."""
    return logit_scale * (F.normalize(t_img.float(), dim=-1) @ F.normalize(t_txt.float(), dim=-1).T)


def logit_kd(student_logits, teacher_logits, temperature):
    """Hinton et al. (2015) KD: T^2 * KL(teacher || student) on softened logits."""
    return (temperature ** 2) * F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                                         F.softmax(teacher_logits / temperature, dim=1),
                                         reduction='batchmean')


def sskd_ss_loss(s_orig, s_rot, t_orig, t_rot, temperature=0.5):
    """N1/SSKD-style self-supervision transfer (Xu et al., ECCV 2020).
    Each net builds a similarity distribution of every rotated sample over the original
    samples in the batch; the student matches the teacher's distribution (KL).
    Each net is compared in its own space, so no student->teacher projection is needed."""
    s_o, s_r = F.normalize(s_orig.float(), dim=-1), F.normalize(s_rot.float(), dim=-1)
    t_o, t_r = F.normalize(t_orig.float(), dim=-1), F.normalize(t_rot.float(), dim=-1)
    sim_s = s_r @ s_o.T / temperature
    sim_t = t_r @ t_o.T / temperature
    return F.kl_div(F.log_softmax(sim_s, dim=1), F.softmax(sim_t, dim=1), reduction='batchmean')


def clipkd_losses(z_proj, t_img, t_txt, targets, logit_scale):
    """N2/CLIP-KD-style (Yang et al., CVPR 2024), adapted to classification:
    FD  - MSE between the projected student embedding and the CLIP image embedding
          (both L2-normalised);
    ICL - the student image embedding is the anchor contrasted against the frozen
          teacher's class-text embeddings (cross-entropy over classes)."""
    p = F.normalize(z_proj.float(), dim=-1)
    t = F.normalize(t_img.float(), dim=-1)
    fd = (p - t).pow(2).sum(1).mean()
    icl = F.cross_entropy(logit_scale * p @ F.normalize(t_txt.float(), dim=-1).T, targets)
    return fd, icl


def linguistic_kd(z, t_img, t_txt, cond_text, logit_scale, temp):
    """VL2Lite linguistic loss: KL(teacher image-text || student image-text)."""
    protos = F.normalize(cond_text(t_txt.float()), dim=-1)
    s_lw = logit_scale * F.normalize(z.float(), dim=-1) @ protos.T
    s_vlm = logit_scale * (F.normalize(t_img.float(), dim=-1)
                           @ F.normalize(t_txt.float(), dim=-1).T)
    return (temp ** 2) * F.kl_div(F.log_softmax(s_lw / temp, dim=1),
                                  F.softmax(s_vlm / temp, dim=1), reduction='batchmean')


def feature_kd(z, t_img, align):
    """Feature KD: MSE + cosine on aligned student features."""
    p = F.normalize(align(z).float(), dim=-1)
    t = F.normalize(t_img.float(), dim=-1)
    return (p - t).pow(2).sum(1).mean(), (1 - (p * t).sum(1)).mean()


def flip_consistency(student, x, z):
    """Horizontal flip is label-preserving for natural object classes."""
    z_aug, _ = student(torch.flip(x, dims=[-1]), return_bottleneck=True)
    return (F.normalize(z.float(), dim=-1)
            - F.normalize(z_aug.float(), dim=-1)).pow(2).sum(1).mean()


# ── Evaluation / benchmark ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, true, top5_hits = [], [], 0
    for xs, ys, _ in loader:
        out = model(xs.to(device))
        if isinstance(out, tuple):
            out = out[1]
        pred.extend(out.argmax(1).cpu().tolist()); true.extend(ys.tolist())
        top5 = out.topk(min(5, out.shape[1]), dim=1).indices.cpu()
        top5_hits += int((top5 == ys.view(-1, 1)).any(dim=1).sum())
    return {'accuracy_pct': 100 * accuracy_score(true, pred),
            'top5_accuracy_pct': 100 * top5_hits / max(len(true), 1),
            'macro_f1_pct': 100 * f1_score(true, pred, average='macro', zero_division=0)}


def benchmark(deployed, device, a):
    deployed.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    energy_fn = None
    if device.type == 'cuda':
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = device.index if device.index is not None else torch.cuda.current_device()
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            energy_fn = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            energy_fn()
        except Exception:
            energy_fn = None

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    lat, eng = [], []
    with torch.inference_mode():
        for _ in range(30):
            deployed(x)
        sync()
        for _ in range(a.benchmark_trials):
            sync()
            e0 = energy_fn() if energy_fn else None
            t0 = time.perf_counter()
            for _ in range(a.benchmark_repeats):
                deployed(x)
            sync()
            dt = time.perf_counter() - t0
            e1 = energy_fn() if energy_fn else None
            lat.append(dt * 1000 / a.benchmark_repeats)
            eng.append((e1 - e0) / a.benchmark_repeats
                       if e0 is not None and e1 is not None and e1 > e0
                       else float('nan'))
    iqr = lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))
    valid_eng = [v for v in eng if np.isfinite(v)]
    return {'latency_ms': float(np.median(lat)), 'latency_iqr_ms': iqr(lat),
            'energy_mJ': float(np.median(valid_eng)) if valid_eng else float('nan'),
            'energy_iqr_mJ': iqr(valid_eng) if valid_eng else float('nan'),
            'latency_trials': lat, 'energy_trials': eng}


BENCH_KEYS = ('latency_ms', 'latency_iqr_ms', 'energy_mJ', 'energy_iqr_mJ')


def bench_trial_rows(tag, bench):
    return [{'tag': tag, 'trial': i + 1, 'latency_ms': lt,
             'energy_mJ': bench['energy_trials'][i] if i < len(bench['energy_trials']) else float('nan')}
            for i, lt in enumerate(bench['latency_trials'])]


def load_deployable(r, out, a):
    obj = torch.load(out / f"{r['tag']}.pt", map_location='cpu', weights_only=True)
    dep = Deployable(StudentMLE(a.student, NUM_CLASSES, int(r['bottleneck_d']), pretrained=False,
                                kind=r.get('bottleneck_kind', 'linear')))
    dep.load_state_dict(obj['student_deployed'])
    return dep


def hardware_info(device, a):
    """P6: what the latency/energy numbers were measured on, and how."""
    info = {'device': str(device), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'benchmark_batch_size': 1,
            'input_shape': [1, 3, 224, 224], 'repeats_per_trial': a.benchmark_repeats,
            'trials': a.benchmark_trials, 'warmup_iterations': 30,
            'latency_method': 'wall clock around repeats_per_trial forward passes with '
                              'torch.cuda.synchronize; median and IQR over trials',
            'energy_method': 'NVML nvmlDeviceGetTotalEnergyConsumption delta over each trial '
                             'window divided by repeats; includes idle/background GPU power, so '
                             'compare models measured on the same idle GPU only; median and IQR '
                             'over trials; NaN where the counter is unsupported'}
    if device.type == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(device)
        try:
            import pynvml
            pynvml.nvmlInit()
            drv = pynvml.nvmlSystemGetDriverVersion()
            info['driver'] = drv.decode() if isinstance(drv, bytes) else drv
        except Exception:
            info['driver'] = 'unknown (pynvml unavailable)'
    return info


def benchmark_saved(results, out, device, a):
    """F3: the deployed network (backbone -> bottleneck -> classifier) depends only on
    (width, bottleneck kind); KD heads are discarded. Benchmark one checkpoint per
    architecture and copy the numbers to every run with that architecture. This is
    ~18x faster and keeps timing noise out of Pareto comparisons between modes."""
    rows, by_arch = [], {}
    for r in results:
        key = (int(r['bottleneck_d']), r.get('bottleneck_kind', 'linear'))
        if key not in by_arch:
            dep = load_deployable(r, out, a)
            b = by_arch[key] = benchmark(dep.to(device), device, a)
            rows += bench_trial_rows(f'arch_d{key[0]}_{key[1]}', b)
            print(f"  benchmarked d={key[0]} ({key[1]}): {b['latency_ms']:.3f} ms, "
                  f"{b['energy_mJ']:.1f} mJ", flush=True)
            del dep
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        r.update({k: by_arch[key][k] for k in BENCH_KEYS})
    print(f'  {len(by_arch)} architecture(s) benchmarked for {len(results)} run(s)', flush=True)
    return rows


# ── Pareto / analysis ─────────────────────────────────────────────────────────
def pareto_mask(frame, acc_col='test_accuracy_pct'):
    """Nondominance on (test accuracy, energy, latency) within the evaluated set."""
    vals = frame[[acc_col, 'energy_mJ', 'latency_ms']].to_numpy(float)
    valid = np.isfinite(vals).all(axis=1)
    obj = np.column_stack([-vals[:, 0], vals[:, 1], vals[:, 2]])
    out = np.full(len(frame), 'unverified_missing_measurement', dtype=object)
    vidx = np.flatnonzero(valid)
    for i in vidx:
        dom = any(np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]) for j in vidx if j != i)
        out[i] = 'dominated' if dom else 'nondominated_within_evaluated_set'
    return out


def pareto_freq_per_seed(frame):
    keys = {(str(r['mode']), str(r['bottleneck_kind']), int(r['bottleneck_d']))
            for _, r in frame.iterrows()}
    freq = {k: 0 for k in keys}
    seeds = frame['seed'].unique()
    for s in seeds:
        sub = frame[frame['seed'] == s].reset_index(drop=True)
        mask = pareto_mask(sub)
        for i, r in sub.iterrows():
            if mask[i] == 'nondominated_within_evaluated_set':
                freq[(str(r['mode']), str(r['bottleneck_kind']), int(r['bottleneck_d']))] += 1
    return {k: round(v / len(seeds), 3) for k, v in freq.items()}


SELECT_COL = 'test_accuracy_pct'    # NV: no validation split; analysis uses test accuracy


def knee_analysis(sub, knee_gain, col=SELECT_COL):
    s = sub.sort_values('bottleneck_d')
    d = s['bottleneck_d'].to_numpy(float)
    acc = s[col].to_numpy(float)
    gains = [{'from_d': int(d[i]), 'to_d': int(d[i + 1]),
              'gain_pp_per_doubling': round(float((acc[i + 1] - acc[i]) / math.log2(d[i + 1] / d[i])), 3)}
             for i in range(len(d) - 1)]
    knee = int(d[-1])
    for i in range(len(gains)):
        if all(g['gain_pp_per_doubling'] < knee_gain for g in gains[i:]):
            knee = gains[i]['from_d']
            break
    return gains, knee


def d_eps_at(sub, tol, col=SELECT_COL):
    """Smallest d whose (test) accuracy is within tol pp of the best."""
    best = sub[col].max()
    return int(sub[sub[col] >= best - tol]['bottleneck_d'].min())


def _acc_at(sub, d, col=SELECT_COL):
    """Accuracy in column col at a given d, or nan if d is not in the grid."""
    row = sub[sub['bottleneck_d'] == d]
    return float(row[col].iloc[0]) if not row.empty else float('nan')


# ── Dimension parsing ──────────────────────────────────────────────────────────
def parse_dims(spec, mle_d, pca_d, erank_d, full_d, fixed_widths=None):
    """B5: parse --dims tokens including 'pca' and 'erank'.
    Always injects fixed_widths into the grid so _acc_at never returns NaN
    for the independent fixed-width baselines."""
    dims = []
    # Always include independent fixed widths in the grid
    for fw in (fixed_widths or []):
        fw = max(2, min(full_d, fw))
        if fw not in dims:
            dims.append(fw)
    for tok in spec.split(','):
        tok = tok.strip().lower()
        if tok == 'mle':
            d = mle_d
        elif tok.startswith('mle*'):
            d = int(round(mle_d * float(tok[4:])))
        elif tok == 'pca':
            d = pca_d
        elif tok == 'erank':
            d = erank_d
        elif tok == 'full':
            d = full_d
        else:
            d = int(tok)
        d = max(2, min(full_d, d))
        if d not in dims:
            dims.append(d)
    return sorted(dims)


def check_resume_signature(out, a, run_signature):
    sigfile, results_csv = out / 'run_signature.txt', out / 'results_per_seed.csv'
    if results_csv.exists() and not a.resume:
        raise RuntimeError('Existing results in --out. Use --resume or a fresh --out directory.')
    if results_csv.exists() and not sigfile.exists():
        raise RuntimeError('Results produced by older script version. Use a fresh --out directory.')
    if sigfile.exists() and sigfile.read_text().strip() != run_signature:
        raise RuntimeError('Training settings differ from existing run. Use a fresh --out directory.')
    safe_write_text(sigfile, run_signature)


# ── Training ───────────────────────────────────────────────────────────────────
def train_one(mode, d_b, seed, kind, teacher_feats, teacher_txt, ds_train, ds_test,
              vlm_dim, device, a, teacher_rot=None):
    """Trains one student and scores it on the test set at the final epoch."""
    seed_everything(seed)
    tag = run_tag(mode, d_b, seed, kind)
    student = StudentMLE(a.student, NUM_CLASSES, d_b,
                         pretrained=not a.no_pretrained_student, kind=kind).to(device)
    cond_text = Condense(vlm_dim, d_b).to(device)   # vl2lite only
    align = nn.Linear(d_b, vlm_dim).to(device)       # feat only
    crd_align = nn.Linear(d_b, vlm_dim).to(device)   # crd only: projects to teacher dim
    clip_align = nn.Linear(d_b, vlm_dim).to(device)  # clipkd only: projects to CLIP space
    ss_dim = 128
    ss_head = nn.Sequential(nn.Linear(d_b, max(d_b, ss_dim)), nn.ReLU(inplace=True),
                            nn.Linear(max(d_b, ss_dim), ss_dim)).to(device)  # sskd only
    if mode == 'sskd' and teacher_rot is None:
        raise ValueError('mode sskd needs rotated teacher features (teacher_rot)')

    backbone_params = list(student.backbone.parameters())
    head_params = list(student.bottleneck.parameters()) + list(student.classifier.parameters())
    if mode == 'vl2lite':
        head_params += list(cond_text.parameters())
    elif mode == 'feat':
        head_params += list(align.parameters())
    elif mode == 'crd':
        head_params += list(crd_align.parameters())
    elif mode == 'clipkd':
        head_params += list(clip_align.parameters())
    elif mode == 'sskd':
        head_params += list(ss_head.parameters())
    params = backbone_params + head_params

    groups = [{'params': backbone_params, 'lr': a.lr},
              {'params': head_params, 'lr': a.lr * a.head_lr_mult}]
    if a.optimizer == 'adamw':
        opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    else:
        opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * e / a.epochs)))

    # F5: every training batch has the same size, so batch-dependent losses (RKD pairs,
    # CRD negatives) see a constant number of samples. Disabled only if the training
    # set is smaller than one batch.
    train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, pin_memory=device.type == 'cuda',
                              persistent_workers=a.num_workers > 0,
                              drop_last=len(ds_train) > a.batch_size)
    test_loader = DataLoader(ds_test, batch_size=a.batch_size, shuffle=False,
                             num_workers=min(a.num_workers, 2), persistent_workers=False)
    test = None
    t_txt = teacher_txt.to(device)
    curves, t_start = [], time.perf_counter()

    for epoch in range(a.epochs):
        student.train(); cond_text.train(); align.train()
        crd_align.train(); clip_align.train(); ss_head.train()
        frac = epoch / max(a.epochs - 1, 1)
        lam_cls = a.lam_cls_start + (a.lam_cls_end - a.lam_cls_start) * frac
        lam_kd = (1.0 - lam_cls) / 2.0
        sums = dict(loss=0.0, cls=0.0, kd1=0.0, kd2=0.0)
        seen = 0

        for xs, ys, ix in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            t_img = teacher_feats[ix].to(device)
            z, logits = student(xs, return_bottleneck=True)
            l_cls = F.cross_entropy(logits, ys, label_smoothing=a.label_smoothing)

            # ── Distillation objective (D1-D3 + existing) ──────────────────
            if mode == 'labels':
                loss, k1, k2 = l_cls, 0.0, 0.0

            elif mode == 'feat':
                mse, cos = feature_kd(z, t_img, align)
                loss = l_cls + a.w_mse * mse + a.w_cos * cos
                k1, k2 = mse.item(), cos.item()

            elif mode == 'rkd':
                # D1: geometry only — no linguistic loss
                l_rkd = relational_kd(z, t_img)
                loss = lam_cls * l_cls + (1.0 - lam_cls) * l_rkd
                k1, k2 = l_rkd.item(), 0.0

            elif mode == 'crd':
                # D2: project student to teacher dim first (fixes shape mismatch)
                z_proj = crd_align(z)
                l_crd = crd_loss(z_proj, t_img, temperature=a.crd_temperature)
                loss = lam_cls * l_cls + (1.0 - lam_cls) * l_crd
                k1, k2 = l_crd.item(), 0.0

            elif mode == 'dkd':
                # D3: decoupled KD — use scaled teacher logits + ground-truth targets
                teacher_logits = clip_teacher_logits(t_img, t_txt, a.kd_logit_scale_used)
                l_dkd = dkd_loss(student_logits=logits,
                                 teacher_logits=teacher_logits,
                                 targets=ys,
                                 temperature=a.temperature,
                                 alpha=a.dkd_alpha,
                                 beta=a.dkd_beta)
                loss = lam_cls * l_cls + (1.0 - lam_cls) * l_dkd
                k1, k2 = l_dkd.item(), 0.0

            elif mode == 'sskd':
                # N1: one random rotation (90/180/270) per sample; teacher features of the
                # same rotation come from the cache. Logit KD from CLIP zero-shot logits on
                # the original image + similarity KL between rotated and original samples.
                ks = torch.randint(1, 4, (xs.size(0),))
                x_rot = torch.stack([torch.rot90(x, int(k), dims=(-2, -1))
                                     for x, k in zip(xs, ks)])
                z_rot, _ = student(x_rot, return_bottleneck=True)
                t_rot = teacher_rot[ks - 1, ix].to(device)
                l_kd = logit_kd(logits, clip_teacher_logits(t_img, t_txt, a.kd_logit_scale_used),
                                a.temperature)
                l_ss = sskd_ss_loss(ss_head(z), ss_head(z_rot), t_img, t_rot, a.sskd_temperature)
                loss = lam_cls * l_cls + (1.0 - lam_cls) * (a.sskd_w_kd * l_kd + a.sskd_w_ss * l_ss)
                k1, k2 = l_kd.item(), l_ss.item()

            elif mode == 'clipkd':
                # N2: FD + ICL against the frozen CLIP teacher
                fd, icl = clipkd_losses(clip_align(z), t_img, t_txt, ys, a.kd_logit_scale_used)
                loss = lam_cls * l_cls + (1.0 - lam_cls) * (a.clipkd_w_fd * fd + a.clipkd_w_icl * icl)
                k1, k2 = fd.item(), icl.item()

            else:  # vl2lite
                # F1: VL2Lite-style keeps its original distance-only relational term
                # (weight 1, no angle). The full RKD weights (25 / 50) belong to mode 'rkd'.
                l_vis = relational_kd(z, t_img, w_dist=1.0, w_angle=0.0)
                l_txt = linguistic_kd(z, t_img, t_txt, cond_text, a.kd_logit_scale_used, a.temperature)
                loss = lam_cls * l_cls + lam_kd * l_vis + lam_kd * l_txt
                k1, k2 = l_vis.item(), l_txt.item()

            if mode != 'labels' and a.w_consistency > 0:
                loss = loss + a.w_consistency * flip_consistency(student, xs, z)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            bs = xs.size(0)
            seen += bs
            sums['loss'] += loss.item() * bs; sums['cls'] += l_cls.item() * bs
            sums['kd1'] += k1 * bs; sums['kd2'] += k2 * bs

        sched.step()
        n = max(seen, 1)
        evaluated = epoch == a.epochs - 1 or (a.eval_every > 0 and (epoch + 1) % a.eval_every == 0)
        if evaluated:
            test = evaluate(student, test_loader, device)
        curves.append({'tag': tag, 'mode': mode, 'bottleneck_d': d_b, 'bottleneck_kind': kind,
                       'seed': seed,
                       'epoch': epoch + 1, 'lam_cls': lam_cls if mode == 'vl2lite' else 1.0,
                       **{k: v / n for k, v in sums.items()},
                       'test_accuracy_pct': test['accuracy_pct'] if evaluated else float('nan')})
        acc_txt = f"test={test['accuracy_pct']:.2f}" if evaluated else 'n/a'
        print(f"[{tag}] ep{epoch + 1:02d}/{a.epochs} loss={sums['loss']/n:.4f} "
              f"cls={sums['cls']/n:.4f} kd1={sums['kd1']/n:.4f} kd2={sums['kd2']/n:.4f} "
              f"acc {acc_txt}", flush=True)

    train_s = time.perf_counter() - t_start
    deployed = Deployable(student)
    out_dir = Path(a.out)
    bench = benchmark(deployed.to(device), device, a) if a.benchmark == 'inline' else None
    safe_save({'student_deployed': deployed.cpu().state_dict(), 'architecture': a.student,
                'bottleneck_d': d_b, 'bottleneck_kind': kind, 'mode': mode, 'seed': seed},
               out_dir / f'{tag}.pt')
    safe_to_csv(pd.DataFrame(curves), out_dir / f'{tag}_history.csv', index=False)
    ce_last = curves[-1]['cls']
    ce_prev = curves[-2]['cls'] if len(curves) > 1 else float('nan')
    ce_drop = 100.0 * (ce_prev - ce_last) / ce_prev if ce_prev and np.isfinite(ce_prev) else float('nan')
    res = {'tag': tag, 'mode': mode, 'seed': seed, 'bottleneck_d': d_b, 'bottleneck_kind': kind,
           'final_train_ce': ce_last, 'last_epoch_ce_drop_pct': ce_drop,
           'has_bottleneck_layer': student.has_bottleneck_layer,
           'test_accuracy_pct': test['accuracy_pct'],
           'test_top5_accuracy_pct': test['top5_accuracy_pct'],
           'test_macro_f1_pct': test['macro_f1_pct'],
           'parameters': sum(p.numel() for p in deployed.parameters()), 'train_seconds': train_s,
           'run_signature': a.run_signature}
    res.update({k: (bench[k] if bench else float('nan')) for k in BENCH_KEYS})
    return res, bench, curves


def _worker(gpu_id, configs, shared, q):
    """M1: every message carries gpu_id, and ('done', gpu_id) is always sent last."""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        a = shared['a']
        train_base, test_base = load_cifar100(a.data_root, download=False)
        ds_train = CIFAR100View(train_base, shared['train_ids'],
                              student_transform(train=True, augment=not a.no_augment))
        ds_test = CIFAR100View(test_base, shared['test_ids'], student_transform())
    except Exception:
        import traceback
        tb = traceback.format_exc()
        for cfg in configs:
            q.put(('err', gpu_id, run_tag(*cfg), f'worker setup failed:\n{tb}'))
        q.put(('done', gpu_id))
        return
    for mode, d, seed, kind in configs:
        try:
            q.put(('ok', gpu_id) + train_one(mode, d, seed, kind, shared['teacher_feats'],
                                             shared['teacher_txt'], ds_train, ds_test,
                                             shared['vlm_dim'], device, a,
                                             teacher_rot=shared.get('teacher_rot')))
        except Exception:
            import traceback
            q.put(('err', gpu_id, run_tag(mode, d, seed, kind), traceback.format_exc()))
    q.put(('done', gpu_id))


def dispatch_workers(worker_fn, gpu_ids, buckets, shared, on_ok, poll_seconds=30.0,
                     stall_seconds=0.0, ctx=None):
    """M1: one worker process per GPU; never blocks forever.

    Returns (lost, failed):
      lost   - tags never reported (worker crashed, was killed, or stalled)
      failed - {tag: traceback} for runs that raised an exception
    Both are rerun by --resume, because neither has a saved result.
    """
    import queue as _queue
    ctx = ctx or mp.get_context('spawn')
    q = ctx.Queue()
    procs = {g: ctx.Process(target=worker_fn, args=(g, b, shared, q))
             for g, b in zip(gpu_ids, buckets)}
    expected = {g: {run_tag(*c) for c in b} for g, b in zip(gpu_ids, buckets)}
    reported = {g: set() for g in gpu_ids}
    failed = {}
    finished, last_seen = set(), {g: time.time() for g in gpu_ids}
    total = sum(len(b) for b in buckets)
    n_seen = 0
    for p in procs.values():
        p.start()

    while len(finished) < len(procs):
        try:
            item = q.get(timeout=poll_seconds)
        except _queue.Empty:
            now = time.time()
            for g, p in procs.items():
                if g in finished:
                    continue
                if not p.is_alive():
                    print(f'[ERROR] worker on GPU {g} exited (code {p.exitcode}) before '
                          f'finishing; {len(expected[g] - reported[g])} run(s) lost.', flush=True)
                    finished.add(g)
                elif stall_seconds > 0 and now - last_seen[g] > stall_seconds:
                    print(f'[ERROR] worker on GPU {g} silent for {(now - last_seen[g]) / 60:.0f} min; '
                          'terminating it.', flush=True)
                    p.terminate(); p.join(10)
                    finished.add(g)
            continue

        kind, g = item[0], item[1]
        last_seen[g] = time.time()
        if kind == 'done':
            finished.add(g)
            continue
        n_seen += 1
        if kind == 'err':
            reported[g].add(item[2])
            failed[item[2]] = item[3]
            print(f'[ERROR] {item[2]} ({n_seen}/{total})\n{item[3]}', flush=True)
        else:
            res = item[2]
            reported[g].add(res['tag'])
            on_ok(*item[2:])
            print(f"  collected {n_seen}/{total}: {res['tag']}", flush=True)

    for p in procs.values():
        p.join(timeout=30)
        if p.is_alive():
            p.terminate(); p.join(10)
    lost = sorted(t for g in gpu_ids for t in expected[g] - reported[g])
    return lost, failed


def analyze_and_write(results, info, a, out, prefix=''):
    """Summary, Pareto, strategy comparison and selection.json from finished runs.
    Shared by a normal training run (prefix='') and --analyze-only (prefix='interim_').
    Needs no GPU and no dataset."""
    seeds, modes = info['seeds'], info['modes']
    mle_d, pca_d, erank_d = info['mle_d'], info['pca_d'], info['erank_d']
    pca_d_raw, erank_d_raw = info['pca_d_raw'], info['erank_d_raw']
    fixed_widths, full_d, vlm_dim = info['fixed_widths'], info['full_d'], info['vlm_dim']
    stab, zs, split_sizes = info['stab'], info['zs'], info['split_sizes']
    complete, missing_tags, min_seeds = info['complete'], info['missing_tags'], info['min_seeds']

    # ── Summary on test accuracy (no validation split) ─────────────────────────
    frame = pd.DataFrame(results)
    expected = len(seeds)
    summary = frame.groupby(['mode', 'bottleneck_kind', 'bottleneck_d']).agg(
        test_accuracy_pct=('test_accuracy_pct', 'mean'),
        test_accuracy_std=('test_accuracy_pct', 'std'),
        test_top5_accuracy_pct=('test_top5_accuracy_pct', 'mean'),
        test_macro_f1_pct=('test_macro_f1_pct', 'mean'),
        latency_ms=('latency_ms', 'mean'), latency_iqr_ms=('latency_iqr_ms', 'mean'),
        energy_mJ=('energy_mJ', 'mean'), energy_iqr_mJ=('energy_iqr_mJ', 'mean'),
        parameters=('parameters', 'first'), train_seconds=('train_seconds', 'mean'),
        final_train_ce=('final_train_ce', 'mean'),
        last_epoch_ce_drop_pct=('last_epoch_ce_drop_pct', 'mean'),
        n_seeds=('seed', 'count')).reset_index()

    lin = summary['bottleneck_kind'].eq('linear')
    all_seeds = summary['n_seeds'] >= min_seeds         # F4/A1: groups with enough seeds
    excluded = summary.loc[~all_seeds, ['mode', 'bottleneck_kind', 'bottleneck_d', 'n_seeds']]
    summary['mle_selected'] = lin & summary['bottleneck_d'].eq(mle_d)
    summary['pca_selected'] = lin & summary['bottleneck_d'].eq(pca_d)
    summary['erank_selected'] = lin & summary['bottleneck_d'].eq(erank_d)
    summary['fixed_selected'] = lin & summary['bottleneck_d'].isin(fixed_widths)

    summary['converged'] = summary['last_epoch_ce_drop_pct'] < 2.0
    if (~summary['converged']).any():
        bad = summary.loc[~summary['converged'], ['mode', 'bottleneck_kind', 'bottleneck_d',
                                                  'last_epoch_ce_drop_pct']]
        print('WARNING: training loss still falling >2% in last epoch:\n'
              f'{bad.to_string(index=False)}\nIncrease --epochs before comparing bottleneck sizes.')
    if (summary['n_seeds'] < expected).any():
        print(f'WARNING: some configurations have fewer than {expected} seeds.')

    summary['pareto_all'] = pareto_mask(summary)
    summary['pareto_within_mode'] = ''
    for m in modes:
        idx = summary.index[summary['mode'] == m]
        summary.loc[idx, 'pareto_within_mode'] = pareto_mask(summary.loc[idx].reset_index(drop=True))
    pfreq = pareto_freq_per_seed(frame) if expected > 1 else {}
    summary['nondominated_freq_per_seed'] = [
        pfreq.get((str(r['mode']), str(r['bottleneck_kind']), int(r['bottleneck_d'])), float('nan'))
        if r['n_seeds'] == expected else float('nan')
        for _, r in summary.iterrows()]

    # ── Selection analysis on test accuracy (NV: post-hoc, no held-out split) ──
    selection = {}
    for m in modes:
        sub = summary[(summary['mode'] == m) & lin & all_seeds]
        if sub.empty:
            continue
        best = float(sub[SELECT_COL].max())
        gains, knee = knee_analysis(sub, a.knee_gain)
        d_eps = d_eps_at(sub, a.accuracy_tolerance)
        oracle_d = int(sub.loc[sub[SELECT_COL].idxmax(), 'bottleneck_d'])   # max test accuracy
        oracle_1pp_d = d_eps_at(sub, 1.0)                                   # smallest d within 1 pp

        def strategy(d):
            row = sub[sub['bottleneck_d'] == d]
            acc = _acc_at(sub, d)
            return {'d': int(d), 'test_accuracy_pct': acc,
                    'test_accuracy_std': float(row['test_accuracy_std'].iloc[0]) if not row.empty
                                         else float('nan'),
                    'gap_to_best_pp': best - acc,
                    'd_minus_oracle_1pp': int(d) - oracle_1pp_d,      # >0 too wide, <0 too narrow
                    'within_1pp_of_best': bool(acc >= best - 1.0)}

        picks = {f'fixed_{fw}': fw for fw in fixed_widths}
        picks.update({'mle': mle_d, 'pca': pca_d, 'erank': erank_d,
                      'oracle_max': oracle_d, 'oracle_1pp': oracle_1pp_d})
        selection[m] = {'strategies': {k: strategy(d) for k, d in picks.items()},
                        'best_test_accuracy_pct': best,
                        'mle_equals_d_eps': mle_d == d_eps,
                        'd_eps': d_eps,
                        'd_eps_by_tolerance': {str(t): d_eps_at(sub, t) for t in (0.5, 1.0, 2.0)},
                        'knee_d': knee, 'knee_gain_threshold_pp': a.knee_gain,
                        'gain_per_doubling': gains}
        idr = summary[(summary['mode'] == m) & summary['bottleneck_kind'].eq('identity') & all_seeds]
        if not idr.empty:                                                     # G1
            id_acc = float(idr[SELECT_COL].iloc[0])
            selection[m]['identity_no_bottleneck'] = {
                'd': full_d, 'test_accuracy_pct': id_acc,
                'test_accuracy_std': float(idr['test_accuracy_std'].iloc[0]),
                'gap_to_best_linear_pp': best - id_acc}

    safe_to_csv(summary, out / f'{prefix}pareto_summary.csv', index=False)

    safe_write_text(out / f'{prefix}selection.json', json.dumps({
        'dataset': 'CIFAR100',
        'num_classes_minus_1': NUM_CLASSES - 1,
        'dimension_estimates': {
            'feature_space': 'L2-normalised CLIP image embeddings',
            'mle_d': mle_d, 'pca_d': pca_d, 'erank_d': erank_d,
            'ablation_raw_features': {'pca_d': pca_d_raw, 'erank_d': erank_d_raw},
            'fixed_widths': fixed_widths, 'vlm_dim': vlm_dim, 'backbone_dim': full_d,
        },
        'mle_stability': stab,
        'teacher_zero_shot': zs,
        'run_signature': a.run_signature,
        'code_fingerprint': a.code_fingerprint,
        'complete': complete,
        'missing_runs': len(missing_tags),
        'excluded_incomplete_groups': excluded.to_dict('records'),
        'accuracy_tolerance_pp': a.accuracy_tolerance,
        'selection_split': 'test (no validation split; post-hoc analysis)',
        'split_sizes': split_sizes,
        'min_seeds_for_selection': min_seeds,
        'analysis': 'interim (--analyze-only)' if prefix else 'final',
        'student_augmentation': 'none (matched views)' if a.no_augment else
                                'RandomResizedCrop(0.72-1.0) + flip; teacher features from '
                                'un-augmented images (run --no-augment for the matched-view ablation)',
        'full_width_note': (f"'d={full_d}' rows with bottleneck_kind='linear' contain a trainable "
                            f'{full_d}x{full_d} projection; bottleneck_kind=\'identity\' rows are '
                            'the true no-bottleneck baseline (backbone -> classifier).'),
        'modes': selection,
        'strategy_references': {
            'fixed': 'Romero et al., FitNets, ICLR 2015',
            'mle': 'Levina & Bickel, NIPS 2004 (proposed)',
            'pca': 'Jolliffe, Principal Component Analysis, Springer 2002',
            'erank': 'Roy & Vetterli, EUSIPCO 2007',
            'oracle': 'Test-grid oracle: width with the highest test accuracy, and smallest '
                      'width within 1 pp of it. Optimistic upper bound; not deployable.',
        },
        'distillation_references': {
            'labels': 'Cross-entropy only (baseline)',
            'feat': 'Feature KD: MSE + cosine alignment',
            'rkd': 'Park et al., RKD, CVPR 2019 (distance + angle)',
            'crd': 'Tian et al., CRD, ICLR 2020 (in-batch contrastive variant)',
            'dkd': 'Zhao et al., DKD, CVPR 2022 (DKD-style: beta=1 by default, no warm-up)',
            'sskd': ('Xu et al., Knowledge Distillation Meets Self-Supervision (SSKD), ECCV 2020 '
                     '(SSKD-style: frozen CLIP teacher without its own SS module, one random '
                     'rotation per sample, logit KD on original images only)'),
            'clipkd': ('Yang et al., CLIP-KD: An Empirical Study of CLIP Model Distillation, '
                       'CVPR 2024 (CLIP-KD-style: FD + ICL adapted to classification; ICL '
                       'contrasts the student image embedding with CLIP class-text embeddings)'),
            'zero_shot_clip': ('Radford et al., CLIP, ICML 2021: frozen teacher zero-shot and '
                               'linear-probe accuracy (teacher_zero_shot in this file)'),
            'vl2lite': 'VL2Lite-style: distance-only relational (weight 1) + linguistic KD',
        },
        'note': ('No validation split was used. Bottleneck selection, d_eps, knee_d and '
                 'the oracles are computed on TEST accuracy, so they are post-hoc analysis '
                 'of the evaluated grid, not held-out model selection, and are optimistic. '
                 'Pareto comparisons use test accuracy, are restricted to the evaluated '
                 'models, and refer to the measured hardware (hardware.json).'),
    }, indent=2))

    print(summary.to_string(index=False))
    print(json.dumps(selection, indent=2))
    return summary, selection


def analyze_only(a, seeds, modes, out):
    """A1: rebuild the analysis from results_per_seed.csv without training or a GPU."""
    need = ['results_per_seed.csv', 'dimension_estimates.json']
    missing = [f for f in need if not (out / f).exists()]
    if missing:
        raise FileNotFoundError(f'--analyze-only needs {missing} in {out}; '
                                'run training first (it writes them before the first run).')
    cfg_path = out / 'config.json'
    if cfg_path.exists():
        trained = json.loads(cfg_path.read_text()).get('student')
        if trained and trained != a.student:
            raise ValueError(f'{out} was trained with --student {trained}, not {a.student}.')
    est = json.loads((out / 'dimension_estimates.json').read_text())
    zs_path = out / 'teacher_zero_shot.json'
    zs = json.loads(zs_path.read_text()) if zs_path.exists() else {}
    full_d = BACKBONE_DIM[a.student]
    mle_d, pca_d, erank_d = int(est['mle_d']), int(est['pca_d']), int(est['erank_d'])
    raw = est.get('ablation_raw_features', {})
    fixed_widths = est.get('fixed_widths') or fixed_widths_for(full_d)
    dims = parse_dims(a.dims, mle_d, pca_d, erank_d, full_d, fixed_widths)

    frame = pd.read_csv(out / 'results_per_seed.csv')
    sig_file = out / 'run_signature.txt'
    if sig_file.exists() and 'run_signature' in frame:
        sig = sig_file.read_text().strip()
        other = frame['run_signature'].astype(str) != sig
        if other.any():
            print(f'WARNING: ignoring {int(other.sum())} row(s) from a different run signature.')
            frame = frame[~other]
        a.run_signature = sig
    else:
        a.run_signature = 'unknown'
    a.code_fingerprint = 'not checked (--analyze-only)'
    results = frame.to_dict('records')
    for r in results:
        if not isinstance(r.get('bottleneck_kind'), str):
            r['bottleneck_kind'] = 'linear'
        for k in BENCH_KEYS:
            r.setdefault(k, float('nan'))

    wanted = {run_tag(m, d, s) for s in seeds for d in dims for m in modes}
    if not a.no_identity_baseline:
        wanted |= {run_tag(m, full_d, s, 'identity') for s in seeds for m in modes}
    results = [r for r in results if r['tag'] in wanted]
    if not results:
        raise RuntimeError('No finished run in results_per_seed.csv matches the requested '
                           '--dims/--modes/--seeds.')
    missing_tags = sorted(wanted - {r['tag'] for r in results})
    safe_write_text(out / 'interim_missing_runs.json', json.dumps(missing_tags, indent=2))
    done_d = sorted({int(r['bottleneck_d']) for r in results if r['bottleneck_kind'] == 'linear'})
    print(f'--analyze-only: {len(results)} finished / {len(wanted)} requested run(s); '
          f'widths so far {done_d}; MLE d={mle_d}')

    info = {'seeds': seeds, 'modes': modes, 'mle_d': mle_d, 'pca_d': pca_d, 'erank_d': erank_d,
            'pca_d_raw': raw.get('pca_d'), 'erank_d_raw': raw.get('erank_d'),
            'fixed_widths': fixed_widths, 'full_d': full_d, 'vlm_dim': est.get('vlm_dim'),
            'stab': est.get('mle_stability', {}), 'zs': zs,
            'split_sizes': est.get('split_sizes', 'unknown'),
            'complete': not missing_tags, 'missing_tags': missing_tags,
            'min_seeds': a.min_seeds or 1}
    return analyze_and_write(results, info, a, out, prefix='interim_')


def code_fingerprint():
    """F2: hash of everything that defines how a run is trained."""
    import inspect
    parts = (relational_kd, crd_loss, dkd_loss, linguistic_kd, feature_kd, flip_consistency,
             clip_teacher_logits, logit_kd, sskd_ss_loss, clipkd_losses, CIFAR100RotView,
             StudentMLE, Condense, train_one, student_transform,
             balanced_indices, build_backbone, load_cifar100)
    return hashlib.sha256(''.join(inspect.getsource(f) for f in parts).encode()).hexdigest()[:12]


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    a = make_parser().parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    modes = [m.strip() for m in a.modes.split(',')]
    invalid_modes = set(modes) - VALID_MODES
    if invalid_modes:
        raise ValueError(f'Unknown modes: {sorted(invalid_modes)}. Valid: {sorted(VALID_MODES)}')
    if not 0.0 <= a.label_smoothing < 1.0:
        raise ValueError('--label-smoothing must be in [0, 1)')
    if a.epochs < 1 or a.benchmark_trials < 1 or a.benchmark_repeats < 1:
        raise ValueError('epochs and benchmark trials/repeats must be positive')
    out = Path(a.out)
    if a.analyze_only:                                                        # A1
        analyze_only(a, seeds, modes, out)
        return
    gpu_ids = [int(g) for g in a.gpus.split(',')]
    out.mkdir(parents=True, exist_ok=True)
    warn_if_synced_folder(out)                                                # W5

    # F2: the signature covers the training code, not only the CLI settings, so
    # --resume refuses to mix runs trained under different loss definitions.
    a.code_fingerprint = code_fingerprint()
    a.run_signature = hashlib.sha256(json.dumps(
        {**{k: getattr(a, k) for k in SIGNATURE_FIELDS}, 'code': a.code_fingerprint},
        sort_keys=True).encode()).hexdigest()[:16]
    check_resume_signature(out, a, a.run_signature)

    seed_everything(a.data_seed)
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    report_gpus(gpu_ids)

    train_base, test_base = load_cifar100(a.data_root, download=True)
    print(f'CIFAR-100: train={len(train_base)}, test={len(test_base)}')

    import open_clip
    teacher, _, teacher_tf = open_clip.create_model_and_transforms(
        a.teacher, pretrained=a.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(a.teacher)
    teacher.eval(); teacher.requires_grad_(False)

    # NV: no validation split; students train on the full (optionally capped) training pool
    train_ix = balanced_indices(labels_of(train_base), a.max_train, a.data_seed)
    test_ix = balanced_indices(labels_of(test_base), a.max_test, a.data_seed)
    ds_train = CIFAR100View(train_base, train_ix, student_transform(train=True, augment=not a.no_augment))
    ds_test = CIFAR100View(test_base, test_ix, student_transform())
    print(f'Train/test: {len(ds_train)}/{len(ds_test)}')

    def cache_for(ix, split):
        sig = hashlib.sha256(ix.tobytes() + f'{a.teacher}/{a.pretrained}/CIFAR100/{split}'.encode()).hexdigest()[:16]
        return out / f'teacher_{split}_{a.teacher}_{a.pretrained}_{sig}.pt'.replace('/', '_')

    teacher_feats = build_teacher_cache(teacher, CIFAR100View(train_base, train_ix, teacher_tf),
                                        device, a.extract_batch_size, a.num_workers,
                                        cache_for(train_ix, 'train'), a.extract_save_every)
    teacher_test = build_teacher_cache(teacher, CIFAR100View(test_base, test_ix, teacher_tf),
                                       device, a.extract_batch_size, a.num_workers,
                                       cache_for(test_ix, 'test'), a.extract_save_every)
    teacher_rot = None
    if 'sskd' in modes:                                                       # N1
        teacher_rot = torch.stack([
            build_teacher_cache(teacher, CIFAR100RotView(train_base, train_ix, teacher_tf, k),
                                device, a.extract_batch_size, a.num_workers,
                                cache_for(train_ix, f'train_rot{90 * k}'), a.extract_save_every)
            for k in (1, 2, 3)])                                              # (3, N, D)
    teacher_txt = encode_class_prompts(teacher, tokenizer, device)
    clip_scale = float(teacher.logit_scale.exp().item())
    a.kd_logit_scale_used = clip_scale if a.kd_logit_scale is None else float(a.kd_logit_scale)
    if a.kd_logit_scale is not None and a.kd_logit_scale < 0.5 * clip_scale:
        print(f'WARNING: --kd-logit-scale {a.kd_logit_scale} far below CLIP scale {clip_scale:.1f}; '
              'teacher targets will be near-uniform. Report as ablation.')
    vlm_dim = teacher_feats.shape[1]

    y_train = labels_of(train_base)[train_ix]
    y_test = labels_of(test_base)[test_ix]
    zs = {'train_zero_shot_pct': zero_shot_accuracy(teacher_feats, teacher_txt, y_train),
          'test_zero_shot_pct': zero_shot_accuracy(teacher_test, teacher_txt, y_test),
          'clip_logit_scale': clip_scale, 'kd_logit_scale_used': a.kd_logit_scale_used,
          'templates': PROMPT_TEMPLATES, 'class_prompt_names': PROMPT_NAMES}
    # N3: CLIP teacher references. Zero-shot uses only class prompts; the linear probe
    # trains a logistic regression on frozen CLIP features of the training set. Neither
    # is guaranteed to be an upper bound on a fine-tuned CNN student.
    from sklearn.linear_model import LogisticRegression
    nrm = lambda f: F.normalize(f.float(), dim=-1).numpy()
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(nrm(teacher_feats), y_train)
    zs['train_linear_probe_pct'] = 100.0 * float(probe.score(nrm(teacher_feats), y_train))
    zs['test_linear_probe_pct'] = 100.0 * float(probe.score(nrm(teacher_test), y_test))
    zs['reference_note'] = ('CLIP ViT teacher, frozen: zero-shot from class prompts and a '
                            'logistic-regression linear probe on L2-normalised image features. '
                            'Reference points, not upper bounds for fine-tuned CNN students.')
    safe_write_text(out / 'teacher_zero_shot.json', json.dumps(zs, indent=2))
    print(f"Teacher feats {tuple(teacher_feats.shape)}, CLIP scale {clip_scale:.1f}, "
          f"KD scale {a.kd_logit_scale_used:.1f}, zero-shot test {zs['test_zero_shot_pct']:.2f}%, "
          f"linear probe test {zs['test_linear_probe_pct']:.2f}%")

    # ── Dimensionality estimation (B1, B2, MLE) ───────────────────────────────
    id_table = mle_dimension(teacher_feats, a.id_ks, a.id_samples, a.data_seed)
    safe_to_csv(id_table, out / 'levina_bickel_mle.csv', index=False)
    mle_float = float(id_table['mle_id'].median())
    mle_d = max(2, min(vlm_dim, math.ceil(mle_float)))
    stab = mle_stability(teacher_feats, a.id_ks, a.id_samples, a.data_seed, a.id_repeats)

    # P1: primary PCA / effective rank on the same L2-normalised space as the MLE;
    # raw-feature values are recorded as an ablation.
    pca_d = max(2, min(vlm_dim, pca_dimension(teacher_feats, a.pca_variance_threshold)))
    erank_d = max(2, min(vlm_dim, effective_rank_dimension(teacher_feats)))
    pca_d_raw = max(2, min(vlm_dim, pca_dimension(teacher_feats, a.pca_variance_threshold,
                                                  normalize=False)))
    erank_d_raw = max(2, min(vlm_dim, effective_rank_dimension(teacher_feats, normalize=False)))
    fixed_widths = fixed_widths_for(BACKBONE_DIM[a.student])                  # B3

    print(id_table.to_string(index=False))
    print(f"MLE d={mle_d} (stability {stab['mean']:.2f} +/- {stab['std']:.2f}) | "
          f"PCA d={pca_d} (raw {pca_d_raw}) | Effective rank d={erank_d} (raw {erank_d_raw}) | "
          f"Fixed widths {fixed_widths}")

    safe_write_text(out / 'dimension_estimates.json', json.dumps({           # B7
        'mle_d': mle_d, 'mle_float': mle_float, 'mle_stability': stab,
        'feature_space': 'L2-normalised CLIP image embeddings (MLE, PCA, effective rank)',
        'pca_d': pca_d, 'pca_variance_threshold': a.pca_variance_threshold,
        'erank_d': erank_d,
        'ablation_raw_features': {'pca_d': pca_d_raw, 'erank_d': erank_d_raw},
        'fixed_widths': fixed_widths, 'fixed_width_grid': FIXED_WIDTHS_INDEPENDENT,
        'vlm_dim': vlm_dim,
        'split_sizes': {'train': int(len(train_ix)), 'test': int(len(test_ix))},
    }, indent=2))
    safe_write_text(out / 'mle_summary.json', json.dumps(
        {'mle_by_k': id_table.to_dict('records'), 'mle_median': mle_float, 'mle_d': mle_d,
         'stability': stab, 'num_classes_minus_1': NUM_CLASSES - 1}, indent=2))

    del teacher
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # B5: parse dims with pca and erank tokens; fixed widths are always included
    full_d = BACKBONE_DIM[a.student]
    dims = parse_dims(a.dims, mle_d, pca_d, erank_d, full_d, fixed_widths)
    configs = [(m, d, s, 'linear') for s in seeds for d in dims for m in modes]
    if not a.no_identity_baseline:                                            # G1
        configs += [(m, full_d, s, 'identity') for s in seeds for m in modes]
    print(f'Dims {dims} (+ identity baseline at {full_d}: {not a.no_identity_baseline}) | '
          f'modes {modes} | seeds {seeds} | {len(configs)} runs on GPUs {gpu_ids}')

    done_rows = []
    if a.resume and (out / 'results_per_seed.csv').exists():
        prev = pd.read_csv(out / 'results_per_seed.csv')
        done = {t for t in prev['tag'] if (out / f'{t}.pt').exists()}
        done_rows = prev[prev['tag'].isin(done)].to_dict('records')
        configs = [c for c in configs if run_tag(*c) not in done]
        print(f'Resume: reusing {len(done_rows)} finished runs, {len(configs)} left to train')

    results, curves_all, bench_rows = list(done_rows), [], []
    for r in done_rows:
        hist = out / f"{r['tag']}_history.csv"
        if hist.exists():
            curves_all.extend(pd.read_csv(hist).to_dict('records'))

    def collect(res, bench, curves):
        results.append(res); curves_all.extend(curves)
        if bench:
            bench_rows.extend(bench_trial_rows(res['tag'], bench))
        safe_to_csv(pd.DataFrame(results), out / 'results_per_seed.csv', index=False)

    lost, failed = [], {}
    if len(gpu_ids) == 1 or device.type != 'cuda':
        for cfg in configs:
            try:
                collect(*train_one(*cfg, teacher_feats, teacher_txt, ds_train, ds_test,
                                   vlm_dim, device, a, teacher_rot=teacher_rot))
            except Exception:                                                  # P6
                import traceback
                failed[run_tag(*cfg)] = traceback.format_exc()
                print(f'[ERROR] {run_tag(*cfg)}\n{failed[run_tag(*cfg)]}', flush=True)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    elif configs:
        shared = {'a': a, 'teacher_feats': teacher_feats, 'teacher_txt': teacher_txt,
                  'vlm_dim': vlm_dim, 'train_ids': train_ix, 'test_ids': test_ix,
                  'teacher_rot': teacher_rot}
        buckets = [configs[i::len(gpu_ids)] for i in range(len(gpu_ids))]
        lost, failed = dispatch_workers(_worker, gpu_ids, buckets, shared, collect,
                                        poll_seconds=a.worker_poll_seconds,
                                        stall_seconds=a.worker_stall_hours * 3600)
    safe_write_text(out / 'lost_runs.json', json.dumps(lost, indent=2))
    safe_write_text(out / 'failed_runs.json', json.dumps(failed, indent=2))
    if lost or failed:
        print(f'WARNING: {len(lost)} lost and {len(failed)} failed run(s); see lost_runs.json '
              'and failed_runs.json. Rerun the same command with --resume to retry them.',
              flush=True)

    wanted = {run_tag(m, d, s) for s in seeds for d in dims for m in modes}
    if not a.no_identity_baseline:
        wanted |= {run_tag(m, full_d, s, 'identity') for s in seeds for m in modes}
    results = [r for r in results if r['tag'] in wanted]
    for r in results:                       # rows resumed from older runs
        r.setdefault('bottleneck_kind', 'linear')
    # Persist completed checkpoints/results before checking completeness. Do not
    # publish an oracle or Pareto analysis from a partial dimension x seed grid
    # unless --allow-partial is given (F4).
    safe_to_csv(pd.DataFrame(results), out / 'results_per_seed.csv', index=False)
    missing_tags = sorted(wanted - {r['tag'] for r in results})
    safe_write_text(out / 'missing_runs.json', json.dumps(missing_tags, indent=2))
    complete = not (lost or failed or missing_tags)
    if not complete and not a.allow_partial:
        raise RuntimeError(
            f'Incomplete experiment: {len(failed)} failed, {len(lost)} lost, '
            f'{len(missing_tags)} missing configuration(s). Completed checkpoints '
            f'and results_per_seed.csv are preserved under {out}. '
            'Correct the failures and rerun with --resume, or pass --allow-partial; '
            'selection/oracle results have NOT been generated from this partial grid.')
    if not complete:
        print(f'WARNING: --allow-partial: analysing an incomplete grid ({len(missing_tags)} '
              'missing run(s)). Widths without all seeds are excluded from selection.',
              flush=True)
    if not results:
        raise RuntimeError('No configuration finished; see [ERROR] messages above.')
    if a.benchmark == 'after':
        print('\nBenchmarking saved models sequentially on an idle GPU ...')
        bench_rows = benchmark_saved(results, out, device, a)
    safe_write_text(out / 'hardware.json', json.dumps(hardware_info(device, a), indent=2))  # P6

    safe_to_csv(pd.DataFrame(curves_all), out / 'training_history.csv', index=False)
    safe_to_csv(pd.DataFrame(bench_rows), out / 'benchmark_trials.csv', index=False)

    info = {'seeds': seeds, 'modes': modes, 'mle_d': mle_d, 'pca_d': pca_d, 'erank_d': erank_d,
            'pca_d_raw': pca_d_raw, 'erank_d_raw': erank_d_raw, 'fixed_widths': fixed_widths,
            'full_d': full_d, 'vlm_dim': vlm_dim, 'stab': stab, 'zs': zs,
            'split_sizes': {'train': int(len(train_ix)), 'test': int(len(test_ix))},
            'complete': complete, 'missing_tags': missing_tags,
            'min_seeds': a.min_seeds or len(seeds)}
    analyze_and_write(results, info, a, out)
    safe_write_text(out / 'config.json', json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2))


if __name__ == '__main__':
    run()
