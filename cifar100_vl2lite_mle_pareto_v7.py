"""VL2Lite-style KD + Levina-Bickel MLE bottleneck — CIFAR-100 edition (v7).

Why CIFAR-100 is the key test: CIFAR-10, Fashion-MNIST and SVHN all have
C-1 = 9, so they cannot separate "classification needs ~C-1 dimensions" from
other explanations. CIFAR-100 has C-1 = 99, while CLIP's MLE intrinsic
dimension is expected to stay near ~20. Prediction: the labels/feat accuracy
plateau moves up toward the class count, while the MLE estimate does not.
The default dimension grid therefore extends well beyond 32 (C100-2).

CIFAR-100 changes relative to the CIFAR-10 v5 script:
  C100-1  100 fine-grained classes; prompt names derived from the class list
          (underscores removed, a/an articles).
  C100-2  Default grid 8,mle*0.75,mle,mle*1.5,40,64,96,128,192,full so the
          C-1 = 99 region is covered. Bottlenecks below 100 are rank-limited for
          a 100-way classifier by construction.
  C100-3  Top-5 accuracy reported alongside top-1 (standard for 100 classes).

RQ1: Does the Levina-Bickel MLE estimate of CLIP vision-embedding intrinsic
dimensionality predict where the student bottleneck stops limiting accuracy,
in particular for geometry-preserving distillation?

v5 corrections to v4 (see R-tags in code)
  R1  --kd-logit-scale defaulted to a fixed 10. With T=2 that makes CLIP's teacher
      distribution nearly uniform (0.155 on the top class in a clear case, 0.114
      for a weak teacher), so the linguistic loss again carried almost no class
      signal -- the v3 FIX-2 problem returned. Default is now CLIP's own learned
      logit scale; an explicit value can still be passed for an ablation, and the
      scale actually used is saved.
  R2  The resume signature included seeds/dims/modes, so adding seeds 43,44 to a
      finished seed-42 run was refused. Only per-run TRAINING settings are in the
      signature now; each run's tag already encodes mode, d and seed.
  R3  Data subsampling and the MLE estimate used seeds[0]. Resuming with a
      different seed list could change mle_d and therefore the whole d grid.
      They now use a fixed --data-seed, independent of training seeds.
  R4  d = full uses no bottleneck layer (identity), as in v4, but this is now
      recorded per row (has_bottleneck_layer) so the architecture change at
      d = full is visible in every results table.
  R5  Docstring commands referred to other file names.
  R6  New layers (bottleneck, classifier, KD heads) get lr x --head-lr-mult (default 10);
      at the backbone lr they stayed undertrained, penalising small d most.
  R7  Each run reports its final training cross-entropy and last-epoch drop;
      the summary flags configurations that had not converged.
v7 (merged): the accuracy-focused training recipe from the user's v6 plus the
Windows-stability fixes. Identical training code across all five datasets.
  Recipe
  A1  AdamW by default (--optimizer adamw|sgd): backbone lr 1e-4, new layers
      lr x --head-lr-mult (10); --weight-decay (default 1e-4).
  A2  Label smoothing on the classification loss (--label-smoothing, default 0.1).
      It is known to tighten class clusters in feature space; run one ablation
      with --label-smoothing 0 to show the plateau location does not depend on it.
  A3  Student-only train augmentation: RandomResizedCrop(224, scale 0.72-1.0)
      plus horizontal flip where flips preserve the class (off for SVHN digits).
      Teacher features stay cached from un-augmented images. --no-augment ablates.
  A4  Every d, including the largest, has a bottleneck Linear layer, so widths
      differ only in width. --full-identity restores the old identity at full d.
  Stability (Windows)
  W1  CUDA_DEVICE_ORDER=PCI_BUS_ID: --gpus indices match nvidia-smi, and NVML
      energy readings come from the GPU doing the computation.
  W2  Startup GPU report with a warning if the selected GPU drives a display.
  W3  Resumable teacher extraction: small batches (--extract-batch-size 32),
      per-batch GPU sync, progress output, partial cache every
      --extract-save-every batches.
  W4  faulthandler prints a stack trace on hard crashes.


Comparability warning: v4/v5 settings (lam_cls_start=0.5, final-epoch-only test,
identity at full d, 50 epochs) differ from the v3 runs behind the earlier CIFAR-10,
FMNIST and SVHN tables. Re-run all three datasets with identical v5 settings before
comparing them in a paper.

Requires: pip install torch torchvision open_clip_torch scikit-learn pandas numpy pynvml

Smoke:  python cifar100_vl2lite_mle_pareto_v7.py --max-train 2000 --max-test 500 --epochs 2 \
            --seeds 42 --dims mle,full --benchmark-repeats 30 --id-samples 1000 --gpus 0
Full:   python cifar100_vl2lite_mle_pareto_v7.py --epochs 50 --seeds 42,43,44 --gpus 0,1 --num-workers 2
Add seeds later:  same command with --seeds 42,43,44 --resume
"""

import os
# W1: number GPUs like nvidia-smi (PCI bus order). Must be set before CUDA starts.
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
import faulthandler
faulthandler.enable()          # W4: print a stack trace on hard crashes
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

# torchvision CIFAR-100 fine-label names (alphabetical; index == label)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
    'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
    'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
    'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
    'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
    'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
    'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
    'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
_NO_ARTICLE = {'cattle'}    # mass noun: "a photo of cattle"


def _prompt_name(c):
    name = c.replace('_', ' ')
    if name in _NO_ARTICLE:
        return name
    return ('an ' if name[0] in 'aeiou' else 'a ') + name


# Grammatical names for prompts, index-aligned with CIFAR100_CLASSES
PROMPT_NAMES = [_prompt_name(c) for c in CIFAR100_CLASSES]
# CLIP-style ensemble; CIFAR images are small and often blurry after upscaling
PROMPT_TEMPLATES = ['a photo of {}.', 'a blurry photo of {}.',
                    'a low resolution photo of {}.', 'a close-up photo of {}.']
NUM_CLASSES = 100
# A3: train-time horizontal flip (horizontal flips preserve the class)
AUG_HFLIP = True
BACKBONE_DIM = {'resnet18': 512, 'mobilenet_v2': 1280}

# R2: settings that change how an individual run is trained. Seeds, dims and modes
# only choose WHICH runs exist (encoded in each tag), so they are not included.
SIGNATURE_FIELDS = ['teacher', 'pretrained', 'student', 'no_pretrained_student',
                    'max_train', 'max_test', 'data_seed', 'epochs',
                    'batch_size', 'lr', 'head_lr_mult', 'temperature', 'lam_cls_start', 'lam_cls_end',
                    'kd_logit_scale', 'w_mse', 'w_cos', 'w_consistency',
                    'id_samples', 'id_ks', 'optimizer', 'weight_decay',
                    'label_smoothing', 'no_augment', 'full_identity']


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
    p.add_argument('--data-seed', type=int, default=42,
                   help='R3: seed for subsampling and the MLE estimate (independent of training seeds)')
    p.add_argument('--teacher', default='ViT-B-32')
    p.add_argument('--pretrained', default='laion2b_s34b_b79k')
    p.add_argument('--student', choices=list(BACKBONE_DIM), default='resnet18')
    p.add_argument('--no-pretrained-student', action='store_true')
    p.add_argument('--id-samples', type=int, default=5000)
    p.add_argument('--id-ks', type=int, nargs='+', default=[5, 10, 20, 50])
    p.add_argument('--id-repeats', type=int, default=5,
                   help='MLE repeats on different random subsamples')
    p.add_argument('--dims', default='8,mle*0.75,mle,mle*1.5,40,64,96,128,192,full',
                   help="Tokens: integer, 'mle', 'mle*<factor>', 'full' (= backbone dim).")
    p.add_argument('--modes', default='labels,feat,vl2lite')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Test only at final epoch by default (0). Positive values are diagnostic only.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--extract-batch-size', type=int, default=32,
                   help='W3: batch size for CLIP feature extraction (smaller = shorter GPU bursts)')
    p.add_argument('--extract-save-every', type=int, default=100,
                   help='W3: save a partial feature cache every N extraction batches')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='A1: learning rate for the pretrained backbone (default 1e-4 with AdamW)')
    p.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.1,
                   help='A2: label smoothing for the classification loss (0 disables)')
    p.add_argument('--no-augment', action='store_true',
                   help='A3: disable student-only train augmentation (ablation)')
    p.add_argument('--full-identity', action='store_true',
                   help='A4: old behaviour, no bottleneck layer at d = backbone dim')
    p.add_argument('--head-lr-mult', type=float, default=10.0,
                   help='R6: new layers (bottleneck, classifier, KD heads) train at lr x this. '
                        'They start from random weights; at the backbone lr they stay '
                        'undertrained, which penalises small bottlenecks most.')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--gpus', default='0')
    p.add_argument('--resume', action='store_true',
                   help='Skip runs already in results_per_seed.csv with a saved .pt. '
                        'Allowed only when training settings match (seeds/dims/modes may differ).')
    # VL2Lite-style KD (lam_cls_start=0.5 deviates from the paper's 0.01; report as "VL2Lite-style")
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lam-cls-start', type=float, default=0.50)
    p.add_argument('--lam-cls-end', type=float, default=0.80)
    p.add_argument('--kd-logit-scale', type=float, default=None,
                   help="R1: image-text logit scale for the linguistic loss. Default: CLIP's "
                        'learned scale (~100). Small values flatten the teacher targets.')
    # feat mode
    p.add_argument('--w-mse', type=float, default=1.0)
    p.add_argument('--w-cos', type=float, default=2.0)
    p.add_argument('--w-consistency', type=float, default=0.0,
                   help='Optional horizontal-flip consistency for KD modes (object class is '
                        'flip-invariant). Off by default to keep the comparison clean.')
    # Analysis / benchmark
    p.add_argument('--accuracy-tolerance', type=float, default=1.0)
    p.add_argument('--knee-gain', type=float, default=2.0,
                   help='knee = smallest d after which every doubling gains < this many pp')
    p.add_argument('--benchmark', choices=['after', 'inline', 'none'], default='after')
    p.add_argument('--benchmark-repeats', type=int, default=2000,
                   help='Inferences per trial; ~2000 gives a multi-second window the NVML '
                        'energy counter can resolve')
    p.add_argument('--benchmark-trials', type=int, default=5)
    return p


# ── Data ──────────────────────────────────────────────────────────────────────
class CIFAR100View(Dataset):
    """One view per dataset (student OR teacher). Returns (image, label, local_index)."""
    def __init__(self, base, indices, tf):
        self.base, self.indices, self.tf = base, np.asarray(indices, dtype=np.int64), tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[int(self.indices[i])]
        return self.tf(img.convert('RGB')), int(label), i


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
    """A3: only the student's training view is augmented. Teacher features are cached
    from un-augmented images, so they act as a fixed target for every crop/flip."""
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
    assert list(tr.classes) == CIFAR100_CLASSES, 'Unexpected CIFAR-100 class list/order'
    return tr, te


def labels_of(ds):
    return np.asarray(ds.targets)


# ── MLE ───────────────────────────────────────────────────────────────────────
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


@torch.no_grad()
def build_teacher_cache(teacher, dataset, device, batch_size, workers, cache_path, save_every=100):
    """W3: extract CLIP image features with progress, per-batch GPU sync and
    resumable partial saves. Rows are in dataset order (no shuffling)."""
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
            torch.cuda.synchronize(device)      # keep each GPU submission short
        else:
            f = teacher.encode_image(timgs)
        chunks.append(f.float().cpu())
        if b % save_every == 0 or b == n_batches:
            chunks = [torch.cat(chunks)]
            n_done = chunks[0].shape[0]
            torch.save({'features': chunks[0], 'n_total': n_total}, partial)
            rate = (n_done - start) / max(time.perf_counter() - t0, 1e-6)
            eta = (n_total - n_done) / max(rate, 1e-6)
            print(f'  {n_done}/{n_total} images ({rate:.0f} img/s, ~{eta / 60:.1f} min left)', flush=True)
    feats = torch.cat(chunks)
    assert feats.shape[0] == n_total, (feats.shape, n_total)
    torch.save({'features': feats}, cache_path)
    partial.unlink(missing_ok=True)
    return feats


def report_gpus(selected):
    """W2: list GPUs (PCI order = nvidia-smi order) and warn if a selected GPU drives a display."""
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
            print(f'  WARNING: CUDA device {i} is {props.name} but NVML device {i} is {nv_name}; '
                  'energy readings may come from a different GPU (check CUDA_VISIBLE_DEVICES).')
        if i in selected and display == 'YES':
            print(f'  WARNING: cuda:{i} drives a display. Long GPU work on it can trigger a Windows '
                  'driver reset (TDR), killing this job. Prefer a GPU with no monitor attached.')


@torch.no_grad()
def encode_class_prompts(teacher, tokenizer, device):
    """Prompt ensemble, averaged and renormalised per class -> (C, D)."""
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
class StudentMLE(nn.Module):
    """backbone -> [Linear(backbone_dim, d_b) unless d_b == backbone_dim] -> Linear(d_b, C)."""
    def __init__(self, name, n_classes, d_b, pretrained=True, full_identity=False):
        super().__init__()
        if name == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            net.fc = nn.Identity()
        else:
            net = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            net.classifier = nn.Identity()
        self.backbone = net
        # A4: identity only at full width and only if --full-identity
        self.has_bottleneck_layer = d_b != BACKBONE_DIM[name] or not full_identity
        self.bottleneck = (nn.Linear(BACKBONE_DIM[name], d_b) if self.has_bottleneck_layer
                           else nn.Identity())
        self.classifier = nn.Linear(d_b, n_classes)

    def forward(self, x, return_bottleneck=False):
        z = self.bottleneck(self.backbone(x))
        logits = self.classifier(z)
        return (z, logits) if return_bottleneck else logits


class Deployable(nn.Module):
    """Inference-only model: no Condense / alignment heads."""
    def __init__(self, s):
        super().__init__()
        self.backbone, self.bottleneck, self.classifier = s.backbone, s.bottleneck, s.classifier

    def forward(self, x):
        return self.classifier(self.bottleneck(self.backbone(x)))


class Condense(nn.Module):
    """Two-layer MLP vlm_dim -> d_b (text prototypes only)."""
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
def relational_kd(z, t_img):
    """RKD-style mean-normalised distance matching against FROZEN teacher geometry."""
    zs = F.normalize(z.float(), dim=-1)
    ts = F.normalize(t_img.float(), dim=-1)
    off = ~torch.eye(zs.shape[0], dtype=torch.bool, device=zs.device)
    d_s, d_t = torch.cdist(zs, zs), torch.cdist(ts, ts)
    d_s = d_s / d_s[off].mean().clamp_min(1e-6)
    d_t = d_t / d_t[off].mean().clamp_min(1e-6)
    return F.smooth_l1_loss(d_s[off], d_t[off])


def linguistic_kd(z, t_img, t_txt, cond_text, logit_scale, temp):
    """Image-text logits; F.kl_div(log q, p) = KL(p_teacher || q_student)."""
    protos = F.normalize(cond_text(t_txt.float()), dim=-1)
    s_lw = logit_scale * F.normalize(z.float(), dim=-1) @ protos.T
    s_vlm = logit_scale * (F.normalize(t_img.float(), dim=-1)
                           @ F.normalize(t_txt.float(), dim=-1).T)
    return (temp ** 2) * F.kl_div(F.log_softmax(s_lw / temp, dim=1),
                                  F.softmax(s_vlm / temp, dim=1), reduction='batchmean')


def feature_kd(z, t_img, align):
    p = F.normalize(align(z).float(), dim=-1)
    t = F.normalize(t_img.float(), dim=-1)
    return (p - t).pow(2).sum(1).mean(), (1 - (p * t).sum(1)).mean()


def flip_consistency(student, x, z):
    """Horizontal flip is label-preserving for natural object classes (unlike digits)."""
    z_aug, _ = student(torch.flip(x, dims=[-1]), return_bottleneck=True)
    return (F.normalize(z.float(), dim=-1)
            - F.normalize(z_aug.float(), dim=-1)).pow(2).sum(1).mean()


# ── Evaluation / benchmark ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, true, top5_hits = [], [], 0
    for xs, ys, _ in loader:
        out = model(xs.to(device))
        if isinstance(out, tuple):
            out = out[1]
        pred.extend(out.argmax(1).cpu().tolist()); true.extend(ys.tolist())
        top5 = out.topk(5, dim=1).indices.cpu()
        top5_hits += int((top5 == ys.view(-1, 1)).any(dim=1).sum())
    return {'accuracy_pct': 100 * accuracy_score(true, pred),
            'top5_accuracy_pct': 100 * top5_hits / len(true),
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


def benchmark_saved(results, out, device, a):
    """Time every saved deployable model sequentially on one idle GPU."""
    rows = []
    for r in results:
        obj = torch.load(out / f"{r['tag']}.pt", map_location='cpu', weights_only=True)
        dep = Deployable(StudentMLE(a.student, NUM_CLASSES, int(r['bottleneck_d']), pretrained=False,
                                    full_identity=a.full_identity))
        dep.load_state_dict(obj['student_deployed'])
        b = benchmark(dep.to(device), device, a)
        r.update({k: b[k] for k in BENCH_KEYS})
        rows += bench_trial_rows(r['tag'], b)
        print(f"  benchmarked {r['tag']}: {b['latency_ms']:.3f} ms, {b['energy_mJ']:.1f} mJ", flush=True)
        del dep
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return rows


# ── Pareto / analysis ─────────────────────────────────────────────────────────
def pareto_mask(frame, acc_col='test_accuracy_pct'):
    """Exhaustive nondominance within the finite evaluated set only."""
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
    keys = {(str(r['mode']), int(r['bottleneck_d'])) for _, r in frame.iterrows()}
    freq = {k: 0 for k in keys}
    seeds = frame['seed'].unique()
    for s in seeds:
        sub = frame[frame['seed'] == s].reset_index(drop=True)
        mask = pareto_mask(sub)
        for i, r in sub.iterrows():
            if mask[i] == 'nondominated_within_evaluated_set':
                freq[(str(r['mode']), int(r['bottleneck_d']))] += 1
    return {k: round(v / len(seeds), 3) for k, v in freq.items()}


def knee_analysis(sub, knee_gain):
    """Accuracy gain per doubling of d; knee = smallest d after which every doubling gains < knee_gain."""
    s = sub.sort_values('bottleneck_d')
    d = s['bottleneck_d'].to_numpy(float)
    acc = s['test_accuracy_pct'].to_numpy(float)
    gains = [{'from_d': int(d[i]), 'to_d': int(d[i + 1]),
              'gain_pp_per_doubling': round(float((acc[i + 1] - acc[i]) / math.log2(d[i + 1] / d[i])), 3)}
             for i in range(len(d) - 1)]
    knee = int(d[-1])
    for i in range(len(gains)):
        if all(g['gain_pp_per_doubling'] < knee_gain for g in gains[i:]):
            knee = gains[i]['from_d']
            break
    return gains, knee


def d_eps_at(sub, tol):
    best = sub['test_accuracy_pct'].max()
    return int(sub[sub['test_accuracy_pct'] >= best - tol]['bottleneck_d'].min())


def parse_dims(spec, mle_d, full_d):
    dims = []
    for tok in spec.split(','):
        tok = tok.strip().lower()
        if tok == 'mle':
            d = mle_d
        elif tok.startswith('mle*'):
            d = int(round(mle_d * float(tok[4:])))
        elif tok == 'full':
            d = full_d
        else:
            d = int(tok)
        d = max(2, min(full_d, d))
        if d not in dims:
            dims.append(d)
    return sorted(dims)


def check_resume_signature(out, a, run_signature):
    """R2: refuse to mix runs trained under different settings."""
    sigfile, results_csv = out / 'run_signature.txt', out / 'results_per_seed.csv'
    if results_csv.exists() and not a.resume:
        raise RuntimeError('Existing results in --out. Use --resume (same training settings) '
                           'or a fresh --out directory.')
    if results_csv.exists() and not sigfile.exists():
        raise RuntimeError('Results in --out were produced by an older script version (no '
                           'run_signature.txt). They used different settings and must not be '
                           'mixed with this run. Use a fresh --out directory.')
    if sigfile.exists() and sigfile.read_text().strip() != run_signature:
        raise RuntimeError('Training settings differ from the existing run in --out. '
                           'Use a fresh --out directory.')
    sigfile.write_text(run_signature)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one(mode, d_b, seed, teacher_feats, teacher_txt, ds_train, ds_test,
              vlm_dim, device, a):
    seed_everything(seed)
    tag = f'{mode}_d{d_b}_seed{seed}'
    student = StudentMLE(a.student, NUM_CLASSES, d_b,
                         pretrained=not a.no_pretrained_student,
                         full_identity=a.full_identity).to(device)
    cond_text = Condense(vlm_dim, d_b).to(device)
    align = nn.Linear(d_b, vlm_dim).to(device)

    # R6: pretrained backbone at a.lr; freshly initialised layers at a.lr * head_lr_mult
    backbone_params = list(student.backbone.parameters())
    head_params = list(student.bottleneck.parameters()) + list(student.classifier.parameters())
    if mode == 'vl2lite':
        head_params += list(cond_text.parameters())
    elif mode == 'feat':
        head_params += list(align.parameters())
    params = backbone_params + head_params

    groups = [{'params': backbone_params, 'lr': a.lr},
              {'params': head_params, 'lr': a.lr * a.head_lr_mult}]
    if a.optimizer == 'adamw':                                            # A1
        opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    else:
        opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=a.weight_decay)
    # Cosine decay of each group to 1% of its own starting rate
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * e / a.epochs)))
    train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, pin_memory=device.type == 'cuda',
                              persistent_workers=a.num_workers > 0)
    # At most 2 short-lived test workers (a second persistent pool caused OSError 22 on Windows)
    test_loader = DataLoader(ds_test, batch_size=a.batch_size, shuffle=False,
                             num_workers=min(a.num_workers, 2), persistent_workers=False)
    t_txt = teacher_txt.to(device)
    curves, test, t_start = [], None, time.perf_counter()

    for epoch in range(a.epochs):
        student.train(); cond_text.train(); align.train()
        frac = epoch / max(a.epochs - 1, 1)
        lam_cls = a.lam_cls_start + (a.lam_cls_end - a.lam_cls_start) * frac
        lam_kd = (1.0 - lam_cls) / 2.0
        sums = dict(loss=0.0, cls=0.0, kd1=0.0, kd2=0.0)

        for xs, ys, ix in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            t_img = teacher_feats[ix].to(device)
            z, logits = student(xs, return_bottleneck=True)
            l_cls = F.cross_entropy(logits, ys, label_smoothing=a.label_smoothing)   # A2

            if mode == 'labels':
                loss, k1, k2 = l_cls, 0.0, 0.0
            elif mode == 'feat':
                mse, cos = feature_kd(z, t_img, align)
                loss = l_cls + a.w_mse * mse + a.w_cos * cos
                k1, k2 = mse.item(), cos.item()
            else:
                l_vis = relational_kd(z, t_img)
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
            sums['loss'] += loss.item() * bs; sums['cls'] += l_cls.item() * bs
            sums['kd1'] += k1 * bs; sums['kd2'] += k2 * bs

        sched.step()
        n = len(ds_train)
        evaluated = epoch == a.epochs - 1 or (a.eval_every > 0 and (epoch + 1) % a.eval_every == 0)
        if evaluated:
            test = evaluate(student, test_loader, device)
        curves.append({'tag': tag, 'mode': mode, 'bottleneck_d': d_b, 'seed': seed,
                       'epoch': epoch + 1, 'lam_cls': lam_cls if mode == 'vl2lite' else 1.0,
                       **{k: v / n for k, v in sums.items()},
                       'test_accuracy_pct': test['accuracy_pct'] if evaluated else float('nan')})
        acc_txt = f"{test['accuracy_pct']:.2f}" if evaluated else 'n/a'
        print(f"[{tag}] ep{epoch + 1:02d}/{a.epochs} loss={sums['loss']/n:.4f} "
              f"cls={sums['cls']/n:.4f} kd1={sums['kd1']/n:.4f} kd2={sums['kd2']/n:.4f} "
              f"test_acc={acc_txt}", flush=True)

    train_s = time.perf_counter() - t_start
    deployed = Deployable(student)
    out_dir = Path(a.out)
    bench = benchmark(deployed.to(device), device, a) if a.benchmark == 'inline' else None
    torch.save({'student_deployed': deployed.cpu().state_dict(), 'architecture': a.student,
                'bottleneck_d': d_b, 'mode': mode, 'seed': seed}, out_dir / f'{tag}.pt')
    pd.DataFrame(curves).to_csv(out_dir / f'{tag}_history.csv', index=False)
    # R7: did the run converge? Relative drop in cross-entropy over the last epoch.
    ce_last = curves[-1]['cls']
    ce_prev = curves[-2]['cls'] if len(curves) > 1 else float('nan')
    ce_drop = 100.0 * (ce_prev - ce_last) / ce_prev if ce_prev and np.isfinite(ce_prev) else float('nan')
    res = {'tag': tag, 'mode': mode, 'seed': seed, 'bottleneck_d': d_b,
           'final_train_ce': ce_last, 'last_epoch_ce_drop_pct': ce_drop,
           'has_bottleneck_layer': student.has_bottleneck_layer,                # R4
           'test_accuracy_pct': test['accuracy_pct'], 'test_top5_accuracy_pct': test['top5_accuracy_pct'],
           'test_macro_f1_pct': test['macro_f1_pct'],
           'parameters': sum(p.numel() for p in deployed.parameters()), 'train_seconds': train_s,
           'run_signature': a.run_signature}
    res.update({k: (bench[k] if bench else float('nan')) for k in BENCH_KEYS})
    return res, bench, curves


def _worker(gpu_id, configs, shared, q):
    """No CLIP model in workers; only student-view datasets are needed."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    a = shared['a']
    train_base, test_base = load_cifar100(a.data_root, download=False)
    ds_train = CIFAR100View(train_base, shared['train_ids'], student_transform(train=True, augment=not a.no_augment))
    ds_test = CIFAR100View(test_base, shared['test_ids'], student_transform())
    for mode, d, seed in configs:
        try:
            q.put(('ok',) + train_one(mode, d, seed, shared['teacher_feats'], shared['teacher_txt'],
                                      ds_train, ds_test, shared['vlm_dim'], device, a))
        except Exception:
            import traceback
            q.put(('err', f'{mode}_d{d}_seed{seed}', traceback.format_exc()))


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    a = make_parser().parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    modes = [m.strip() for m in a.modes.split(',')]
    invalid_modes = set(modes) - {'labels', 'feat', 'vl2lite'}
    if invalid_modes:
        raise ValueError(f'Unknown modes: {sorted(invalid_modes)}')
    if not 0.0 <= a.label_smoothing < 1.0:
        raise ValueError('--label-smoothing must be in [0, 1)')
    if a.epochs < 1 or a.benchmark_trials < 1 or a.benchmark_repeats < 1:
        raise ValueError('epochs and benchmark trials/repeats must be positive')
    gpu_ids = [int(g) for g in a.gpus.split(',')]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # R2: check resume safety before any expensive work
    a.run_signature = hashlib.sha256(json.dumps(
        {k: getattr(a, k) for k in SIGNATURE_FIELDS}, sort_keys=True).encode()).hexdigest()[:16]
    check_resume_signature(out, a, a.run_signature)

    seed_everything(a.data_seed)
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    report_gpus(gpu_ids)                                                  # W2

    train_base, test_base = load_cifar100(a.data_root, download=True)
    print(f'CIFAR-100: train={len(train_base)}, test={len(test_base)}')

    import open_clip
    teacher, _, teacher_tf = open_clip.create_model_and_transforms(
        a.teacher, pretrained=a.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(a.teacher)
    teacher.eval(); teacher.requires_grad_(False)

    # R3: subsampling uses data_seed, not the training seeds
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
    teacher_txt = encode_class_prompts(teacher, tokenizer, device)
    clip_scale = float(teacher.logit_scale.exp().item())
    # R1: default to CLIP's learned scale; an explicit value is an ablation
    a.kd_logit_scale_used = clip_scale if a.kd_logit_scale is None else float(a.kd_logit_scale)
    if a.kd_logit_scale is not None and a.kd_logit_scale < 0.5 * clip_scale:
        print(f'WARNING: --kd-logit-scale {a.kd_logit_scale} is far below CLIP\'s {clip_scale:.1f}; '
              'teacher targets will be close to uniform. Report this as an ablation.')
    vlm_dim = teacher_feats.shape[1]

    y_train = labels_of(train_base)[train_ix]
    y_test = labels_of(test_base)[test_ix]
    zs = {'train_zero_shot_pct': zero_shot_accuracy(teacher_feats, teacher_txt, y_train),
          'test_zero_shot_pct': zero_shot_accuracy(teacher_test, teacher_txt, y_test),
          'clip_logit_scale': clip_scale, 'kd_logit_scale_used': a.kd_logit_scale_used,
          'templates': PROMPT_TEMPLATES, 'class_prompt_names': PROMPT_NAMES}
    (out / 'teacher_zero_shot.json').write_text(json.dumps(zs, indent=2))
    print(f"Teacher feats {tuple(teacher_feats.shape)}, CLIP logit scale {clip_scale:.1f}, "
          f"KD scale used {a.kd_logit_scale_used:.1f}, zero-shot test acc {zs['test_zero_shot_pct']:.2f}%")

    # R3: MLE uses data_seed so mle_d (and the d grid) never depends on the seed list
    id_table = mle_dimension(teacher_feats, a.id_ks, a.id_samples, a.data_seed)
    id_table.to_csv(out / 'levina_bickel_mle.csv', index=False)
    mle_float = float(id_table['mle_id'].median())
    mle_d = max(2, min(vlm_dim, math.ceil(mle_float)))
    stab = mle_stability(teacher_feats, a.id_ks, a.id_samples, a.data_seed, a.id_repeats)
    print(id_table.to_string(index=False))
    print(f"MLE dimension: {mle_d} (stability {stab['mean']:.2f} +/- {stab['std']:.2f})")
    (out / 'mle_summary.json').write_text(json.dumps(
        {'mle_by_k': id_table.to_dict('records'), 'mle_median': mle_float, 'mle_d': mle_d,
         'stability': stab, 'num_classes_minus_1': NUM_CLASSES - 1}, indent=2))
    del teacher
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    dims = parse_dims(a.dims, mle_d, BACKBONE_DIM[a.student])
    configs = [(m, d, s) for s in seeds for d in dims for m in modes]
    print(f'Dims {dims} | modes {modes} | seeds {seeds} | {len(configs)} runs on GPUs {gpu_ids}')

    done_rows = []
    if a.resume and (out / 'results_per_seed.csv').exists():
        prev = pd.read_csv(out / 'results_per_seed.csv')
        done = {t for t in prev['tag'] if (out / f'{t}.pt').exists()}
        done_rows = prev[prev['tag'].isin(done)].to_dict('records')
        configs = [c for c in configs if f'{c[0]}_d{c[1]}_seed{c[2]}' not in done]
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
        pd.DataFrame(results).to_csv(out / 'results_per_seed.csv', index=False)

    if len(gpu_ids) == 1 or device.type != 'cuda':
        for m, d, s in configs:
            collect(*train_one(m, d, s, teacher_feats, teacher_txt, ds_train, ds_test,
                               vlm_dim, device, a))
    elif configs:
        shared = {'a': a, 'teacher_feats': teacher_feats, 'teacher_txt': teacher_txt,
                  'vlm_dim': vlm_dim, 'train_ids': train_ix, 'test_ids': test_ix}
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        buckets = [configs[i::len(gpu_ids)] for i in range(len(gpu_ids))]
        procs = [ctx.Process(target=_worker, args=(g, b, shared, q)) for g, b in zip(gpu_ids, buckets)]
        for p in procs:
            p.start()
        for n in range(len(configs)):
            item = q.get()
            if item[0] == 'err':
                print(f'[ERROR] {item[1]}\n{item[2]}', flush=True)
            else:
                collect(*item[1:])
                print(f'  collected {n + 1}/{len(configs)}: {item[1]["tag"]}', flush=True)
        for p in procs:
            p.join()

    # Keep only runs belonging to the requested grid (seeds/dims/modes of this call)
    wanted = {f'{m}_d{d}_seed{s}' for s in seeds for d in dims for m in modes}
    results = [r for r in results if r['tag'] in wanted]
    if not results:
        raise RuntimeError('No configuration finished; see [ERROR] messages above.')
    if a.benchmark == 'after':
        print('\nBenchmarking saved models sequentially on an idle GPU ...')
        bench_rows = benchmark_saved(results, out, device, a)

    pd.DataFrame(results).to_csv(out / 'results_per_seed.csv', index=False)
    pd.DataFrame(curves_all).to_csv(out / 'training_history.csv', index=False)
    pd.DataFrame(bench_rows).to_csv(out / 'benchmark_trials.csv', index=False)

    frame = pd.DataFrame(results)
    expected = len(seeds)
    summary = frame.groupby(['mode', 'bottleneck_d']).agg(
        has_bottleneck_layer=('has_bottleneck_layer', 'first'),
        test_accuracy_pct=('test_accuracy_pct', 'mean'),
        test_accuracy_std=('test_accuracy_pct', 'std'),          # NaN for a single seed
        test_top5_accuracy_pct=('test_top5_accuracy_pct', 'mean'),
        test_macro_f1_pct=('test_macro_f1_pct', 'mean'),
        latency_ms=('latency_ms', 'mean'), latency_iqr_ms=('latency_iqr_ms', 'mean'),
        energy_mJ=('energy_mJ', 'mean'), energy_iqr_mJ=('energy_iqr_mJ', 'mean'),
        parameters=('parameters', 'first'), train_seconds=('train_seconds', 'mean'),
        final_train_ce=('final_train_ce', 'mean'),
        last_epoch_ce_drop_pct=('last_epoch_ce_drop_pct', 'mean'),
        n_seeds=('seed', 'count')).reset_index()
    summary['mle_selected'] = summary['bottleneck_d'].eq(mle_d)
    # R7: flag runs whose training loss was still falling clearly at the end
    summary['converged'] = summary['last_epoch_ce_drop_pct'] < 2.0
    if (~summary['converged']).any():
        bad = summary.loc[~summary['converged'], ['mode', 'bottleneck_d', 'last_epoch_ce_drop_pct']]
        print('WARNING: training loss still falling >2% in the last epoch for:\n'
              f'{bad.to_string(index=False)}\nIncrease --epochs before comparing bottleneck sizes.')
    if (summary['n_seeds'] < expected).any():
        print(f'WARNING: some configurations have fewer than {expected} seeds; check [ERROR] lines.')

    summary['pareto_all'] = pareto_mask(summary)
    summary['pareto_within_mode'] = ''
    for m in modes:
        idx = summary.index[summary['mode'] == m]
        summary.loc[idx, 'pareto_within_mode'] = pareto_mask(summary.loc[idx].reset_index(drop=True))
    pfreq = pareto_freq_per_seed(frame) if expected > 1 else {}
    summary['nondominated_freq_per_seed'] = [
        pfreq.get((str(r['mode']), int(r['bottleneck_d'])), float('nan'))
        if r['n_seeds'] == expected else float('nan')
        for _, r in summary.iterrows()]
    summary.to_csv(out / 'pareto_summary.csv', index=False)

    selection = {}
    for m in modes:
        sub = summary[summary['mode'] == m]
        if sub.empty:
            continue
        best = float(sub['test_accuracy_pct'].max())
        has_mle = (sub['bottleneck_d'] == mle_d).any()
        acc_mle = float(sub.loc[sub['bottleneck_d'] == mle_d, 'test_accuracy_pct'].iloc[0]) \
            if has_mle else float('nan')
        gains, knee = knee_analysis(sub, a.knee_gain)
        d_eps = d_eps_at(sub, a.accuracy_tolerance)
        selection[m] = {
            'best_accuracy_pct': best, 'mle_d': mle_d, 'mle_accuracy_pct': acc_mle,
            'mle_gap_pp': best - acc_mle,
            'd_eps': d_eps, 'mle_equals_d_eps': mle_d == d_eps,
            'd_eps_by_tolerance': {str(t): d_eps_at(sub, t) for t in (0.5, 1.0, 2.0)},
            'knee_d': knee, 'knee_gain_threshold_pp': a.knee_gain,
            'gain_per_doubling': gains}
    (out / 'selection.json').write_text(json.dumps(
        {'dataset': 'CIFAR100', 'num_classes_minus_1': NUM_CLASSES - 1,
         'mle_d': mle_d, 'mle_stability': stab, 'teacher_zero_shot': zs,
         'run_signature': a.run_signature,
         'accuracy_tolerance_pp': a.accuracy_tolerance, 'modes': selection,
         'note': 'd_eps and knee_d are computed on the test set (no validation split); '
                 'report them as post-hoc analysis, not model selection. Pareto claims '
                 'cover only the evaluated grid. ' + ('The largest d has no bottleneck layer.'
                 if a.full_identity else 'Every d, including the largest, has a bottleneck layer.')}, indent=2))
    (out / 'config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    run()
