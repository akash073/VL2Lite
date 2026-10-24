"""VL2Lite-style KD + Levina-Bickel MLE bottleneck — SVHN edition (v8).

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
  B3  Added fixed-width baseline: nearest standard width to mle_d from
      {8, 16, 32, 64, 128, 256, 512}.
  B4  Validation-grid oracle: best d on the evaluated grid (upper bound).
  B5  parse_dims now accepts tokens 'pca' and 'erank' in --dims.
  B6  All five selection strategies are reported in selection.json per mode.
  B7  pca_d and erank_d saved to dimension_estimates.json.
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

Smoke:
  python svhn_vl2lite_mle_pareto_v8.py --max-train 2000 --max-test 500 --epochs 2 \\
      --seeds 42 --dims mle,pca,erank,full --modes labels,rkd,crd,dkd,vl2lite \\
      --benchmark-repeats 30 --id-samples 1000 --gpus 0

Full:
  python svhn_vl2lite_mle_pareto_v8.py --epochs 50 --seeds 42,43,44 \\
      --modes labels,feat,rkd,crd,dkd,vl2lite \\
      --dims 8,mle*0.5,mle*0.75,mle,pca,erank,mle*1.25,mle*1.5,32,128,full \\
      --gpus 0,1 --num-workers 2
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

# torchvision maps SVHN's original label 10 to 0, so index == digit
SVHN_CLASSES = [str(i) for i in range(10)]
PROMPT_NAMES = SVHN_CLASSES
PROMPT_TEMPLATES = ['a photo of the number: "{}".', 'a photo of the house number "{}".',
                    'a street view photo of the digit {}.', 'a blurry photo of the number {}.']
NUM_CLASSES = 10
AUG_HFLIP = False
BACKBONE_DIM = {'resnet18': 512, 'resnet50': 2048, 'mobilenet_v2': 1280,
                'mobilenet_v3_large': 960, 'efficientnet_b0': 1280, 'convnext_tiny': 768}

# Standard fixed widths for the fixed-width baseline (B3)
FIXED_WIDTH_GRID = [8, 16, 32, 64, 128, 256, 512]

# All valid distillation modes
VALID_MODES = {'labels', 'feat', 'rkd', 'crd', 'dkd', 'vl2lite'}


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
                    'use_extra',
                    'max_train', 'max_test', 'data_seed', 'epochs',
                    'batch_size', 'lr', 'head_lr_mult', 'temperature', 'lam_cls_start', 'lam_cls_end',
                    'kd_logit_scale', 'w_mse', 'w_cos', 'w_consistency',
                    'id_samples', 'id_ks', 'optimizer', 'weight_decay',
                    'label_smoothing', 'no_augment', 'full_identity',
                    # S1: new fields
                    'crd_temperature', 'dkd_alpha', 'dkd_beta', 'pca_variance_threshold']


def seed_everything(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--out', default='svhn_vl2lite_mle_results')
    p.add_argument('--max-train', type=int, default=0)
    p.add_argument('--max-test', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--teacher', default='ViT-B-32')
    p.add_argument('--pretrained', default='laion2b_s34b_b79k')
    p.add_argument('--student', choices=list(BACKBONE_DIM), default='resnet18')
    p.add_argument('--no-pretrained-student', action='store_true')
    p.add_argument('--use-extra', action='store_true',
                   help='Include SVHN extra split in training')
    p.add_argument('--id-samples', type=int, default=5000)
    p.add_argument('--id-ks', type=int, nargs='+', default=[5, 10, 20, 50])
    p.add_argument('--id-repeats', type=int, default=5)
    p.add_argument('--dims', default='8,mle*0.5,mle*0.75,mle,pca,erank,mle*1.25,mle*1.5,32,128,full',
                   help="Tokens: integer, 'mle', 'mle*<factor>', 'pca', 'erank', 'full'.")
    p.add_argument('--modes', default='labels,feat,rkd,crd,dkd,vl2lite',
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
    p.add_argument('--full-identity', action='store_true')
    p.add_argument('--head-lr-mult', type=float, default=10.0)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--gpus', default='0')
    p.add_argument('--resume', action='store_true')
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
class SVHNView(Dataset):
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
    if train and augment:
        spatial = [transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.9, 1.1))]
        if AUG_HFLIP:
            spatial.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        spatial = [transforms.Resize((224, 224))]
    return transforms.Compose(spatial + [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class _ConcatLabelled(Dataset):
    """Concatenation that keeps a .labels array (torch ConcatDataset does not)."""
    def __init__(self, parts):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])
        self.labels = np.concatenate([np.asarray(p.labels) for p in parts])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        j = int(np.searchsorted(self.offsets, i, side='right') - 1)
        return self.parts[j][i - int(self.offsets[j])]


def load_svhn(root, download, use_extra=False):
    tr = datasets.SVHN(root=root, split='train', download=download)
    te = datasets.SVHN(root=root, split='test', download=download)
    if use_extra:
        tr = _ConcatLabelled([tr, datasets.SVHN(root=root, split='extra', download=download)])
    labels = np.asarray(tr.labels)
    assert labels.min() == 0 and labels.max() == NUM_CLASSES - 1, \
        f'Unexpected SVHN label range {labels.min()}..{labels.max()}'
    return tr, te


def labels_of(ds):
    return np.asarray(ds.labels)


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


def pca_dimension(embeddings, variance_threshold=0.95) -> int:
    """B1: smallest number of PCs explaining variance_threshold of total variance.
    Reference: Jolliffe (2002), Principal Component Analysis, 2nd ed., Springer."""
    x = embeddings.float().numpy().astype('float64')
    x -= x.mean(axis=0)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    explained = np.cumsum(s ** 2) / (s ** 2).sum()
    d = int(np.searchsorted(explained, variance_threshold) + 1)
    return max(2, d)


def effective_rank_dimension(embeddings) -> int:
    """B2: exp(Shannon entropy of normalised singular values).
    Reference: Roy & Vetterli (2007), EUSIPCO, pp. 606-610."""
    x = embeddings.float().numpy().astype('float64')
    x -= x.mean(axis=0)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    erank = int(round(float(np.exp(-np.sum(p * np.log(p))))))
    return max(2, erank)


def fixed_width_nearest(mle_d: int, full_d: int) -> int:
    """B3: nearest standard power-of-two width to mle_d.
    Reference: Romero et al., ICLR 2015 (FitNets)."""
    candidates = [w for w in FIXED_WIDTH_GRID if w <= full_d]
    if not candidates:
        return 8
    return min(candidates, key=lambda w: abs(w - mle_d))


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
class StudentMLE(nn.Module):
    def __init__(self, name, n_classes, d_b, pretrained=True, full_identity=False):
        super().__init__()
        self.backbone = build_backbone(name, pretrained)
        self.has_bottleneck_layer = d_b != BACKBONE_DIM[name] or not full_identity
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
def relational_kd(z, t_img):
    """D1/RKD: mean-normalised pairwise distance matching.
    Reference: Park et al., CVPR 2019."""
    zs = F.normalize(z.float(), dim=-1)
    ts = F.normalize(t_img.float(), dim=-1)
    off = ~torch.eye(zs.shape[0], dtype=torch.bool, device=zs.device)
    d_s, d_t = torch.cdist(zs, zs), torch.cdist(ts, ts)
    d_s = d_s / d_s[off].mean().clamp_min(1e-6)
    d_t = d_t / d_t[off].mean().clamp_min(1e-6)
    return F.smooth_l1_loss(d_s[off], d_t[off])


def crd_loss(z, t_img, temperature=0.07):
    """D2/CRD: symmetric contrastive loss between student and teacher embeddings.
    Reference: Tian et al., ICLR 2020."""
    z_n = F.normalize(z.float(), dim=-1)
    t_n = F.normalize(t_img.float(), dim=-1)
    N = z_n.shape[0]
    logits = z_n @ t_n.T / temperature          # (N, N)
    labels = torch.arange(N, device=z_n.device)
    loss_s2t = F.cross_entropy(logits, labels)
    loss_t2s = F.cross_entropy(logits.T, labels)
    return (loss_s2t + loss_t2s) / 2.0


def dkd_loss(logits, t_img, t_txt, temperature=4.0, alpha=1.0, beta=1.0):
    """D3/DKD: decoupled KD separating TCKD (target class) and NCKD (non-target).
    Reference: Zhao et al., CVPR 2022."""
    # Teacher soft labels from image-text similarity
    t_soft = F.softmax(
        (F.normalize(t_img.float(), dim=-1) @ F.normalize(t_txt.float(), dim=-1).T) / temperature,
        dim=1)
    s_soft = F.softmax(logits / temperature, dim=1)
    gt = t_soft.argmax(dim=1)

    # TCKD: binary distribution over correct class vs. rest
    s_t = s_soft.gather(1, gt.unsqueeze(1))
    t_t = t_soft.gather(1, gt.unsqueeze(1))
    tckd = F.binary_cross_entropy(s_t.clamp(1e-7, 1 - 1e-7), t_t.clamp(1e-7, 1 - 1e-7))

    # NCKD: distribution over non-target classes only
    mask = torch.ones_like(s_soft, dtype=torch.bool)
    mask.scatter_(1, gt.unsqueeze(1), False)
    s_nt = s_soft[mask].view(s_soft.shape[0], -1)
    t_nt = t_soft[mask].view(t_soft.shape[0], -1)
    # Renormalise to sum to 1 over non-target classes
    s_nt = s_nt / s_nt.sum(dim=1, keepdim=True).clamp_min(1e-7)
    t_nt = t_nt / t_nt.sum(dim=1, keepdim=True).clamp_min(1e-7)
    nckd = F.kl_div(s_nt.log().clamp_min(-100), t_nt, reduction='batchmean')

    return alpha * tckd + beta * nckd


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


def shift_consistency(student, x, z, max_shift=8):
    """Label-preserving small translation (digit flips change class identity)."""
    dy, dx = (int(v) for v in torch.randint(-max_shift, max_shift + 1, (2,)))
    z_aug, _ = student(torch.roll(x, shifts=(dy, dx), dims=(-2, -1)), return_bottleneck=True)
    return (F.normalize(z.float(), dim=-1)
            - F.normalize(z_aug.float(), dim=-1)).pow(2).sum(1).mean()


# ── Evaluation / benchmark ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, true = [], []
    for xs, ys, _ in loader:
        out = model(xs.to(device))
        if isinstance(out, tuple):
            out = out[1]
        pred.extend(out.argmax(1).cpu().tolist()); true.extend(ys.tolist())
    return {'accuracy_pct': 100 * accuracy_score(true, pred),
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


def _acc_at(sub, d):
    """Accuracy at a given d, or nan if d not in grid."""
    row = sub[sub['bottleneck_d'] == d]
    return float(row['test_accuracy_pct'].iloc[0]) if not row.empty else float('nan')


# ── Dimension parsing ──────────────────────────────────────────────────────────
def parse_dims(spec, mle_d, pca_d, erank_d, full_d):
    """B5: parse --dims tokens including 'pca' and 'erank'."""
    dims = []
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
    sigfile.write_text(run_signature)


# ── Training ───────────────────────────────────────────────────────────────────
def train_one(mode, d_b, seed, teacher_feats, teacher_txt, ds_train, ds_test,
              vlm_dim, device, a):
    seed_everything(seed)
    tag = f'{mode}_d{d_b}_seed{seed}'
    student = StudentMLE(a.student, NUM_CLASSES, d_b,
                         pretrained=not a.no_pretrained_student,
                         full_identity=a.full_identity).to(device)
    cond_text = Condense(vlm_dim, d_b).to(device)   # vl2lite only
    align = nn.Linear(d_b, vlm_dim).to(device)       # feat only

    backbone_params = list(student.backbone.parameters())
    head_params = list(student.bottleneck.parameters()) + list(student.classifier.parameters())
    if mode == 'vl2lite':
        head_params += list(cond_text.parameters())
    elif mode == 'feat':
        head_params += list(align.parameters())
    params = backbone_params + head_params

    groups = [{'params': backbone_params, 'lr': a.lr},
              {'params': head_params, 'lr': a.lr * a.head_lr_mult}]
    if a.optimizer == 'adamw':
        opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    else:
        opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * e / a.epochs)))

    train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, pin_memory=device.type == 'cuda',
                              persistent_workers=a.num_workers > 0)
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
                # D2: contrastive student-teacher matching
                l_crd = crd_loss(z, t_img, temperature=a.crd_temperature)
                loss = lam_cls * l_cls + (1.0 - lam_cls) * l_crd
                k1, k2 = l_crd.item(), 0.0

            elif mode == 'dkd':
                # D3: decoupled KD (TCKD + NCKD)
                l_dkd = dkd_loss(logits, t_img, t_txt,
                                 temperature=a.temperature,
                                 alpha=a.dkd_alpha,
                                 beta=a.dkd_beta)
                loss = lam_cls * l_cls + (1.0 - lam_cls) * l_dkd
                k1, k2 = l_dkd.item(), 0.0

            else:  # vl2lite
                l_vis = relational_kd(z, t_img)
                l_txt = linguistic_kd(z, t_img, t_txt, cond_text, a.kd_logit_scale_used, a.temperature)
                loss = lam_cls * l_cls + lam_kd * l_vis + lam_kd * l_txt
                k1, k2 = l_vis.item(), l_txt.item()

            if mode != 'labels' and a.w_consistency > 0:
                loss = loss + a.w_consistency * shift_consistency(student, xs, z)

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
    ce_last = curves[-1]['cls']
    ce_prev = curves[-2]['cls'] if len(curves) > 1 else float('nan')
    ce_drop = 100.0 * (ce_prev - ce_last) / ce_prev if ce_prev and np.isfinite(ce_prev) else float('nan')
    res = {'tag': tag, 'mode': mode, 'seed': seed, 'bottleneck_d': d_b,
           'final_train_ce': ce_last, 'last_epoch_ce_drop_pct': ce_drop,
           'has_bottleneck_layer': student.has_bottleneck_layer,
           'test_accuracy_pct': test['accuracy_pct'], 'test_macro_f1_pct': test['macro_f1_pct'],
           'parameters': sum(p.numel() for p in deployed.parameters()), 'train_seconds': train_s,
           'run_signature': a.run_signature}
    res.update({k: (bench[k] if bench else float('nan')) for k in BENCH_KEYS})
    return res, bench, curves


def _worker(gpu_id, configs, shared, q):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    a = shared['a']
    train_base, test_base = load_svhn(a.data_root, download=False, use_extra=a.use_extra)
    ds_train = SVHNView(train_base, shared['train_ids'],
                          student_transform(train=True, augment=not a.no_augment))
    ds_test = SVHNView(test_base, shared['test_ids'], student_transform())
    for mode, d, seed in configs:
        try:
            q.put(('ok',) + train_one(mode, d, seed, shared['teacher_feats'], shared['teacher_txt'],
                                      ds_train, ds_test, shared['vlm_dim'], device, a))
        except Exception:
            import traceback
            q.put(('err', f'{mode}_d{d}_seed{seed}', traceback.format_exc()))


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
    gpu_ids = [int(g) for g in a.gpus.split(',')]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    a.run_signature = hashlib.sha256(json.dumps(
        {k: getattr(a, k) for k in SIGNATURE_FIELDS}, sort_keys=True).encode()).hexdigest()[:16]
    check_resume_signature(out, a, a.run_signature)

    seed_everything(a.data_seed)
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    report_gpus(gpu_ids)

    train_base, test_base = load_svhn(a.data_root, download=True, use_extra=a.use_extra)
    print(f'SVHN: train={len(train_base)}, test={len(test_base)}')

    import open_clip
    teacher, _, teacher_tf = open_clip.create_model_and_transforms(
        a.teacher, pretrained=a.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(a.teacher)
    teacher.eval(); teacher.requires_grad_(False)

    train_ix = balanced_indices(labels_of(train_base), a.max_train, a.data_seed)
    test_ix = balanced_indices(labels_of(test_base), a.max_test, a.data_seed)
    ds_train = SVHNView(train_base, train_ix, student_transform(train=True, augment=not a.no_augment))
    ds_test = SVHNView(test_base, test_ix, student_transform())
    print(f'Train/test: {len(ds_train)}/{len(ds_test)}')

    def cache_for(ix, split):
        sig = hashlib.sha256(ix.tobytes() + f'{a.teacher}/{a.pretrained}/SVHN/extra={a.use_extra}/{split}'.encode()).hexdigest()[:16]
        return out / f'teacher_{split}_{a.teacher}_{a.pretrained}_{sig}.pt'.replace('/', '_')

    teacher_feats = build_teacher_cache(teacher, SVHNView(train_base, train_ix, teacher_tf),
                                        device, a.extract_batch_size, a.num_workers,
                                        cache_for(train_ix, 'train'), a.extract_save_every)
    teacher_test = build_teacher_cache(teacher, SVHNView(test_base, test_ix, teacher_tf),
                                       device, a.extract_batch_size, a.num_workers,
                                       cache_for(test_ix, 'test'), a.extract_save_every)
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
    (out / 'teacher_zero_shot.json').write_text(json.dumps(zs, indent=2))
    print(f"Teacher feats {tuple(teacher_feats.shape)}, CLIP scale {clip_scale:.1f}, "
          f"KD scale {a.kd_logit_scale_used:.1f}, zero-shot test {zs['test_zero_shot_pct']:.2f}%")

    # ── Dimensionality estimation (B1, B2, MLE) ───────────────────────────────
    id_table = mle_dimension(teacher_feats, a.id_ks, a.id_samples, a.data_seed)
    id_table.to_csv(out / 'levina_bickel_mle.csv', index=False)
    mle_float = float(id_table['mle_id'].median())
    mle_d = max(2, min(vlm_dim, math.ceil(mle_float)))
    stab = mle_stability(teacher_feats, a.id_ks, a.id_samples, a.data_seed, a.id_repeats)

    pca_d = pca_dimension(teacher_feats, a.pca_variance_threshold)          # B1
    pca_d = max(2, min(vlm_dim, pca_d))
    erank_d = effective_rank_dimension(teacher_feats)                         # B2
    erank_d = max(2, min(vlm_dim, erank_d))
    fixed_d = fixed_width_nearest(mle_d, BACKBONE_DIM[a.student])            # B3

    print(id_table.to_string(index=False))
    print(f"MLE d={mle_d} (stability {stab['mean']:.2f} +/- {stab['std']:.2f}) | "
          f"PCA d={pca_d} | Effective rank d={erank_d} | Fixed width d={fixed_d}")

    (out / 'dimension_estimates.json').write_text(json.dumps({           # B7
        'mle_d': mle_d, 'mle_float': mle_float, 'mle_stability': stab,
        'pca_d': pca_d, 'pca_variance_threshold': a.pca_variance_threshold,
        'erank_d': erank_d,
        'fixed_d': fixed_d, 'fixed_width_grid': FIXED_WIDTH_GRID,
        'vlm_dim': vlm_dim,
    }, indent=2))
    (out / 'mle_summary.json').write_text(json.dumps(
        {'mle_by_k': id_table.to_dict('records'), 'mle_median': mle_float, 'mle_d': mle_d,
         'stability': stab, 'num_classes_minus_1': NUM_CLASSES - 1}, indent=2))

    del teacher
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # B5: parse dims with pca and erank tokens
    dims = parse_dims(a.dims, mle_d, pca_d, erank_d, BACKBONE_DIM[a.student])
    configs = [(m, d, s) for s in seeds for d in dims for m in modes]
    print(f'Dims {dims} | modes {modes} | seeds {seeds} | {len(configs)} runs on GPUs {gpu_ids}')
    print(f'Bottleneck strategies: MLE={mle_d}, PCA={pca_d}, Erank={erank_d}, Fixed={fixed_d}, '
          f'Oracle=best on grid (upper bound)')

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
        procs = [ctx.Process(target=_worker, args=(g, b, shared, q))
                 for g, b in zip(gpu_ids, buckets)]
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
        test_accuracy_std=('test_accuracy_pct', 'std'),
        test_macro_f1_pct=('test_macro_f1_pct', 'mean'),
        latency_ms=('latency_ms', 'mean'), latency_iqr_ms=('latency_iqr_ms', 'mean'),
        energy_mJ=('energy_mJ', 'mean'), energy_iqr_mJ=('energy_iqr_mJ', 'mean'),
        parameters=('parameters', 'first'), train_seconds=('train_seconds', 'mean'),
        final_train_ce=('final_train_ce', 'mean'),
        last_epoch_ce_drop_pct=('last_epoch_ce_drop_pct', 'mean'),
        n_seeds=('seed', 'count')).reset_index()

    summary['mle_selected'] = summary['bottleneck_d'].eq(mle_d)
    summary['pca_selected'] = summary['bottleneck_d'].eq(pca_d)
    summary['erank_selected'] = summary['bottleneck_d'].eq(erank_d)
    summary['fixed_selected'] = summary['bottleneck_d'].eq(fixed_d)

    summary['converged'] = summary['last_epoch_ce_drop_pct'] < 2.0
    if (~summary['converged']).any():
        bad = summary.loc[~summary['converged'], ['mode', 'bottleneck_d', 'last_epoch_ce_drop_pct']]
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
        pfreq.get((str(r['mode']), int(r['bottleneck_d'])), float('nan'))
        if r['n_seeds'] == expected else float('nan')
        for _, r in summary.iterrows()]
    summary.to_csv(out / 'pareto_summary.csv', index=False)

    # ── Selection analysis: all 5 strategies per mode (B4, B6) ───────────────
    selection = {}
    for m in modes:
        sub = summary[summary['mode'] == m]
        if sub.empty:
            continue
        best = float(sub['test_accuracy_pct'].max())
        gains, knee = knee_analysis(sub, a.knee_gain)
        d_eps = d_eps_at(sub, a.accuracy_tolerance)

        # B4: oracle = best d on the evaluated grid (upper bound)
        oracle_d = int(sub.loc[sub['test_accuracy_pct'].idxmax(), 'bottleneck_d'])
        oracle_acc = best

        acc_mle = _acc_at(sub, mle_d)
        acc_pca = _acc_at(sub, pca_d)
        acc_erank = _acc_at(sub, erank_d)
        acc_fixed = _acc_at(sub, fixed_d)

        selection[m] = {
            # ── Strategy 1: Fixed width (baseline, B3) ─────────────────────
            'fixed_d': fixed_d,
            'fixed_accuracy_pct': acc_fixed,
            'fixed_gap_pp': oracle_acc - acc_fixed,
            # ── Strategy 2: MLE-guided (proposed) ──────────────────────────
            'mle_d': mle_d,
            'mle_accuracy_pct': acc_mle,
            'mle_gap_pp': oracle_acc - acc_mle,
            'mle_equals_d_eps': mle_d == d_eps,
            # ── Strategy 3: PCA-based (B1) ──────────────────────────────────
            'pca_d': pca_d,
            'pca_accuracy_pct': acc_pca,
            'pca_gap_pp': oracle_acc - acc_pca,
            # ── Strategy 4: Effective-rank-based (B2) ──────────────────────
            'erank_d': erank_d,
            'erank_accuracy_pct': acc_erank,
            'erank_gap_pp': oracle_acc - acc_erank,
            # ── Strategy 5: Validation-grid oracle (upper bound, B4) ────────
            'oracle_d': oracle_d,
            'oracle_accuracy_pct': oracle_acc,
            # ── Shared analysis ─────────────────────────────────────────────
            'best_accuracy_pct': best,
            'd_eps': d_eps,
            'd_eps_by_tolerance': {str(t): d_eps_at(sub, t) for t in (0.5, 1.0, 2.0)},
            'knee_d': knee, 'knee_gain_threshold_pp': a.knee_gain,
            'gain_per_doubling': gains,
        }

    (out / 'selection.json').write_text(json.dumps({
        'dataset': 'SVHN', 'use_extra': a.use_extra,
        'num_classes_minus_1': NUM_CLASSES - 1,
        'dimension_estimates': {
            'mle_d': mle_d, 'pca_d': pca_d, 'erank_d': erank_d, 'fixed_d': fixed_d,
            'vlm_dim': vlm_dim,
        },
        'mle_stability': stab,
        'teacher_zero_shot': zs,
        'run_signature': a.run_signature,
        'accuracy_tolerance_pp': a.accuracy_tolerance,
        'modes': selection,
        'strategy_references': {
            'fixed': 'Romero et al., FitNets, ICLR 2015',
            'mle': 'Levina & Bickel, NIPS 2004 (proposed)',
            'pca': 'Jolliffe, Principal Component Analysis, Springer 2002',
            'erank': 'Roy & Vetterli, EUSIPCO 2007',
            'oracle': 'Unreachable upper bound — uses test labels',
        },
        'distillation_references': {
            'labels': 'Cross-entropy only (baseline)',
            'feat': 'Feature KD: MSE + cosine alignment',
            'rkd': 'Park et al., RKD, CVPR 2019',
            'crd': 'Tian et al., CRD, ICLR 2020',
            'dkd': 'Zhao et al., DKD, CVPR 2022',
            'vl2lite': 'VL2Lite-style: relational + linguistic KD',
        },
        'note': ('d_eps and knee_d computed on test set — post-hoc analysis only. '
                 'Pareto claims cover only the evaluated grid. '
                 'Oracle is an upper bound; it uses test accuracy and is not deployable.'),
    }, indent=2))

    (out / 'config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2))

    print(summary.to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    run()
