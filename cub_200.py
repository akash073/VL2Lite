"""VL2Lite + Levina-Bickel MLE bottleneck — CUB-200-2011 edition.

RQ1: Does the Levina-Bickel MLE estimate of CLIP vision-embedding intrinsic
dimensionality predict the smallest sufficient student bottleneck d_eps for
VLM->CNN distillation?

Fixes in v3 (see FIX tags in code)
  FIX-1  L_vis collapse: relational KD target is now the FROZEN teacher geometry
         (RKD-style, mean-normalised distances). Previously the target came from
         a trainable Condense MLP, so student and Condense could both collapse to
         constants and drive L_vis to zero -> VL2Lite d=512 fell to 72.9%.
  FIX-2  L_txt had no signal: CLIP cosines (~0.2-0.3) divided by T=2 give a
         near-uniform teacher distribution (0.105 on the true class). Cosines
         are now multiplied by CLIP's logit scale (~100) before /T, as in CLIP.
  FIX-3  KL direction comment corrected: F.kl_div(log q_s, p_t) = KL(p_t || q_s),
         the standard Hinton KD direction.
  FIX-4  nondominated_freq_per_seed was NaN: summary.apply used r.mode, which is
         pandas' Series.mode() method, not the column.
  FIX-5  RQ1 needs a dimension GRID to locate the accuracy elbow; two points
         (mle_d, 512) cannot. --dims now defaults to mle,8,16,32,64,128,256,full.
  FIX-6  Added a 'feat' mode (CE + MSE + cosine feature KD, SAM3-LiteText-style)
         as a simple distillation control, so MLE conclusions do not depend on
         VL2Lite alone.
  FIX-7  Selection uses epsilon-sufficient dimension d_eps (Eq. in methodology),
         not "is mle_d the single best dim".
  FIX-8  NVML handle and CUDA device now follow the run's actual GPU.
  FIX-9  Pareto reported both across all configs and within each mode.

Smoke:  python vl2lite_mle_pareto.py --max-train 2000 --max-test 500 --epochs 2  --seeds 42 --dims mle,full --benchmark-repeats 30 --id-samples 1000 --gpus 0
Full:   python cub_200.py --epochs 10 --seeds 42,43,44 --gpus 0
"""

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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

NUM_CLASSES = 200
BACKBONE_DIM = {'resnet18': 512, 'mobilenet_v2': 1280}


def seed_everything(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='./data/CUB_200_2011',
                   help='Folder containing images.txt, classes.txt and train_test_split.txt')
    p.add_argument('--out', default='cub200_vl2lite_mle_results')
    p.add_argument('--max-train', type=int, default=0, help='Optional class-balanced subsampling of official train')
    p.add_argument('--max-test', type=int, default=0, help='Optional subsampling of official test for smoke runs only')
    p.add_argument('--teacher', default='ViT-B-32')
    p.add_argument('--pretrained', default='laion2b_s34b_b79k')
    p.add_argument('--student', choices=list(BACKBONE_DIM), default='resnet18')
    p.add_argument('--no-pretrained-student', action='store_true')
    p.add_argument('--id-samples', type=int, default=5000)
    p.add_argument('--id-ks', type=int, nargs='+', default=[5, 10, 20, 50])
    # Dimension ablation and training controls
    p.add_argument('--dims', default='mle,8,16,32,64,128,256,full',
                   help="Bottleneck grid. 'mle' = MLE estimate, 'full' = backbone dim.")
    p.add_argument('--modes', default='labels,feat,vl2lite',
                   help='labels = CE only; feat = CE+MSE+cos feature KD; vl2lite = VL2Lite')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--gpus', default='0')
    # VL2Lite
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lam-cls-start', type=float, default=0.01)
    p.add_argument('--lam-cls-end', type=float, default=0.80)
    # feat mode
    p.add_argument('--w-mse', type=float, default=1.0)
    p.add_argument('--w-cos', type=float, default=2.0)
    # Optional flip consistency (both KD modes). 0 keeps the comparison clean.
    p.add_argument('--w-consistency', type=float, default=0.0)
    # Analysis / benchmark
    p.add_argument('--accuracy-tolerance', type=float, default=1.0)
    p.add_argument('--benchmark-repeats', type=int, default=150)
    p.add_argument('--benchmark-trials', type=int, default=5)
    return p


# ── CUB-200-2011 official annotated split ─────────────────────────────────────
def resolve_cub_root(root):
    root = Path(root).expanduser()
    candidates = [root, root / 'CUB_200_2011', root / 'cub2002011' / 'CUB_200_2011',
                  root / 'CUB_200_2011' / 'CUB_200_2011']
    for c in candidates:
        if all((c / f).is_file() for f in
               ('images.txt', 'image_class_labels.txt', 'train_test_split.txt', 'classes.txt')) \
               and (c / 'images').is_dir():
            return c.resolve()
    raise FileNotFoundError('Could not find CUB metadata files at: ' + str(root) +
                            '. Unzip Kaggle archive and set --data-root to CUB_200_2011.')


def read_id_mapping(path, value_parser=str):
    output = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        i, value = line.split(maxsplit=1)
        output[int(i)] = value_parser(value)
    return output


class CUB200Base(Dataset):
    """Image paths, official train/test flags, labels 0..199, class names."""
    def __init__(self, root, train):
        root = resolve_cub_root(root)
        relpaths = read_id_mapping(root / 'images.txt')
        labels = read_id_mapping(root / 'image_class_labels.txt', int)
        split = read_id_mapping(root / 'train_test_split.txt', int)
        class_mapping = read_id_mapping(root / 'classes.txt')
        if len(class_mapping) != 200:
            raise ValueError(f'Expected 200 CUB classes, found {len(class_mapping)}')
        self.classes = [class_mapping[c].split('.', 1)[-1].replace('_', ' ')
                        for c in range(1, 201)]
        ids = sorted(i for i in relpaths if split[i] == int(train))
        self.files = [root / 'images' / relpaths[i] for i in ids]
        self.targets = [labels[i] - 1 for i in ids]
        if not self.files or any(not p.is_file() for p in self.files):
            raise RuntimeError('CUB split empty or image files missing; check extraction.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as im:
            image = im.convert('RGB')
        return image, self.targets[index]


# ── Data ──────────────────────────────────────────────────────────────────────
class CUB200Paired(Dataset):
    """Returns (student_view, teacher_view, label, local_index)."""
    def __init__(self, base, indices, student_tf, teacher_tf):
        self.base, self.indices = base, np.asarray(indices, dtype=np.int64)
        self.student_tf, self.teacher_tf = student_tf, teacher_tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[int(self.indices[i])]
        rgb = img.convert('RGB') if hasattr(img, 'convert') else img
        return self.student_tf(rgb), self.teacher_tf(rgb), label, i


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


def student_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# ── MLE ───────────────────────────────────────────────────────────────────────
def mle_dimension(embeddings: torch.Tensor, ks, n: int, seed: int) -> pd.DataFrame:
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


@torch.no_grad()
def build_teacher_cache(teacher, dataset, device, batch_size, workers, cache_path):
    if cache_path.exists():
        obj = torch.load(cache_path, map_location='cpu', weights_only=True)
        if isinstance(obj, dict) and obj.get('features') is not None \
                and obj['features'].shape[0] == len(dataset):
            print(f'Loaded teacher cache: {cache_path.name}')
            return obj['features']
        print('Cache mismatch — rebuilding.')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    chunks = []
    teacher.eval()
    for _, timgs, _, _ in loader:
        timgs = timgs.to(device)
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda'):
                f = teacher.encode_image(timgs)
        else:
            f = teacher.encode_image(timgs)
        chunks.append(f.float().cpu())
    feats = torch.cat(chunks)
    torch.save({'features': feats}, cache_path)
    return feats


# ── Models ────────────────────────────────────────────────────────────────────
class StudentMLE(nn.Module):
    """backbone -> Linear(backbone_dim, d_b) -> Linear(d_b, C)."""
    def __init__(self, name, n_classes, d_b, pretrained=True):
        super().__init__()
        if name == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            net.fc = nn.Identity()
        else:
            net = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            net.classifier = nn.Identity()
        self.backbone = net
        self.bottleneck = nn.Linear(BACKBONE_DIM[name], d_b)
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
    """Two-layer MLP vlm_dim -> d_b (used for text prototypes only, see FIX-1)."""
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
    """FIX-1: RKD-style distance loss against FROZEN teacher geometry.

    Distances are computed on L2-normalised features and divided by their mean
    off-diagonal distance, making the loss scale-invariant. Because the target
    is the frozen teacher (no trainable projection), there is no trivial
    collapsed solution. Distance matrices are BxB, so no dimension matching
    between d_b and vlm_dim is needed.
    """
    zs = F.normalize(z.float(), dim=-1)
    ts = F.normalize(t_img.float(), dim=-1)
    off = ~torch.eye(zs.shape[0], dtype=torch.bool, device=zs.device)
    d_s, d_t = torch.cdist(zs, zs), torch.cdist(ts, ts)
    d_s = d_s / d_s[off].mean().clamp_min(1e-6)
    d_t = d_t / d_t[off].mean().clamp_min(1e-6)
    return F.smooth_l1_loss(d_s[off], d_t[off])


def linguistic_kd(z, t_img, t_txt, cond_text, logit_scale, temp):
    """FIX-2 / FIX-3: CLIP-scaled image-text logits, KL(p_teacher || q_student)."""
    protos = F.normalize(cond_text(t_txt.float()), dim=-1)                     # (C, d_b)
    s_lw = logit_scale * F.normalize(z.float(), dim=-1) @ protos.T               # (B, C)
    s_vlm = logit_scale * (F.normalize(t_img.float(), dim=-1)
                           @ F.normalize(t_txt.float(), dim=-1).T)               # (B, C)
    # F.kl_div(input=log q, target=p) computes KL(p || q): teacher -> student.
    return (temp ** 2) * F.kl_div(F.log_softmax(s_lw / temp, dim=1),
                                  F.softmax(s_vlm / temp, dim=1), reduction='batchmean')


def feature_kd(z, t_img, align):
    """FIX-6: simple feature imitation through a train-only alignment head."""
    p = F.normalize(align(z).float(), dim=-1)
    t = F.normalize(t_img.float(), dim=-1)
    return (p - t).pow(2).sum(1).mean(), (1 - (p * t).sum(1)).mean()


def flip_consistency(student, x, z):
    z_flip, _ = student(torch.flip(x, dims=[-1]), return_bottleneck=True)
    return (F.normalize(z.float(), dim=-1)
            - F.normalize(z_flip.float(), dim=-1)).pow(2).sum(1).mean()


# ── Evaluation / benchmark ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, true = [], []
    for xs, _, ys, _ in loader:
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
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)            # FIX-8
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
            if e0 is not None and e1 is not None and e1 > e0:
                eng.append((e1 - e0) / a.benchmark_repeats)
    iqr = lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))
    return {'latency_ms': float(np.median(lat)), 'latency_iqr_ms': iqr(lat),
            'energy_mJ': float(np.median(eng)) if eng else float('nan'),
            'energy_iqr_mJ': iqr(eng) if eng else float('nan'),
            'latency_trials': lat, 'energy_trials': eng}


# ── Pareto ────────────────────────────────────────────────────────────────────
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
    """Fraction of seeds on which each (mode, d) is nondominated. Uses r['mode']."""
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


# ── Training ──────────────────────────────────────────────────────────────────
def train_one(mode, d_b, seed, teacher_feats, teacher_txt, ds_train, ds_test,
              vlm_dim, device, a):
    seed_everything(seed)
    tag = f'{mode}_d{d_b}_seed{seed}'
    student = StudentMLE(a.student, NUM_CLASSES, d_b,
                         pretrained=not a.no_pretrained_student).to(device)
    cond_text = Condense(vlm_dim, d_b).to(device)          # vl2lite only
    align = nn.Linear(d_b, vlm_dim).to(device)             # feat only (train-only head)

    params = list(student.parameters())
    if mode == 'vl2lite':
        params += list(cond_text.parameters())
    elif mode == 'feat':
        params += list(align.parameters())

    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr / 100)
    train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(ds_test, batch_size=a.batch_size, shuffle=False,
                             num_workers=a.num_workers)
    t_txt = teacher_txt.to(device)
    curves, t_start = [], time.perf_counter()

    for epoch in range(a.epochs):
        student.train(); cond_text.train(); align.train()
        frac = epoch / max(a.epochs - 1, 1)
        lam_cls = a.lam_cls_start + (a.lam_cls_end - a.lam_cls_start) * frac
        lam_kd = (1.0 - lam_cls) / 2.0
        sums = dict(loss=0.0, cls=0.0, kd1=0.0, kd2=0.0)

        for xs, _, ys, ix in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            t_img = teacher_feats[ix].to(device)
            z, logits = student(xs, return_bottleneck=True)
            l_cls = F.cross_entropy(logits, ys)

            if mode == 'labels':
                loss, k1, k2 = l_cls, 0.0, 0.0
            elif mode == 'feat':
                mse, cos = feature_kd(z, t_img, align)
                loss = l_cls + a.w_mse * mse + a.w_cos * cos
                k1, k2 = mse.item(), cos.item()
            else:  # vl2lite
                l_vis = relational_kd(z, t_img)
                l_txt = linguistic_kd(z, t_img, t_txt, cond_text, a.logit_scale, a.temperature)
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
        test = evaluate(student, test_loader, device)
        curves.append({'tag': tag, 'mode': mode, 'bottleneck_d': d_b, 'seed': seed,
                       'epoch': epoch + 1, 'lam_cls': lam_cls if mode == 'vl2lite' else 1.0,
                       **{k: v / n for k, v in sums.items()},
                       'test_accuracy_pct': test['accuracy_pct']})
        print(f"[{tag}] ep{epoch + 1:02d}/{a.epochs} loss={sums['loss']/n:.4f} "
              f"cls={sums['cls']/n:.4f} kd1={sums['kd1']/n:.4f} kd2={sums['kd2']/n:.4f} "
              f"test_acc={test['accuracy_pct']:.2f}", flush=True)

    train_s = time.perf_counter() - t_start
    deployed = Deployable(student).to(device)
    bench = benchmark(deployed, device, a)
    out_dir = Path(a.out)
    torch.save({'student_deployed': deployed.cpu().state_dict(), 'architecture': a.student,
                'bottleneck_d': d_b, 'mode': mode, 'seed': seed}, out_dir / f'{tag}.pt')
    pd.DataFrame(curves).to_csv(out_dir / f'{tag}_history.csv', index=False)
    return ({'tag': tag, 'mode': mode, 'seed': seed, 'bottleneck_d': d_b,
             'test_accuracy_pct': test['accuracy_pct'], 'test_macro_f1_pct': test['macro_f1_pct'],
             'latency_ms': bench['latency_ms'], 'latency_iqr_ms': bench['latency_iqr_ms'],
             'energy_mJ': bench['energy_mJ'], 'energy_iqr_mJ': bench['energy_iqr_mJ'],
             'parameters': sum(p.numel() for p in deployed.parameters()),
             'train_seconds': train_s}, bench, curves)


def _worker(gpu_id, configs, shared, q):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    a = shared['a']
    import open_clip
    _, _, teacher_tf = open_clip.create_model_and_transforms(a.teacher, pretrained=a.pretrained,
                                                             device='cpu')
    train_base = CUB200Base(a.data_root, train=True)
    test_base = CUB200Base(a.data_root, train=False)
    ds_train = CUB200Paired(train_base, shared['train_ids'], student_transform(), teacher_tf)
    ds_test = CUB200Paired(test_base, shared['test_ids'], student_transform(), teacher_tf)
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
    gpu_ids = [int(g) for g in a.gpus.split(',')]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seed_everything(seeds[0])

    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)                                   # FIX-8

    a.data_root = str(resolve_cub_root(a.data_root))
    train_base = CUB200Base(a.data_root, train=True)
    test_base = CUB200Base(a.data_root, train=False)
    assert train_base.classes == test_base.classes
    class_names = train_base.classes
    print(f'CUB: {len(class_names)} species, train={len(train_base)}, test={len(test_base)}')

    import open_clip
    teacher, _, teacher_tf = open_clip.create_model_and_transforms(
        a.teacher, pretrained=a.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(a.teacher)
    teacher.eval(); teacher.requires_grad_(False)

    train_ix = balanced_indices(train_base.targets, a.max_train, seeds[0])
    test_ix = balanced_indices(test_base.targets, a.max_test, seeds[0])
    ds_train = CUB200Paired(train_base, train_ix, student_transform(), teacher_tf)
    ds_test = CUB200Paired(test_base, test_ix, student_transform(), teacher_tf)
    print(f'Train/test: {len(ds_train)}/{len(ds_test)}')

    sig = hashlib.sha256(train_ix.tobytes() + f'{a.teacher}/{a.pretrained}/{a.data_root}/CUB'.encode()).hexdigest()[:16]
    cache = out / f'teacher_{a.teacher}_{a.pretrained}_{sig}.pt'.replace('/', '_')
    teacher_feats = build_teacher_cache(teacher, ds_train, device, a.batch_size,
                                        a.num_workers, cache)
    with torch.no_grad():
        teacher_txt = teacher.encode_text(
            tokenizer([f'A photo of a {c} bird' for c in class_names]).to(device)).float().cpu()
        a.logit_scale = float(teacher.logit_scale.exp().item())       # FIX-2
    assert teacher_txt.shape[0] == NUM_CLASSES, f'Expected {NUM_CLASSES} class prototypes'
    vlm_dim = teacher_feats.shape[1]
    print(f'Teacher feats {tuple(teacher_feats.shape)}, logit scale {a.logit_scale:.1f}')

    id_table = mle_dimension(teacher_feats, a.id_ks, a.id_samples, seeds[0])
    id_table.to_csv(out / 'levina_bickel_mle.csv', index=False)
    mle_float = float(id_table['mle_id'].median())
    mle_d = max(2, min(vlm_dim, math.ceil(mle_float)))
    print(id_table.to_string(index=False)); print(f'MLE dimension: {mle_d}')
    (out / 'mle_summary.json').write_text(json.dumps(
        {'mle_by_k': id_table.to_dict('records'), 'mle_median': mle_float, 'mle_d': mle_d}, indent=2))
    del teacher
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Dimension grid (test-derived sufficient dimension is exploratory only)
    dims = []
    for tok in a.dims.split(','):
        tok = tok.strip().lower()
        d = mle_d if tok == 'mle' else BACKBONE_DIM[a.student] if tok == 'full' else int(tok)
        if d not in dims:
            dims.append(d)
    if any(d < 2 or d > BACKBONE_DIM[a.student] for d in dims):
        raise ValueError(f'Dimensions must be within [2, {BACKBONE_DIM[a.student]}]')
    dims.sort()
    configs = [(m, d, s) for s in seeds for d in dims for m in modes]
    print(f'Dims {dims} | modes {modes} | seeds {seeds} | {len(configs)} runs on GPUs {gpu_ids}')

    results, curves_all, bench_rows = [], [], []

    def collect(res, bench, curves):
        results.append(res); curves_all.extend(curves)
        for i, lt in enumerate(bench['latency_trials']):
            bench_rows.append({'tag': res['tag'], 'trial': i + 1, 'latency_ms': lt,
                               'energy_mJ': bench['energy_trials'][i]
                               if i < len(bench['energy_trials']) else float('nan')})
        pd.DataFrame(results).to_csv(out / 'results_per_seed.csv', index=False)

    if len(gpu_ids) == 1 or device.type != 'cuda':
        for m, d, s in configs:
            collect(*train_one(m, d, s, teacher_feats, teacher_txt, ds_train, ds_test,
                               vlm_dim, device, a))
    else:
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

    pd.DataFrame(curves_all).to_csv(out / 'training_history.csv', index=False)
    pd.DataFrame(bench_rows).to_csv(out / 'benchmark_trials.csv', index=False)

    frame = pd.DataFrame(results)
    summary = frame.groupby(['mode', 'bottleneck_d']).agg(
        test_accuracy_pct=('test_accuracy_pct', 'mean'),
        test_accuracy_std=('test_accuracy_pct', 'std'),
        test_macro_f1_pct=('test_macro_f1_pct', 'mean'),
        latency_ms=('latency_ms', 'mean'), latency_iqr_ms=('latency_iqr_ms', 'mean'),
        energy_mJ=('energy_mJ', 'mean'), energy_iqr_mJ=('energy_iqr_mJ', 'mean'),
        parameters=('parameters', 'first'), train_seconds=('train_seconds', 'mean'),
        n_seeds=('seed', 'count')).reset_index()
    summary['test_accuracy_std'] = summary['test_accuracy_std'].fillna(0.0)
    summary['mle_selected'] = summary['bottleneck_d'].eq(mle_d)

    # Test-set descriptive Pareto across all configs AND within each mode
    summary['pareto_all'] = pareto_mask(summary)
    summary['pareto_within_mode'] = ''
    for m in modes:
        idx = summary.index[summary['mode'] == m]
        summary.loc[idx, 'pareto_within_mode'] = pareto_mask(summary.loc[idx].reset_index(drop=True))
    pfreq = pareto_freq_per_seed(frame)
    summary['nondominated_freq_per_seed'] = [                          # FIX-4
        pfreq.get((str(r['mode']), int(r['bottleneck_d'])), float('nan'))
        for _, r in summary.iterrows()]
    summary.to_csv(out / 'pareto_summary.csv', index=False)

    # Exploratory, test-set post-hoc epsilon-sufficient dimension per mode
    selection = {}
    for m in modes:
        sub = summary[summary['mode'] == m]
        if sub.empty:
            continue
        best = float(sub['test_accuracy_pct'].max())
        d_eps = int(sub[sub['test_accuracy_pct'] >= best - a.accuracy_tolerance]['bottleneck_d'].min())
        has_mle = (sub['bottleneck_d'] == mle_d).any()
        acc_mle = float(sub.loc[sub['bottleneck_d'] == mle_d, 'test_accuracy_pct'].iloc[0]) \
            if has_mle else float('nan')
        selection[m] = {'best_accuracy_pct': best, 'd_eps': d_eps, 'mle_d': mle_d,
                        'mle_accuracy_pct': acc_mle, 'mle_gap_pp': best - acc_mle,
                        'mle_within_tolerance': bool(best - acc_mle <= a.accuracy_tolerance),
                        'mle_equals_d_eps': mle_d == d_eps}
    (out / 'selection.json').write_text(json.dumps(
        {'accuracy_tolerance_pp': a.accuracy_tolerance, 'modes': selection,
         'note': 'd_eps is computed on the test set (no validation split); report it as '
                 'post-hoc analysis, not model selection. Pareto claims cover only the '
                 'evaluated grid.'}, indent=2))
    (out / 'class_names.json').write_text(json.dumps(class_names, indent=2))
    (out / 'config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    run()
