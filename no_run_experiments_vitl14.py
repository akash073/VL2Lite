"""Run every (dataset x student model) experiment with the CLIP ViT-L-14 teacher.

Edit the SETTINGS section below, then run:
    python run_experiments_vitl14.py            # run everything
    python run_experiments_vitl14.py --dry-run  # only print what would run

Before each run it prints the script name, dataset, student model, epochs, output
folder and the full command. Each run's console output is also saved to
<output folder>/run_log.txt. A run that fails does not stop the others; a summary
table is printed at the end and saved to <RESULTS_ROOT>/runs_summary.csv.

Re-running is safe: finished experiments are skipped, and unfinished ones (or ones
needing extra seeds/modes) continue with --resume.

Teacher feature cache: the cached CLIP features depend only on the teacher and the
dataset, not on the student. To avoid re-extracting them for every student, this
runner copies the cache files from an earlier folder of the same dataset into each
new output folder before training starts.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ══════════════════════════════ SETTINGS ══════════════════════════════════════
# Datasets to run, in order. Comment out lines you do not want.
# name: (training script, extra flags for that dataset)
DATASETS = {

    'fmnist':                ('fmnist_vl2lite_mle_pareto_v8.py',       []),
   # 'svhn':                  ('svhn_vl2lite_mle_pareto_v8.py',         []),
    #'cifar10':               ('cifar10_vl2lite_mle_pareto_v8.py',      []),
    #'cifar100':              ('cifar100_vl2lite_mle_pareto_v8.py',     []),
    #'tinyimagenet':          ('tinyimagenet_vl2lite_mle_pareto_v8.py', []),
}

# Student models to run on every dataset. Options: resnet18, resnet50, mobilenet_v2,
# mobilenet_v3_large, efficientnet_b0, convnext_tiny
STUDENTS = ['resnet18', 'mobilenet_v2', 'mobilenet_v3_large', 'efficientnet_b0', 'convnext_tiny', 'resnet50']

# Reverse
#STUDENTS = ['resnet50', 'convnext_tiny', 'efficientnet_b0', 'mobilenet_v3_large', 'mobilenet_v2', 'resnet18']

# Teacher (open_clip name and pretrained tag)
TEACHER = 'ViT-L-14'
PRETRAINED = 'laion2b_s32b_b82k'
EXTRACT_BATCH_SIZE = 32              # lower (16 or 8) if teacher extraction runs out of memory

# Epochs
EPOCHS_DEFAULT = 20

SEEDS = '42,43,44'
MODES = 'labels,feat,vl2lite'       # e.g. 'labels' for a quicker first pass
GPUS = '0'                           # e.g. '0,1' for two GPUs
EXTRA_FLAGS = ['--teacher', TEACHER, '--pretrained', PRETRAINED,
               '--extract-batch-size', str(EXTRACT_BATCH_SIZE)]   # added to EVERY run
DATA_ROOT = './data'
RESULTS_ROOT = 'results_vitl14'      # folders: results_vitl14/<dataset>_<student>_e<epochs>
SKIP_FINISHED = True                 # skip runs that already have a complete summary
SHARE_TEACHER_CACHE = True           # reuse teacher feature caches across students
# ══════════════════════════════════════════════════════════════════════════════


def out_folder(dataset, student, epochs):
    return Path(RESULTS_ROOT) / f'{dataset}_{student}_e{epochs}'


def share_teacher_cache(dataset, out):
    """Copy finished teacher feature caches from another folder of the same dataset.
    Partial caches (*.partial.pt) are never copied."""
    if not SHARE_TEACHER_CACHE:
        return
    root = Path(RESULTS_ROOT)
    for folder in sorted(root.glob(f'{dataset}_*')):
        if folder.resolve() == out.resolve() or not folder.is_dir():
            continue
        for f in folder.glob('teacher_*.pt'):
            if f.name.endswith('.partial.pt'):
                continue
            dest = out / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                print(f'  reused teacher cache: {f.name} (from {folder.name})', flush=True)


def run_state(out):
    """'new', 'done' or 'resume' for an output folder, given the requested seeds and modes."""
    summary, per_seed = out / 'pareto_summary.csv', out / 'results_per_seed.csv'
    if summary.exists():
        with open(summary, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        want_seeds = len([s for s in SEEDS.split(',') if s.strip()])
        want_modes = {m.strip() for m in MODES.split(',') if m.strip()}
        have_modes = {r['mode'] for r in rows}
        enough = rows and min(int(float(r['n_seeds'])) for r in rows) >= want_seeds
        if enough and want_modes <= have_modes:
            return 'done'
        return 'resume'
    return 'resume' if per_seed.exists() else 'new'


def build_command(dataset, student, epochs, out, resume):
    script, dataset_flags = DATASETS[dataset]
    cmd = [sys.executable, '-u', script,
           '--student', student, '--epochs', str(epochs),
           '--seeds', SEEDS, '--modes', MODES, '--gpus', GPUS,
           '--data-root', DATA_ROOT,
           '--out', str(out)] + list(dataset_flags) + list(EXTRA_FLAGS)
    if resume:
        cmd.append('--resume')
    return cmd


def run_and_log(cmd, log_path):
    """Run a command, showing its output live and saving it to log_path."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n{' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', env=env)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
        return proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true', help='Print the runs without executing them')
    a = p.parse_args()

    missing = sorted({s for s, _ in DATASETS.values() if not Path(s).exists()})
    if missing and not a.dry_run:
        sys.exit(f'Training script(s) not found in {Path.cwd()}: {missing}')

    jobs = [(d, s) for d in DATASETS for s in STUDENTS]
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    print(f'{len(jobs)} experiments: {len(DATASETS)} dataset(s) x {len(STUDENTS)} student model(s)')
    print(f'Teacher {TEACHER} / {PRETRAINED} | seeds {SEEDS} | modes {MODES} | GPUs {GPUS}')
    print(f'Results in {Path(RESULTS_ROOT).resolve()}')

    summary, t_all = [], time.time()
    for i, (dataset, student) in enumerate(jobs, 1):
        epochs = EPOCHS_DEFAULT
        out = out_folder(dataset, student, epochs)
        state = run_state(out)
        script = DATASETS[dataset][0]
        cmd = build_command(dataset, student, epochs, out, resume=(state == 'resume'))

        print('\n' + '=' * 78)
        print(f'[{i}/{len(jobs)}]  Script  : {script}')
        print(f'         Dataset : {dataset}')
        print(f'         Student : {student}')
        print(f'         Teacher : {TEACHER} / {PRETRAINED}')
        print(f'         Epochs  : {epochs}')
        print(f'         Output  : {out}')
        print(f"         Status  : {'already finished' if state == 'done' else state}")
        print(f"         Command : {' '.join(cmd)}")
        print('=' * 78, flush=True)

        row = {'dataset': dataset, 'student': student, 'epochs': epochs, 'script': script, 'folder': str(out)}
        if state == 'done' and SKIP_FINISHED:
            summary.append({**row, 'status': 'skipped (finished)', 'minutes': 0})
            continue
        if a.dry_run:
            summary.append({**row, 'status': 'dry run', 'minutes': 0})
            continue

        out.mkdir(parents=True, exist_ok=True)
        share_teacher_cache(dataset, out)
        t0 = time.time()
        try:
            code = run_and_log(cmd, out / 'run_log.txt')
            status = 'ok' if code == 0 else f'FAILED (exit {code})'
        except KeyboardInterrupt:
            summary.append({**row, 'status': 'interrupted', 'minutes': round((time.time() - t0) / 60, 1)})
            print('\nInterrupted by user; rerun this script to continue where it stopped.')
            break
        minutes = round((time.time() - t0) / 60, 1)
        summary.append({**row, 'status': status, 'minutes': minutes})
        print(f'\n--> {dataset} / {student}: {status} after {minutes} min '
              f'(log: {out / "run_log.txt"})', flush=True)

    print('\n' + '=' * 78 + '\nSUMMARY')
    width = max(len(r['dataset']) for r in summary) if summary else 10
    for r in summary:
        print(f"  {r['dataset']:<{width}}  {r['student']:<20} e{r['epochs']:<4} {r['status']:<22} {r['minutes']:>7} min")
    print(f'Total time: {(time.time() - t_all) / 60:.1f} min')
    with open(Path(RESULTS_ROOT) / 'runs_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'student', 'epochs', 'script', 'folder', 'status', 'minutes'])
        w.writeheader(); w.writerows(summary)
    failed = [r for r in summary if r['status'].startswith('FAILED')]
    if failed:
        print(f'{len(failed)} run(s) failed; see run_log.txt in their folders. Rerun this script to retry them.')
        sys.exit(1)


if __name__ == '__main__':
    main()
