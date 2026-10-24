"""Run every (dataset x student model) experiment, one after another.

Edit the SETTINGS section below, then run:
    python run_experiments.py            # run everything
    python run_experiments.py --dry-run  # only print what would run

Before each run it prints the script name, dataset, student model, epochs, output
folder and the full command. Each run's console output is also saved to
<output folder>/run_log.txt. A run that fails does not stop the others; a summary
table is printed at the end and saved to <RESULTS_ROOT>/runs_summary.csv.

Re-running is safe: finished experiments are skipped, and unfinished ones (or ones
needing extra seeds/modes) continue with --resume.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ══════════════════════════════ SETTINGS ══════════════════════════════════════
# Datasets to run, in order. Comment out lines you do not want.
# name: (training script, extra flags for that dataset)
DATASETS = {
    'fmnist':         ('fmnist_vl2lite_mle_pareto_v8.py',       []),
    # 'svhn':         ('svhn_vl2lite_mle_pareto_v8.py',         []),
    # 'cifar10':      ('cifar10_vl2lite_mle_pareto_v8.py',      []),
    # 'cifar100':     ('cifar100_vl2lite_mle_pareto_v8.py',     []),
    # 'tinyimagenet': ('tinyimagenet_vl2lite_mle_pareto_v8.py', []),
}

# Student models to run on every dataset.
# Options: resnet18, resnet50, mobilenet_v2, mobilenet_v3_large,
#          efficientnet_b0, convnext_tiny
STUDENTS = [
    'resnet18',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'efficientnet_b0',
    'convnext_tiny',
    'resnet50',
]

# ── Core training settings ────────────────────────────────────────────────────
# Fix 2: 50 epochs for convergence (v8 convergence check fires at 20)
EPOCHS_DEFAULT = 1

# Fix 4: 3 seeds for variance analysis (single seed gives NaN std)
SEEDS = '42'#'42,43,44'

# Fix 1: all 6 distillation modes including rkd, crd, dkd
MODES = 'labels,feat,rkd,crd,dkd,vl2lite,sskd,clipkd'

# Fix 5: explicit dims so pca and erank tokens are always passed
# Grid starts at 2 so a plateau at the smallest width can be detected (d_1pp < 8)
DIMS = ('2,4,8,12,16,24,32,48,64,96,128,'
        'mle*0.5,mle*0.75,mle,pca,erank,mle*1.25,mle*1.5,full')

# Ablations: each gets its own output folder. 'main' uses student-only augmentation;
# 'matched_views' trains on un-augmented images, so teacher and student see the same view.
ABLATIONS = {
    'main':          [],
    'matched_views': ['--no-augment'],
}

GPUS = '0'          # e.g. '0,1' for two GPUs
NUM_WORKERS = 2     # dataloader workers per run

# Extra flags added to EVERY run — add ablations here if needed
# e.g. ['--label-smoothing', '0', '--no-augment']
EXTRA_FLAGS = []

DATA_ROOT    = './data'
RESULTS_ROOT = 'results'    # output: results/<dataset>_<student>_e<epochs>/
SKIP_FINISHED = True        # skip runs whose pareto_summary.csv is complete
# ══════════════════════════════════════════════════════════════════════════════


def out_folder(dataset: str, student: str, epochs: int, ablation: str = 'main') -> Path:
    suffix = '' if ablation == 'main' else f'_{ablation}'
    return Path(RESULTS_ROOT) / f'{dataset}_{student}_e{epochs}{suffix}'


def run_state(out: Path, dataset: str, student: str) -> str:
    """Return 'done', 'resume', or 'new' for an output folder.

    'done'   — final analysis exists, is marked complete, nothing is missing, and it
               covers enough seeds and all requested modes.
    'resume' — results_per_seed.csv exists but the run is incomplete, or the analysis
               was produced with --allow-partial. interim_* files from --analyze-only
               never count as finished.
    'new'    — nothing exists yet.
    """
    summary_csv  = out / f'{dataset}_{student}_pareto_summary.csv'
    per_seed_csv = out / f'{dataset}_{student}_results_per_seed.csv'

    # A partial analysis (--allow-partial) writes complete: false; never treat it as done.
    try:
        sel = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
        if sel.get('complete') is False:
            return 'resume'
    except FileNotFoundError:
        pass
    except Exception:
        return 'resume'
    try:
        if json.loads((out / 'missing_runs.json').read_text(encoding='utf-8')):
            return 'resume'
    except FileNotFoundError:
        pass
    except Exception:
        return 'resume'

    if summary_csv.exists():
        try:
            with open(summary_csv, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return 'resume'

        if not rows:
            return 'resume'

        want_seeds = len([s for s in SEEDS.split(',') if s.strip()])
        want_modes = {m.strip() for m in MODES.split(',') if m.strip()}
        have_modes = {r.get('mode', '') for r in rows}

        # Fix 3: safe int conversion for n_seeds
        try:
            min_seeds = min(int(r['n_seeds']) for r in rows if r.get('n_seeds', '').strip())
        except (ValueError, KeyError):
            return 'resume'

        if min_seeds >= want_seeds and want_modes <= have_modes:
            return 'done'
        return 'resume'

    return 'resume' if per_seed_csv.exists() else 'new'


def build_command(dataset: str, student: str, epochs: int, out: Path, resume: bool,
                  ablation: str = 'main') -> list:
    """Build the subprocess command for one (dataset, student) run."""
    script, dataset_flags = DATASETS[dataset]
    cmd = [
        sys.executable, '-u', script,
        '--student',     student,
        '--epochs',      str(epochs),
        '--seeds',       SEEDS,
        '--modes',       MODES,
        '--dims',        DIMS,          # Fix 5: always pass dims explicitly
        '--gpus',        GPUS,
        '--num-workers', str(NUM_WORKERS),
        '--data-root',   DATA_ROOT,
        '--out',         str(out),
    ]
    cmd += list(dataset_flags)
    cmd += list(EXTRA_FLAGS)
    cmd += list(ABLATIONS[ablation])
    if resume:
        cmd.append('--resume')
    return cmd


def run_and_log(cmd: list, log_path: Path) -> int:
    """Run a command, stream output to console and append to log_path."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
                  f"{' '.join(str(c) for c in cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
        return proc.wait()


def print_separator(char='=', width=78):
    print(char * width)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--dry-run', action='store_true',
                   help='Print the runs without executing them')
    a = p.parse_args()

    # Check all required scripts exist before starting
    missing = sorted({script for script, _ in DATASETS.values()
                      if not Path(script).exists()})
    if missing and not a.dry_run:
        sys.exit(f'Training script(s) not found in {Path.cwd()}:\n  ' +
                 '\n  '.join(missing))

    jobs = [(dataset, student, abl) for abl in ABLATIONS for dataset in DATASETS
            for student in STUDENTS]
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)

    print_separator()
    print(f'Experiments  : {len(jobs)}  '
          f'({len(DATASETS)} dataset(s) x {len(STUDENTS)} student model(s) x '
          f'{len(ABLATIONS)} ablation(s): {list(ABLATIONS)})')
    print(f'Seeds        : {SEEDS}')
    print(f'Modes        : {MODES}')
    print(f'Dims         : {DIMS}')
    print(f'Epochs       : {EPOCHS_DEFAULT}')
    print(f'GPUs         : {GPUS}')
    print(f'Results root : {Path(RESULTS_ROOT).resolve()}')
    print_separator()

    summary_rows = []
    t_all = time.time()

    for i, (dataset, student, ablation) in enumerate(jobs, 1):
        epochs = EPOCHS_DEFAULT
        out    = out_folder(dataset, student, epochs, ablation)
        state  = run_state(out,dataset,student)
        script = DATASETS[dataset][0]
        resume = (state == 'resume')
        cmd    = build_command(dataset, student, epochs, out, resume=resume, ablation=ablation)

        print()
        print_separator()
        print(f'[{i}/{len(jobs)}]  Script  : {script}')
        print(f'         Dataset : {dataset}')
        print(f'         Student : {student}')
        print(f'         Ablation: {ablation}')
        print(f'         Epochs  : {epochs}')
        print(f'         Output  : {out}')
        print(f'         Status  : {state}')
        print(f"         Command : {' '.join(str(c) for c in cmd)}")
        print('=' * 78, flush=True)

        base_row = {
            'dataset': dataset,
            'student': student,
            'ablation': ablation,
            'epochs':  epochs,
            'script':  script,
            'folder':  str(out),
        }

        if state == 'done' and SKIP_FINISHED:
            print('  -> Skipping: already finished.')
            summary_rows.append({**base_row, 'status': 'skipped (finished)', 'minutes': 0})
            continue

        if a.dry_run:
            summary_rows.append({**base_row, 'status': 'dry run', 'minutes': 0})
            continue

        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            exit_code = run_and_log(cmd, out / 'run_log.txt')
            status = 'ok' if exit_code == 0 else f'FAILED (exit {exit_code})'
        except KeyboardInterrupt:
            minutes = round((time.time() - t0) / 60, 1)
            summary_rows.append({**base_row, 'status': 'interrupted', 'minutes': minutes})
            print('\nInterrupted by user. Rerun this script to continue.')
            break

        minutes = round((time.time() - t0) / 60, 1)
        summary_rows.append({**base_row, 'status': status, 'minutes': minutes})
        print(f'\n--> {dataset} / {student}: {status} in {minutes} min  '
              f'(log: {out / "run_log.txt"})', flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print_separator()
    print('SUMMARY')
    print_separator('-')
    if summary_rows:
        dw = max(len(r['dataset']) for r in summary_rows)
        sw = max(len(r['student']) for r in summary_rows)
        for r in summary_rows:
            print(f"  {r['dataset']:<{dw}}  {r['student']:<{sw}}  {r['ablation']:<13}  "
                  f"e{r['epochs']:<4}  {r['status']:<25}  {r['minutes']:>7.1f} min")

    total_min = (time.time() - t_all) / 60
    print_separator('-')
    print(f'Total time: {total_min:.1f} min')
    print_separator()

    # Save summary CSV
    summary_csv = Path(RESULTS_ROOT) / 'runs_summary.csv'
    fieldnames  = ['dataset', 'student', 'ablation', 'epochs', 'script', 'folder', 'status', 'minutes']
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f'Summary saved to: {summary_csv}')

    failed = [r for r in summary_rows if r['status'].startswith('FAILED')]
    if failed:
        print(f'\n{len(failed)} run(s) FAILED. Check run_log.txt in their folders. '
              'Rerun this script to retry them.')
        sys.exit(1)


if __name__ == '__main__':
    main()
