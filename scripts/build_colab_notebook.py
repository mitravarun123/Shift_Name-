"""Build a self-contained Colab driver for the uploadable source bundle."""
from pathlib import Path
import nbformat as nbf

nb=nbf.v4.new_notebook()
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''
# ShiftGuard — Master Colab: ResNet-18 and Risk-Controlled Selective Prediction

**One seed (42), 40 epochs, from-scratch CIFAR-adapted ResNet-18, five reliability methods.**
All figures are **600-DPI PNG only**; CSV/JSON tables and raw predictions are saved too.

1. Select **Runtime → Change runtime type → GPU** (a GPU must actually be available).
2. Run cells in order. Upload **`shiftguard_colab_bundle.zip`** when prompted.
3. Authorize Google Drive access. Keep the same ZIP, run name and settings when resuming.
4. The long-running cell trains/resumes, evaluates once training is complete, and creates figures.
5. Review the gallery and download the results ZIP.

No Colab GPU or 40-epoch outcome is claimed by this delivered notebook. Local tests and a CPU smoke run validate the workflow; the full experiment runs in **your** Colab account. Free GPU resources and runtime lifetimes are not guaranteed: [Colab FAQ](https://research.google.com/colaboratory/faq.html).
''')
md('''
## 1. Upload and extract the project ZIP
Use the provided small source bundle. An existing project ZIP can also work if it contains `src/shiftguard/colab_workflow.py` and the matching scripts/configs. Datasets, old results, environments and checkpoints are skipped. No repository URL is required.
''')
code('''
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['MPLCONFIGDIR'] = '/content/shiftguard_mpl'
from pathlib import Path
import sys, json, subprocess
from google.colab import files
import io, zipfile, stat, hashlib, uuid
uploaded = files.upload()
archives = [(name, content) for name, content in uploaded.items() if name.lower().endswith('.zip')]
if len(archives) != 1:
    raise ValueError('Upload exactly one project ZIP.')
EXTRACT_ROOT = Path('/content/shiftguard_upload') / uuid.uuid4().hex
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
excluded = {'.git', '.venv', 'venv', '__pycache__', '__MACOSX', '.aws', '.codex', '.agents',
            'data', 'results', 'checkpoints', 'dist', '.pytest_cache'}
allowed = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.toml', '.ipynb'}
with zipfile.ZipFile(io.BytesIO(archives[0][1])) as archive:
    members = []
    for info in archive.infolist():
        relative = Path(info.filename)
        if not relative.parts or relative.is_absolute() or '..' in relative.parts or ':' in relative.parts[0]:
            raise ValueError('Archive contains an unsafe path.')
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('Archive contains a symbolic link.')
        if info.is_dir() or any(p in excluded for p in relative.parts) or relative.suffix.lower() not in allowed:
            continue
        members.append(info)
    if sum(m.file_size for m in members) > 64 * 1024**2:
        raise ValueError('Source exceeds 64 MiB; use the provided source-only bundle.')
    for info in members:
        target = (EXTRACT_ROOT / info.filename).resolve()
        if EXTRACT_ROOT.resolve() not in target.parents:
            raise ValueError('Unsafe archive path.')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(info))
projects = [p.parents[2] for p in EXTRACT_ROOT.rglob('colab_workflow.py')
            if p.parent.name == 'shiftguard' and p.parent.parent.name == 'src']
if len(projects) != 1:
    raise ValueError('Could not identify one ShiftGuard source root. Use the provided ZIP.')
PROJECT = projects[0]
manifest = PROJECT / 'BUNDLE_MANIFEST.json'
if manifest.exists():
    for relative, expected in json.loads(manifest.read_text())['sha256'].items():
        path = (PROJECT / relative).resolve()
        if PROJECT.resolve() not in path.parents:
            raise ValueError('Unsafe bundle manifest path.')
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundle file differs: {relative}. Re-upload the original source bundle.')
print('Project:', PROJECT)
print('Source ZIP SHA-256:', hashlib.sha256(archives[0][1]).hexdigest())
del uploaded, archives
''')
md('''
## 2. Install small dependencies and run tests
The requirements file deliberately excludes torch/torchvision. Tests include score direction, split independence, exact CPU epoch-boundary recovery and confidence-bound checks. GPU execution itself is validated by the actual training run.
''')
code('''
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PROJECT / 'requirements-colab.txt')])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', str(PROJECT)])
subprocess.check_call([sys.executable, '-m', 'pytest', '-q'], cwd=PROJECT)
''')
md('''
## 3. Mount Drive and fix the run configuration
Checkpoints/results are stored in a dedicated Drive folder. The CIFAR dataset stays on the local VM for faster I/O. **Reuse the same run name, ZIP and batch size when resuming.** Changes to the training configuration/source are rejected instead of silently changing the experiment. If a fresh run hits GPU out-of-memory, choose a smaller batch and a **new run name** before starting again.

The default is the agreed **from-scratch** ResNet-18. No pretrained weights are downloaded. It uses a 3×3 stride-1 CIFAR stem, no initial max pool, 32×32 images and 512-dimensional penultimate features. Pretrained/ImageNet-transfer experiments require a separate documented protocol.
''')
code('''
from google.colab import drive
drive.mount('/content/drive')
import yaml, re
RUN_NAME = 'resnet18_40e_seed42_risk_v1'  # reuse unchanged after a disconnect
BATCH_SIZE = 128
DRIVE_BASE = Path('/content/drive/MyDrive/ShiftGuard')
if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', RUN_NAME):
    raise ValueError('Use a plain run name with letters, digits, underscores or hyphens.')
RUN_DIR = DRIVE_BASE / RUN_NAME
DATA_DIR = Path('/content/shiftguard_data')
RUN_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((PROJECT / 'configs/colab_resnet18_40.yaml').read_text())
cfg.update(output_dir=str(RUN_DIR), data_dir=str(DATA_DIR), batch_size=BATCH_SIZE)
RUN_CONFIG = RUN_DIR / 'run_config.yaml'
if (RUN_DIR / 'config.json').exists() and json.loads((RUN_DIR / 'config.json').read_text()) != cfg:
    raise ValueError('Run settings differ from saved experiment. Restore settings or use a new RUN_NAME.')
RUN_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(yaml.safe_dump(cfg, sort_keys=False))
print('Persistent checkpoint/result folder:', RUN_DIR)
''')
md('''
## 4. Experimental protocol and the precise guarantee

| Partition | Images | Purpose |
|---|---:|---|
| Training | 45,000 | CNN gradients; crop/flip augmentation |
| Selection | 2,000 | Checkpoint selection only |
| Reference | 1,000 | Feature scaling/centroids from correct predictions |
| Estimator fit | 1,000 | Fit two- and three-signal logistic models; form candidate grids |
| Risk calibration | 1,000 | Statistical bounds only; nested budgets 250/500/1,000 |
| Official test | 10,000 | Final evaluation after rules are frozen |

**Important split refinement:** risk calibration is reserved uniformly without looking at labels **before** stratifying the remaining pool. Other validation roles remain stratified. Sizes match our prior protocol, but split identities change; historical seed-42 metrics are not directly reused. The calibration partition never fits the CNN, centroids, score model, candidate grids or learning rate.

For each fixed rule `accept if error_score ≤ threshold`, let `N` be the accepted calibration count and `K` its errors. Compute the one-sided **Clopper–Pearson** upper bound with failure level `δ/M`, where `M` counts all prespecified method × threshold × calibration-budget rules. It is 315 when each of five methods has 21 distinct thresholds and all three budgets are used; ties can reduce it. The same bound covers both risk targets, **5% and 10%**. Choose maximum empirical calibration coverage among eligible rules.

Under independent IID calibration and future clean examples, with probability at least **95% over calibration sampling**, every selected nonempty rule has true population error rate among accepted predictions at most its target. This is **not** a per-image correctness guarantee, an observed-test-error bound, or a guarantee under arbitrary shift. A fixed CIFAR benchmark does not establish the IID population assumption. If no rule qualifies, report **not certified**, accept nothing, and leave conditional risk undefined.

This is an established finite-grid multiple-testing construction, inspired by [Learn then Test](https://arxiv.org/abs/2110.01052); it is not a claimed new theorem. Our empirical question is whether representation distance improves coverage at equal controlled risk and limited calibration labels. Confidence/entropy/distance fusion alone is not claimed novel.
''')
md('''
## 5. Download the official CIFAR-10 archive
The optional MIT Open Learning mirror serves the unmodified archive; the downloader checks torchvision's official archive MD5 before installing it. If the mirror is unavailable, torchvision uses the official dataset host during loading. Nothing is fitted on test labels.
''')
code('''
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity
base = DATA_DIR / CIFAR10.base_folder
valid_files = all(check_integrity(str(base / name), md5)
                  for name, md5 in CIFAR10.train_list + CIFAR10.test_list)
valid_archive = check_integrity(str(DATA_DIR / CIFAR10.filename), CIFAR10.tgz_md5)
if valid_files or valid_archive:
    print('Verified CIFAR-10 cache already present.')
else:
    result = subprocess.run([sys.executable, str(PROJECT / 'scripts/download_cifar.py'),
                             '--data-dir', str(DATA_DIR)], cwd=PROJECT)
    if result.returncode:
        print('Mirror download did not complete; the official torchvision downloader will be used.')
''')
md('''
## 6. Train/resume all 40 epochs, freeze the selected checkpoint, evaluate and plot
This is the long-running cell. There is **no early stopping**: 40 epochs are completed, and the highest selection-accuracy checkpoint is used for final evaluation. AdamW and a cosine learning-rate schedule are fixed before test evaluation.

`last.pt` contains model/optimizer/scheduler states, all RNG states, sample-order generator state, completed epoch count and history. `best.pt` holds the validation-selected classifier. Both are saved to Drive each completed epoch using temporary-file replacement. A disconnect loses at most the unfinished epoch; rerun setup and this cell with the original ZIP/config to resume. Exact numeric recovery across different GPUs/software versions is not promised; each session's environment is recorded.

The cell automatically extracts cached predictions/features, fits all five score methods, freezes risk certificates **before opening the test dataset**, evaluates, and generates all PNG figures. A repeated completed run reuses checkpoints/caches; it does not train another 40 epochs.
''')
code('''
import signal

def stream_run(command):
    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        print('Interrupted. Re-run this cell to resume from the last committed epoch.')
        raise

stream_run([sys.executable, '-u', str(PROJECT / 'scripts/run_colab_experiment.py'),
            '--config', str(RUN_CONFIG), '--stage', 'all'])
training = json.loads((RUN_DIR / 'training.json').read_text())
assert training['training_complete'] and training['epochs_completed'] == 40
assert (RUN_DIR / 'evaluation_complete.json').exists()
print('Completed 40 epochs. Best selection epoch:', training['best_epoch'])
''')
md('''
## 7. Actual metrics and risk-control audit
Every method uses the same frozen classifier/test images. The two-signal estimator is standardized logistic regression on MSP+entropy; ShiftGuard adds standardized predicted-class feature distance. Both use C=1 and the same fitting subset. AUROC treats errors as positive. AURC is mean prefix risk; smaller is better. An empirical test risk above a target is not by itself proof that a population-risk certificate failed.
''')
code('''
import pandas as pd
from IPython.display import display, Markdown
clean = json.loads((RUN_DIR / 'metrics/clean.json').read_text())
print(json.dumps(clean['classifier'], indent=2))
display(pd.read_csv(RUN_DIR / 'metrics/clean_methods.csv'))
risk_table = pd.read_csv(RUN_DIR / 'metrics/clean_risk_control.csv')
display(risk_table[risk_table.calibration_n == 1000][
    ['method','target_risk','status','calibration_accepted','calibration_upper_risk',
     'test_accepted','test_coverage','test_risk']])
display(Markdown((RUN_DIR / 'experiment_report.md').read_text()))
''')
md('''
## 8. Paper-quality PNG gallery and figure-only regeneration
Exports use **600 DPI**, consistent colors/line styles/fonts, clear units, subtle grids, cell annotations on every heatmap and independent captions. Tables are PNG as well as CSV/JSON. There are **no PDF exports**. One-seed plots do not invent across-seed error bars. Inspect saved files at their intended paper size before submission; unusually long custom labels may require style adjustments.

Figures: ROC; risk–coverage; annotated risk-at-coverage heatmap; calibration diagrams; annotated confusion matrix; primary metrics table; controlled-risk coverage heatmap; calibration-budget coverage curves; risk-control audit table; training curves. No eligible certificates is an actual result, not missing data.
''')
code('''
# Regeneration reads saved predictions and metrics only; no training or inference.
stream_run([sys.executable, str(PROJECT / 'scripts/generate_paper_figures.py'),
            '--run-dir', str(RUN_DIR)])
from IPython.display import Image
for path in sorted((RUN_DIR / 'figures/clean').glob('*.png')):
    print(path.name)
    display(Image(filename=str(path), width=1100))
for path in (RUN_DIR / 'figures/training').glob('*.png'):
    display(Image(filename=str(path), width=1100))
assert not list((RUN_DIR / 'figures').rglob('*.pdf')), 'Unexpected PDF in this new run folder'
''')
md('''
## 9. Optional synthetic-corruption stress tests (off by default)
These reuse the clean classifier, score models and calibration thresholds. They **do not** recalibrate using corrupted labels. A clean IID risk certificate does **not** apply automatically under shift. This is a corruption stress test, not a distribution-shift detector or comprehensive OOD evaluation. Clean and corruption outputs remain separate. Enabling this cell does not change or retrain the clean model.

Severity-1 settings: Gaussian noise std 0.04 on [0,1]; blur radius 0.5 pixels; brightness ×1.2; contrast ×0.8, applied before normalization. Noise is reproducible from the image and seed.
''')
code('''
RUN_STRESS_TESTS = False
if RUN_STRESS_TESTS:
    stream_run([sys.executable, str(PROJECT / 'scripts/run_colab_corruptions.py'),
                '--config', str(RUN_CONFIG), '--corruptions',
                'noise:1', 'blur:1', 'brightness:1', 'contrast:1'])
else:
    print('Optional stress tests skipped; clean results are complete.')
''')
md('''
## 10. Download results
All files already persist in the dedicated Drive run folder. This cell creates a ZIP with figures, captions, CSV/JSON tables, score/feature caches and the report. Checkpoint inclusion is optional because resumable checkpoints remain on Drive. Existing historical PDF figures outside this new run are not exported.
''')
code('''
import shutil
INCLUDE_CHECKPOINTS = False  # True includes best.pt AND the larger resumable last.pt
export = Path('/content') / f'{RUN_NAME}_results.zip'
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
    for path in sorted(RUN_DIR.rglob('*')):
        if not path.is_file() or path.suffix in {'.pdf','.svg','.tmp'}:
            continue
        if not INCLUDE_CHECKPOINTS and path.suffix == '.pt':
            continue
        archive.write(path, arcname=str(Path(RUN_NAME) / path.relative_to(RUN_DIR)))
persistent_export = DRIVE_BASE / export.name
shutil.copy2(export, persistent_export)
print('Results ZIP (MiB):', round(export.stat().st_size / 1024**2, 2))
print('Drive copy:', persistent_export)
files.download(str(export))
''')
md('''
## Interpretation and research limits
- This is one training seed: preliminary results, not across-seed robustness.
- Compare accepted coverage at the same risk target and the **same simultaneous calibration rule**; report failed certification honestly.
- Do not tune score features, grids, C, epoch budget or hyperparameters using calibration/test outcomes. New choices require new independent calibration and a documented follow-up.
- Confidence-bound conservatism and small calibration budgets can produce zero eligible coverage even for a useful score.
- Synthetic corruptions do not inherit the clean IID guarantee and do not establish novelty or OOD detection.
- Main references: [Learn then Test](https://arxiv.org/abs/2110.01052), [Selective Classification](https://arxiv.org/abs/1705.08500), [ResNet](https://arxiv.org/abs/1512.03385), [PyTorch checkpoint guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).
''')
nb['cells']=cells
nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python'},'colab':{'name':'ShiftGuard_Master_Colab.ipynb','provenance':[]},
                'accelerator':'GPU'}
nbf.validate(nb)
for i,cell in enumerate(cells):
    if cell.cell_type=='code':
        compile(cell.source,f'colab_cell_{i}','exec')
path=Path('colab/ShiftGuard_Master_Colab.ipynb')
path.parent.mkdir(parents=True,exist_ok=True)
nbf.write(nb,path)
print(path, len(cells), 'cells; code cells compile')
