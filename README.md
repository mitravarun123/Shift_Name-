# ShiftGuard

A CPU-first research implementation asking whether a small post-hoc logistic regression on maximum softmax probability (MSP), predictive entropy, and feature distance can detect classification errors and improve selective prediction. These are hypotheses, not demonstrated novelty or superiority. CIFAR-10 is the only dataset. The initial phase uses a compact CNN; the authorized follow-up adds a from-scratch ResNet-18 CPU pilot.

## Installation on macOS

Use Python 3.10–3.12, a standard virtual environment, and the matched torch/torchvision pair below. CPU is the default; no pretrained weights, CUDA, cloud account, or paid GPU is needed.

```bash
cd /Users/mitravarun/Desktop/shiftfolder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

The exact environment used for any actual run is recorded in its `environment.json`; it may differ from these installation pins. This workspace originally had torch 2.11.0 with torchvision 0.16.1; its optional torchvision image extension warns on import. CIFAR uses PIL rather than that extension. A fresh installation uses a matched pair.

## Protocol

Official CIFAR-10: 50,000 training images, 10,000 test images, ten classes. Source: [dataset website](https://www.cs.toronto.edu/~kriz/cifar.html). A seeded stratified split reserves 5,000 official training images for validation. Within validation, seeded stratified partitions allocate:

| Partition | Full count | Purpose |
|---|---:|---|
| Training | 45,000 | CNN gradient updates and training augmentation |
| Selection | 2,000 | Best checkpoint / early stopping only |
| Reference | 1,000 | Feature standardization and centroids from correct predictions only |
| Estimator fit | 1,000 | Standardized logistic regression against classifier error outcomes |
| Calibration / thresholds | 1,000 | Held-out calibration assessment and empirical acceptance thresholds |
| Official test | 10,000 | Final independent clean evaluation |

The reference and estimator subsets are disjoint to avoid centroid self-distance bias. No estimator hyperparameters are selected in this initial version: logistic C=1 is prespecified. No temperature scaling or probability recalibration is fitted; calibration is measured. Labels from test never fit parameters, centroids, or thresholds. Test data are opened after classifier selection, and all four methods share the same examples. Repeated test inspection when developing follow-ups can still compromise scientific independence; pre-register future comparisons.

The smoke configuration explicitly uses 1,000 training, 1,000 validation (400/200/200/200), and 1,000 stratified official test images, two epochs, seed 42. It checks the pipeline, not the research hypothesis. It never alters the full configuration. The smoke test subset overlaps the full official test set; full hyperparameters were prespecified before smoke evaluation and were not tuned from smoke outcomes. Stored test IDs refer to the official test namespace, distinct from training IDs.

Training uses three convolution–ReLU–pool stages (16, 32, 64 channels), adaptive pooling, a 64-dimensional feature layer, and a ten-class head. Cross-entropy with AdamW; training-only random padded crops and horizontal flips. Normalization uses fixed CIFAR channel mean `(0.4914,0.4822,0.4465)` and std `(0.2470,0.2435,0.2616)`. Best selection accuracy wins; ties retain the earlier epoch. Default limit 15 epochs, patience 4, batch 64, four torch threads, zero loader workers for macOS.

## Commands

Run from the repository root. Initial downloading requires internet access and roughly 170 MB compressed storage plus extracted files. If the official server is slow, first run `python scripts/download_cifar.py`: it pins the current MIT Open Learning mirror revision, checks the official archive MD5, and records download provenance. The dataset remains the identical official CIFAR-10 archive. Prediction caches and datasets are ignored by Git.

```bash
# Smoke test: creates results/smoke_seed42
python scripts/run_smoke_test.py

# Principal full experiment, training through evaluation and figures
python scripts/run_experiment.py --config configs/default.yaml

# Training only; official test not loaded
python scripts/run_experiment.py --config configs/default.yaml --stage train

# Evaluate saved CNN and reliability estimator without retraining
python scripts/run_experiment.py --config configs/default.yaml --stage evaluate

# Regenerate PNG/PDF figures from saved scores only
python scripts/run_experiment.py --config configs/default.yaml --stage plot

# Write a preliminary report from actual saved results
python scripts/generate_report.py

# Three full seeds, per-seed runs and mean/sample-standard-deviation summary
python scripts/reproduce_results.py --config configs/default.yaml --seeds 42 43 44
```

An existing checkpoint is protected against accidental retraining. To repeat from scratch use `--output-dir results/new_run`. Evaluation reuses cached predictions; it checks checkpoint SHA-256 and exact split IDs. Keep the saved config unchanged; a changed experiment must use a new output directory. Checkpoints/joblib artifacts are trusted local files and must not be loaded from untrusted sources. A full three-seed reproduction resumes completed checkpoints through evaluation; it does not resume interrupted training.

## Scores and metrics

All scores use higher = more likely incorrect. MSP is converted to `1−MSP`; entropy uses stable float64 softmax and clipped logs. Reference features are standardized by the mean/std across correctly classified reference examples (std floored at 0.001). Each class centroid uses correct reference examples of that class. Distance is RMS Euclidean distance in standardized coordinates to the **predicted** class centroid. With fewer than two correct reference examples for a class, use the pooled correct centroid; counts and fallback classes are saved. If there are no correct reference examples, fail explicitly. This diagonal scaling ignores correlations and class multimodality and is not Mahalanobis distance or Trust Score.

Combined uses a StandardScaler and unweighted logistic regression on `[MSP, entropy, distance]`; 1 labels an incorrect CNN prediction. Coefficients and input scales are recorded in `reliability.json`. Its output is an estimated error probability, with binary Brier and ECE evaluated separately from classifier calibration. No perfect-calibration assumption is made.

Error-detection AUROC treats errors as positive; single-class outcomes produce JSON null. Accept examples in ascending error score. Risk is accepted error fraction; coverage is accepted fraction. AURC is the arithmetic mean of N prefix risks (right endpoint discrete integral, no artificial zero-coverage point). Exact score ties use expected risk under random within-tie ordering to avoid label-dependent tie breaks. Requested risks at 50/70/80/90% use `ceil(coverage*N)` accepted items.

`oracle_coverage_at_target_risk` is a **descriptive** test-curve statistic, not a deployment threshold. Actual thresholds use only the threshold-validation partition, accept entire tied groups, maximize empirical coverage below target risk 0.1 with at least 100 observations. An absent feasible threshold accepts none (risk null). Empirical validation risk gives no statistical risk guarantee; test threshold coverage/risk is reported separately, including accepted count.

Classifier NLL uses the ground-truth softmax probability clipped to [1e-15, 1] for finite logs; multiclass Brier is the mean sum of squared class errors. ECE uses 15 equal-width confidence bins on [0,1], final bin including 1, empty bins omitted, count-weighted absolute confidence/accuracy gaps. Binary error Brier averages squared probability errors. Report correct/incorrect counts alongside detection metrics. Forward latency excludes one warmup, transfers/data preparation, and scoring; extraction wall time and combined scoring time are separate. Process peak RSS is lifetime resident memory (macOS bytes / Linux KiB), not incremental allocation or GPU peak memory.

## Outputs and figures

Each configured output directory contains `config.json`, `environment.json`, `splits.npz`, `training.json`, `best.pt`, `reliability.joblib`, `reliability.json`, `resources.json`, and:

- `predictions/`: cached logits, features, labels, official indices; scored predictions, probabilities, errors, all detector scores.
- `metrics/`: actual metrics in JSON and CSV; null means undefined or infeasible.
- `figures/clean/`: ROC, risk–coverage, selected operating points, classifier/error calibration, and compact result table; PNG and vector PDF, with text captions.

Plots are generated only from saved actual predictions. Feature dimension is small, and only held-out partitions are cached. No placeholder performance figures are shipped. `notebooks/analysis.ipynb` reads saved results.

## Optional synthetic corruption

For a completed clean run, evaluate additional corruptions without retraining:

```bash
python scripts/run_corruptions.py --run-dir results/full_seed42 --corruptions noise:1 blur:2 brightness:1 contrast:3
```

Alternatively, include corruptions in a fresh run by creating a copy of the default YAML with a fresh output directory and e.g.:

```yaml
corruptions:
  - {kind: noise, severity: 1}
  - {kind: blur, severity: 2}
  - {kind: brightness, severity: 1}
  - {kind: contrast, severity: 3}
```

Noise standard deviations are 0.04/0.08/0.12 on [0,1]; blur radii 0.5/1/1.5 pixels; brightness multipliers 1.2/1.4/1.6; contrast 0.8/0.6/0.4. Applied before normalization, with noise seeded by image content and experiment seed. The frozen clean classifier/estimator/thresholds are reused. Clean and corruption metrics live separately, on identical test indices. These synthetic tests are not an independent OOD dataset or comprehensive robustness validation. Optional corruptions have not been run unless recorded in the experiment report.

## Research status and limitations

See `paper/experiment_report.md` for what actually executed, `paper/experiment_log.md` for failures and environment details, and `paper/outline.md` for the paper plan and verified reference links. One seed is preliminary; smoke data cannot justify moving to larger architectures. Low classifier accuracy changes the error-detection task. Centroid fallback, 1,000-example fitting subsets, shared source distribution, ECE binning, empirical threshold selection, and correlated uncertainty features limit conclusions. No novelty, SOTA accuracy, distribution-shift robustness, or estimator superiority is assumed.

Future architectures can implement the same `(logits, features)` forward interface and use these caches/metrics; preprocessing and feature differences must be recorded. ResNet-18 is now implemented for the follow-up; VGG remains unimplemented. Before expansion, repeat the full small-CNN protocol for three seeds, add paired uncertainty estimates and MSP+entropy versus all-three ablations, and review representation-based error detectors directly.

## Follow-up: three seeds, ablation, and ResNet-18

The authorized follow-up adds an uncertainty-only logistic estimator (`MSP+Entropy`) and compares it against the original three-signal `Combined` estimator. Both use the same frozen classifier and disjoint fitting/threshold subsets, with C=1 and no test-based model selection. The original four-method results stay in their run directories; five-method ablation results go to `results/ablation/<run-name>/`.

```bash
# Reuse seed 42 and train missing small-CNN seeds 43 and 44
python scripts/reproduce_results.py --prefix results/full --seeds 42 43 44
# Fit the uncertainty-only ablation from existing caches without CNN training
python scripts/run_ablation.py --run-dir results/full_seed42
```

`results/full_summary.json` reports mean and sample standard deviation (ddof=1), including paired differences `Combined − MSP+Entropy`. AUROC differences above zero favor distance; AURC differences below zero favor distance. Three seeds give limited evidence, not a conclusive significance test.

ResNet-18 uses torchvision with `weights=None`, a CIFAR-specific 3×3 stride-1 convolution, no initial max pooling, a 512-dimensional penultimate feature, and a ten-class head. It has 11,173,962 parameters. The original small CNN still defaults to its unchanged architecture. References: [torchvision ResNet-18 API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Images remain 32×32 with the same normalization and augmentation; no ImageNet preprocessing or pretrained weights are used.

The measured CPU throughput projects roughly 45 minutes per 45,000-image epoch, versus roughly 30 seconds for the small CNN. `paper/resnet18_compute_benchmark.json` is a synthetic compute benchmark, not classification evidence. MPS was unavailable in this installed environment.

```bash
# Explicit CPU feasibility pilot: 5,000 training, full 5,000 validation,
# full 10,000 test, batch 16, two epochs; not a controlled accuracy comparison
python scripts/run_experiment.py --config configs/resnet18_pilot.yaml
python scripts/run_ablation.py --run-dir results/resnet18_pilot_seed42

# Optional full-data 15-epoch CPU experiment; potentially overnight
python scripts/run_experiment.py --config configs/resnet18.yaml
```

The pilot keeps the exact seed-42 validation partitions and official test indices, but intentionally uses a documented stratified subset of the 45,000 training images to bound compute. Its different training budget prevents concluding that ResNet improves or worsens classifier accuracy relative to the full small-CNN runs. A full-data/full-budget comparison is a separate configuration; settings are never silently reduced. VGG16 remains unimplemented.

After all three seeds and the pilot complete, generate aggregate figures and the follow-up report:

```bash
python scripts/summarize_followup.py
```

This produces `paper/followup_report.md` and `results/followup/` CSV/JSON summaries, mean ROC and risk–coverage plots, and metric comparison plots in PNG/PDF. Bands/error bars show sample standard deviation across training seeds, not confidence intervals. Per-seed error labels can differ because each CNN is different. The test image identities are checked across runs.

For a fair **matched pilot budget**, the follow-up also provides the small CNN with the same 5,000 training images, two epochs and batch size 16:

```bash
python scripts/run_experiment.py --config configs/small_cnn_matched_pilot.yaml
python scripts/run_ablation.py --run-dir results/small_cnn_matched_pilot_seed42
```

Only compare these two pilot classifiers directly at their shared sample/epoch budget; they still differ in compute cost. The limited recipe is not optimized to establish ResNet's best achievable accuracy. See `paper/followup_protocol.md` for the declared protocol.

A single sequential command reproduces the full follow-up (three full small-CNN seeds, both matched pilots, ablations and report):

```bash
python scripts/run_followup.py
```

It reuses completed local checkpoints and runs on CPU. The larger pilot can take roughly 15 minutes here; the full-data ResNet configuration is deliberately excluded from this bounded command. Actual durations are recorded rather than assumed.

## Master Colab notebook: 40 epochs and risk-controlled selective prediction

Open [`colab/ShiftGuard_Master_Colab.ipynb`](colab/ShiftGuard_Master_Colab.ipynb) in Colab and upload `dist/shiftguard_colab_bundle.zip` at its prompt. It is a one-seed, from-scratch ResNet-18 workflow with epoch-level Drive recovery, AdamW/cosine scheduling, five methods and independent risk calibration. See `colab/README_COLAB.md` for precise assumptions, split refinement and steps. The 45k/5k/10k sizes are unchanged, but risk calibration is now reserved before label stratification, so IDs differ from historical results.

The new Colab pipeline and `scripts/generate_paper_figures.py` export **600-DPI PNG only**, with annotated heatmaps and no PDFs. Historical figures from earlier phases are preserved. The source-only Colab ZIP excludes datasets, checkpoints, historical results and legacy PDF-generating entry points. The full 40-epoch CUDA run is to be executed in the user's Colab session; local validation is a CPU smoke experiment.

```bash
python scripts/run_colab_experiment.py --config configs/colab_cpu_smoke.yaml
python scripts/generate_paper_figures.py --run-dir results/colab_workflow_smoke
```
# Shift_Name-
