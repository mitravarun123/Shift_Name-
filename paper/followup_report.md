# ShiftGuard follow-up: replication, ablation, and ResNet-18

## Three-seed small-CNN replication

Seeds [42, 43, 44]; 45,000 training, validation roles 2,000/1,000/1,000/1,000, and the same 10,000 official test examples. Architecture, preprocessing, AdamW settings, epoch limit and early stopping match the original experiment. Seed 42 reuses its original frozen checkpoint. No test-based hyperparameter selection.

| Method | Error AUROC mean ± SD ↑ | AURC mean ± SD ↓ |
|---|---:|---:|
| MSP | 0.80553 ± 0.00060 | 0.11029 ± 0.00328 |
| Entropy | 0.79727 ± 0.00346 | 0.11293 ± 0.00411 |
| Distance | 0.55352 ± 0.00489 | 0.23555 ± 0.00149 |
| Combined | 0.80381 ± 0.00153 | 0.11111 ± 0.00360 |
| MSP+Entropy | 0.80423 ± 0.00081 | 0.11069 ± 0.00319 |

| Seed | Test accuracy | Test errors / N | Best selection accuracy | Epochs | Training minutes |
|---|---:|---:|---:|---:|---:|
| 42 | 71.56% | 2844 / 10000 | 71.70% | 15 | 7.62 |
| 43 | 72.29% | 2771 / 10000 | 71.40% | 15 | 7.69 |
| 44 | 71.06% | 2894 / 10000 | 73.45% | 15 | 7.37 |

Classifier accuracy: 71.64% ± 0.62% (sample SD across training seeds).

MSP+Entropy and Combined are standardized unweighted logistic regressions with C=1. Their coefficients and empirical thresholds are fitted on the same frozen-classifier fitting and threshold subsets. The added feature is the standardized predicted-class centroid distance. Both methods are evaluated on identical examples.

| Paired difference: all three − uncertainty only | Mean | Sample SD | Per-seed values |
|---|---:|---:|---|
| auroc | -0.000425 | 0.000808 | -0.001354, -0.000036, +0.000115 |
| aurc | +0.000415 | 0.000859 | +0.001405, -0.000124, -0.000037 |

Distance improved AUROC in 1/3 seeds and AURC in 2/3 seeds.

Three seeds estimate variability but provide limited statistical power. No significance claim is made. Within-test sampling uncertainty and across-training-seed uncertainty are distinct. Test results have now been inspected; future method changes should be selected using validation and described as follow-up analyses.

## ResNet-18 CPU feasibility pilot

From-scratch CIFAR-stem ResNet-18: 11,173,962 parameters, 512-dimensional features, 2 epochs on 5,000 explicitly selected training images. Batch 16, four CPU threads, same full validation roles and official test images as seed 42. No pretrained weights or resized ImageNet inputs.

Training: 7.77 minutes; peak process RSS: 544.9 MiB. Test accuracy: 38.79%; NLL: 1.6410. Best validation selection accuracy: 38.45%.

This short, reduced-training-data pilot cannot be compared directly with the three full-data small-CNN runs. The matched-budget comparison is below. It cannot establish whether a fully trained ResNet improves classifier accuracy or detector ranking over the small CNN. Full-data 15-epoch CPU training is provided in `configs/resnet18.yaml`, but was not run here.

| Pilot method | Error AUROC | AURC |
|---|---:|---:|
| MSP | 0.64659 | 0.49017 |
| Entropy | 0.63776 | 0.49731 |
| Distance | 0.56360 | 0.55555 |
| Combined | 0.66381 | 0.47925 |
| MSP+Entropy | 0.64582 | 0.49020 |

## Matched pilot architecture comparison

Both classifiers use the identical 5,000 training image indices, two epochs, batch 16, full validation roles and 10,000 official test identities. Index equality was checked. This matches sample/epoch budgets, not training time or FLOPs. Each classifier has its own fitted detector.

| Classifier | Test accuracy | Training minutes | Forward ms/image | Peak RSS MiB |
|---|---:|---:|---:|---:|
| Small CNN | 34.42% | 0.29 | 0.413 | 487.2 |
| ResNet-18 | 38.79% | 7.77 | 26.199 | 544.9 |

Observed pilot accuracy difference: +4.37 percentage points. Training-time ratio: 26.8×; forward-latency ratio: 63.5×. These are single-run timings; the small-CNN pilot briefly overlapped full-CNN replication, and timings are not a controlled repeated hardware benchmark.

| Pilot classifier | ∆ AUROC: distance added | ∆ AURC: distance added |
|---|---:|---:|
| Small CNN | +0.015955 | -0.007967 |
| ResNet-18 | +0.017996 | -0.010948 |

Distance added useful ranking information in both short pilots on this seed, unlike its inconsistent effect in the three full-data small-CNN runs. This suggests investigating training-budget/representation-quality dependence; it does not isolate accuracy as the cause or establish benefit in a fully trained ResNet. Error-detection tasks differ across classifiers because their incorrect-prediction sets differ.

These single-seed, two-epoch pilot results are not a converged architecture comparison. They cannot support state-of-the-art or comprehensive robustness claims.

## Recommendation

Treat distance as an unproven addition. Prioritize validation-only ablations of representation distance before claiming superiority or novelty. Larger models require an explicit compute budget and a full-data training recipe; the pilot establishes execution feasibility and limited-budget observations, not converged performance. VGG16 has not been implemented or trained.

## Reproduction

```bash
python scripts/run_followup.py
# Regenerate report and aggregate figures from saved results only
python scripts/summarize_followup.py
```

Tests: `paper/extension_unit_tests.txt`. Aggregate PNG/PDF figures and CSV/JSON metrics: `results/followup/`. Per-seed plots and all method scores: `results/ablation/`. Actual throughput benchmark: `paper/resnet18_compute_benchmark.json`.
