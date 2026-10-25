# ShiftGuard preliminary experiment report

Generated: 2026-10-14T00:16:56+05:30.

This document records the initial one-seed stage; see `paper/followup_report.md` for the later authorized replication and architecture pilot. All reported performance comes from saved experiment outputs. One training seed is preliminary; no statistical robustness, novelty, or distribution-shift robustness is established.

## Smoke pipeline check

Configuration: `results/smoke_seed42/config.json`; seed 42, device cpu, 40,682 CNN parameters, 2 epochs completed. Training wall time: 1.50 seconds. Best checkpoint-selection accuracy: 22.00%.

Test N=1,000: 212 correct, 788 incorrect; accuracy 21.20%. NLL 2.1029, multiclass Brier 0.8573, classifier ECE 0.0243.

| Method | Error AUROC ↑ | AURC ↓ | Risk 50% | Risk 70% | Risk 80% | Risk 90% |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.6205 | 0.7167 | 0.7300 | 0.7586 | 0.7650 | 0.7778 |
| Entropy | 0.6074 | 0.7184 | 0.7360 | 0.7643 | 0.7688 | 0.7756 |
| Distance | 0.5604 | 0.7499 | 0.7480 | 0.7714 | 0.7800 | 0.7867 |
| Combined | 0.6454 | 0.7118 | 0.7280 | 0.7414 | 0.7550 | 0.7744 |

Estimated-error binary Brier: 0.1667; ECE: 0.0592. Forward latency: 0.2600 ms/image; all four scores: 0.0023 ms/image. Latencies depend on batching/hardware; cached reevaluation preserves extraction latency and remeasures scoring.

Correct reference counts by class: [5, 1, 0, 0, 0, 9, 16, 0, 12, 0]. Pooled-centroid fallback classes: [1, 2, 3, 4, 7, 9].

| Method | Descriptive test coverage at risk ≤ target | Validation-threshold test coverage | Test threshold risk | Accepted N |
|---|---:|---:|---:|---:|
| MSP | N/A | 0.0000 | N/A | 0 |
| Entropy | N/A | 0.0000 | N/A | 0 |
| Distance | N/A | 0.0000 | N/A | 0 |
| Combined | N/A | 0.0000 | N/A | 0 |

Target risk=0.1; minimum validation acceptance=100. Test-curve coverage is descriptive only. The deployed threshold is fixed from a separate validation partition, may reject everything, and has no guaranteed test-risk bound.

Figures: `results/smoke_seed42/figures/clean/` (PNG + PDF with captions); metrics: `results/smoke_seed42/metrics/clean.json` and CSV. Prediction and feature caches allow reruns without CNN training.

Recorded process peak RSS: 435.6 MiB (process lifetime, including imports/data/plots; not incremental training memory).

This reduced two-epoch run validates execution only. Its low classifier accuracy and centroid fallback make it unsuitable for research conclusions.

## Full clean experiment

Configuration: `results/full_seed42/config.json`; seed 42, device cpu, 40,682 CNN parameters, 15 epochs completed. Training wall time: 457.31 seconds. Best checkpoint-selection accuracy: 71.70%.

Test N=10,000: 7,156 correct, 2,844 incorrect; accuracy 71.56%. NLL 0.8219, multiclass Brier 0.3915, classifier ECE 0.0178.

| Method | Error AUROC ↑ | AURC ↓ | Risk 50% | Risk 70% | Risk 80% | Risk 90% |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.8061 | 0.1110 | 0.0972 | 0.1644 | 0.2016 | 0.2418 |
| Entropy | 0.7935 | 0.1151 | 0.1008 | 0.1754 | 0.2094 | 0.2443 |
| Distance | 0.5518 | 0.2345 | 0.2502 | 0.2777 | 0.2831 | 0.2870 |
| Combined | 0.8022 | 0.1133 | 0.0960 | 0.1681 | 0.2042 | 0.2429 |

Estimated-error binary Brier: 0.1597; ECE: 0.0438. Forward latency: 0.1919 ms/image; all four scores: 0.0025 ms/image. Latencies depend on batching/hardware; cached reevaluation preserves extraction latency and remeasures scoring.

Correct reference counts by class: [75, 88, 56, 60, 73, 78, 80, 65, 83, 77]. Pooled-centroid fallback classes: [].

| Method | Descriptive test coverage at risk ≤ target | Validation-threshold test coverage | Test threshold risk | Accepted N |
|---|---:|---:|---:|---:|
| MSP | 0.5110 | 0.4768 | 0.0893 | 4768 |
| Entropy | 0.4980 | 0.4600 | 0.0863 | 4600 |
| Distance | 0.0222 | 0.0000 | N/A | 0 |
| Combined | 0.5126 | 0.4828 | 0.0911 | 4828 |

Target risk=0.1; minimum validation acceptance=100. Test-curve coverage is descriptive only. The deployed threshold is fixed from a separate validation partition, may reject everything, and has no guaranteed test-risk bound.

Figures: `results/full_seed42/figures/clean/` (PNG + PDF with captions); metrics: `results/full_seed42/metrics/clean.json` and CSV. Prediction and feature caches allow reruns without CNN training.

Recorded process peak RSS: 553.8 MiB (process lifetime, including imports/data/plots; not incremental training memory).

Combined minus strongest individual AUROC: -0.0039; combined minus lowest individual AURC: +0.0023. These are one-seed point estimates, without significance claims.

## Recommendation

Revise the method before expanding to larger architectures. Run seeds 43 and 44 with the prespecified protocol, compare MSP+entropy against all three signals, and add paired uncertainty estimates. Only expand if distance contributes consistently at acceptable cost. No novelty claim is justified by the initial combination.

## Execution and limitations

Unit-test results: `paper/unit_tests.txt`. Smoke evaluation rerun and optional severity-1 noise, blur, brightness, and contrast smoke evaluations completed if their saved JSON outputs exist; these are pipeline checks only. Full corruption results are separate from clean results and must be explicitly run.

The first smoke attempt failed at download due to sandbox DNS/network restrictions. A permitted network retry reached the official server but downloaded at roughly 40–50 KB/s. The complete unmodified CIFAR-10 archive was obtained from the MIT Open Learning mirror and matched official MD5 `c58f30108f718f92721af3b95e74349a`. Dataset provenance is in `paper/dataset_provenance.json`. No data subset or model settings were changed to improve results.

The installed torch/torchvision pair emits a warning for the optional torchvision image extension. PIL-backed CIFAR loading and actual training worked. The README recommends a matched pair in a clean venv. Cross-version exact numeric reproduction is not assured.

Only one training seed is reported here. Three-seed training, ablations, formal risk guarantees, probability recalibration, direct stronger-detector comparisons, and exhaustive literature review remain future work. No pretrained weights or larger architectures were used in this initial experiment. Later architecture extensions are reported separately in paper/followup_report.md.
