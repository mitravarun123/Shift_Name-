# ShiftGuard Colab experiment — full

Architecture: resnet18, from scratch; seed 42; 40 epochs, best epoch 39.

Test N=10000, accuracy=92.92%, correct=9292, errors=708. NLL=0.3789, multiclass Brier=0.1202, ECE=0.0488.

| Method | AUROC ↑ | AURC ↓ |
|---|---:|---:|
| MSP | 0.92299 | 0.00860 |
| Entropy | 0.92332 | 0.00857 |
| Distance | 0.82242 | 0.01845 |
| MSP+Entropy | 0.92354 | 0.00856 |
| Combined | 0.88145 | 0.01357 |

Risk calibration: simultaneous confidence 95% across 315 prespecified method/threshold/budget rules. Certified risk refers to the future population error fraction among accepted predictions under IID assumptions; it is not a guarantee for each image or a bound on this particular test sample.

A failed certificate rejects all examples; conditional risk is undefined, not zero. Grids were formed from estimator-fitting scores, never from calibration/test outcomes. Calibration was reserved without labels before stratifying the remaining pool. This changes the historical split identities while keeping the full 45k/5k/10k sizes.

One seed is preliminary. CIFAR is a fixed benchmark and IID population assumptions cannot be verified by this experiment. No clean-data guarantee is asserted under arbitrary shift. Synthetic corruption results, if enabled, are stress tests using frozen clean thresholds, not OOD detection.

No novelty claim is made for confidence fusion or the established Bonferroni/binomial construction. The empirical question is whether distance increases coverage at the same controlled risk, including when calibration labels are limited.

See metrics/*_risk_control.csv, risk_certificates.json, predictions/, and figures/ (600-DPI PNG only).
