# ShiftGuard Colab experiment — smoke

Architecture: small_cnn, from scratch; seed 42; 2 epochs, best epoch 2.

Test N=1000, accuracy=27.10%, correct=271, errors=729. NLL=2.1541, multiclass Brier=0.8677, ECE=0.1162.

| Method | AUROC ↑ | AURC ↓ |
|---|---:|---:|
| MSP | 0.63439 | 0.65440 |
| Entropy | 0.62101 | 0.66170 |
| Distance | 0.61168 | 0.66849 |
| MSP+Entropy | 0.63696 | 0.65033 |
| Combined | 0.64863 | 0.64087 |

Risk calibration: simultaneous confidence 95% across 110 prespecified method/threshold/budget rules. Certified risk refers to the future population error fraction among accepted predictions under IID assumptions; it is not a guarantee for each image or a bound on this particular test sample.

A failed certificate rejects all examples; conditional risk is undefined, not zero. Grids were formed from estimator-fitting scores, never from calibration/test outcomes. Calibration was reserved without labels before stratifying the remaining pool. This changes the historical split identities while keeping the full 45k/5k/10k sizes.

One seed is preliminary. CIFAR is a fixed benchmark and IID population assumptions cannot be verified by this experiment. No clean-data guarantee is asserted under arbitrary shift. Synthetic corruption results, if enabled, are stress tests using frozen clean thresholds, not OOD detection.

No novelty claim is made for confidence fusion or the established Bonferroni/binomial construction. The empirical question is whether distance increases coverage at the same controlled risk, including when calibration labels are limited.

See metrics/*_risk_control.csv, risk_certificates.json, predictions/, and figures/ (600-DPI PNG only).
