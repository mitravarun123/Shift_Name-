# ShiftGuard: preliminary paper outline

## 1. Abstract template
We investigate whether a lightweight post-hoc estimator combining predictive uncertainty and representation distance improves error detection for a compact CIFAR-10 classifier. We compare MSP, entropy, standardized class-centroid distance, and logistic combination under independent validation roles. Classifier accuracy: TBD; error AUROC: TBD; AURC: TBD; across-seed differences and uncertainty: TBD. Findings and supported implications: TBD. No novelty or superiority claim until justified.

## 2. Introduction
Research question: does feature distance add useful information beyond softmax uncertainty, under CPU and memory constraints? Secondary question: does it improve selective prediction? Prespecify the methods and metrics before examining test outcomes. Hypotheses H1: combined AUROC exceeds strongest individual baseline; H2: combined AURC is lower. Counterhypothesis: signals are redundant or feature geometry is too weak.

## 3. Related work and review checklist
Metadata below checked against primary arXiv pages on 2026-10-14. This is an initial checklist, not an exhaustive novelty review.

- [ ] MSP: Dan Hendrycks, Kevin Gimpel. **A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks**. ICLR 2017 (preprint 2016). [arXiv:1610.02136](https://arxiv.org/abs/1610.02136). Established softmax baseline for error detection; reproduce its score direction.
- [ ] Entropy and post-hoc calibration: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. **On Calibration of Modern Neural Networks**. ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599). Read temperature scaling and its validation requirements. Additional primary reference for predictive entropy: TBD after verification.
- [ ] Feature geometry: Heinrich Jiang, Been Kim, Melody Y. Guan, Maya Gupta. **To Trust Or Not To Trust A Classifier**. NeurIPS 2018. [arXiv:1805.11783](https://arxiv.org/abs/1805.11783). Compare distance-based trust scoring; this implementation is not Trust Score.
- [ ] Class-conditional distance and learned combination: Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin. **A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks**. NeurIPS 2018. [arXiv:1807.03888](https://arxiv.org/abs/1807.03888). Review covariance-based scores, learned combinations, and differing OOD/adversarial supervision. Inspect full paper before making detailed equivalence claims.
- [ ] Selective classification: Yonatan Geifman, Ran El-Yaniv. **Selective Classification for Deep Neural Networks**. arXiv repository, 2017; [arXiv:1705.08500](https://arxiv.org/abs/1705.08500). Review risk–coverage and probabilistic guarantees; current empirical thresholding offers no such guarantees.
- [ ] Search for methods that directly combine entropy, confidence, and representation-space information for in-distribution error detection; exact overlap: TBD.
- [ ] Review AURC conventions, tie treatment, excess AURC, and finite-sample uncertainty.
- [ ] Add direct comparisons to stronger established detectors before claiming a contribution.

The implemented combination is an experimental baseline. If results repeat established findings, narrow the question to when a low-cost centroid signal adds information conditional on confidence, especially under limited validation budgets. Cite verified links; no unverified DOI/venue entries are needed.

## 4. Methodology
Three pooled conv blocks and shared penultimate features. Confidence/entropy, standardized predicted-class centroid RMS distance, standardized C=1 logistic regression. Separate selection/reference/fit/threshold subsets. Fall back to pooled correct centroid for underrepresented classes. No test fitting. Estimated error probability is evaluated for calibration.

## 5. Experimental setup
CIFAR-10 45k/5k/10k, validation roles 2k/1k/1k/1k. CPU, batch 64, 15 epochs maximum, patience 4. Fixed seeds 42/43/44 proposed; actual executed seeds TBD in report. Versions, model parameters, elapsed time, and peak RSS saved. Clean test is principal; synthetic corruption optional and separate. The initial phase had no architecture extension; the authorized follow-up adds a ResNet-18 CPU feasibility pilot, reported separately.

## 6. Results table template

| Method | Error AUROC ↑ | AURC ↓ | Risk 50% ↓ | Risk 70% ↓ | Risk 80% ↓ | Risk 90% ↓ |
|---|---|---|---|---|---|---|
| MSP | TBD | TBD | TBD | TBD | TBD | TBD |
| Entropy | TBD | TBD | TBD | TBD | TBD | TBD |
| Distance | TBD | TBD | TBD | TBD | TBD | TBD |
| Combined | TBD | TBD | TBD | TBD | TBD | TBD |

Main-paper entries require full data and across-seed uncertainty, not smoke-test numbers. Include counts, classifier NLL/Brier/ECE, binary estimator calibration, inference cost, empirical validation threshold risk/coverage and descriptive coverage-at-risk separately.

## 7. Ablation plan
TBD: confidence+entropy only; confidence+distance; entropy+distance; all three. Hold CNN, splits and examples fixed. Study reference/fit sample budgets, pooled versus class centroids, and scale floor sensitivity using validation only. Use paired test bootstraps for fixed-model comparison and separate across-training-seed variation; do not treat them interchangeably.

## 8. Limitations and threats to validity
Single dataset; compact underfit classifier; nonindependent repeated test development; limited reference data; centroid geometry and fallback; correlated MSP/entropy; empirical thresholds without guarantees; calibration bin sensitivity; no OOD claim from clean results. Cross-platform deterministic equivalence is not guaranteed. No statistical robustness from one seed. Literature review unfinished.

## 9. Conclusion
TBD after full experiments. Decide whether to revise distance/estimator, narrow research question, or justify larger-model replication. Do not infer performance from implementation success.

## 10. References
Primary papers and verified links are listed in Section 3. Further references: TBD.

## Follow-up architecture reference

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. **Deep Residual Learning for Image Recognition**. arXiv technical report, 2015. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Verified against the primary arXiv record on 2026-10-14. The local ResNet-18 uses a documented CIFAR stem adaptation, not the original ImageNet preprocessing.
