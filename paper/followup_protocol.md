# Follow-up protocol, declared before new-seed and ResNet test comparisons

The seed-42 original test result was already inspected. This is a follow-up replication/ablation, not a previously unseen confirmatory study. Preserve original hyperparameters and all data roles. Do not tune C, distance scaling, architecture, training duration, or thresholds based on new test outcomes.

## Small-CNN replication

Seeds 42/43/44, 45,000 training / 5,000 validation / 10,000 test; validation roles 2,000 selection, 1,000 correct-reference statistics, 1,000 logistic fit, 1,000 threshold assessment. Seed changes training randomness and validation partition assignment. Seed 42 reuses its frozen original checkpoint. Epoch cap 15, patience 4, batch 64, AdamW lr 0.001, weight decay 0.0001, CPU four threads. Same official test identities for every run.

Ablation: standardized unweighted C=1 logistic regression using MSP+entropy versus MSP+entropy+distance. Fit both on the same 1,000-example fitting subset; thresholds use the same 1,000-example assessment subset. Distance statistics use only correctly classified reference examples. Prespecified primary differences: all-three minus uncertainty-only AUROC (positive favors distance) and AURC (negative favors distance). Report individual differences, mean, and sample SD across seeds; three seeds are insufficient for strong significance claims. Additional 50/70/80/90% risk metrics are secondary. No post-hoc feature/hyperparameter selection.

## ResNet compute feasibility

The local from-scratch CIFAR-stem ResNet-18 CPU benchmark suggests about 45 minutes per full-data epoch. MPS is unavailable. An optional compute-budget question was sent; absent a preference, the announced default is a limited CPU feasibility pilot: 5,000 stratified images selected exclusively from the 45,000 training partition, two epochs, batch 16, same seed-42 full validation and full test. The 15-epoch/full-data configuration remains separate and unexecuted unless selected explicitly.

To avoid comparing different training budgets, additionally run a small-CNN counterpart using exactly the pilot's 5,000 image indices, two epochs, batch 16, optimizer settings, validation partitions and test indices. Compare these pilot architectures only within this matched budget, and clearly distinguish them from the three full-data small-CNN runs. Training wall time remains different; matching sample/epoch budgets is not matching FLOPs or time. The limited training recipe is not optimized for ResNet, so it cannot establish a larger model's best achievable performance.

ResNet-18: no pretrained weights, CIFAR 3x3 stride-1 stem, no initial max pool, 512-D penultimate features, ten classes. Same 32x32 preprocessing/augmentation, AdamW and checkpoint-selection rule. No VGG implementation. Reuse the score/metric/schema interface, and rerun the same two-versus-three-signal logistic ablation separately for each pilot classifier.

Archive configs, environment versions, split indices, checkpoints, validation/test logits/features, score arrays, JSON/CSV metrics, actual-result PNG/PDF plots and report. Actual completion status and limitations must be reported without inventing results.
