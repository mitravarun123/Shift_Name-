# ShiftGuard Colab handoff

1. Open **ShiftGuard_Master_Colab.ipynb** in Google Colab (File → Upload notebook).
2. Choose a GPU runtime. Run the notebook cells in order.
3. At the source-upload prompt upload **shiftguard_colab_bundle.zip**. Upload this source ZIP, not a ZIP of datasets or old experiment outputs.
4. Authorize Drive mounting, review the run name/batch size, then run the training cell.
5. After a disconnect, reconnect to a GPU, upload the exact same source ZIP, mount the same Drive and reuse the same settings. Training resumes after the last committed epoch. Re-running a completed run reuses saved weights/features.
6. Inspect the results and figure gallery, then download the results ZIP. Files already persist on Drive.

## Fixed experiment

- From-scratch, CIFAR-stem ResNet-18, seed 42, all 40 epochs; AdamW and cosine learning rate. No pretrained weights downloaded.
- 45,000 training / 5,000 held-out training-pool images / 10,000 official test images.
- Holdout roles: selection 2,000; reference 1,000; estimator fit 1,000; risk calibration 1,000.
- Independent calibration is reserved without labels before stratifying the rest. Thus IDs differ from historical experiments while counts remain the same.
- Five methods: MSP, entropy, standardized predicted-class distance, MSP+entropy logistic regression, and all three logistic regression.
- Risk targets 5% and 10%; calibration budgets 250/500/1,000; 21 fit-data threshold quantiles/method; simultaneous 95% confidence using a finite-family Bonferroni/Clopper–Pearson construction.
- The theoretical claim is about true accepted population risk under IID calibration/deployment assumptions, not correctness of every image, a bound on this particular test sample, or arbitrary shifts. Fixed CIFAR benchmarking does not verify those sampling assumptions. The statistical construction is established, not claimed novel.
- No certificate means reject all, zero accepted coverage, and undefined conditional risk. It is an honest possible result, especially with few calibration labels.

## Figures: high-resolution PNG only

600 DPI, consistent fonts and colorblind-oriented palette, line styles, clear units/labels, subtle grids, annotated heatmaps, captions and saved calibration-bin counts. No invented cross-seed error bars.

1. ROC curves with AUROC in the legend.
2. Risk–coverage curves.
3. Annotated risk-at-50/70/80/90%-coverage heatmap.
4. Classifier and both logistic-estimator calibration diagrams.
5. Annotated ten-class confusion matrix.
6. Primary results table as PNG.
7. Controlled-risk accepted-coverage heatmap, including “not certified”.
8. Coverage versus calibration-label budget.
9. Risk-control audit table as PNG.
10. Training loss, selection accuracy and learning-rate diagnostics.

Numerical tables: CSV/JSON. Predicted labels, probabilities, scores and features: NPZ. No PDFs are exported by the Colab pipeline or its figure script. Local smoke previews carry a prominent **SMOKE TEST** label and are not full ResNet results.

Regenerate figures without training/inference:

```bash
python scripts/generate_paper_figures.py --run-dir /path/to/your/run
```

Optional frozen-rule corruption tests (no clean guarantee is asserted under shift):

```bash
python scripts/run_colab_corruptions.py --config /path/to/run/run_config.yaml --corruptions noise:1 blur:1 brightness:1 contrast:1
```

## Checkpoints and reproducibility

`last.pt` stores model, optimizer, scheduler, RNG and DataLoader sample-order states plus the completed epoch/history. `best.pt` stores the selection-chosen classifier. Epoch commits use temporary-file replacement on Drive. Only load your own trusted checkpoints/joblib artifacts. Cross-hardware/software bit-exactness is not promised; environment versions are recorded per session. Configuration/source mismatches fail explicitly; no silent batch, epoch, or dataset reductions occur.

CIFAR remains on the VM for faster I/O; it may need downloading after a runtime reset. Checkpoints, split indices, certificates, results and PNG figures persist in the dedicated Drive run folder. No paid service is required, but free Colab GPU availability/runtime duration are not guaranteed.

## What was tested locally

See `VALIDATION.md` in the bundle: unit tests, exact CPU interrupted-vs-uninterrupted recovery, notebook code syntax/format validation, ZIP bootstrap validation and actual CIFAR-10 end-to-end smoke evaluation/PNG rendering. The 40-epoch GPU run and interactive Google authentication require your Colab session; they are not claimed executed locally.

## Primary references

- Learn then Test: https://arxiv.org/abs/2110.01052
- Selective Classification for Deep Neural Networks: https://arxiv.org/abs/1705.08500
- ResNet: https://arxiv.org/abs/1512.03385
- PyTorch checkpoint guide: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- Colab resource limits: https://research.google.com/colaboratory/faq.html
