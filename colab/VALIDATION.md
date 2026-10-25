# Validation status

- Unit suite: 20 tests passed locally, including exact CPU uninterrupted-versus-interrupted epoch recovery, split disjointness, label-blind risk calibration, score direction, feature extraction, and exact binomial upper-bound coverage checks.
- Actual CIFAR-10 CPU smoke workflow: completed training, feature caches, five methods, risk calibration and final evaluation. This reduced smoke experiment is not a full ResNet result. Failure to certify a risk target is an expected valid result for the undertrained smoke classifier.
- Generated ten real-result figures: PNG only, with 600-DPI metadata and at least 4,000 horizontal pixels. No PDF was emitted. Layout samples were visually inspected, including annotated heatmaps, calibration diagrams and tables.
- Master notebook: nbformat schema valid; all code cells compile as Python. ZIP upload/extraction and manifest verification are checked locally with a mocked Colab upload interface.
- Full 40-epoch CUDA training and interactive Google Drive authorization require the user's Colab session and are not claimed executed here. GPU hardware and exact cross-session numeric reproducibility are not guaranteed.
