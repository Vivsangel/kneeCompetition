# kneeCompetition
RSA KNEE COMPETITION
RSNA 2026 Knee MRI AI Challenge — Multimodal Report-Supervised Pipeline

Multi-label detection of twelve knee-MRI findings (ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture). Metric: macro-averaged AUROC.

Core idea

Training studies ship with radiology reports; the test set does not. Text is therefore not an input modality — it is a label factory.

Stage 1 — label mining. A multilingual NegEx rule labeler plus an mDeBERTa-v3 classifier, trained on the small gold-labelled subset, emit soft labels for every report-only study.
Stage 1.5 (optional) — ConVIRT-style image↔report contrastive pretraining.
Stage 2 — vision. 2.5D slice encoder → slice-axis Transformer → series token (conditioned on plane / fluid-sensitivity / fat-suppression) → gated-attention MIL over series → 12 logits, distilled from the Stage 1 soft targets.
Layout
rsna2026_knee_mri_pipeline.py — full pipeline, # %% cell markers (VS Code / jupytext)
rsna2026_knee_mri_pipeline.ipynb — same content as a notebook

All configuration lives in the CFG class at the top. Set CFG.COMP_DIR to the competition data path and CFG.DEBUG = True for a smoke test.

Two things that decide the score

Laterality canonicalisation. Four targets are compartment-specific. In LPS coordinates medial is +x for a right knee and −x for a left one, so left knees are mirrored (coronal/axial) or slice-order-reversed (sagittal). Skip this and Medial OA and Lateral OA collapse into the same feature. Corollary: never use horizontal flip augmentation.

The report lexicon. Every labeling error is baked permanently into the vision model's supervision. Run the lexicon regression suite (run_lexicon_tests()) after any edit, and read the per-label precision/recall table before training anything.

Selected references

Bien et al., PLOS Med 2018 (MRNet) · Irvin et al., AAAI 2019 (CheXpert) · Smit et al., EMNLP 2020 (CheXbert) · Chapman et al., J Biomed Inform 2001 (NegEx) · Ilse et al., ICML 2018 (attention MIL) · Lu et al., Nat Biomed Eng 2021 (CLAM) · Tiu et al., Nat Biomed Eng 2022 (CheXzero) · Ben-Baruch et al., ICCV 2021 (ASL) · Hunter et al., Osteoarthritis Cartilage 2011 (MOAKS).

Licence

MIT
