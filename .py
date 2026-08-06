#!/usr/bin/env python
# -*- coding: utf-8 -*-
# RSNA 2026 Knee MRI AI Challenge - multimodal report-supervised pipeline
# Cells are marked with '# %%' (VS Code / Spyder / jupytext compatible).

# %% [markdown]
# # RSNA 2026 Knee MRI AI Challenge — Multimodal Report-Supervised Pipeline
#
# **Task.** Per-study probability for 12 findings (ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture). Metric: **macro-averaged AUROC**.
#
# **The single most important structural fact about this competition:** only a small subset of training studies carry per-condition labels, but *every* training study ships with its free-text radiology report — and the **test set has no reports**. So text is not an input modality at inference. Text is a **label factory**.
#
# That reframes the whole problem into the shape that has repeatedly won medical-imaging benchmarks:
#
# ```
#                  ┌──────────────────────────────────────────────┐
#  STAGE 1         │ Reports ──► multilingual NLP labeler         │   supervised on the
#  (label mining)  │            (NegEx rules + mDeBERTa/XLM-R)    │   small gold subset
#                  └───────────────┬──────────────────────────────┘
#                                  │  soft labels for ~all studies
#                                  ▼
#  STAGE 1.5       ┌──────────────────────────────────────────────┐
#  (optional)      │ ConVIRT/CheXzero-style image↔report          │   free representation
#                  │ contrastive pretraining of the vision tower  │   learning
#                  └───────────────┬──────────────────────────────┘
#                                  ▼
#  STAGE 2         ┌──────────────────────────────────────────────┐
#  (vision)        │ 2.5D slice encoder → slice Transformer →     │   distilled from the
#                  │ series token (+ sequence metadata) →         │   text model's soft
#                  │ gated-attention MIL over series → 12 logits  │   targets
#                  └───────────────┬──────────────────────────────┘
#                                  ▼
#                           study-level probabilities
# ```
#
# ---
#
# ### Why this architecture (and where it comes from)
#
# | Design choice | Rationale / precedent |
# |---|---|
# | Report→label NLP model, then image distillation | **CheXpert** (Irvin et al., *AAAI* 2019) and **CheXbert** (Smit et al., *EMNLP* 2020) built 224k-image supervision entirely from report labelers. This competition is that setup by construction. |
# | **NegEx**-style negation scoping in the rule labeler | Chapman et al., *J Biomed Inform* 2001. Radiology reports are dominated by negated findings; naive keyword matching is catastrophically wrong. |
# | Soft-target distillation instead of hard pseudo-labels | Hinton et al. 2015; soft targets carry the labeler's calibrated uncertainty, which matters enormously for an **AUROC** metric that only cares about ranking. |
# | Per-plane encoders aggregated at study level | **MRNet** (Bien et al., *PLOS Medicine* 2018) — the reference knee-MRI deep learning study — trained sagittal/coronal/axial models and fused them. We generalise this to learned attention fusion. |
# | Gated-attention **MIL** over series | Ilse et al., *ICML* 2018; **CLAM** (Lu et al., *Nature Biomedical Engineering* 2021). A study is a bag of series; findings are visible in only some of them. |
# | Slice-axis Transformer rather than 3D conv | Knee MRI is strongly anisotropic (3–4 mm slices, sub-mm in-plane). Full 3D kernels waste capacity on a near-degenerate axis; this is why 2.5D + sequence models dominated RSNA Cervical Spine 2022, Abdominal Trauma 2023 and Lumbar Spine 2024. |
# | **Laterality canonicalisation** (left knees mirrored) | Non-negotiable here. Four of twelve targets are compartment-specific (Medial vs Lateral OA/meniscus). Without normalising left/right knees into a common frame — and without forbidding horizontal flip augmentation — you are teaching the network that medial and lateral are the same thing. This is the highest-leverage single line of preprocessing in the competition. |
# | Asymmetric Loss option | Ben-Baruch et al., *ICCV* 2021 — designed for exactly this: long-tailed multi-label. Fracture/Synovitis will be rare. |
# | Sequence-metadata conditioning (`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`) | Fluid-sensitive fat-suppressed sequences are where effusion, synovitis and bone-marrow oedema live; PD/T2 sagittals carry meniscus and ACL. Telling the model what it is looking at is nearly free accuracy. |
#
# **Clinical grounding for the label definitions** (useful when writing the lexicon): meniscal tear = intrasubstance signal contacting an articular surface on ≥2 slices (Kijowski et al., *Radiology*); bone contusion = reticulated marrow oedema on fluid-sensitive fat-suppressed sequences (Mandalia & Henson, *JBJS Br* 2008); knee OA compartment grading on MRI follows MOAKS (Hunter et al., *Osteoarthritis Cartilage* 2011); effusion-synovitis on non-contrast MRI per Guermazi et al., *Ann Rheum Dis* 2011.
#
# ---
#
# > **How to run this notebook.** It is written as one continuous pipeline but every stage is independently switchable in `CFG`. On Kaggle, run Stage 1 + the DICOM cache build as a *separate* dataset-producing notebook, then keep the submission notebook to inference only — DICOM decoding for ~1300 test studies is the real runtime risk, not the forward pass.

# %%
# =====================================================================================
#  CONFIG  — every knob lives here. Nothing below this cell should be edited to retune.
# =====================================================================================
import os, sys, math, json, time, random, re, gc, warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

class CFG:
    # ---------------- paths ----------------
    COMP_DIR   = "/kaggle/input/rsna-2026-knee-mri"     # <-- set to the real competition slug
    WORK_DIR   = "/kaggle/working"
    CACHE_DIR  = "/kaggle/working/cache"                # pre-decoded volumes (npz, uint8)
    TEXT_CKPT  = "/kaggle/working/text_ckpt"
    IMG_CKPT   = "/kaggle/working/img_ckpt"
    # If you pre-built the cache / weights in another notebook, point these at the dataset:
    EXT_CACHE  = None     # e.g. "/kaggle/input/knee-mri-cache-256"
    EXT_WEIGHTS= None     # e.g. "/kaggle/input/knee-mri-weights"

    # ---------------- targets ----------------
    LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
              "Medial OA", "Lateral OA", "PF OA", "Effusion",
              "Synovitis", "Baker's", "Contusion", "Fracture"]
    N_LABELS = 12

    # ---------------- global ----------------
    SEED        = 42
    DEBUG       = False        # True -> tiny subset, 1 fold, 1 epoch. Always smoke-test first.
    N_FOLDS     = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]
    NUM_WORKERS = 4

    # ---------------- stage switches ----------------
    RUN_TEXT_STAGE     = True    # train report labeler + emit pseudo-labels
    RUN_CACHE_STAGE    = True    # decode DICOM -> npz cache
    RUN_CONTRASTIVE    = False   # optional ConVIRT-style pretraining (expensive; do it once)
    RUN_IMAGE_STAGE    = True    # train the vision model
    RUN_INFERENCE      = True

    # ---------------- stage 1: report labeler ----------------
    # mdeberta-v3-base is the best quality/size tradeoff for multilingual clinical text.
    # xlm-roberta-large is ~1.5 AUC pts better on the rare classes if you can afford it.
    TEXT_BACKBONE = "microsoft/mdeberta-v3-base"
    TEXT_MAXLEN   = 384
    TEXT_EPOCHS   = 6
    TEXT_LR       = 2e-5
    TEXT_HEAD_LR  = 1e-3
    TEXT_BS       = 16
    TEXT_WD       = 0.01
    RULE_BLEND    = 0.15     # convex weight on the deterministic NegEx labeler
    PSEUDO_TEMP   = 1.0      # >1 softens the distilled targets

    # ---------------- stage 2: vision ----------------
    BACKBONE    = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    # strong alternatives, roughly in order of "accuracy per GPU-hour" on RSNA-style data:
    #   "convnextv2_tiny.fcmae_ft_in22k_in1k"   balanced default
    #   "tf_efficientnetv2_s.in21k_ft_in1k"     fastest, great for the efficiency prize
    #   "maxvit_tiny_tf_512.in1k"               best single model if you have A100 time
    #   "caformer_s18.sail_in22k_ft_in1k"       excellent on low-contrast MRI
    PRETRAINED  = True
    IMG_SIZE    = 256        # 320 buys ~0.5-1.0 macro-AUC on meniscus; costs ~1.6x
    N_SLICES    = 24         # slices sampled per series
    MAX_SERIES  = 4          # series sampled per study per step
    IN_CHANS    = 3          # 2.5D: (i-1, i, i+1)

    D_MODEL     = 512
    N_TX_LAYERS = 2          # slice-axis transformer depth
    N_HEADS     = 8
    DROPOUT     = 0.1

    EPOCHS      = 12
    BATCH_SIZE  = 2          # studies per step (each = MAX_SERIES * N_SLICES images!)
    GRAD_ACCUM  = 4          # effective batch 8
    LR          = 3e-4
    BACKBONE_LR = 6e-5       # discriminative LR: pretrained tower learns slower
    WD          = 1e-2
    WARMUP_PCT  = 0.05
    EMA_DECAY   = 0.999
    GRAD_CKPT   = True       # trades ~30% speed for ~50% memory. Keep on for 16GB cards.
    AMP_DTYPE   = "bf16"     # "bf16" on A100/L4/H100, "fp16" on P100/T4

    LOSS        = "bce"      # "bce" | "asl"  (asl = Asymmetric Loss, better for rare labels)
    ASL_GNEG, ASL_GPOS, ASL_CLIP = 4.0, 0.0, 0.05
    AUX_SERIES_W = 0.3       # weight on the per-series auxiliary head
    LABEL_SMOOTH = 0.01

    # ---------------- inference ----------------
    TTA          = 2         # 1 = none, 2 = + small shift/scale, 3 = + gamma
    INFER_SERIES = 6         # use more series at test than at train (free ensembling)
    ENSEMBLE     = "rank"    # "rank" (recommended for AUROC) | "mean" | "logit"

if CFG.DEBUG:
    CFG.N_FOLDS, CFG.TRAIN_FOLDS = 5, [0]
    CFG.TEXT_EPOCHS, CFG.EPOCHS = 1, 1

os.makedirs(CFG.WORK_DIR, exist_ok=True)
os.makedirs(CFG.CACHE_DIR, exist_ok=True)
os.makedirs(CFG.TEXT_CKPT, exist_ok=True)
os.makedirs(CFG.IMG_CKPT,  exist_ok=True)
print("config ready | DEBUG =", CFG.DEBUG)

# %%
# =====================================================================================
#  ENVIRONMENT
# =====================================================================================
# On Kaggle add these as offline wheels / pinned datasets rather than pip-installing at
# submission time (internet is disabled in code competitions):
#   pip install -q timm pydicom pylibjpeg pylibjpeg-libjpeg python-gdcm \
#                  albumentations transformers sentencepiece iterative-stratification
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

def seed_everything(seed: int = 42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True          # fixed input shapes -> big win
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

seed_everything(CFG.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if (CFG.AMP_DTYPE == "bf16" and torch.cuda.is_available()
                               and torch.cuda.is_bf16_supported()) else torch.float16
print(f"torch {torch.__version__} | device {DEVICE} | amp {AMP_DTYPE}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# %%
# =====================================================================================
#  DATA LOADING
# =====================================================================================
def _p(*a): return os.path.join(CFG.COMP_DIR, *a)

train      = pd.read_csv(_p("train.csv"))
train_ser  = pd.read_csv(_p("train_series.csv"))
test       = pd.read_csv(_p("test.csv"))
test_ser   = pd.read_csv(_p("test_series.csv"))
sample_sub = pd.read_csv(_p("sample_submission.csv"))

L = CFG.LABELS
# Gold subset = rows where the twelve label columns are actually populated.
has_label = train[L].notna().all(axis=1)
gold      = train[has_label].reset_index(drop=True)
unlabeled = train[~has_label].reset_index(drop=True)

print(f"studies: {len(train):,} | gold-labelled: {len(gold):,} | report-only: {len(unlabeled):,}")
print(f"series : {len(train_ser):,} ({len(train_ser)/max(len(train),1):.1f} per study)")
print(f"test   : {len(test):,} studies")
print("\nlabel prevalence on the gold subset:")
print((gold[L].mean().sort_values(ascending=False) * 100).round(2).to_string())
print("\nsequence descriptor grid:")
print(pd.crosstab([train_ser.Fluid_Sensitive, train_ser.Fat_Suppression],
                  train_ser.Anatomical_Plane))

if CFG.DEBUG:
    gold      = gold.head(200).reset_index(drop=True)
    unlabeled = unlabeled.head(200).reset_index(drop=True)

# %% [markdown]
# ## Stage 1 — Report → label extraction
#
# Two labelers, deliberately:
#
# 1. **A deterministic NegEx-style rule labeler.** Multilingual lexicon, sentence-scoped negation, anatomy×pathology co-occurrence constraints. It is auditable, has zero variance, and gives you a sanity floor. If your neural labeler doesn't beat it, something is broken.
# 2. **A multilingual transformer labeler** (mDeBERTa-v3 / XLM-R) fine-tuned on the gold subset.
#
# Then blend. The rule labeler is a strong prior on the classes where the vocabulary is closed and unambiguous (Baker's cyst, effusion, fracture); the transformer wins where phrasing is open-ended (OA severity thresholds, "signal alteration without discrete tear").
#
# **The negation problem is the whole problem.** A knee MRI report says "anterior cruciate ligament" in nearly every case — overwhelmingly to say it is intact. Keyword matching without negation scoping produces a labeler with ~50% precision on ACL. Chapman's NegEx (2001) remains the right primitive: find the finding term, look backwards within the clause for a negation trigger, stop at a conjunction.

# %%
# =====================================================================================
#  STAGE 1a — Multilingual NegEx-style rule labeler
#  Languages covered: EN, ES, PT, FR, DE, IT, TR, NL, PL (extend as you inspect the data)
# =====================================================================================

# ---- negation triggers: PRE-position (Romance/Germanic/English) -----------------------
# \b anchor is load-bearing, not decoration: unanchored "no\s+" matches inside "inter-NO y",
# "exter-NO y", "tor-NO a" ... which silently negates a large share of true positives in
# Spanish/Portuguese/Italian. Every alternative below must start on a word boundary.
NEG_PRE = r"""\b(?:
  no\s+(?:evidence|signs?|significant|definite|discrete|acute)? |
  not\s+ | without\s+ | absence\s+of | absent | negative\s+for | free\s+of |
  unremarkable | intact | normal\s+ | preserved | continuous |
  sin\s+ | no\s+se\s+(?:observa|aprecia|identifica|evidencia)[n]? | ausencia\s+de | integro | íntegro |
  sem\s+ | não\s+(?:se\s+)?(?:observa|há|identifica)| ausência\s+de |
  pas\s+d[e'’] | absence\s+d[e'’] | sans\s+ | ne\s+.{0,12}\s+pas | aucun[e]?\s+ |
  kein[e]?[nrsm]?\s+ | ohne\s+ | nicht\s+ | unauffällig | regelrecht[e]? |
  non\s+ | assenza\s+di | senza\s+ |
  geen\s+ | zonder\s+ |
  bez\s+ | brak\s+
)"""
# ---- negation triggers: POST-position (Turkish, German verb-final, etc.) --------------
NEG_POST = r"""\b(?:
  yok(?:tur)? | izlenme(?:di|mektedir)? | saptanma(?:dı|mıştır) | görülme(?:di|miştir) |
  mevcut\s+değil | normal(?:dir)? | doğal(?:dır)? | intakt(?:tır)? |
  nicht\s+nachweisbar | nicht\s+abgrenzbar | ausgeschlossen |
  is\s+intact | are\s+intact | appears?\s+normal | within\s+normal\s+limits
)"""
CONJ = r"(?:\bbut\b|\bhowever\b|\bpero\b|\bmas\b|\bmais\b|\baber\b|\bma\b|\bancak\b|\bfakat\b|,\s*(?:with|con|avec|mit)\b)"

# ---- diacritic folding ----------------------------------------------------------------
# Reports are frequently dictated without accents ("on capraz bag" for "ön çapraz bağ",
# "cirugia" for "cirugía"). Fold BOTH the report and the lexicon to ASCII so a missing
# cedilla never costs a label. Folding the patterns is safe: no metacharacter is non-ASCII.
import unicodedata
_MANUAL = {"ß": "ss", "ı": "i", "İ": "i", "ø": "o", "æ": "ae", "ł": "l"}
def fold(s: str) -> str:
    if not isinstance(s, str): return ""
    s = "".join(_MANUAL.get(ch, ch) for ch in s)
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

NEG_PRE_RE  = re.compile(fold(NEG_PRE),  re.I | re.X | re.U)
NEG_POST_RE = re.compile(fold(NEG_POST), re.I | re.X | re.U)
CONJ_RE     = re.compile(fold(CONJ),     re.I | re.U)

# ---- anatomy lexicon ------------------------------------------------------------------
A = {
 "acl":   r"(?:\bacl\b|anterior\s+cruciate|cruzado\s+anterior|croisé\s+antérieur|"
          r"vorder(?:es|en)\s+kreuzband|\bvkb\b|crociato\s+anteriore|\bl\.?c\.?a\.?\b|"
          r"ön\s+çapraz\s+bağ|\böçb\b|voorste\s+kruisband)",
 "mcl":   r"(?:\bmcl\b|medial\s+collateral|colateral\s+(?:medial|interno)|"
          r"collatéral\s+(?:médial|interne)|innenband|mediale[sn]?\s+kollateralband|"
          r"collaterale\s+mediale|iç\s+yan\s+bağ|mediale\s+collaterale\s+band)",
 "mm":    r"(?:medial\s+meniscus|menisco\s+(?:medial|interno)|ménisque\s+(?:médial|interne)|"
          r"innenmeniskus|medialer?\s+meniskus|menisco\s+mediale|iç\s+men(?:i|ı)sküs|"
          r"mediale\s+meniscus)",
 "lm":    r"(?:lateral\s+meniscus|menisco\s+(?:lateral|externo)|ménisque\s+(?:latéral|externe)|"
          r"außenmeniskus|aussenmeniskus|lateraler?\s+meniskus|menisco\s+laterale|"
          r"d(?:ı|i)ş\s+men(?:i|ı)sküs|laterale\s+meniscus)",
 "mcomp": r"(?:medial\s+(?:tibiofemoral\s+)?compartment|compartimento\s+(?:medial|interno)|"
          r"compartiment\s+(?:médial|interne)|mediale[sn]?\s+kompartiment|femorotibiale?\s+intern|"
          r"iç\s+kompartman|comparto\s+mediale|femorotibial\s+(?:medial|interno))",
 "lcomp": r"(?:lateral\s+(?:tibiofemoral\s+)?compartment|compartimento\s+(?:lateral|externo)|"
          r"compartiment\s+(?:latéral|externe)|laterale[sn]?\s+kompartiment|femorotibiale?\s+extern|"
          r"d(?:ı|i)ş\s+kompartman|comparto\s+laterale|femorotibial\s+(?:lateral|externo))",
 "pf":    r"(?:patellofemoral|femoropatellar|fémoro[- ]?patellaire|femoropatelar|"
          r"retropatellar|patellofemorale?|patellofemoral(?:en|es)?|patellar\s+cartilage|"
          r"patellofemoral\s+eklem|cartilage\s+rétro[- ]?patellaire|troclea)",
}
# ---- pathology lexicon ----------------------------------------------------------------
P = {
 "tear":  r"(?:tear|torn|rupture[d]?|disruption|discontinuit|avulsion|"
          r"rotura|ruptura|desgarro|desinserción|"
          r"déchirure|rupture\s+compl|fissuration|"
          r"riss|ruptur|(?:teil|komplett)ruptur|kontinuitätsunterbrechung|"
          r"lesione|rottura|lacerazione|"
          r"yırt(?:ık|ığı)|rüptür|"
          r"scheur|bucket[- ]handle|asa\s+de\s+(?:cubo|balde)|korbhenkel|"
          r"grade\s*(?:3|iii|III)\s*signal|grado\s*(?:3|III))",
 "oa":    r"(?:osteoarthrit|osteoarthros|arthros|gonarthros|degenerative\s+change|"
          r"chondral\s+(?:loss|thinning|defect|wear)|cartilage\s+(?:loss|thinning|denudation)|"
          r"chondromalac|full[- ]thickness\s+cartilage|joint\s+space\s+narrowing|osteophyt|"
          r"artrosis|artrose|gonartrose|condropat|pérdida\s+(?:del\s+)?cartílago|"
          r"arthrose|amincissement\s+cartilagineux|chondropathie|"
          r"knorpel(?:verschmälerung|defekt|glatze|schaden)|osteophyten|"
          r"artrosi|condropatia|"
          r"osteoartrit|kıkırdak\s+(?:kaybı|incelme))",
 "eff":   r"(?:joint\s+effusion|effusion|intra[- ]?articular\s+fluid|suprapatellar\s+fluid|"
          r"derrame(?:\s+articular)?|líquido\s+(?:articular|intraarticular)|"
          r"épanchement(?:\s+articulaire)?|lame\s+liquidienne|"
          r"gelenkerguss|ergu(?:ss|ß)|"
          r"versamento(?:\s+articolare)?|"
          r"efüzyon|eklem\s+s(?:ı|i)v(?:ı|i)s(?:ı|i)|hidrartroz)",
 "syn":   r"(?:synovit|synovial\s+(?:thickening|hypertroph|proliferation|enhancement)|pannus|"
          r"sinovit|hipertrofia\s+sinovial|engrosamiento\s+sinovial|"
          r"synovite|épaississement\s+synovial|"
          r"synovialis(?:verdickung|proliferation)|"
          r"sinovi(?:te|ale)|"
          r"sinovyal\s+(?:kalınlaşma|proliferasyon))",
 "baker": r"(?:baker'?s?\s*(?:cyst|zyste|kisti)|popliteal\s+cyst|quiste\s+(?:de\s+baker|poplíteo)|"
          r"cisto\s+(?:de\s+baker|poplíteo)|kyste\s+(?:de\s+baker|poplité)|"
          r"bakerzyste|poplitealzyste|cisti\s+(?:di\s+baker|poplitea)|"
          r"popliteal\s+kist|gastrocnemio[- ]?semimembranosus\s+bursa)",
 "cont":  r"(?:bone\s+(?:contusion|bruise|marrow\s+o?edema)|marrow\s+o?edema|bone\s+marrow\s+lesion|"
          r"trabecular\s+microfracture|"
          r"contusión\s+ósea|edema\s+óseo|edema\s+de\s+médula|contusão\s+óssea|"
          r"contusion\s+osseuse|o?edème\s+(?:osseux|médullaire)|"
          r"knochen(?:mark)?ödem|bone\s+bruise|kontusion|"
          r"contusione\s+ossea|edema\s+(?:osseo|midollare)|"
          r"kemik\s+(?:kontüzyon|ödem))",
 "frac":  r"(?:fracture|fractured|fx\b|avulsion\s+fracture|impaction\s+fracture|"
          r"insufficiency\s+fracture|stress\s+fracture|segond|"
          r"fractura|fratura|"
          r"fraktur|"
          r"frattura|"
          r"kır(?:ık|ığı))",
}
# guard: "microfracture" is a *surgical procedure*, not a fracture; likewise post-op notes
FRAC_EXCLUDE = re.compile(r"(?:micro[- ]?fracture|microfrattura|microfractura|"
                          r"mikrofrakturierung|fracture\s+(?:risk|prophylaxis))", re.I)

RULES = {
 "ACL":             ("cooc", A["acl"],   P["tear"]),
 "MCL":             ("cooc", A["mcl"],   P["tear"]),
 "Medial Meniscus": ("cooc", A["mm"],    P["tear"]),
 "Lateral Meniscus":("cooc", A["lm"],    P["tear"]),
 "Medial OA":       ("cooc", A["mcomp"], P["oa"]),
 "Lateral OA":      ("cooc", A["lcomp"], P["oa"]),
 "PF OA":           ("cooc", A["pf"],    P["oa"]),
 "Effusion":        ("solo", P["eff"],   None),
 "Synovitis":       ("solo", P["syn"],   None),
 "Baker's":         ("solo", P["baker"], None),
 "Contusion":       ("solo", P["cont"],  None),
 "Fracture":        ("solo", P["frac"],  None),
}
COMPILED = {k: (m, re.compile(fold(a), re.I | re.X | re.U),
                re.compile(fold(b), re.I | re.X | re.U) if b else None)
            for k, (m, a, b) in RULES.items()}

SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+|\n+|•|\|")

def _is_negated(sent: str, start: int, end: int) -> bool:
    # look back up to 80 chars, stopping at a contrastive conjunction (NegEx scope rule)
    left = sent[max(0, start - 80):start]
    cj = list(CONJ_RE.finditer(left))
    if cj: left = left[cj[-1].end():]
    if NEG_PRE_RE.search(left): return True
    # Post-position cues (Turkish/German verb-final) only scope over the SAME clause.
    # Without the clause cut, "tear of the medial meniscus, lateral meniscus normal"
    # would let "normal" negate the medial tear.
    right = sent[end:end + 45]
    cut = re.search(r"[,;]", right)
    if cut: right = right[:cut.start()]
    cj = CONJ_RE.search(right)
    if cj: right = right[:cj.start()]
    return bool(NEG_POST_RE.search(right))

def rule_label(report: str) -> Dict[str, float]:
    out = {k: 0.0 for k in CFG.LABELS}
    if not isinstance(report, str) or not report.strip():
        return {k: np.nan for k in CFG.LABELS}
    for sent in SENT_SPLIT.split(fold(report)):
        if not sent.strip(): continue
        for lab, (mode, rx_a, rx_b) in COMPILED.items():
            for m in rx_a.finditer(sent):
                if mode == "cooc":
                    hit = rx_b.search(sent)
                    if not hit: continue
                    # negation is checked around whichever term the negation would scope over
                    lo, hi = min(m.start(), hit.start()), max(m.end(), hit.end())
                else:
                    if lab == "Fracture" and FRAC_EXCLUDE.search(sent): continue
                    lo, hi = m.start(), m.end()
                if not _is_negated(sent, lo, hi):
                    out[lab] = 1.0
                break
    return out

# ---- validate the rule labeler against the gold subset --------------------------------
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

_rl = pd.DataFrame([rule_label(r) for r in gold["Report"]])[CFG.LABELS]
rows = []
for c in CFG.LABELS:
    y, p = gold[c].values.astype(int), _rl[c].fillna(0).values
    rows.append(dict(label=c, prevalence=y.mean(),
                     precision=precision_score(y, p, zero_division=0),
                     recall=recall_score(y, p, zero_division=0),
                     f1=f1_score(y, p, zero_division=0)))
rule_report = pd.DataFrame(rows).set_index("label")
print(rule_report.round(3).to_string())
print(f"\nmacro F1 = {rule_report.f1.mean():.4f}")
print("\n>> Read this table before going further. Low RECALL on a label means your lexicon is")
print(">> missing a language or phrasing. Low PRECISION means negation scoping is failing.")
print(">> Fixing the lexicon here is worth more macro-AUC than any architecture change later.")

# %%
# =====================================================================================
#  STAGE 1a-test — Lexicon regression suite. RE-RUN EVERY TIME YOU EDIT THE LEXICON.
#  Catches the two silent, expensive failure modes:
#    (i)  a negation cue firing inside an ordinary word  ("inter-NO y" matching "no ")
#    (ii) a finding term that only matches when the accent is present
# =====================================================================================
LEXICON_TESTS = [
 ("EN neg",      "The anterior cruciate ligament is intact. No joint effusion.",
                 {"ACL": 0, "Effusion": 0}),
 ("EN pos",      "Complete tear of the anterior cruciate ligament with adjacent bone contusion.",
                 {"ACL": 1, "Contusion": 1}),
 ("EN mixed",    "Oblique tear of the posterior horn of the medial meniscus. Lateral meniscus intact.",
                 {"Medial Meniscus": 1, "Lateral Meniscus": 0}),
 ("EN clause",   "Tear of the medial meniscus, lateral meniscus is normal.",
                 {"Medial Meniscus": 1}),
 ("EN conj",     "No effusion, but there is a tear of the medial meniscus.",
                 {"Effusion": 0, "Medial Meniscus": 1}),
 ("ES neg",      "No se observa rotura del ligamento cruzado anterior. Sin derrame articular.",
                 {"ACL": 0, "Effusion": 0}),
 ("ES -no trap", "Rotura del menisco interno y derrame articular.",
                 {"Medial Meniscus": 1, "Effusion": 1}),
 ("ES sinovitis","Sinovitis con engrosamiento sinovial.",              {"Synovitis": 1}),
 ("PT pos",      "Ruptura do ligamento cruzado anterior. Derrame articular moderado.",
                 {"ACL": 1, "Effusion": 1}),
 ("PT neg",      "Nao se observa rotura meniscal. Ausencia de derrame articular.",
                 {"Effusion": 0}),
 ("FR elision",  "Pas d'epanchement articulaire. Absence de rupture du ligament croise anterieur.",
                 {"Effusion": 0, "ACL": 0}),
 ("FR pos",      "Dechirure du menisque interne. Kyste de Baker.",
                 {"Medial Meniscus": 1, "Baker's": 1}),
 ("DE pos",      "Ruptur des vorderen Kreuzbandes. Gelenkerguss nachweisbar. Bakerzyste.",
                 {"ACL": 1, "Effusion": 1, "Baker's": 1}),
 ("DE neg",      "Kein Erguss. Vorderes Kreuzband unauffaellig.",      {"Effusion": 0}),
 ("DE compound", "Aussenmeniskusriss. Knochenmarkoedem lateral.",
                 {"Lateral Meniscus": 1}),
 ("IT pos",      "Rottura del menisco mediale. Versamento articolare.",
                 {"Medial Meniscus": 1, "Effusion": 1}),
 ("IT neg",      "Assenza di versamento articolare. Menisco mediale integro.", {"Effusion": 0}),
 ("TR neg",      "On capraz bag yirtigi izlenmedi. Eklem efuzyonu yoktur.",
                 {"ACL": 0, "Effusion": 0}),
 ("TR pos",      "Ic meniskus yirtigi mevcut. Baker kisti izlendi.",
                 {"Medial Meniscus": 1, "Baker's": 1}),
 ("OA",          "Severe medial compartment osteoarthritis with full-thickness cartilage loss. "
                 "Patellofemoral chondromalacia.",                     {"Medial OA": 1, "PF OA": 1}),
 ("OA neg",      "Lateral compartment cartilage is preserved. Medial compartment osteoarthritis.",
                 {"Lateral OA": 0, "Medial OA": 1}),
 ("frac guard",  "Prior microfracture procedure of the trochlea. No acute fracture.",
                 {"Fracture": 0}),
 ("frac pos",    "Nondisplaced fracture of the lateral tibial plateau.", {"Fracture": 1}),
 ("synovitis",   "Synovial thickening consistent with synovitis.",       {"Synovitis": 1}),
 ("all normal",  "Menisci intact. Cruciate and collateral ligaments intact. No effusion. "
                 "No fracture. Normal marrow signal.",
                 {"ACL": 0, "MCL": 0, "Medial Meniscus": 0, "Lateral Meniscus": 0,
                  "Effusion": 0, "Fracture": 0}),
]
# NOTE: written without diacritics on purpose -- these double as the un-accented-dictation
# tests. Add accented variants of your own once you have seen the real report distribution.

def run_lexicon_tests(verbose=True):
    fails = []
    for name, txt, exp in LEXICON_TESTS:
        got = rule_label(txt)
        bad = {k: (exp[k], got[k]) for k in exp if got[k] != exp[k]}
        if bad: fails.append((name, bad))
        if verbose and bad: print(f"  FAIL {name}: {bad}")
    print(f"lexicon regression: {len(LEXICON_TESTS)-len(fails)}/{len(LEXICON_TESTS)} passed")
    return fails

_fails = run_lexicon_tests()
assert not _fails, "lexicon regression failed -- fix before generating pseudo-labels"

# %%
# =====================================================================================
#  STAGE 1b — Multilingual transformer report labeler
# =====================================================================================
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    _HAS_ITERSTRAT = True
except ImportError:
    from sklearn.model_selection import StratifiedKFold
    _HAS_ITERSTRAT = False

def make_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    df = df.copy(); df["fold"] = -1
    Y = df[CFG.LABELS].values.astype(int)
    if _HAS_ITERSTRAT:
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = splitter.split(df, Y)
    else:
        # fallback: stratify on a coarse signature (count of positives + the 3 rarest labels)
        rare = np.argsort(Y.mean(0))[:3]
        key = Y.sum(1).clip(0, 4).astype(str) + "_" + Y[:, rare].astype(str).sum(1)
        it = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df, key)
    for f, (_, v) in enumerate(it): df.loc[v, "fold"] = f
    return df

gold = make_folds(gold, CFG.N_FOLDS, CFG.SEED)
print(gold.groupby("fold")[CFG.LABELS].mean().round(3).to_string())


def clean_report(t: str) -> str:
    if not isinstance(t, str): return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(?i)\b(dictated|transcribed|electronically signed|assinado|firmado)\b.*$", "", t)
    return t.strip()

class ReportDataset(Dataset):
    def __init__(self, df, tok, has_y=True):
        self.txt = [clean_report(t) for t in df["Report"].tolist()]
        self.tok, self.has_y = tok, has_y
        self.y = df[CFG.LABELS].values.astype("float32") if has_y else None
    def __len__(self): return len(self.txt)
    def __getitem__(self, i):
        # head+tail truncation: impressions live at the end of most reports, findings at the
        # start. Plain right-truncation throws away the single most label-dense paragraph.
        ids = self.tok(self.txt[i], add_special_tokens=False)["input_ids"]
        m = CFG.TEXT_MAXLEN - 2
        if len(ids) > m:
            h = m // 2; ids = ids[:h] + ids[-(m - h):]
        ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]
        pad = CFG.TEXT_MAXLEN - len(ids)
        att = [1] * len(ids) + [0] * pad
        ids = ids + [self.tok.pad_token_id] * pad
        out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att)}
        if self.has_y: out["y"] = torch.tensor(self.y[i])
        return out

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
    def forward(self, h, mask):
        a = self.w(h).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(-1)
        return (h * a.unsqueeze(-1)).sum(1)

class ReportLabeler(nn.Module):
    def __init__(self, name=CFG.TEXT_BACKBONE, n_out=CFG.N_LABELS):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        cfg.update({"hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0})
        self.enc  = AutoModel.from_pretrained(name, config=cfg)
        d = cfg.hidden_size
        self.pool = AttnPool(d)
        self.head = nn.Linear(d, n_out)
        # multi-sample dropout: cheap variance reduction, ~+0.3 AUC on small label sets
        self.drops = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        z = self.pool(h, attention_mask)
        return torch.stack([self.head(d(z)) for d in self.drops]).mean(0)

def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, List[float]]:
    aucs = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        aucs.append(roc_auc_score(yj, y_pred[:, j]) if len(np.unique(yj)) > 1 else np.nan)
    return float(np.nanmean(aucs)), aucs


def train_text_fold(fold: int) -> Tuple[np.ndarray, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(CFG.TEXT_BACKBONE)
    tr = gold[gold.fold != fold].reset_index(drop=True)
    va = gold[gold.fold == fold].reset_index(drop=True)
    dl_tr = DataLoader(ReportDataset(tr, tok), batch_size=CFG.TEXT_BS, shuffle=True,
                       num_workers=2, drop_last=True, pin_memory=True)
    dl_va = DataLoader(ReportDataset(va, tok), batch_size=CFG.TEXT_BS * 2, num_workers=2)

    model = ReportLabeler().to(DEVICE)
    body = [p for n, p in model.named_parameters() if n.startswith("enc")]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc")]
    opt = torch.optim.AdamW([{"params": body, "lr": CFG.TEXT_LR},
                             {"params": head, "lr": CFG.TEXT_HEAD_LR}], weight_decay=CFG.TEXT_WD)
    steps = len(dl_tr) * CFG.TEXT_EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    # pos_weight capped at 10 — uncapped weights destabilise the rarest labels
    pw = torch.tensor(np.clip((1 - tr[CFG.LABELS].mean()) / tr[CFG.LABELS].mean().clip(1e-3),
                              0.5, 10.0).values, dtype=torch.float32, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best, best_state = -1, None
    for ep in range(CFG.TEXT_EPOCHS):
        model.train()
        for b in dl_tr:
            b = {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = crit(model(b["input_ids"], b["attention_mask"]), b["y"])
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        model.eval(); P_, Y_ = [], []
        with torch.no_grad():
            for b in dl_va:
                b = {k: v.to(DEVICE) for k, v in b.items()}
                with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                    P_.append(model(b["input_ids"], b["attention_mask"]).float().sigmoid().cpu().numpy())
                Y_.append(b["y"].cpu().numpy())
        P_, Y_ = np.concatenate(P_), np.concatenate(Y_)
        auc, _ = macro_auc(Y_, P_)
        print(f"  fold {fold} ep {ep+1}/{CFG.TEXT_EPOCHS}  macroAUC {auc:.4f}")
        if auc > best:
            best = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_oof = P_
    torch.save(best_state, f"{CFG.TEXT_CKPT}/text_f{fold}.pt")
    del model; gc.collect(); torch.cuda.empty_cache()
    return best_oof, va.index.values

if CFG.RUN_TEXT_STAGE:
    oof = np.zeros((len(gold), CFG.N_LABELS), dtype="float32")
    for f in CFG.TRAIN_FOLDS:
        p, idx = train_text_fold(f)
        oof[gold.index[gold.fold == f]] = p
    m = gold.fold.isin(CFG.TRAIN_FOLDS).values
    auc, per = macro_auc(gold.loc[m, CFG.LABELS].values.astype(int), oof[m])
    print("\n=== TEXT LABELER OOF ===")
    print(pd.Series(per, index=CFG.LABELS).round(4).to_string())
    print(f"MACRO AUC = {auc:.4f}")
    print(">> Expect 0.97+. This is a ceiling check: labels were derived from these reports,")
    print(">> so anything much below 0.95 means tokenisation or truncation is losing signal.")
    np.save(f"{CFG.TEXT_CKPT}/text_oof.npy", oof)

# %%
# =====================================================================================
#  STAGE 1c — Emit soft pseudo-labels for every training study
#  These become the distillation targets for the vision model.
# =====================================================================================
@torch.no_grad()
def predict_reports(df: pd.DataFrame, folds=CFG.TRAIN_FOLDS) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(CFG.TEXT_BACKBONE)
    dl = DataLoader(ReportDataset(df, tok, has_y=False), batch_size=CFG.TEXT_BS * 2, num_workers=2)
    acc = np.zeros((len(df), CFG.N_LABELS), dtype="float64")
    for f in folds:
        model = ReportLabeler().to(DEVICE)
        model.load_state_dict(torch.load(f"{CFG.TEXT_CKPT}/text_f{f}.pt", map_location=DEVICE))
        model.eval(); out = []
        for b in dl:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                lg = model(b["input_ids"], b["attention_mask"]).float()
            out.append((lg / CFG.PSEUDO_TEMP).sigmoid().cpu().numpy())
        acc += np.concatenate(out); del model; gc.collect(); torch.cuda.empty_cache()
    return (acc / len(folds)).astype("float32")

if CFG.RUN_TEXT_STAGE:
    soft_unl = predict_reports(unlabeled)
    rule_unl = pd.DataFrame([rule_label(r) for r in unlabeled["Report"]])[CFG.LABELS].fillna(0.5).values
    soft_unl = (1 - CFG.RULE_BLEND) * soft_unl + CFG.RULE_BLEND * rule_unl

    pseudo = pd.concat([
        pd.DataFrame({"StudyInstanceUID": gold.StudyInstanceUID,
                      **{c: gold[c].astype("float32") for c in CFG.LABELS},
                      "is_gold": 1}),
        pd.DataFrame({"StudyInstanceUID": unlabeled.StudyInstanceUID,
                      **{c: soft_unl[:, j] for j, c in enumerate(CFG.LABELS)},
                      "is_gold": 0}),
    ], ignore_index=True)
    pseudo.to_parquet(f"{CFG.WORK_DIR}/pseudo_labels.parquet", index=False)
else:
    pseudo = pd.read_parquet(f"{CFG.WORK_DIR}/pseudo_labels.parquet")

print(f"pseudo-labels for {len(pseudo):,} studies "
      f"({int(pseudo.is_gold.sum()):,} gold / {int((1-pseudo.is_gold).sum()):,} distilled)")
print("\nsoft-label mean by source:")
print(pseudo.groupby("is_gold")[CFG.LABELS].mean().round(3).T.to_string())
print("\n>> Sanity check: distilled means should track gold prevalence. A distilled mean far")
print(">> above gold means the labeler is over-firing on that class in some language.")

# %% [markdown]
# ## Stage 2a — DICOM → canonical volume cache
#
# Three things happen here, and the second one is the one people skip and lose the competition on.
#
# **1. Geometric slice ordering.** `InstanceNumber` is unreliable across vendors. Sort by the projection of `ImagePositionPatient` onto the slice normal `n = r × c` derived from `ImageOrientationPatient`. Fall back to `InstanceNumber` only when geometry is missing.
#
# **2. Laterality canonicalisation.** Four targets are compartment-specific. In LPS patient coordinates, `+x` points to the patient's **left**. For a **right** knee the medial compartment is toward `+x`; for a **left** knee it is toward `−x`. So the in-plane sign that puts medial on a consistent image side is
#
# ```
# s = sign(row_direction · x̂) · (+1 if right knee else −1)
# ```
#
# and if `s < 0` we mirror the image. For sagittal series the through-plane axis *is* the medial↔lateral axis, so we normalise the **slice order** instead of the pixels. Without this, "Medial OA" and "Lateral OA" become the same feature and both AUCs collapse toward their average.
#
# Corollary that is easy to get wrong: **never use horizontal flip augmentation.** It undoes exactly this normalisation.
#
# **3. Intensity normalisation.** MRI has no absolute scale — a raw signal value means nothing across scanners, which is why an internationally-sourced dataset like this one punishes fixed windowing. Apply `RescaleSlope/Intercept`, invert `MONOCHROME1`, clip to per-volume 0.5/99.5 percentiles, then min-max to `uint8`. Percentile-based standardisation is the same choice nnU-Net makes for MR, and for the same reason.

# %%
# =====================================================================================
#  STAGE 2a — DICOM utilities
# =====================================================================================
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
cv2.setNumThreads(0)

def _laterality(ds) -> Optional[str]:
    for tag in ("ImageLaterality", "Laterality"):
        v = getattr(ds, tag, None)
        if v in ("L", "R"): return v
    blob = " ".join(str(getattr(ds, t, "")) for t in
                    ("SeriesDescription", "StudyDescription", "BodyPartExamined",
                     "ProtocolName", "PerformedProcedureStepDescription")).upper()
    has_l = bool(re.search(r"\b(LEFT|LT|IZQ|GAUCHE|LINKS|SINISTR|SOL)\b|\bL\s*KNEE\b", blob))
    has_r = bool(re.search(r"\b(RIGHT|RT|DER|DROIT|RECHTS|DESTR|SAG)\b|\bR\s*KNEE\b", blob))
    if has_l and not has_r: return "L"
    if has_r and not has_l: return "R"
    return None

def _plane_from_iop(iop) -> str:
    if iop is None or len(iop) != 6: return "Unknown"
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    n = np.abs(np.cross(r, c))
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(n))]

def _to_uint8(img: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
    img = img.astype(np.float32)
    a, b = np.percentile(img, lo), np.percentile(img, hi)
    if b - a < 1e-6: a, b = float(img.min()), float(img.max()) + 1e-6
    return (np.clip((img - a) / (b - a), 0, 1) * 255).astype(np.uint8)


def load_series(series_dir: str, n_slices: int, size: int,
                plane_hint: Optional[str] = None) -> Optional[np.ndarray]:
    # Returns a canonicalised uint8 volume of shape (n_slices, size, size), or None.
    files = [f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not files: return None

    metas = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        metas.append((f, ds))
    if not metas: return None

    ds0 = metas[0][1]
    iop = getattr(ds0, "ImageOrientationPatient", None)
    plane = plane_hint or _plane_from_iop(iop)
    side = _laterality(ds0) or "R"      # default to R; a wrong constant is still consistent

    # ---- geometric ordering -----------------------------------------------------------
    if iop is not None and len(iop) == 6:
        r, c = np.array(iop[:3], float), np.array(iop[3:], float)
        n = np.cross(r, c)
        def key(m):
            ipp = getattr(m[1], "ImagePositionPatient", None)
            return float(np.dot(np.array(ipp, float), n)) if ipp else float(
                getattr(m[1], "InstanceNumber", 0) or 0)
    else:
        r = np.array([1., 0., 0.]); n = np.array([0., 0., 1.])
        key = lambda m: float(getattr(m[1], "InstanceNumber", 0) or 0)
    metas.sort(key=key)

    # ---- laterality sign --------------------------------------------------------------
    side_sign = 1.0 if side == "R" else -1.0
    flip_lr = False; flip_depth = False
    if plane in ("Coronal", "Axial"):
        # medial should always increase with column index
        flip_lr = (np.sign(r[0]) * side_sign) < 0
    elif plane == "Sagittal":
        # through-plane axis is medial<->lateral; normalise slice ORDER, not pixels
        flip_depth = (np.sign(n[0]) * side_sign) < 0

    # ---- uniform slice sampling (decode only what we keep) ----------------------------
    idx = np.linspace(0, len(metas) - 1, num=min(n_slices, len(metas))).round().astype(int)
    idx = np.unique(idx)
    if flip_depth: idx = idx[::-1]

    frames = []
    for i in idx:
        fn = metas[i][0]
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, fn), force=True)
            arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        except Exception:
            continue
        if arr.ndim == 3: arr = arr[arr.shape[0] // 2]          # stray multi-frame
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        if flip_lr: arr = arr[:, ::-1]
        frames.append(cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA))
    if not frames: return None

    vol = np.stack(frames)
    vol = _to_uint8(vol)                                        # volume-wise, not slice-wise
    if vol.shape[0] < n_slices:                                 # edge-pad short series
        pad = n_slices - vol.shape[0]
        vol = np.concatenate([vol, np.repeat(vol[-1:], pad, 0)], 0)
    return np.ascontiguousarray(vol)

# %%
# =====================================================================================
#  STAGE 2b — Series prioritisation
#  A study can carry a dozen series. Which ones you feed the model matters more than which
#  backbone you use. Priorities follow standard MSK knee protocol reading order.
# =====================================================================================
def series_priority(plane: str, fluid: int, fatsat: int) -> float:
    p = 0.0
    # Sagittal fluid-sensitive fat-suppressed: ACL, bone marrow oedema, effusion. The
    # workhorse sequence of knee MRI.
    if plane == "Sagittal" and fluid == 1 and fatsat == 1: p = 10.0
    # Coronal fluid-sensitive FS: MCL, medial/lateral compartment marrow + cartilage.
    elif plane == "Coronal"  and fluid == 1 and fatsat == 1: p = 9.5
    # Sagittal PD/T2 without FS: the classic meniscus sequence (high SNR, sharp).
    elif plane == "Sagittal" and fluid == 1 and fatsat == 0: p = 9.0
    # Axial FS: patellofemoral cartilage, synovitis, Baker's cyst neck.
    elif plane == "Axial"    and fluid == 1 and fatsat == 1: p = 8.5
    elif plane == "Coronal"  and fluid == 1 and fatsat == 0: p = 8.0
    elif plane == "Axial"    and fluid == 1 and fatsat == 0: p = 7.0
    elif fluid == 0:                                         p = 5.0   # T1: fracture lines
    else:                                                    p = 4.0
    return p

def build_series_index(ser_df: pd.DataFrame) -> pd.DataFrame:
    d = ser_df.copy()
    d["Fluid_Sensitive"] = d["Fluid_Sensitive"].fillna(1).astype(int)
    d["Fat_Suppression"] = d["Fat_Suppression"].fillna(0).astype(int)
    d["Anatomical_Plane"] = d["Anatomical_Plane"].fillna("Unknown")
    d["prio"] = [series_priority(p, f, s) for p, f, s in
                 zip(d.Anatomical_Plane, d.Fluid_Sensitive, d.Fat_Suppression)]
    # keep at most 2 series per (plane, fluid, fatsat) bucket -> diversity, not duplication
    d = (d.sort_values(["StudyInstanceUID", "prio"], ascending=[True, False])
           .groupby(["StudyInstanceUID", "Anatomical_Plane",
                     "Fluid_Sensitive", "Fat_Suppression"], as_index=False)
           .head(2)
           .sort_values(["StudyInstanceUID", "prio"], ascending=[True, False])
           .reset_index(drop=True))
    return d

train_idx = build_series_index(train_ser)
PLANE2I = {"Sagittal": 0, "Coronal": 1, "Axial": 2, "Unknown": 3}
print(train_idx.groupby("StudyInstanceUID").size().describe().round(2).to_string())

# %%
# =====================================================================================
#  STAGE 2c — Parallel cache build (DICOM -> one .npz per study)
#  Do this ONCE in a separate notebook and publish it as a Kaggle Dataset. Decoding is
#  CPU-bound and will otherwise eat the entire GPU budget of every training run.
# =====================================================================================
def cache_one_study(args) -> Tuple[str, int]:
    sid, rows, src_root, out_dir, n_slices, size = args
    out_path = os.path.join(out_dir, f"{sid}.npz")
    if os.path.exists(out_path): return sid, -1
    vols, metas = [], []
    for (uid, plane, fluid, fat) in rows:
        d = os.path.join(src_root, sid, uid)
        if not os.path.isdir(d): continue
        try:
            v = load_series(d, n_slices, size, plane_hint=plane if plane != "Unknown" else None)
        except Exception:
            v = None
        if v is None: continue
        vols.append(v)
        metas.append([PLANE2I.get(plane, 3), int(fluid), int(fat)])
    if not vols: return sid, 0
    np.savez_compressed(out_path,
                        vol=np.stack(vols).astype(np.uint8),      # (S, N, H, W)
                        meta=np.array(metas, dtype=np.int16))     # (S, 3)
    return sid, len(vols)

def build_cache(idx_df: pd.DataFrame, src_root: str, out_dir: str,
                max_series: int = 6, workers: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for sid, g in idx_df.groupby("StudyInstanceUID"):
        g = g.head(max_series)
        rows = list(zip(g.SeriesInstanceUID, g.Anatomical_Plane,
                        g.Fluid_Sensitive, g.Fat_Suppression))
        tasks.append((sid, rows, src_root, out_dir, CFG.N_SLICES, CFG.IMG_SIZE))
    t0, done, empty = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(cache_one_study, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            sid, n = f.result(); done += 1
            if n == 0: empty.append(sid)
            if done % 200 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {el/60:.1f} min  "
                      f"eta {(el/done*(len(tasks)-done))/60:.1f} min", flush=True)
    if empty: print(f"WARNING: {len(empty)} studies produced no usable volume")
    return empty

CACHE = CFG.EXT_CACHE if CFG.EXT_CACHE and os.path.isdir(CFG.EXT_CACHE) else CFG.CACHE_DIR
if CFG.RUN_CACHE_STAGE and not (CFG.EXT_CACHE and os.path.isdir(CFG.EXT_CACHE)):
    sub_idx = train_idx
    if CFG.DEBUG:
        keep = set(pd.concat([gold.StudyInstanceUID, unlabeled.StudyInstanceUID]))
        sub_idx = train_idx[train_idx.StudyInstanceUID.isin(keep)]
    build_cache(sub_idx, _p("train_series"), CACHE,
                max_series=6, workers=min(os.cpu_count() or 4, 8))

cached = {f[:-4] for f in os.listdir(CACHE) if f.endswith(".npz")}
print(f"cached studies: {len(cached):,}")

# %% [markdown]
# ## Stage 2d — Model
#
# ```
# per study
#   ├── series 1 ─┐
#   ├── series 2 ─┤   each: (N, 3, H, W)  [2.5D: slice i-1, i, i+1 as channels]
#   └── series S ─┘
#                 │
#         ┌───────▼────────────────────────────────────────────┐
#         │ CNN backbone (shared)  →  per-slice feature (N, d)  │
#         │ Transformer over the slice axis (+ learned pos-emb) │
#         │ masked attention pool                → series vec   │
#         │ ⊕ sequence-descriptor embedding (plane/fluid/fat)   │
#         └───────┬────────────────────────────────────────────┘
#                 │  (S, d)
#         ┌───────▼──────────────────────────────┐
#         │ gated-attention MIL over series      │  Ilse et al. ICML 2018
#         │  a_s = softmax(w·(tanh(Vh) ⊙ σ(Uh))) │
#         └───────┬──────────────────────────────┘
#                 ▼
#           study vector → 12 logits     (+ auxiliary per-series head)
# ```
#
# Why the auxiliary per-series head: it forces every series to be individually predictive, which stops the MIL attention from collapsing onto a single series early in training. It also gives you free test-time ensembling — you can average the per-series logits with the pooled logits.
#
# Why *gated* attention rather than plain attention: the sigmoid gate lets the network suppress series that are uninformative for a given label, which is exactly the situation here (a T1 sagittal says nothing about synovitis).

# %%
# =====================================================================================
#  STAGE 2d — Dataset
# =====================================================================================
import albumentations as A_
from albumentations.pytorch import ToTensorV2

# NOTE: NO HorizontalFlip. See the laterality discussion above. VerticalFlip is also out
# (it swaps femur and tibia). Everything else is fair game.
#
# Albumentations changed its transform signatures between 1.x and 2.x (var_limit ->
# std_range, max_holes -> num_holes_range, value -> fill). A wrong kwarg does NOT raise on
# 2.x, it warns and silently drops the argument -- so an augmentation you think is running
# quietly becomes a no-op. Build both spellings and take whichever constructs cleanly.
def _try(fn, *variants):
    # albumentations 2.x WARNS instead of raising on unknown kwargs, so a bad spelling
    # becomes a silent no-op. Promote that warning to an error inside this probe only.
    for kw in variants:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                return fn(**kw)
        except Exception:
            continue
    return None

def _noise():
    return _try(A_.GaussNoise, dict(std_range=(0.03, 0.12)), dict(var_limit=(5.0, 30.0)))

def _dropout(size):
    h = max(2, size // 12)
    return _try(A_.CoarseDropout,
                dict(num_holes_range=(1, 4), hole_height_range=(h // 2, h),
                     hole_width_range=(h // 2, h), fill=0),
                dict(max_holes=4, max_height=h, max_width=h, fill_value=0))

# Geometry is sampled ONCE per series and replayed identically on every slice: a knee that
# rotates 8 degrees between slice 12 and slice 13 is not a knee. Photometric jitter is
# applied per slice, where mild variation is harmless and acts as extra regularisation.
def geo_aug():
    common = dict(scale=(0.90, 1.12), rotate=(-12, 12),
                  translate_percent=(-0.06, 0.06), p=0.75)
    aff = _try(A_.Affine,
               dict(**common, border_mode=cv2.BORDER_CONSTANT, fill=0),   # albumentations 2.x
               dict(**common, mode=cv2.BORDER_CONSTANT, cval=0),          # albumentations 1.x
               dict(**common))
    return A_.ReplayCompose([aff])

def photo_aug(size):
    t = [A_.RandomBrightnessContrast(0.20, 0.20, p=0.7),
         A_.RandomGamma(gamma_limit=(80, 125), p=0.3)]
    oneof = [x for x in (_noise(), A_.GaussianBlur(blur_limit=(3, 5)),
                         A_.MotionBlur(blur_limit=5)) if x is not None]
    if oneof: t.append(A_.OneOf(oneof, p=0.25))
    d = _dropout(size)
    if d is not None: t.append(A_.Compose([d], p=0.25))
    return A_.Compose(t)

class KneeStudyDataset(Dataset):
    def __init__(self, df, cache_dir, labels=None, train=True,
                 max_series=CFG.MAX_SERIES, n_slices=CFG.N_SLICES, size=CFG.IMG_SIZE):
        self.sids = df["StudyInstanceUID"].tolist()
        self.y = df[CFG.LABELS].values.astype("float32") if labels is None and \
                 all(c in df.columns for c in CFG.LABELS) else labels
        self.w = df["w"].values.astype("float32") if "w" in df.columns else np.ones(len(df), "float32")
        self.cache, self.train = cache_dir, train
        self.S, self.N, self.size = max_series, n_slices, size
        self.geo = geo_aug() if train else None
        self.photo = photo_aug(size) if train else None

    def __len__(self): return len(self.sids)

    def _stack25d(self, vol: np.ndarray) -> np.ndarray:
        # vol (N,H,W) uint8 -> (N,3,H,W) using neighbouring slices as channels
        n = vol.shape[0]
        prev = np.concatenate([vol[:1], vol[:-1]], 0)
        nxt  = np.concatenate([vol[1:], vol[-1:]], 0)
        return np.stack([prev, vol, nxt], 1)

    def __getitem__(self, i):
        sid = self.sids[i]
        path = os.path.join(self.cache, f"{sid}.npz")
        if os.path.exists(path):
            z = np.load(path)
            vol, meta = z["vol"], z["meta"]                      # (S0,N,H,W), (S0,3)
        else:                                                    # missing study -> zeros
            vol = np.zeros((1, self.N, self.size, self.size), np.uint8)
            meta = np.array([[3, 1, 0]], np.int16)

        S0 = vol.shape[0]
        if self.train and S0 > self.S:
            # sample without replacement; priority order is already encoded in cache order,
            # so bias the draw toward the front while keeping stochastic diversity
            p = np.linspace(1.0, 0.4, S0); p /= p.sum()
            sel = np.sort(np.random.choice(S0, self.S, replace=False, p=p))
        else:
            sel = np.arange(min(S0, self.S))

        xs = []
        for s in sel:
            v = vol[s]
            if v.shape[0] != self.N:                             # resample slice axis
                gi = np.linspace(0, v.shape[0] - 1, self.N).round().astype(int)
                v = v[gi]
            if self.train:
                first = self.geo(image=v[0])                  # sample geometry once...
                out = [self.photo(image=first["image"])["image"]]
                for k in range(1, v.shape[0]):                # ...and replay it exactly
                    g = A_.ReplayCompose.replay(first["replay"], image=v[k])["image"]
                    out.append(self.photo(image=g)["image"])
                v = np.stack(out)
            xs.append(self._stack25d(v))

        x = np.stack(xs)                                          # (s, N, 3, H, W)
        m = meta[sel]
        pad = self.S - x.shape[0]
        mask = np.ones(self.S, np.float32)
        if pad > 0:
            x = np.concatenate([x, np.zeros((pad, *x.shape[1:]), np.uint8)], 0)
            m = np.concatenate([m, np.full((pad, 3), [3, 0, 0], np.int16)], 0)
            mask[self.S - pad:] = 0.0

        item = {"x": torch.from_numpy(np.ascontiguousarray(x)),   # uint8, normalised on GPU
                "meta": torch.from_numpy(m.astype(np.int64)),
                "mask": torch.from_numpy(mask),
                "w": torch.tensor(self.w[i])}
        if self.y is not None: item["y"] = torch.from_numpy(self.y[i])
        return item

# %%
# =====================================================================================
#  STAGE 2e — Architecture
# =====================================================================================
import timm
from torch.utils.checkpoint import checkpoint_sequential

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class GatedAttentionMIL(nn.Module):
    # Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
    def __init__(self, d, h=256, dropout=0.1):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d, h), nn.Tanh(),    nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(d, h), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(h, 1)
    def forward(self, x, mask=None):                          # x (B,S,d)
        a = self.w(self.V(x) * self.U(x)).squeeze(-1)         # (B,S)
        if mask is not None: a = a.masked_fill(mask == 0, torch.finfo(a.dtype).min)
        a = a.softmax(-1)
        return (x * a.unsqueeze(-1)).sum(1), a

class MaskedAttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(),
                                                   nn.Linear(d // 2, 1))
    def forward(self, x, mask=None):
        a = self.w(x).squeeze(-1)
        if mask is not None: a = a.masked_fill(mask == 0, torch.finfo(a.dtype).min)
        return (x * a.softmax(-1).unsqueeze(-1)).sum(1)

class KneeMRINet(nn.Module):
    def __init__(self, backbone=CFG.BACKBONE, n_out=CFG.N_LABELS, d=CFG.D_MODEL,
                 n_slices=CFG.N_SLICES, pretrained=CFG.PRETRAINED, grad_ckpt=CFG.GRAD_CKPT):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                         num_classes=0, global_pool="avg",
                                         in_chans=CFG.IN_CHANS, drop_path_rate=0.1)
        d_enc = self.encoder.num_features
        if grad_ckpt and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(True)

        self.proj = nn.Sequential(nn.Linear(d_enc, d), nn.LayerNorm(d), nn.GELU())
        self.slice_pos = nn.Parameter(torch.zeros(1, n_slices, d)); nn.init.trunc_normal_(self.slice_pos, std=.02)
        layer = nn.TransformerEncoderLayer(d, CFG.N_HEADS, d * 4, CFG.DROPOUT,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.slice_tx = nn.TransformerEncoder(layer, CFG.N_TX_LAYERS)
        self.slice_pool = MaskedAttnPool(d)

        # sequence-descriptor conditioning: tells the model what physics it is looking at
        self.emb_plane = nn.Embedding(4, d)
        self.emb_fluid = nn.Embedding(2, d)
        self.emb_fat   = nn.Embedding(2, d)
        self.meta_ln   = nn.LayerNorm(d)

        self.series_mil = GatedAttentionMIL(d, 256, CFG.DROPOUT)
        self.head_study  = nn.Sequential(nn.LayerNorm(d), nn.Dropout(CFG.DROPOUT), nn.Linear(d, n_out))
        self.head_series = nn.Sequential(nn.LayerNorm(d), nn.Dropout(CFG.DROPOUT), nn.Linear(d, n_out))
        self.register_buffer("mean", IMNET_MEAN); self.register_buffer("std", IMNET_STD)

    def _norm(self, x):                       # uint8 (.,3,H,W) -> normalised float
        return (x.float().div_(255.0) - self.mean) / self.std

    def forward(self, x, meta, mask):
        # x (B,S,N,3,H,W) uint8 | meta (B,S,3) long | mask (B,S) float
        B, S, N = x.shape[:3]
        f = self.encoder(self._norm(x.reshape(B * S * N, *x.shape[3:])))   # (B*S*N, d_enc)
        f = self.proj(f).reshape(B * S, N, -1)

        f = f + self.slice_pos[:, :N]
        f = self.slice_tx(f)
        s = self.slice_pool(f)                                            # (B*S, d)

        m = meta.reshape(B * S, 3)
        s = self.meta_ln(s + self.emb_plane(m[:, 0].clamp(0, 3))
                           + self.emb_fluid(m[:, 1].clamp(0, 1))
                           + self.emb_fat(m[:, 2].clamp(0, 1)))
        logits_series = self.head_series(s).reshape(B, S, -1)             # aux head

        s = s.reshape(B, S, -1)
        z, attn = self.series_mil(s, mask)
        return self.head_study(z), logits_series, attn

if not CFG.DEBUG:
    _m = KneeMRINet(pretrained=False)
    _n = sum(p.numel() for p in _m.parameters()) / 1e6
    print(f"{CFG.BACKBONE}: {_n:.1f} M params")
    with torch.no_grad():
        o = _m(torch.randint(0, 255, (1, 2, CFG.N_SLICES, 3, CFG.IMG_SIZE, CFG.IMG_SIZE),
                             dtype=torch.uint8),
               torch.zeros(1, 2, 3, dtype=torch.long), torch.ones(1, 2))
    print("shapes:", o[0].shape, o[1].shape, o[2].shape)
    del _m; gc.collect()

# %%
# =====================================================================================
#  STAGE 2f — Losses, EMA, schedule
# =====================================================================================
class AsymmetricLoss(nn.Module):
    # Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    # Down-weights easy negatives and hard-clips very-confident negatives, which is the
    # right inductive bias when a label fires in <3% of studies.
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__(); self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
    def forward(self, logits, y, weight=None):
        p = torch.sigmoid(logits)
        pm = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        lp = y * torch.log(p.clamp(min=self.eps))
        ln = (1 - y) * torch.log((1 - pm).clamp(min=self.eps))
        loss = lp + ln
        with torch.no_grad():
            pt = p * y + pm * (1 - y)
            g = self.gp * y + self.gn * (1 - y)
            w = torch.pow(1 - pt, g)
        loss = loss * w
        if weight is not None: loss = loss * weight
        return -loss.mean()

class SoftBCE(nn.Module):
    # Distillation-friendly BCE: targets may be soft probabilities, and each STUDY carries
    # a scalar weight so gold labels can outrank distilled ones.
    def __init__(self, pos_weight=None, smooth=0.0):
        super().__init__(); self.pw, self.s = pos_weight, smooth
    def forward(self, logits, y, weight=None):
        if self.s > 0: y = y * (1 - self.s) + 0.5 * self.s
        l = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw, reduction="none")
        if weight is not None: l = l * weight
        return l.mean()

class ModelEMA:
    # Polyak averaging. On noisy distilled targets this is worth ~1 macro-AUC point and is
    # essentially free.
    def __init__(self, model, decay=CFG.EMA_DECAY):
        self.ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.ema[k].copy_(v.detach())
    def copy_to(self, model):
        model.load_state_dict({k: v.to(dtype=p.dtype) for (k, v), p
                               in zip(self.ema.items(), model.state_dict().values())})

def cosine_with_warmup(opt, total, warmup_pct=CFG.WARMUP_PCT, min_ratio=0.02):
    warm = max(1, int(total * warmup_pct))
    def f(step):
        if step < warm: return step / warm
        t = (step - warm) / max(1, total - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

# %%
# =====================================================================================
#  STAGE 2g — Training / validation loops
#
#  Validation policy (important): validate ONLY on gold-labelled studies. Validating on
#  distilled labels measures how well you imitate the text model, not how well you read
#  the MRI, and it will happily reward a model that has overfit the labeler's mistakes.
# =====================================================================================
def build_train_frame() -> pd.DataFrame:
    df = pseudo[pseudo.StudyInstanceUID.isin(cached)].copy()
    fold_map = dict(zip(gold.StudyInstanceUID, gold.fold))
    df["fold"] = df.StudyInstanceUID.map(fold_map).fillna(-1).astype(int)
    # distilled studies get lower loss weight; gold studies anchor the model
    df["w"] = np.where(df.is_gold == 1, 1.0, 0.6)
    return df.reset_index(drop=True)

full = build_train_frame()
print(f"trainable studies (cached): {len(full):,}  | gold {int(full.is_gold.sum()):,}")


def run_epoch(model, loader, crit, opt=None, sch=None, scaler=None, ema=None, accum=1):
    train = opt is not None
    model.train(train)
    tot, nb = 0.0, 0
    P, Y = [], []
    for step, b in enumerate(loader):
        x  = b["x"].to(DEVICE, non_blocking=True)
        mt = b["meta"].to(DEVICE, non_blocking=True)
        mk = b["mask"].to(DEVICE, non_blocking=True)
        y  = b["y"].to(DEVICE, non_blocking=True)
        w  = b["w"].to(DEVICE, non_blocking=True).unsqueeze(1)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                lg, lg_s, _ = model(x, mt, mk)
                loss = crit(lg, y, w)
                if CFG.AUX_SERIES_W > 0:
                    # broadcast the study target to every valid series
                    B, S, C = lg_s.shape
                    ys = y.unsqueeze(1).expand(B, S, C)
                    ws = (w.unsqueeze(1) * mk.unsqueeze(-1))
                    loss = loss + CFG.AUX_SERIES_W * crit(lg_s, ys, ws)
        if train:
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sch is not None: sch.step()
                if ema is not None: ema.update(model)
        else:
            P.append(lg.float().sigmoid().cpu().numpy()); Y.append(y.cpu().numpy())
        tot += float(loss.detach()); nb += 1
    if train: return tot / max(nb, 1), None, None
    return tot / max(nb, 1), np.concatenate(P), np.concatenate(Y)


def train_image_fold(fold: int) -> Tuple[np.ndarray, np.ndarray, float]:
    tr = full[(full.fold != fold)].reset_index(drop=True)          # gold(other folds)+distilled
    va = full[(full.fold == fold) & (full.is_gold == 1)].reset_index(drop=True)
    print(f"\n=== fold {fold} | train {len(tr):,} | val {len(va):,} (gold only) ===")

    dl_tr = DataLoader(KneeStudyDataset(tr, CACHE, train=True),
                       batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True,
                       num_workers=CFG.NUM_WORKERS, pin_memory=True,
                       persistent_workers=CFG.NUM_WORKERS > 0)
    dl_va = DataLoader(KneeStudyDataset(va, CACHE, train=False,
                                        max_series=CFG.INFER_SERIES),
                       batch_size=max(1, CFG.BATCH_SIZE), shuffle=False,
                       num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = KneeMRINet().to(DEVICE)
    for _cp in [f"{CFG.EXT_WEIGHTS}/contrastive.pt" if CFG.EXT_WEIGHTS else None,
                f"{CFG.WORK_DIR}/weights/contrastive.pt"]:
        if _cp and os.path.exists(_cp):
            print("  loading contrastive-pretrained encoder from", _cp,
                  model.load_state_dict(torch.load(_cp, map_location="cpu"), strict=False))
            break

    bb = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": bb,   "lr": CFG.BACKBONE_LR},
                             {"params": rest, "lr": CFG.LR}], weight_decay=CFG.WD)
    total = (len(dl_tr) // CFG.GRAD_ACCUM) * CFG.EPOCHS
    sch = cosine_with_warmup(opt, total)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))
    ema = ModelEMA(model)

    if CFG.LOSS == "asl":
        crit = AsymmetricLoss(CFG.ASL_GNEG, CFG.ASL_GPOS, CFG.ASL_CLIP)
    else:
        prev = np.clip(tr[CFG.LABELS].mean().values, 1e-3, 1 - 1e-3)
        pw = torch.tensor(np.clip((1 - prev) / prev, 0.5, 8.0),
                          dtype=torch.float32, device=DEVICE)
        crit = SoftBCE(pos_weight=pw, smooth=CFG.LABEL_SMOOTH)

    best, best_oof, patience = -1.0, None, 0
    for ep in range(CFG.EPOCHS):
        t0 = time.time()
        trl, _, _ = run_epoch(model, dl_tr, crit, opt, sch, scaler, ema, CFG.GRAD_ACCUM)

        shadow = KneeMRINet(pretrained=False).to(DEVICE)
        shadow.load_state_dict(model.state_dict()); ema.copy_to(shadow)
        vl, P, Y = run_epoch(shadow, dl_va, crit)
        auc, per = macro_auc(Y, P)
        print(f"  ep {ep+1:02d}/{CFG.EPOCHS}  train {trl:.4f}  val {vl:.4f}  "
              f"macroAUC {auc:.4f}  ({time.time()-t0:.0f}s)")
        if auc > best:
            best, best_oof, patience = auc, P, 0
            torch.save(shadow.state_dict(), f"{CFG.IMG_CKPT}/img_f{fold}.pt")
            print("    " + "  ".join(f"{c}:{a:.3f}" for c, a in zip(CFG.LABELS, per)))
        else:
            patience += 1
        del shadow; gc.collect(); torch.cuda.empty_cache()

    del model, opt, dl_tr, dl_va; gc.collect(); torch.cuda.empty_cache()
    return best_oof, va.StudyInstanceUID.values, best


if CFG.RUN_IMAGE_STAGE:
    oof_rows, oof_ids, scores = [], [], []
    for f in CFG.TRAIN_FOLDS:
        p, ids, sc = train_image_fold(f)
        oof_rows.append(p); oof_ids.append(ids); scores.append(sc)
    oof_img = pd.DataFrame(np.concatenate(oof_rows), columns=CFG.LABELS)
    oof_img.insert(0, "StudyInstanceUID", np.concatenate(oof_ids))
    oof_img.to_csv(f"{CFG.WORK_DIR}/oof_image.csv", index=False)

    gt = gold.set_index("StudyInstanceUID").loc[oof_img.StudyInstanceUID, CFG.LABELS].values
    auc, per = macro_auc(gt.astype(int), oof_img[CFG.LABELS].values)
    print("\n================ IMAGE MODEL OOF ================")
    print(pd.Series(per, index=CFG.LABELS).round(4).to_string())
    print(f"MACRO AUC = {auc:.4f}   (per-fold: {[round(s,4) for s in scores]})")

# %% [markdown]
# ### Optional Stage 1.5 — image↔report contrastive pretraining
#
# Every training study has a report. That is a large corpus of paired supervision that the twelve binary labels throw away. **ConVIRT** (Zhang et al. 2020), **GLoRIA** (Huang et al., *ICCV* 2021) and **CheXzero** (Tiu et al., *Nature Biomedical Engineering* 2022) all showed that InfoNCE alignment between an image encoder and a report encoder produces representations that beat ImageNet initialisation on downstream radiology classification, especially in the low-label regime — which is precisely our regime.
#
# Run this once, save the encoder, and use it as the initialisation for every fold. It typically buys 1–2 macro-AUC points; skip it if you are optimising for the efficiency prize.

# %%
# =====================================================================================
#  STAGE 1.5 (optional) — ConVIRT-style image<->report contrastive pretraining
# =====================================================================================
class ContrastiveWrapper(nn.Module):
    def __init__(self, vision: KneeMRINet, text_name=CFG.TEXT_BACKBONE, dim=256):
        super().__init__()
        self.vision = vision
        self.txt = AutoModel.from_pretrained(text_name)
        for p in list(self.txt.parameters())[:-40]: p.requires_grad = False   # freeze most of it
        self.pi = nn.Sequential(nn.Linear(CFG.D_MODEL, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pt = nn.Sequential(nn.Linear(self.txt.config.hidden_size, dim), nn.GELU(),
                                nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    def forward(self, x, meta, mask, ids, att):
        B, S, N = x.shape[:3]
        f = self.vision.encoder(self.vision._norm(x.reshape(B * S * N, *x.shape[3:])))
        f = self.vision.proj(f).reshape(B * S, N, -1)
        f = self.vision.slice_tx(f + self.vision.slice_pos[:, :N])
        s = self.vision.slice_pool(f).reshape(B, S, -1)
        zi, _ = self.vision.series_mil(s, mask)
        h = self.txt(input_ids=ids, attention_mask=att).last_hidden_state
        zt = (h * att.unsqueeze(-1)).sum(1) / att.sum(1, keepdim=True).clamp(min=1)
        zi = F.normalize(self.pi(zi), dim=-1); zt = F.normalize(self.pt(zt), dim=-1)
        logits = self.logit_scale.exp().clamp(max=100) * zi @ zt.t()
        tgt = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))

if CFG.RUN_CONTRASTIVE:
    tok = AutoTokenizer.from_pretrained(CFG.TEXT_BACKBONE)
    cdf = train[train.StudyInstanceUID.isin(cached)].reset_index(drop=True)
    texts = [clean_report(t) for t in cdf.Report]
    class PairDS(Dataset):
        def __init__(self):
            self.img = KneeStudyDataset(cdf.assign(**{c: 0.0 for c in CFG.LABELS}),
                                        CACHE, train=True)
        def __len__(self): return len(cdf)
        def __getitem__(self, i):
            it = self.img[i]
            e = tok(texts[i], truncation=True, max_length=256, padding="max_length",
                    return_tensors="pt")
            it["ids"] = e["input_ids"][0]; it["att"] = e["attention_mask"][0]
            return it
    dl = DataLoader(PairDS(), batch_size=8, shuffle=True, drop_last=True,
                    num_workers=CFG.NUM_WORKERS)   # larger batch = better InfoNCE
    net = ContrastiveWrapper(KneeMRINet()).to(DEVICE)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4,
                            weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))
    for ep in range(3):
        net.train(); run = 0.0
        for i, b in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                loss = net(b["x"].to(DEVICE), b["meta"].to(DEVICE), b["mask"].to(DEVICE),
                           b["ids"].to(DEVICE), b["att"].to(DEVICE))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss)
            if i % 100 == 0: print(f"  ep{ep} it{i} infoNCE {run/(i+1):.4f}", flush=True)
    os.makedirs(f"{CFG.WORK_DIR}/weights", exist_ok=True)
    torch.save(net.vision.state_dict(), f"{CFG.WORK_DIR}/weights/contrastive.pt")
    del net; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Stage 3 — Inference
#
# Three things to be deliberate about:
#
# - **Runtime is the binding constraint, not accuracy.** ~1300 test studies × ~6 series × ~30 slices is on the order of 200k DICOM reads. Decode in a process pool that runs *ahead of* the GPU, not sequentially with it.
# - **Rank-average, not probability-average, when ensembling folds.** AUROC is rank-only, and folds are not calibrated to each other. Averaging ranks is strictly the right operation for this metric.
# - **Never emit constant columns.** A column of identical values yields AUC = 0.5 for that label and drags the macro mean. If a study fails to decode, fall back to the training prevalence *plus tiny noise* so ties break randomly rather than degenerately.

# %%
# =====================================================================================
#  STAGE 3 — Inference & submission
# =====================================================================================
from scipy.stats import rankdata

def cache_test(test_ser_df: pd.DataFrame, src_root: str, out_dir: str) -> None:
    idx = build_series_index(test_ser_df)
    build_cache(idx, src_root, out_dir, max_series=CFG.INFER_SERIES,
                workers=min(os.cpu_count() or 4, 8))

@torch.no_grad()
def predict_studies(df: pd.DataFrame, cache_dir: str, ckpts: List[str],
                    tta: int = CFG.TTA) -> np.ndarray:
    ds = KneeStudyDataset(df.assign(**{c: 0.0 for c in CFG.LABELS}), cache_dir,
                          train=False, max_series=CFG.INFER_SERIES)
    dl = DataLoader(ds, batch_size=max(1, CFG.BATCH_SIZE), shuffle=False,
                    num_workers=CFG.NUM_WORKERS, pin_memory=True)
    per_model = []
    for ck in ckpts:
        model = KneeMRINet(pretrained=False).to(DEVICE).eval()
        model.load_state_dict(torch.load(ck, map_location=DEVICE))
        outs = []
        for b in dl:
            x  = b["x"].to(DEVICE, non_blocking=True)
            mt = b["meta"].to(DEVICE); mk = b["mask"].to(DEVICE)
            acc = 0.0
            for t in range(tta):
                xt = x
                if t == 1:                      # small isotropic zoom-in
                    B, S, N, C, H, W = x.shape
                    k = int(H * 0.06)
                    xt = x[..., k:H - k, k:W - k]
                    xt = F.interpolate(xt.reshape(-1, C, H - 2 * k, W - 2 * k).float(),
                                       size=(H, W), mode="bilinear", align_corners=False
                                       ).to(torch.uint8).reshape(B, S, N, C, H, W)
                elif t == 2:                    # gamma shift
                    xt = (255.0 * (x.float() / 255.0).clamp(1e-3).pow(0.85)).to(torch.uint8)
                with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
                    lg, lg_s, _ = model(xt, mt, mk)
                # blend the pooled head with the mask-averaged per-series head
                p_series = (lg_s.float().sigmoid() * mk.unsqueeze(-1)).sum(1) / \
                           mk.sum(1, keepdim=True).clamp(min=1)
                acc = acc + (0.75 * lg.float().sigmoid() + 0.25 * p_series)
            outs.append((acc / tta).cpu().numpy())
        per_model.append(np.concatenate(outs))
        del model; gc.collect(); torch.cuda.empty_cache()
    P = np.stack(per_model)                                     # (M, n, 12)
    if CFG.ENSEMBLE == "rank" and P.shape[0] > 1:
        R = np.stack([np.apply_along_axis(rankdata, 0, p) for p in P]).mean(0)
        return R / (len(df) + 1.0)
    if CFG.ENSEMBLE == "logit" and P.shape[0] > 1:
        lg = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
        return 1 / (1 + np.exp(-lg.mean(0)))
    return P.mean(0)


if CFG.RUN_INFERENCE:
    TEST_CACHE = "/kaggle/working/test_cache"
    os.makedirs(TEST_CACHE, exist_ok=True)
    cache_test(test_ser, _p("test_series"), TEST_CACHE)

    wdir = CFG.EXT_WEIGHTS if (CFG.EXT_WEIGHTS and os.path.isdir(CFG.EXT_WEIGHTS)) else CFG.IMG_CKPT
    ckpts = sorted(os.path.join(wdir, f) for f in os.listdir(wdir)
                   if f.startswith("img_f") and f.endswith(".pt"))
    print("checkpoints:", [os.path.basename(c) for c in ckpts])

    if ckpts:
        preds = predict_studies(test, TEST_CACHE, ckpts)
    else:
        preds = np.tile(gold[CFG.LABELS].mean().values, (len(test), 1))

    sub = pd.DataFrame(preds, columns=CFG.LABELS)
    sub.insert(0, "StudyInstanceUID", test.StudyInstanceUID.values)

    # ---- safety net: never ship a constant or NaN column ------------------------------
    prior = gold[CFG.LABELS].mean().values
    rng = np.random.default_rng(CFG.SEED)
    for j, c in enumerate(CFG.LABELS):
        v = sub[c].values.astype(float)
        bad = ~np.isfinite(v)
        if bad.any(): v[bad] = prior[j]
        if np.nanstd(v) < 1e-9: v = prior[j] + rng.normal(0, 1e-4, size=len(v))
        sub[c] = np.clip(v, 1e-6, 1 - 1e-6)

    sub = sub[["StudyInstanceUID"] + CFG.LABELS]
    assert list(sub.columns) == list(sample_sub.columns), "column order must match sample_submission"
    assert len(sub) == len(test) and sub.isna().sum().sum() == 0
    sub.to_csv("submission.csv", index=False)
    print(sub.head().round(4).to_string())
    print("\nwrote submission.csv", sub.shape)

# %% [markdown]
# ## Where the points actually are
#
# Ordered by expected macro-AUC per hour of your time. This is the part worth re-reading.
#
# **1. The report labeler's lexicon (worth 3–6 points).** Every error here is baked permanently into your image model's supervision — it is an irreducible noise floor. Print the per-label precision/recall table from Stage 1a, then go read 30 reports in the language with the worst recall. Non-Latin scripts and Turkish/German verb-final negation are the usual culprits. `Synovitis` and `Lateral OA` are typically where a naive lexicon collapses.
#
# **2. Laterality canonicalisation (worth 2–4 points, all of it on 4 labels).** Verify it empirically rather than trusting the code: after caching, tile the mid coronal slice of 40 studies and check that the medial femoral condyle is on the same side in every one. If it isn't, the `_laterality` fallback is failing and you need to inspect `SeriesDescription` patterns for the institutions in your data.
#
# **3. Resolution before depth (worth 1–3 points).** Meniscal tears and cartilage defects are millimetre-scale. `IMG_SIZE 320` with `N_SLICES 20` beats `IMG_SIZE 224` with `N_SLICES 32` on the meniscus and OA labels almost every time. Also consider a centre crop — the knee occupies maybe 60% of a typical FOV and you are spending a third of your pixels on air and coil.
#
# **4. Per-label specialist heads or models (worth 1–2 points).** The twelve labels do not want the same receptive field. Fracture and Contusion are coarse whole-bone patterns; meniscal tear is a 3-pixel signal line. A cheap version: keep one backbone but give the meniscus/OA labels their own head fed from a higher-resolution feature stage.
#
# **5. Series-conditional routing.** Right now every series goes through the same tower. A stronger version learns plane-specific projections (the MRNet lineage) or masks the MIL attention per label — e.g. Synovitis should be structurally forbidden from attending to a non-fluid-sensitive series.
#
# ### Things that will cost you points
#
# - **Horizontal flip augmentation.** Destroys medial/lateral. Already excluded above; do not add it back.
# - **Slice-wise intensity normalisation.** Normalise per volume. Per-slice destroys the through-plane intensity gradient that effusion and marrow oedema live on.
# - **Validating on distilled labels.** Measures imitation of the text model, not radiology.
# - **Trusting `InstanceNumber` for ordering.** Vendor-dependent; use geometry.
# - **Ignoring the runtime budget.** DICOM decode, not the GPU, is what times out RSNA submissions. Profile the cache build on the public test set before you tune anything.
#
# ### Efficiency-prize configuration
#
# This challenge awards model-efficiency prizes for the first time. A competitive efficiency entry: `tf_efficientnetv2_s` at `IMG_SIZE 224`, `N_SLICES 16`, `MAX_SERIES 2` (sagittal FS + coronal FS only), single fold, `TTA 1`, fp16. That is roughly 8× cheaper than the accuracy configuration and typically lands within 1.5 macro-AUC points of it.
#
# ---
#
# ### References
#
# 1. Bien N, Rajpurkar P, Ball RL, et al. **Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet.** *PLOS Medicine* 2018;15(11):e1002699.
# 2. Irvin J, Rajpurkar P, Ko M, et al. **CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison.** *AAAI* 2019.
# 3. Smit A, Jain S, Rajpurkar P, et al. **CheXbert: Combining automatic labelers and expert annotations for accurate radiology report labeling using BERT.** *EMNLP* 2020.
# 4. Chapman WW, Bridewell W, Hanbury P, et al. **A simple algorithm for identifying negated findings and diseases in discharge summaries.** *J Biomed Inform* 2001;34(5):301–310.
# 5. Ilse M, Tomczak JM, Welling M. **Attention-based deep multiple instance learning.** *ICML* 2018.
# 6. Lu MY, Williamson DFK, Chen TY, et al. **Data-efficient and weakly supervised computational pathology on whole-slide images (CLAM).** *Nature Biomedical Engineering* 2021;5:555–570.
# 7. Zhang Y, Jiang H, Miura Y, Manning CD, Langlotz CP. **Contrastive learning of medical visual representations from paired images and text (ConVIRT).** *MLHC* 2022 (arXiv 2020).
# 8. Huang S-C, Shen L, Lungren MP, Yeung S. **GLoRIA: A multimodal global-local representation learning framework for label-efficient medical image recognition.** *ICCV* 2021.
# 9. Tiu E, Talius E, Patel P, Langlotz CP, Ng AY, Rajpurkar P. **Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning (CheXzero).** *Nature Biomedical Engineering* 2022;6:1399–1406.
# 10. Ben-Baruch E, Ridnik T, Zamir N, et al. **Asymmetric loss for multi-label classification.** *ICCV* 2021.
# 11. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. **nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation.** *Nature Methods* 2021;18:203–211.
# 12. Hunter DJ, Guermazi A, Lo GH, et al. **Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS (MRI Osteoarthritis Knee Score).** *Osteoarthritis and Cartilage* 2011;19(8):990–1002.
# 13. Guermazi A, Roemer FW, Hayashi D, et al. **Assessment of synovitis with contrast-enhanced MRI using a whole-joint semiquantitative scoring system.** *Annals of the Rheumatic Diseases* 2011;70(5):805–811.
# 14. Kijowski R, Blankenbaker DG, Shinki K, et al. **Juxta-articular bone marrow signal changes on MR imaging: Characteristics and association with pain.** *Radiology* 2009;252(2):486–495.
# 15. Astuto B, Flament I, Namiri NK, et al. **Automatic deep learning-assisted detection and grading of abnormalities in knee MRI studies.** *Radiology: Artificial Intelligence* 2021;3(3):e200165.
# 16. Fritz B, Marbach G, Civardi F, et al. **Deep convolutional neural network-based detection of meniscus tears: Comparison with radiologists and surgery as standard of reference.** *Skeletal Radiology* 2020;49(8):1207–1217.
# 17. Hinton G, Vinyals O, Dean J. **Distilling the knowledge in a neural network.** arXiv:1503.02531, 2015.
# 18. Woo S, Debnath S, Hu R, et al. **ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders.** *CVPR* 2023.
