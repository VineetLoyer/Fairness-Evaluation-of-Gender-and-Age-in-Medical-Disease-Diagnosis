"""
Fairness Evaluation of Gender and Age in Medical Disease Diagnosis
==================================================================

Authors : Vedant Bhenia, Prem Doshi, Vineet Kumar Loyer
Course  : DSCI 531 - Fair Informatics (University of Southern California, 2026)

Pipeline overview:
    Part 1 - UCI Heart Failure (tabular)
        * EDA and preprocessing
        * Baseline models (Logistic Regression, Gradient Boosting)
        * Subgroup fairness audit (gender + age)
        * Threshold sweep analysis
        * Mitigation: ThresholdOptimizer, ExponentiatedGradient
        * Interpretability: SHAP, LIME
    Part 2 - PAPILA Retinal Images (imaging)
        * ResNet18 transfer learning with augmentation
        * Gender- and age-stratified evaluation
        * Grad-CAM visualisation
    Part 3 - Comprehensive fairness summary table

Usage:
    python fairness_evaluation.py

Expected directory layout:
    ./data/heart.csv                 (UCI Heart Failure dataset)
    ./data/papila/clinical_data.csv  (PAPILA clinical metadata)
    ./data/papila/images/            (PAPILA fundus images)
    ./results/                       (figures will be written here)

Run `python src/download_papila.py` to fetch and prepare the PAPILA dataset.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    true_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve paths relative to the repo root so the script works from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEART_CSV = DATA_DIR / "heart.csv"
PAPILA_DIR = DATA_DIR / "papila"
PAPILA_IMG_DIR = PAPILA_DIR / "images"
PAPILA_CLINICAL_CSV = PAPILA_DIR / "clinical_data.csv"

RANDOM_SEED = 42
IMG_SIZE = 224                           # ResNet18 default input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]    # Pretrained-model normalisation
IMAGENET_STD = [0.229, 0.224, 0.225]
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def savePlot(figName: str) -> None:
    """Persist the current matplotlib figure into the results directory."""
    outPath = RESULTS_DIR / figName
    plt.savefig(outPath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {outPath}")


# ===========================================================================
# PART 1: UCI HEART FAILURE PREDICTION
# ===========================================================================
def runHeartPipeline() -> dict:
    """
    End-to-end fairness audit and mitigation on the UCI Heart Failure dataset.

    Returns
    -------
    dict
        Predictions, sensitive attributes, and ground truth needed by the
        final summary section.
    """
    print("\n" + "=" * 70)
    print("PART 1: UCI HEART FAILURE PREDICTION")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1.1 Load data
    # -----------------------------------------------------------------------
    heartDf = pd.read_csv(HEART_CSV)
    print(f"Dataset shape: {heartDf.shape}")

    # -----------------------------------------------------------------------
    # 1.2 Exploratory data analysis
    # -----------------------------------------------------------------------
    _plotHeartEda(heartDf)

    # -----------------------------------------------------------------------
    # 1.3 Preprocessing: label-encode categoricals, stratified split, scaling
    # -----------------------------------------------------------------------
    encoded = heartDf.copy()
    labelEncoder = LabelEncoder()
    for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
        encoded[col] = labelEncoder.fit_transform(encoded[col])

    X = encoded.drop(columns=["HeartDisease", "AgeGroup"], errors="ignore")
    y = encoded["HeartDisease"]
    sensitiveSex = heartDf["Sex"]  # keep original labels for readable reports

    X_train, X_test, y_train, y_test, sfTrain, sfTest = train_test_split(
        X, y, sensitiveSex,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------------------------------------------
    # 1.4 Baseline models
    # -----------------------------------------------------------------------
    print("\n[1.4] Training baseline models")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    lr.fit(X_train_scaled, y_train)
    yPredLr = lr.predict(X_test_scaled)
    yProbLr = lr.predict_proba(X_test_scaled)[:, 1]

    gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    gb.fit(X_train_scaled, y_train)
    yPredGb = gb.predict(X_test_scaled)
    yProbGb = gb.predict_proba(X_test_scaled)[:, 1]

    for name, yPred, yProb in [
        ("Logistic Regression", yPredLr, yProbLr),
        ("Gradient Boosting", yPredGb, yProbGb),
    ]:
        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy_score(y_test, yPred):.4f}")
        print(f"AUC     : {roc_auc_score(y_test, yProb):.4f}")
        print(classification_report(y_test, yPred))

    # -----------------------------------------------------------------------
    # 1.5 Subgroup fairness (by gender)
    # -----------------------------------------------------------------------
    print("\n[1.5] Subgroup fairness by gender")
    for name, yPred in [("Logistic Regression", yPredLr), ("Gradient Boosting", yPredGb)]:
        _reportSubgroup(name + " (gender)", y_test, yPred, sfTest)

    # -----------------------------------------------------------------------
    # 1.5b Subgroup fairness (by age)
    # -----------------------------------------------------------------------
    print("\n[1.5b] Subgroup fairness by age group")
    ageTest = encoded.loc[X_test.index, "Age"].values
    ageBins = pd.cut(ageTest, bins=[0, 50, 65, 100], labels=["<50", "50-65", "65+"])

    for name, yPred in [("Logistic Regression", yPredLr), ("Gradient Boosting", yPredGb)]:
        _reportSubgroup(name + " (age)", y_test, yPred, ageBins)

    _plotAgeFairness(y_test, yPredLr, yPredGb, ageBins)

    # -----------------------------------------------------------------------
    # 1.6 Mitigation: ThresholdOptimizer (post-processing)
    # -----------------------------------------------------------------------
    print("\n[1.6] Mitigation: ThresholdOptimizer")
    thresholdOpt = ThresholdOptimizer(
        estimator=lr,
        constraints="equalized_odds",
        prefit=True,
        predict_method="predict_proba",
    )
    thresholdOpt.fit(X_train_scaled, y_train, sensitive_features=sfTrain)
    yPredTo = thresholdOpt.predict(X_test_scaled, sensitive_features=sfTest)
    _reportSubgroup("ThresholdOptimizer", y_test, yPredTo, sfTest)
    _plotMitigationComparison(y_test, yPredLr, yPredTo, sfTest)

    # -----------------------------------------------------------------------
    # 1.7 Mitigation: ExponentiatedGradient (in-processing)
    # -----------------------------------------------------------------------
    print("\n[1.7] Mitigation: ExponentiatedGradient")
    expGradient = ExponentiatedGradient(
        estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        constraints=EqualizedOdds(),
    )
    expGradient.fit(X_train_scaled, y_train, sensitive_features=sfTrain)
    yPredEg = expGradient.predict(X_test_scaled)
    _reportSubgroup("ExponentiatedGradient", y_test, yPredEg, sfTest)

    # -----------------------------------------------------------------------
    # 1.8 Interpretability: SHAP + LIME
    # -----------------------------------------------------------------------
    print("\n[1.8] Interpretability (SHAP / LIME)")
    _runShap(lr, gb, X_train_scaled, X_test_scaled, X.columns.tolist())
    _runLime(lr, X_train_scaled, X_test_scaled, y_test, yPredLr, X.columns.tolist())

    return {
        "y_test": y_test,
        "yPredLr": yPredLr,
        "yPredGb": yPredGb,
        "yPredTo": yPredTo,
        "yPredEg": yPredEg,
        "sfTest": sfTest,
        "ageBins": ageBins,
    }


def _plotHeartEda(heartDf: pd.DataFrame) -> None:
    """Quick 3-panel EDA view of the heart failure dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    heartDf.groupby(["Sex", "HeartDisease"]).size().unstack(fill_value=0).plot(
        kind="bar", ax=axes[0], color=["#4CAF50", "#F44336"], rot=0
    )
    axes[0].set_title("Heart Disease by Gender")
    axes[0].legend(["No Disease", "Disease"])

    heartDf["AgeGroup"] = pd.cut(
        heartDf["Age"],
        bins=[20, 30, 40, 50, 60, 70, 80],
        labels=["20-30", "30-40", "40-50", "50-60", "60-70", "70-80"],
    )
    heartDf.groupby(["AgeGroup", "HeartDisease"]).size().unstack(fill_value=0).plot(
        kind="bar", ax=axes[1], color=["#4CAF50", "#F44336"]
    )
    axes[1].set_title("Heart Disease by Age Group")
    axes[1].legend(["No Disease", "Disease"])
    axes[1].tick_params(axis="x", rotation=45)

    heartDf["Sex"].value_counts().plot(
        kind="pie", ax=axes[2], autopct="%1.1f%%", colors=["#2196F3", "#FF9800"]
    )
    axes[2].set_title("Gender Distribution")
    axes[2].set_ylabel("")

    plt.tight_layout()
    savePlot("eda_heart_disease.png")


def _reportSubgroup(name, yTrue, yPred, sensitive) -> None:
    """Pretty-print per-subgroup metrics plus DPD/EOD."""
    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate,
        },
        y_true=yTrue,
        y_pred=yPred,
        sensitive_features=sensitive,
    )
    print(f"\n[{name}]")
    print(frame.by_group)
    print(f"  Demographic Parity Diff : {demographic_parity_difference(yTrue, yPred, sensitive_features=sensitive):.4f}")
    print(f"  Equalized Odds Diff     : {equalized_odds_difference(yTrue, yPred, sensitive_features=sensitive):.4f}")


def _plotAgeFairness(yTest, yPredLr, yPredGb, ageBins) -> None:
    """Side-by-side subgroup bars by age group."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for idx, (name, yPred) in enumerate([("Logistic Regression", yPredLr), ("Gradient Boosting", yPredGb)]):
        frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,
                "TPR": true_positive_rate,
                "FPR": false_positive_rate,
            },
            y_true=yTest,
            y_pred=yPred,
            sensitive_features=ageBins,
        )
        frame.by_group.plot(kind="bar", ax=axes[idx], rot=0)
        axes[idx].set_title(f"{name} - Fairness by Age Group")
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    savePlot("age_fairness_heart.png")


def _plotMitigationComparison(yTest, yPredBefore, yPredAfter, sfTest) -> None:
    """Gender-wise bar chart before and after ThresholdOptimizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, yPred) in zip(axes, [
        ("Before Mitigation (Logistic Regression)", yPredBefore),
        ("After ThresholdOptimizer (Equalized Odds)", yPredAfter),
    ]):
        frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,
                "TPR": true_positive_rate,
                "FPR": false_positive_rate,
            },
            y_true=yTest,
            y_pred=yPred,
            sensitive_features=sfTest,
        )
        frame.by_group.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(title)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    savePlot("mitigation_comparison.png")


def _runShap(lr, gb, X_train_scaled, X_test_scaled, featureNames) -> None:
    """Global SHAP summary for both baseline models."""
    # Logistic Regression: linear explainer
    linearExplainer = shap.LinearExplainer(lr, X_train_scaled)
    shapLr = linearExplainer.shap_values(X_test_scaled)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shapLr, X_test_scaled, feature_names=featureNames, show=False)
    plt.title("SHAP Feature Importance (Logistic Regression)")
    plt.tight_layout()
    savePlot("shap_summary_lr.png")

    # Gradient Boosting: tree explainer
    treeExplainer = shap.TreeExplainer(gb)
    shapGb = treeExplainer.shap_values(X_test_scaled)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shapGb, X_test_scaled, feature_names=featureNames, show=False)
    plt.title("SHAP Feature Importance (Gradient Boosting)")
    plt.tight_layout()
    savePlot("shap_summary_gb.png")


def _runLime(lr, X_train_scaled, X_test_scaled, yTest, yPredLr, featureNames) -> None:
    """Local LIME explanation for one test sample."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train_scaled,
        feature_names=featureNames,
        class_names=["No Disease", "Disease"],
        mode="classification",
    )
    sampleIdx = 0
    exp = explainer.explain_instance(
        X_test_scaled[sampleIdx], lr.predict_proba, num_features=10
    )
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    plt.title(
        f"LIME Explanation - Sample {sampleIdx} "
        f"(True: {yTest.iloc[sampleIdx]}, Pred: {yPredLr[sampleIdx]})"
    )
    plt.tight_layout()
    savePlot("lime_explanation_sample.png")


# ===========================================================================
# PART 2: PAPILA RETINAL IMAGE CNN PIPELINE
# ===========================================================================
class RetinalDataset(Dataset):
    """PyTorch dataset with optional on-the-fly augmentation for PAPILA images."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = images        # shape (N, 3, H, W), pixel range [0, 1]
        self.labels = labels
        self.augment = augment
        self.augTransform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            imgPil = transforms.ToPILImage()(torch.tensor(img))
            imgPil = self.augTransform(imgPil)
            img = np.array(imgPil, dtype=np.float32).transpose(2, 0, 1) / 255.0

        # ImageNet normalisation per channel (required for pretrained ResNet18).
        imgTensor = torch.tensor(img, dtype=torch.float32)
        for c in range(3):
            imgTensor[c] = (imgTensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return imgTensor, torch.tensor(label, dtype=torch.float32)


class RetinalCnn(nn.Module):
    """ResNet18 backbone with a lightweight classification head for glaucoma."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Drop the original 1000-class FC layer.
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze the earliest blocks (stem + layer1); fine-tune layer2-4.
        for name, param in self.features.named_parameters():
            if name.startswith(("0.", "1.", "2.", "3.", "4.")):
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(1)  # raw logits -> BCEWithLogitsLoss


def loadPapilaData(imgDir: Path, clinicalCsv: Path, imgSize: int = IMG_SIZE):
    """Load PAPILA metadata + both eyes (OD/OS) per patient."""
    clinical = pd.read_csv(clinicalCsv)
    images, labels, genders, ages, ids = [], [], [], [], []

    for _, row in clinical.iterrows():
        imgId = row["ID"]
        numStr = imgId[1:] if isinstance(imgId, str) and imgId.startswith("#") else f"{int(imgId):03d}"
        for eye in ["OD", "OS"]:
            imgPath = imgDir / f"RET{numStr}{eye}.jpg"
            if not imgPath.exists():
                continue
            img = Image.open(imgPath).convert("RGB").resize((imgSize, imgSize))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            images.append(arr)
            labels.append(row["Diagnosis"])
            genders.append(row["Gender"])
            ages.append(row["Age"])
            ids.append(f"{imgId}_{eye}")

    return np.array(images), np.array(labels), np.array(genders), np.array(ages), ids


def runPapilaPipeline() -> dict:
    """CNN training, fairness evaluation, and Grad-CAM on PAPILA images."""
    print("\n" + "=" * 70)
    print("PART 2: PAPILA RETINAL IMAGE DATASET")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    if not PAPILA_CLINICAL_CSV.exists():
        raise FileNotFoundError(
            f"PAPILA clinical data not found at {PAPILA_CLINICAL_CSV}. "
            "Run `python src/download_papila.py` to prepare the dataset."
        )

    Ximg, yImg, genderImg, ageImg, imgIds = loadPapilaData(
        PAPILA_IMG_DIR, PAPILA_CLINICAL_CSV
    )
    print(f"Loaded {len(Ximg)} images")
    print(f"Class distribution : {pd.Series(yImg).value_counts().to_dict()}")
    print(f"Gender distribution: {pd.Series(genderImg).value_counts().to_dict()}")

    _plotPapilaEda(ageImg, genderImg, yImg)

    # -----------------------------------------------------------------------
    # Patient-level train/test split (avoids leakage between OD and OS).
    # -----------------------------------------------------------------------
    clinical = pd.read_csv(PAPILA_CLINICAL_CSV)
    patientIds = clinical["ID"].values
    patientLabels = clinical["Diagnosis"].values
    pidTrain, pidTest = train_test_split(
        patientIds, test_size=0.2, random_state=RANDOM_SEED, stratify=patientLabels
    )
    pidTrainSet = set(pidTrain)
    trainMask = np.array([iid.rsplit("_", 1)[0] in pidTrainSet for iid in imgIds])
    testMask = ~trainMask

    XTrainImg, XTestImg = Ximg[trainMask], Ximg[testMask]
    yTrainImg, yTestImg = yImg[trainMask], yImg[testMask]
    gTest, aTest = genderImg[testMask], ageImg[testMask]

    # 80/20 train/validation inside the training portion.
    np.random.seed(RANDOM_SEED)
    idx = np.arange(len(XTrainImg))
    np.random.shuffle(idx)
    splitPoint = int(0.8 * len(idx))
    trainIdx, valIdx = idx[:splitPoint], idx[splitPoint:]

    trainLoader = DataLoader(
        RetinalDataset(XTrainImg[trainIdx], yTrainImg[trainIdx], augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    valLoader = DataLoader(
        RetinalDataset(XTrainImg[valIdx], yTrainImg[valIdx], augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    testLoader = DataLoader(
        RetinalDataset(XTestImg, yTestImg, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # -----------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # -----------------------------------------------------------------------
    model = RetinalCnn().to(DEVICE)
    nPos = yTrainImg.sum()
    nNeg = len(yTrainImg) - nPos
    posWeight = torch.tensor([nNeg / nPos], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=posWeight)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = _trainCnn(model, trainLoader, valLoader, criterion, optimizer, scheduler)
    _plotTrainingCurves(history)

    # -----------------------------------------------------------------------
    # Evaluation (gender + age)
    # -----------------------------------------------------------------------
    yProbCnn, yPredCnn = _evaluateCnn(model, testLoader, yTestImg)
    _reportSubgroup("CNN (gender)", yTestImg, yPredCnn, pd.Series(gTest))
    _plotCnnFairness(yTestImg, yPredCnn, gTest, "cnn_fairness_by_gender.png", "Gender")

    ageBinsCnn = pd.cut(aTest, bins=[0, 50, 65, 100], labels=["<50", "50-65", "65+"])
    _reportSubgroup("CNN (age)", yTestImg, yPredCnn, ageBinsCnn)
    _plotCnnFairness(yTestImg, yPredCnn, ageBinsCnn, "cnn_fairness_by_age.png", "Age Group")

    _runGradCam(model, XTestImg, yTestImg, yPredCnn, yProbCnn, gTest)

    return {
        "yTestImg": yTestImg,
        "yPredCnn": yPredCnn,
        "gTest": gTest,
        "ageBinsCnn": ageBinsCnn,
    }


def _plotPapilaEda(ageImg, genderImg, yImg) -> None:
    df = pd.DataFrame({"Age": ageImg, "Gender": genderImg, "Diagnosis": yImg})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for diag, color, label in [(0, "#4CAF50", "Healthy"), (1, "#F44336", "Glaucoma")]:
        axes[0].hist(df[df["Diagnosis"] == diag]["Age"], bins=15, alpha=0.6, color=color, label=label)
    axes[0].set_title("Age Distribution by Diagnosis")
    axes[0].set_xlabel("Age")
    axes[0].legend()

    df.groupby(["Gender", "Diagnosis"]).size().unstack(fill_value=0).plot(
        kind="bar", ax=axes[1], color=["#4CAF50", "#F44336"], rot=0
    )
    axes[1].set_title("Gender Distribution by Diagnosis")
    axes[1].legend(["Healthy", "Glaucoma"])

    plt.tight_layout()
    savePlot("eda_papila.png")


def _trainCnn(model, trainLoader, valLoader, criterion, optimizer, scheduler):
    """Standard training loop with best-on-validation checkpointing."""
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    bestValAcc = 0.0
    bestState = None

    for epoch in range(EPOCHS):
        # --- train ---
        model.train()
        trainLoss, correct, total = 0.0, 0, 0
        for xb, yb in trainLoader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        history["train_loss"].append(trainLoss / total)
        history["train_acc"].append(correct / total)

        # --- validate ---
        model.eval()
        valLoss, valCorrect, valTotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in valLoader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                valLoss += criterion(logits, yb).item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                valCorrect += (preds == yb).sum().item()
                valTotal += xb.size(0)
        history["val_loss"].append(valLoss / valTotal)
        history["val_acc"].append(valCorrect / valTotal)

        if history["val_acc"][-1] > bestValAcc:
            bestValAcc = history["val_acc"][-1]
            bestState = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:2d}/{EPOCHS} | "
                f"Train Loss {history['train_loss'][-1]:.4f} Acc {history['train_acc'][-1]:.4f} | "
                f"Val Loss {history['val_loss'][-1]:.4f} Acc {history['val_acc'][-1]:.4f}"
            )

    if bestState is not None:
        model.load_state_dict(bestState)
    print(f"Best validation accuracy: {bestValAcc:.4f}")
    return history


def _plotTrainingCurves(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_acc"], label="Train")
    axes[0].plot(history["val_acc"], label="Val")
    axes[0].set_title("CNN Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Val")
    axes[1].set_title("CNN Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    savePlot("cnn_training_curves.png")


def _evaluateCnn(model, testLoader, yTestImg):
    """Return sigmoid probabilities and threshold-0.5 predictions."""
    model.eval()
    logitsAll = []
    with torch.no_grad():
        for xb, _ in testLoader:
            logitsAll.append(model(xb.to(DEVICE)).cpu().numpy())
    logits = np.concatenate(logitsAll)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    print(f"\nCNN overall accuracy: {accuracy_score(yTestImg, preds):.4f}")
    print(f"CNN overall AUC     : {roc_auc_score(yTestImg, probs):.4f}")
    return probs, preds


def _plotCnnFairness(yTest, yPred, sensitive, figName, attrLabel) -> None:
    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate,
        },
        y_true=yTest,
        y_pred=yPred,
        sensitive_features=sensitive,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    frame.by_group.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(f"CNN Subgroup Fairness Metrics (by {attrLabel})")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savePlot(figName)


def _runGradCam(model, XTestImg, yTestImg, yPredCnn, yProbCnn, gTest) -> None:
    """Overlay Grad-CAM heatmaps on a diverse set of four test samples."""
    print("\n[2.7] Grad-CAM visualisation")

    # Pick one example for each (true, pred) combination where available.
    targets = [(0, 0), (1, 1), (0, 1), (1, 0)]
    indices = []
    for trueLabel, predLabel in targets:
        for j in range(len(yTestImg)):
            if yTestImg[j] == trueLabel and yPredCnn[j] == predLabel and j not in indices:
                indices.append(j)
                break
    # Pad with any remaining indices if some categories are absent.
    for j in range(len(yTestImg)):
        if len(indices) >= 4:
            break
        if j not in indices:
            indices.append(j)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, i in enumerate(indices[:4]):
        imgDisplay = XTestImg[i].transpose(1, 2, 0)
        imgTensor = torch.tensor(XTestImg[i], dtype=torch.float32)
        for c in range(3):
            imgTensor[c] = (imgTensor[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        imgTensor = imgTensor.unsqueeze(0)

        status = "correct" if yTestImg[i] == yPredCnn[i] else "wrong"
        axes[0, col].imshow(imgDisplay)
        axes[0, col].set_title(
            f"[{status}] True: {yTestImg[i]} Pred: {yPredCnn[i]}\n"
            f"Gender: {gTest[i]} | p={yProbCnn[i]:.2f}"
        )
        axes[0, col].axis("off")

        heatmap = _makeGradCamHeatmap(model, imgTensor)
        heatmapPil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
        heatmapResized = np.array(heatmapPil, dtype=np.float32) / 255.0
        axes[1, col].imshow(imgDisplay)
        axes[1, col].imshow(heatmapResized, cmap="jet", alpha=0.4)
        axes[1, col].set_title("Grad-CAM")
        axes[1, col].axis("off")

    plt.suptitle("Grad-CAM Visualizations on Retinal Images", fontsize=14)
    plt.tight_layout()
    savePlot("gradcam_visualization.png")


def _makeGradCamHeatmap(model, imgTensor) -> np.ndarray:
    """
    Grad-CAM over the last convolutional block (ResNet18 layer4).
    Returns a [0, 1]-normalised 2D heatmap.
    """
    model.eval()
    activations, gradients = {}, {}

    def forwardHook(_module, _inp, output):
        activations["value"] = output

    def backwardHook(_module, _gradIn, gradOut):
        gradients["value"] = gradOut[0]

    targetLayer = model.features[7]  # ResNet18 layer4
    fwdHandle = targetLayer.register_forward_hook(forwardHook)
    bwdHandle = targetLayer.register_full_backward_hook(backwardHook)

    imgTensor = imgTensor.to(DEVICE).requires_grad_(True)
    logits = model(imgTensor)
    model.zero_grad()
    logits.backward()

    fwdHandle.remove()
    bwdHandle.remove()

    grads = gradients["value"].cpu().data.numpy()[0]    # (C, H, W)
    acts = activations["value"].cpu().data.numpy()[0]   # (C, H, W)
    weights = np.mean(grads, axis=(1, 2))               # (C,)

    heatmap = np.zeros(acts.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        heatmap += w * acts[c]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap


# ===========================================================================
# PART 3: FINAL FAIRNESS SUMMARY
# ===========================================================================
def disparateImpact(yPred, sensitive) -> float:
    """Min/max selection-rate ratio across subgroups (four-fifths rule)."""
    rates = pd.Series(yPred).groupby(sensitive).mean()
    return rates.min() / rates.max() if rates.max() > 0 else 0.0


def buildSummary(heart: dict, papila: dict) -> pd.DataFrame:
    """Combine heart + PAPILA results into one comparison table."""
    yTest = heart["y_test"]
    sfTest = heart["sfTest"]
    ageBins = heart["ageBins"]
    yTestImg = papila["yTestImg"]
    gTest = papila["gTest"]
    ageBinsCnn = papila["ageBinsCnn"]

    rows = []

    # Gender rows
    for name, yPred in [
        ("LR (Baseline)", heart["yPredLr"]),
        ("GB (Baseline)", heart["yPredGb"]),
        ("LR + ThresholdOpt", heart["yPredTo"]),
        ("LR + ExpGradient", heart["yPredEg"]),
    ]:
        rows.append({
            "Model": name,
            "Attribute": "Gender",
            "Accuracy": accuracy_score(yTest, yPred),
            "DemParity Diff": demographic_parity_difference(yTest, yPred, sensitive_features=sfTest),
            "EqOdds Diff": equalized_odds_difference(yTest, yPred, sensitive_features=sfTest),
            "Disparate Impact": disparateImpact(yPred, sfTest.values),
        })
    rows.append({
        "Model": "CNN (Retinal)",
        "Attribute": "Gender",
        "Accuracy": accuracy_score(yTestImg, papila["yPredCnn"]),
        "DemParity Diff": demographic_parity_difference(yTestImg, papila["yPredCnn"], sensitive_features=gTest),
        "EqOdds Diff": equalized_odds_difference(yTestImg, papila["yPredCnn"], sensitive_features=gTest),
        "Disparate Impact": disparateImpact(papila["yPredCnn"], gTest),
    })

    # Age rows
    for name, yPred in [
        ("LR (Baseline)", heart["yPredLr"]),
        ("GB (Baseline)", heart["yPredGb"]),
    ]:
        rows.append({
            "Model": name,
            "Attribute": "Age",
            "Accuracy": accuracy_score(yTest, yPred),
            "DemParity Diff": demographic_parity_difference(yTest, yPred, sensitive_features=ageBins),
            "EqOdds Diff": equalized_odds_difference(yTest, yPred, sensitive_features=ageBins),
            "Disparate Impact": disparateImpact(yPred, ageBins.values),
        })
    rows.append({
        "Model": "CNN (Retinal)",
        "Attribute": "Age",
        "Accuracy": accuracy_score(yTestImg, papila["yPredCnn"]),
        "DemParity Diff": demographic_parity_difference(yTestImg, papila["yPredCnn"], sensitive_features=ageBinsCnn),
        "EqOdds Diff": equalized_odds_difference(yTestImg, papila["yPredCnn"], sensitive_features=ageBinsCnn),
        "Disparate Impact": disparateImpact(papila["yPredCnn"], ageBinsCnn.values),
    })

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("FAIRNESS COMPARISON SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    # Save as an image table for quick reference in the report / README.
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    table = ax.table(
        cellText=summary.round(4).values,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title("Fairness Evaluation Summary (Gender & Age)", fontsize=14, pad=20)
    plt.tight_layout()
    savePlot("fairness_summary_table.png")

    summary.to_csv(RESULTS_DIR / "fairness_summary.csv", index=False)
    print(f"  saved -> {RESULTS_DIR / 'fairness_summary.csv'}")
    return summary


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    heart = runHeartPipeline()
    papila = runPapilaPipeline()
    buildSummary(heart, papila)
    print("\nPipeline finished. Figures + summary are in the ./results directory.")


if __name__ == "__main__":
    main()
