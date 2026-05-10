"""
Download and prepare the PAPILA retinal fundus dataset.

This helper script fetches the PAPILA archive from figshare, extracts it, and
produces a simplified `data/papila/clinical_data.csv` along with a flat
`data/papila/images/` directory that the main pipeline expects.

Dataset source (Kovalyk et al., 2022):
    https://doi.org/10.6084/m9.figshare.14798004.v2

PAPILA raw clinical metadata is provided as two Excel workbooks (OD and OS
eyes). We merge them into a single per-patient record and binarise the
diagnosis (0 = healthy, 1 = glaucoma or glaucoma-suspect).

Usage:
    python src/download_papila.py
"""

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ZIP_FILE = DATA_DIR / "PAPILA.zip"
EXTRACT_DIR = DATA_DIR / "papila_raw"
TARGET_DIR = DATA_DIR / "papila"
TARGET_IMG_DIR = TARGET_DIR / "images"
CLINICAL_CSV = TARGET_DIR / "clinical_data.csv"

DOWNLOAD_URL = "https://ndownloader.figshare.com/files/35013982"


def downloadWithProgress(url: str, filename: Path) -> None:
    """Stream a URL to disk while displaying MB/percent progress."""
    print(f"Downloading PAPILA archive (~590 MB) to {filename} ...")

    def reportHook(blockNum, blockSize, totalSize):
        downloaded = blockNum * blockSize
        if totalSize > 0:
            pct = min(100, downloaded * 100 / totalSize)
            mbDown = downloaded / (1024 * 1024)
            mbTotal = totalSize / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:.1f}% ({mbDown:.1f}/{mbTotal:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, filename, reportHook)
    print("\nDownload complete.")


def extractArchive(zipPath: Path, extractTo: Path) -> None:
    """Extract the PAPILA zip archive if it has not been extracted already."""
    if extractTo.exists():
        print(f"{extractTo} already exists, skipping extraction.")
        return
    print(f"Extracting {zipPath} -> {extractTo}")
    with zipfile.ZipFile(zipPath, "r") as zf:
        zf.extractall(extractTo)


def findClinicalFiles(rootDir: Path) -> list:
    """Locate PAPILA clinical Excel/CSV files inside the extracted archive."""
    clinicalFiles = []
    for root, _dirs, files in os.walk(rootDir):
        rootLower = root.lower()
        for f in files:
            fpath = Path(root) / f
            if f.endswith((".xlsx", ".xls", ".csv")) and (
                "clinical" in f.lower() or "clinical" in rootLower
            ):
                clinicalFiles.append(fpath)
    return clinicalFiles


def findImageDirs(rootDir: Path) -> list:
    """Find directories containing PAPILA fundus images."""
    imgDirs = []
    for root, _dirs, files in os.walk(rootDir):
        rootLower = root.lower()
        if any(keyword in rootLower for keyword in ("fundusimages", "images")):
            jpgFiles = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if jpgFiles:
                imgDirs.append((Path(root), jpgFiles))
    return imgDirs


def buildClinicalCsv(clinicalFiles: list) -> pd.DataFrame:
    """
    Merge PAPILA OD and OS clinical workbooks into a per-patient record.

    PAPILA encoding:
        Gender    : 0 = Male, 1 = Female
        Diagnosis : 0 = Healthy, 1 = Glaucoma, 2 = Suspect

    We binarise diagnosis to {Healthy, Glaucoma-or-suspect} and take the worst
    label across both eyes (OD, OS) for each patient.
    """
    patientData = {}

    for cf in clinicalFiles:
        print(f"Reading clinical file: {cf}")
        try:
            if cf.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(cf, engine="openpyxl")
            else:
                df = pd.read_csv(cf)
        except Exception as exc:
            print(f"  error reading {cf}: {exc}")
            continue

        df.columns = [c.strip() for c in df.columns]

        # Best-effort column mapping since the raw workbook uses descriptive names.
        idCol, ageCol, genderCol, diagCol = None, None, None, None
        for col in df.columns:
            cl = col.lower()
            if idCol is None and ("id" in cl or "patient" in cl):
                idCol = col
            if ageCol is None and "age" in cl:
                ageCol = col
            if genderCol is None and ("gender" in cl or "sex" in cl):
                genderCol = col
            if diagCol is None and "diag" in cl:
                diagCol = col

        if idCol is None:
            idCol = df.columns[0]

        for _, row in df.iterrows():
            pid = row[idCol]
            if pd.isna(pid):
                continue
            pid = int(pid) if not isinstance(pid, str) else pid

            record = patientData.setdefault(pid, {})

            if ageCol and not pd.isna(row.get(ageCol, None)):
                record["Age"] = int(row[ageCol])
            if genderCol and not pd.isna(row.get(genderCol, None)):
                # PAPILA: 0 = Male, 1 = Female
                record["Gender"] = "Female" if row[genderCol] == 1 else "Male"
            if diagCol and not pd.isna(row.get(diagCol, None)):
                currentDiag = 1 if int(row[diagCol]) >= 1 else 0
                existing = record.get("Diagnosis")
                record["Diagnosis"] = max(existing, currentDiag) if existing is not None else currentDiag

    rows = [
        {"ID": pid, "Gender": data["Gender"], "Age": data["Age"], "Diagnosis": data["Diagnosis"]}
        for pid, data in sorted(patientData.items())
        if {"Age", "Gender", "Diagnosis"}.issubset(data)
    ]

    if not rows:
        raise RuntimeError("Could not build clinical_data.csv: no complete records found.")

    return pd.DataFrame(rows)


def copyImages(imgDirs: list, targetDir: Path) -> int:
    """Flatten PAPILA image subfolders into a single images directory."""
    targetDir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for imgRoot, imgFiles in imgDirs:
        for imgFile in imgFiles:
            src = imgRoot / imgFile
            dst = targetDir / imgFile
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    return copied


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download archive if missing.
    if not ZIP_FILE.exists():
        downloadWithProgress(DOWNLOAD_URL, ZIP_FILE)
    else:
        print(f"{ZIP_FILE} already exists, skipping download.")

    # 2. Extract archive.
    extractArchive(ZIP_FILE, EXTRACT_DIR)

    # 3. Build clinical_data.csv.
    clinicalFiles = findClinicalFiles(EXTRACT_DIR)
    if not clinicalFiles:
        raise RuntimeError(f"No clinical Excel/CSV files found under {EXTRACT_DIR}.")
    clinicalDf = buildClinicalCsv(clinicalFiles)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    clinicalDf.to_csv(CLINICAL_CSV, index=False)
    print(f"\nWrote {CLINICAL_CSV} ({len(clinicalDf)} patients)")
    print(f"  Gender    : {clinicalDf['Gender'].value_counts().to_dict()}")
    print(f"  Diagnosis : {clinicalDf['Diagnosis'].value_counts().to_dict()}")
    print(f"  Age range : {clinicalDf['Age'].min()} - {clinicalDf['Age'].max()}")

    # 4. Flatten images.
    imgDirs = findImageDirs(EXTRACT_DIR)
    if not imgDirs:
        raise RuntimeError(f"No fundus image directories found under {EXTRACT_DIR}.")
    copiedCount = copyImages(imgDirs, TARGET_IMG_DIR)
    print(f"Copied {copiedCount} images to {TARGET_IMG_DIR}")

    print("\nPAPILA dataset ready.")
    print("You can now run: python src/fairness_evaluation.py")


if __name__ == "__main__":
    main()
