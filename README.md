# DC-CFL (experimental code)

This repository provides experimental code for DC-CFL style evaluation.

## Quick start

### Option A (recommended): Clone this repository

```bash
git clone https://github.com/soutasuga/DC-CFL.git
cd DC-CFL
```

### Option B: Download as ZIP

Download from GitHub: **Code** → **Download ZIP**, then unzip and open the folder.

## Python / environment

This project is tested with:

Python 3.10.19 (recommended: Anaconda / Miniconda environment)

NumPy 1.26.4

pandas 2.3.3

SciPy 1.15.3

scikit-learn 1.7.2

PyTorch 2.9.0 (CPU build)

If you use a newer Python version (e.g., 3.12+), some packages may not provide prebuilt wheels on Windows and installation can fail.

## Setup (recommended): conda + environment.yml

This is the most reproducible setup on Windows.

```bash
conda env create -f environment.yml
conda activate dccfl310
```

Then install / confirm dependencies (if needed):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Note: environment.yml already pins the main versions. requirements.txt is kept for pip-based workflows and documentation consistency.

## Setup (alternative): conda + manual Python version

```bash
conda create -n dccfl310 python=3.10.19 -y
conda activate dccfl310
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data (Demo: Covertype)

The demo notebook is written for the **Covertype** dataset (UCI / sklearn `fetch_covtype`-style tabular data).  
This repository does **NOT** include datasets.

Place your dataset at:

- `data/train.csv`

### Dataset format assumptions (CSV)

The notebook assumes the following column layout (0-based column positions, i.e., `iloc`):

- **Features (X):** columns `1..54` (Python slice: `data.iloc[:, 1:55]`)
- **Label (y):** column `55` (Python slice: `data.iloc[:, 55]`)

If your CSV has a different layout (e.g., no ID column, different label column), modify these lines in the notebook accordingly.

## Label preprocessing

The label vector is converted to **0-based consecutive class indices** using `sklearn.preprocessing.LabelEncoder`.  
This ensures labels become `{0, 1, ..., C-1}` even if the original labels are non-consecutive or non-integer.

## Representation dimensions (must be adjusted per dataset)

The following dimensions are **dataset-dependent** and should be adjusted when you use a different dataset.

- **Local dimensionality reduction (PCA):**  
  Let `D` be the number of feature dimensions used in `X`.  
  We set the PCA output dimension to `D - 1`.

- **Integrated representation dimension (SVD-based):**  
  Let `S` be the singular values from SVD of the concatenated anchor intermediate representations.  
  We set the integrated dimension `d` to the number of singular values satisfying:  
  `S_k >= 1e-2`.

## Run the demo (Jupyter)

After placing data/train.csv, start Jupyter and open the notebook:

```bash
jupyter lab
```

Then open:

notebooks/demo_DC-CFL.ipynb

## Note on working directory (important)

Depending on how Jupyter is launched, the notebook's current working directory may be the repository root or the notebooks/ folder.
The demo notebook includes a small setup cell to automatically move to the repository root so that:

data/train.csv 

can be found reliably.

## Run the experiment pipeline (CLI)

You can also run the experiment pipeline as a Python module from the repository root:

```bash
python -m src.experiment
```
