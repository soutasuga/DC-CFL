\# DC-CFL (experimental code)



This repository provides experimental code for DC-CFL style evaluation.



\## Requirements

```bash

pip install -r requirements.txt



```md

## Data (Demo: Covertype)

The demo notebook is written for the **Covertype** dataset (UCI / sklearn `fetch_covtype`-style tabular data).
This repository does NOT include datasets.

Place your dataset at:
- `data/train.csv`

### Dataset format assumptions (CSV)

The notebook assumes the following column layout (0-based column positions, i.e., `iloc`):
- **Features (X):** columns `1..54`  (Python slice: `data.iloc[:, 1:55]`)
- **Label (y):** column `55`         (Python slice: `data.iloc[:, 55]`)

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
