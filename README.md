# Driver Classification Using ConvTran Ensembles

A deep-learning and ensemble-learning framework for **driver classification from multivariate vehicle time-series data**. This repository contains the code developed for my diploma thesis, exploring whether combining learned representations from multiple ConvTran models can improve classification performance compared with a conventional machine-learning baseline.

## Project Overview

The objective is to identify the driver (classes **A–J**) from time-series data collected while driving. Each observation contains multiple vehicle signals recorded over time.

The main approach is a two-stage classification pipeline:

1. **Representation learning:** train three ConvTran models on separate driver groups.
2. **Ensemble classification:** extract a feature embedding from each trained model, concatenate the embeddings, and train a Random Forest meta-classifier to predict one of the ten driver classes.

The ensemble is evaluated against two alternatives:

- A **Random Forest baseline** trained directly on statistical summaries of the same time windows.
- A **single ConvTran model** trained to classify all ten driver classes directly.

## Methodology

### 1. Data preparation and exploratory analysis

The data preparation workflow includes data inspection, feature analysis and the creation of fixed-length time-series windows.

The `features.py` script contains exploratory feature-analysis steps, including checks for missing values and outliers, as well as feature-related analysis such as mutual information.

The windowing scripts transform continuous driving records into overlapping samples:

| Parameter | Value |
|---|---:|
| Window length | 60 time steps |
| Sampling frequency | 1 Hz |
| Window duration | 60 seconds |
| Sliding-window stride | 15 time steps |
| Overlap | 75% |

Windows are generated within their corresponding continuous data blocks and data splits. A window is not allowed to cross a `block_id` or split boundary. This preserves the temporal structure of the data and prevents samples from being created across unrelated recording segments.

The resulting arrays are stored as compressed NumPy `.npz` files containing:

- `X`: time-series windows, initially shaped as `(number_of_windows, 60, number_of_features)`.
- `y`: the corresponding driver-class labels.

### 2. Data splits

The pipeline uses four separate splits:

- `convtran_train`: used to train the ConvTran models.
- `convtran_val`: used for validation and model selection / early stopping.
- `meta_train`: used to train the ensemble's meta-classifier.
- `test`: held out for final evaluation.

The meta-classifier is trained on embeddings extracted from `meta_train`, while the final ensemble performance is measured on the separate `test` split.

### 3. ConvTran model training

The main experiment trains three ConvTran models on separate driver groups:

| Model | Driver classes | Number of classes |
|---|---|---:|
| `convtran_cleaned_1` | A, B, C | 3 |
| `convtran_cleaned_2` | D, E, F | 3 |
| `convtran_cleaned_3` | G, H, I, J | 4 |

Each model learns temporal representations from its assigned group. The trained models are saved as checkpoints in `outputs/models/`.

The main training configuration in `train.py` includes:

| Hyperparameter | Value |
|---|---:|
| Epochs | 100 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Embedding size | 16 |
| Attention heads | 8 |
| Feed-forward dimension | 256 |
| Dropout | 0.2 |
| Random seed | 42 |
| Early-stopping patience | 25 |

The implementation automatically selects CUDA when an available GPU is detected; otherwise, it uses the CPU.

### 4. ConvTran architecture

The ConvTran implementation is based on the architecture introduced in *Improving Position Encoding of Transformers for Multivariate Time Series Classification*.

At a high level, each input window passes through the following stages:

1. **Temporal convolution:** learns local patterns along the time dimension.
2. **Spatial convolution:** combines information across the input features / vehicle signals.
3. **Temporal positional encoding (tAPE):** adds temporal-position information to the learned representation.
4. **Multi-head attention with eRPE:** models relationships between different time steps using relative positional information.
5. **Feed-forward block:** further transforms the representation.
6. **Global Average Pooling:** aggregates the temporal representation into a fixed-size vector.
7. **Classifier:** produces class scores when the model is used for direct classification.

The model also supports returning the representation immediately before the final classification layer. This is used by the ensemble pipeline to extract embeddings.

### 5. Embedding extraction and ensemble classification

After training, `extract_embeddings.py` loads the three saved ConvTran checkpoints and processes the windows from the full dataset.

For every window, each model produces a **16-dimensional embedding**. The three embeddings are concatenated into a single feature vector:

`16 + 16 + 16 = 48 features`

The resulting embedding datasets are saved under `outputs/embeddings/`, including `meta_train.npz` and `test.npz`.

The `ensembles.py` script then:

1. Loads the combined `meta_train` embeddings and their labels.
2. Trains a `RandomForestClassifier` with 200 trees and `class_weight="balanced"`.
3. Predicts the labels of the held-out test embeddings.
4. Reports accuracy and a classification report.
5. Generates a confusion matrix.
6. Saves the trained meta-classifier as `outputs/models/meta_classifier.joblib`.

This approach uses the ConvTran networks as representation extractors and the Random Forest as the final classifier.

### 6. Baseline and additional evaluation

To put the ensemble results into context, the repository includes two additional experiments.

**Random Forest baseline — `random_forest_baseline.py`**

This baseline uses the same temporal windows as the ensemble. Instead of learned ConvTran embeddings, each window is represented by the mean and standard deviation of every input feature. With 30 input features, this produces a 60-dimensional vector per window. A Random Forest is trained on the `meta_train` split and evaluated on the same held-out `test` split.

**Single ConvTran — `train_single_contran.py` and `evaluate_single_contran.py`**

This experiment trains one ConvTran model directly on the complete ten-class problem (A–J), rather than using three group-specific models and a meta-classifier. It provides an additional deep-learning comparison.

**Combined evaluation — `run_all_evaluations.py`**

This script brings together the evaluation of the individual group-specific ConvTran models and the full classification approaches, allowing their results to be compared using a common test setup.

## Repository Structure

```text
.
├── data/
│   ├── windows/
│   │   ├── cleaned_1/
│   │   ├── cleaned_2/
│   │   ├── cleaned_3/
│   │   └── full/
│   └── ...
│
├── models/
│   ├── model.py
│   ├── attention.py
│   └── positional_encoding.py
│
├── outputs/
│   ├── models/
│   ├── embeddings/
│   └── plots/
│
├── presentation_figures/
│
├── checkpoints.py
├── ensembles.py
├── evaluate_single_contran.py
├── extract_embeddings.py
├── features.py
├── random_forest_baseline.py
├── run_all_evaluations.py
├── test_model.py
├── train.py
├── train_single_contran.py
├── windowing.py
└── windowing_full.py
```

The exact contents of `data/` and `outputs/` depend on the datasets, checkpoints and experiment artifacts available in a particular checkout.

### Main scripts

| Script | Purpose |
|---|---|
| `features.py` | Exploratory data and feature analysis |
| `windowing.py` | Creates windows for the separate driver groups |
| `windowing_full.py` | Creates windows for the complete ten-class dataset |
| `train.py` | Trains the three group-specific ConvTran models |
| `extract_embeddings.py` | Extracts and concatenates embeddings from the trained models |
| `ensembles.py` | Trains and evaluates the Random Forest meta-classifier |
| `random_forest_baseline.py` | Trains and evaluates the statistical-feature Random Forest baseline |
| `train_single_contran.py` | Trains one ConvTran on all ten classes |
| `evaluate_single_contran.py` | Evaluates the single full-problem ConvTran |
| `run_all_evaluations.py` | Runs the combined model evaluation |
| `test_model.py` | Provides an additional model-testing entry point |
| `checkpoints.py` | Checkpoint-related utilities |

## Getting Started

### Requirements

The project is implemented in Python and uses PyTorch for the ConvTran models, NumPy and Pandas for data handling, and scikit-learn for the Random Forest classifiers and evaluation metrics.

Install the main dependencies in your Python environment:

```bash
pip install torch numpy pandas scikit-learn joblib matplotlib seaborn
```

For GPU training, install the PyTorch build compatible with your CUDA setup. The code automatically falls back to CPU when CUDA is unavailable.

### Running the pipeline

Run the scripts from the repository root so that the relative data and output paths resolve correctly.

The general execution order for the ensemble experiment is:

```bash
# 1. Create the windowed datasets
python windowing.py
python windowing_full.py

# 2. Train the three group-specific ConvTran models
python train.py

# 3. Extract the combined embeddings
python extract_embeddings.py

# 4. Train and evaluate the ensemble
python ensembles.py
```

To run the comparison experiments:

```bash
# Random Forest baseline
python random_forest_baseline.py

# Single ConvTran trained on all ten classes
python train_single_contran.py
python evaluate_single_contran.py

# Combined evaluation
python run_all_evaluations.py
```

**Note:** The scripts expect the corresponding preprocessed input files and trained checkpoints to exist at the paths defined in the code. Run the data-preparation and training stages first, or provide the required files in the expected locations. Some exploratory scripts may also refer to intermediate CSV files created during the original preprocessing workflow.

## Results and Outputs

Depending on which experiments have been run, the repository stores:

- Trained ConvTran checkpoints in `outputs/models/`.
- The trained Random Forest meta-classifier in `outputs/models/`.
- Extracted embedding arrays in `outputs/embeddings/`.
- Confusion matrices and other evaluation plots in `outputs/plots/`.
- Figures prepared for the thesis presentation in `presentation_figures/`.

The evaluation scripts report accuracy and per-class precision, recall and F1-score, together with confusion matrices. No single performance value is hard-coded in this README because the results depend on the model checkpoints and dataset version used for evaluation.

## Acknowledgements and References

### ConvTran implementation

The ConvTran architecture in `models/model.py` is **customized from the original ConvTran implementation** by Navid Mohammadi Foumani and collaborators:

- Original repository: [Navidfoumani/ConvTran](https://github.com/Navidfoumani/ConvTran)
- Research paper: N. M. Foumani, C. W. Tan, G. I. Webb and M. Salehi, “Improving Position Encoding of Transformers for Multivariate Time Series Classification,” *Data Mining and Knowledge Discovery*, vol. 38, pp. 22–48, 2024.

The relevant source repository is also acknowledged directly in the corresponding model source code. The implementation in this thesis project has been adapted to the vehicle-driver classification task and extended to support the embedding-extraction and ensemble workflow described above.

Please refer to the original repository and its license for the original implementation and licensing information.

## License

This repository is an academic thesis project. If you reuse or redistribute parts of the code, please also review and comply with the license and attribution requirements of the original ConvTran implementation and any other third-party components used.
