# Email Spam Detection

Binary text classification using **Naive Bayes** and **Logistic Regression** on the SMS Spam Collection dataset. The project demonstrates a full ML pipeline — from raw text to TF-IDF features — and includes a controlled experiment showing how noisy features degrade performance and how chi-squared feature selection recovers it.

## Dataset

[SMS Spam Collection — UCI ML Repository / Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

| Split | Ham | Spam | Total |
|-------|-----|------|-------|
| Full  | 4825 | 747 | 5572 |
| Train (80%) | 3860 | 597 | 4457 |
| Test  (20%) | 965  | 150 | 1115 |

> The dataset is not committed. See [Setup](#setup) for how to obtain it.

## Pipeline

```
Raw SMS text
    └── TF-IDF Vectorisation (5000 features, unigrams + bigrams)
            ├── Naive Bayes (MultinomialNB)
            └── Logistic Regression
                    ├── Baseline evaluation
                    ├── + 2000 random noise columns  →  performance drop
                    └── Chi-squared feature selection (k=2000)  →  recovery
```

## Results

### Baseline (clean TF-IDF features)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Naive Bayes | 98.30% | 0.9851 | 0.8859 | 0.9329 | 0.9898 |
| Logistic Regression | 97.31% | 1.0000 | 0.7987 | 0.8881 | 0.9850 |

### After injecting 2000 random noise columns

| Model | F1 | Change |
|-------|----|--------|
| Naive Bayes | 0.9357 | +0.003 (robust) |
| Logistic Regression | 0.2576 | **-0.631 (catastrophic drop)** |

LR spreads its L2 penalty across all 7000 features, severely diluting the signal. NB naturally down-weights uninformative features.

### After chi-squared feature selection (top 2000 from noisy matrix)

| Model | F1 | Recovery |
|-------|----|---------|
| Naive Bayes | 0.9203 | Restored |
| Logistic Regression | 0.8359 | Mostly restored |

### Top spam-indicating words (Naive Bayes log-probability ratio)

> `claim`, `prize`, `guaranteed`, `ringtone`, `awarded`, `1000 cash`, `dating`, `http` ...

## Key Insights

- **Naive Bayes is well-suited for sparse text** — its conditional independence assumption aligns with TF-IDF's bag-of-words representation and gives strong baseline results.
- **Logistic Regression is sensitive to high-dimensional noise** — adding irrelevant features drops its F1 from 0.89 to 0.26 because L2 regularisation cannot zero out useless weights completely.
- **Feature selection matters more than feature quantity** — chi-squared selection restores both models by eliminating most noise columns, confirming that fewer, relevant features beat many irrelevant ones.

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/shakib79ai/Email-Spam-Detection.git
cd Email-Spam-Detection
```

### 2. Create a virtual environment and install dependencies

```bash
# Using uv (recommended)
uv venv .venv
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Download the dataset

Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the project root.

Alternatively, download automatically from the UCI ML Repository:

```python
import urllib.request, zipfile, io, pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
with urllib.request.urlopen(url) as r:
    with zipfile.ZipFile(io.BytesIO(r.read())) as z:
        z.extractall('.')
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
df.to_csv('spam.csv', index=False)
```

### 4. Run

```bash
python Email-Spam-Detection.py
```

## Output files

| File | Description |
|------|-------------|
| `eda_plots.png` | Class distribution + message length histogram |
| `confusion_baseline.png` | Confusion matrices for both baseline models |
| `roc_baseline.png` | ROC curves for both baseline models |
| `top_spam_words.png` | Top 20 spam-indicating words by NB log-ratio |
| `f1_comparison.png` | F1 score bar chart across all 6 conditions |

## Tech stack

- Python 3.8+
- scikit-learn — TF-IDF, models, feature selection, metrics
- pandas / numpy — data handling
- matplotlib / seaborn — visualisation
- scipy — sparse matrix operations
