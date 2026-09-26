# IMDB Sentiment Analysis with RNN (PyTorch, from scratch)

A sentiment classifier built from scratch in PyTorch that analyzes IMDB movie reviews and classifies them as **positive** or **negative**. The project covers the full pipeline — text preprocessing, vectorization, a custom recurrent neural network, training, and evaluation — implemented without any pre-built NLP pipeline shortcuts.

> **Status:** current version uses a TF-IDF + RNN pipeline. An Embedding + LSTM upgrade is planned (see [Future Improvements](#future-improvements)).

## Overview

Given a movie review, the goal is to predict whether the sentiment is:
- **Positive** — the reviewer enjoyed the movie
- **Negative** — the reviewer found the show terrible or poorly executed

This is framed as a binary classification problem, solved using a recurrent neural network trained end-to-end on the IMDB Dataset of 50,000 movie reviews.

## Project Flow

1. Load data
2. Text preprocessing
3. Convert to Tensors, Datasets, DataLoaders
4. Build the RNN
5. Train the RNN
6. Evaluate the RNN
7. Run inference on new reviews

## Dataset

- **Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Shape after cleaning:** 49,582 rows × 2 columns (418 duplicate rows removed)
- **Columns:** `review` (raw text), `sentiment` (label: `positive` / `negative`)

> The dataset CSV is not included in this repo. Download it from Kaggle and place it in the project root before running the notebook.

## Text Preprocessing

Each review goes through the following steps, in order:

| Step | What it does | Why |
|------|---------------|-----|
| 1. Lowercasing | Converts every word to lowercase (`Good`, `GOOD`, `good` → `good`) | So the model treats casing variants as the same word |
| 2. Remove URLs | Strips out `http...` links using regex | URLs add noise, no sentiment value |
| 3. Remove punctuation | Keeps only `A-Z`, `a-z`, `0-9`, and spaces | Punctuation isn't part of the vocabulary being modeled |
| 4. Remove HTML tags | Strips tags like `<br />` using regex | Leftover HTML markup from scraped reviews |
| 5. Remove stopwords | Removes common low-information words (`a`, `an`, `the`, `and`, `or`, `is`, etc.) using NLTK's stopword list | Reduces noise, keeps sentiment-bearing words |
| 6. Stemming | Reduces words to their root form (`running` → `run`, `played` → `play`) using the Porter Stemmer | Normalizes different forms of the same word |
| 7. Label encoding | Converts `sentiment` column into `1` (positive) / `0` (negative) using `LabelEncoder` | Numeric targets required for training |
| 8. Vectorization | Converts cleaned text into numeric features | Turns text into a representation the model can consume |

### Regex-based cleaning

```python
import re

def remove_urls(text):
    return re.sub(r"http\S+", "", text)

def remove_punctuations(text):
    return re.sub(r"[^A-Za-z0-9\s]", "", text)

def remove_html(text):
    return re.sub(r"<.*?>", "", text)
```

### Stopword removal (token-based)

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
```

### Stemming

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def stemming(text):
    tokens = word_tokenize(text)
    stemmed_words = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_words)
```

## Dataset & DataLoaders

```python
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

train_set = TensorDataset(
    torch.from_numpy(x_train).float(),
    torch.from_numpy(y_train.values).float()
)
test_set = TensorDataset(
    torch.from_numpy(x_test).float(),
    torch.from_numpy(y_test.values).float()
)

train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = DataLoader(test_set, shuffle=True, batch_size=64)
```

## Model Architecture — Many-to-One RNN

Since each review (many inputs) maps to a single sentiment label (one output), this is a **many-to-one** recurrent architecture.

```python
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # last timestep's hidden state
        return out
```

- `batch_first=True` reshapes PyTorch's default `(seq_len, batch, input_size)` expectation into `(batch, seq_len, input_size)`.
- `self.rnn(x, h0)` returns the hidden state at every timestep plus the final hidden state — only the last timestep's output is passed to the fully connected layer, since that represents the model's final understanding of the review.
- Before feeding a batch into the model, an extra sequence dimension is added since the RNN expects 3D input: `xb = xb.unsqueeze(1)`.

## Training

```python
input_size = x_train.shape[1]
model = RNN(input_size)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 10

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()

        xb = xb.unsqueeze(1)
        outputs = model(xb)
        p = torch.sigmoid(outputs.squeeze(1))  # convert raw outputs to probabilities

        loss = criterion(p, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # prevent exploding gradients
        optimizer.step()

    print(f"{epoch+1}/{epochs} and loss = {loss.item()}")
```

## Evaluation

```python
model.eval()
with torch.no_grad():
    correct_vals = 0
    total_vals = 0

    for xb, yb in test_loader:
        xb = xb.unsqueeze(1)
        outputs = model(xb)
        predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()

        total_vals += yb.size(0)
        correct_vals += (predicted == yb).sum().item()

    print(f"accuracy = {correct_vals / total_vals * 100}")
```

## Results

| Metric | Value |
|--------|-------|
| Train/test split | 80% / 20% |
| Epochs | 10 |
| Batch size | 64 |
| Final test accuracy | _fill in once training completes_ |

## Tech Stack

- Python
- PyTorch (model, training loop)
- scikit-learn (train/test split, label encoding)
- NLTK (tokenization, stopwords, stemming)
- pandas (data loading/cleaning)
- Matplotlib (loss/accuracy plots)

## Project Structure

```
├── Sentiment_Analysis.ipynb   # Main notebook: preprocessing → training → evaluation
├── 02) IMDB Dataset.csv       # Dataset (not included — download separately)
└── README.md
```

## How to Run

1. Clone the repo (see "Uploading to GitHub" below if you haven't pushed it yet).
2. Install dependencies:
   ```bash
   pip install torch scikit-learn nltk pandas matplotlib
   ```
3. Download the IMDB dataset CSV from Kaggle and place it in the project root.
4. Open and run the notebook top to bottom.

## Future Improvements

- Replace the current TF-IDF + RNN pipeline with tokenized word sequences fed through an `nn.Embedding` layer followed by an `nn.LSTM` — this lets the recurrence model actual word order instead of a single flattened vector.
- Track and plot training loss and test accuracy per epoch.
- Move training to GPU (`model.to(device)`, batches moved with `.to(device)`) for faster iteration.
- Add early stopping and a validation split separate from the test set.
- Experiment with bidirectional LSTM/GRU layers.

## License

This project is open source and available under the MIT License.
