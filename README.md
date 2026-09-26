# IMDB Sentiment Analysis with LSTM (PyTorch, from scratch)

A sentiment classifier built from scratch in PyTorch that analyzes IMDB movie reviews and classifies them as **positive** or **negative**. The pipeline covers text preprocessing, vocabulary building, sequence encoding, an Embedding + LSTM model, training with gradient clipping, and evaluation — all implemented without any pre-built NLP pipeline shortcuts.

## Overview

Given a movie review, the goal is to predict whether the sentiment is:
- **Positive** — the reviewer enjoyed the movie
- **Negative** — the reviewer found the show terrible or poorly executed

This is a binary classification problem, solved with a recurrent architecture (Embedding → LSTM → Linear) trained end-to-end on the IMDB Dataset of 50,000 movie reviews.

## Results

| Metric | Value |
|--------|-------|
| Epochs | 10 |
| Batch size | 64 |
| Sequence length | 200 tokens |
| Vocabulary size | 10,000 |
| **Final train loss** | **0.0808** |
| **Final test accuracy** | **86.15%** |

Loss dropped steadily from ~0.69 (coin-flip) at epoch 1 to 0.08 by epoch 10, with test accuracy climbing from ~50% to 86%+ — confirming the model is genuinely learning sentence-level sentiment rather than guessing.

## Project Flow

1. Load data (via `kagglehub`)
2. Text preprocessing
3. Build vocabulary + encode reviews as token ID sequences
4. Convert to Tensors, Datasets, DataLoaders
5. Build the Embedding + LSTM model
6. Train with gradient clipping
7. Evaluate and plot loss/accuracy curves

## Dataset

- **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (loaded directly via `kagglehub`, no manual download needed)
- **Shape after cleaning:** 49,582 rows × 2 columns (418 duplicate rows removed)
- **Columns:** `review` (raw text), `sentiment` (label: `positive` / `negative`)

```python
import kagglehub
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
```

## Text Preprocessing

Each review goes through the following steps, in order:

| Step | What it does | Why |
|------|---------------|-----|
| 1. Lowercasing | Converts every word to lowercase | Treats casing variants as the same word |
| 2. Remove URLs | Strips `http...` links via regex | URLs add noise, no sentiment value |
| 3. Remove punctuation | Keeps only `A-Z`, `a-z`, `0-9`, and spaces | Punctuation isn't part of the modeled vocabulary |
| 4. Remove HTML tags | Strips tags like `<br />` via regex | Leftover markup from scraped reviews |
| 5. Remove stopwords | Removes common low-information words (`a`, `an`, `the`, `is`, etc.) via NLTK | Reduces noise, keeps sentiment-bearing words |
| 6. Stemming | Reduces words to root form (`running` → `run`) via Porter Stemmer | Normalizes different forms of the same word |
| 7. Label encoding | Converts `sentiment` into `1` (positive) / `0` (negative) via `LabelEncoder` | Numeric targets for training |
| 8. Vocabulary + sequence encoding | Builds a word→index vocabulary and converts each review into a fixed-length sequence of token IDs | Real sequential input for the LSTM (see below) |

```python
import re

def remove_urls(text):
    return re.sub(r"http\S+", "", text)

def remove_punctuations(text):
    return re.sub(r"[^A-Za-z0-9\s]", "", text)

def remove_html(text):
    return re.sub(r"<.*?>", "", text)
```

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
```

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def stemming(text):
    tokens = text.split()
    stemmed_words = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_words)
```

## Vocabulary & Sequence Encoding

Unlike a TF-IDF approach (which discards word order), this version tokenizes each review and maps it to a sequence of integer token IDs — preserving order so the LSTM can actually model it.

```python
from collections import Counter

all_tokens = []
tokenized_reviews = []

for review in df["review"]:
    tokens = review.split()
    tokenized_reviews.append(tokens)
    all_tokens.extend(tokens)

vocab_size = 10000  # top 10k most frequent words
word_counts = Counter(all_tokens)
most_common = word_counts.most_common(vocab_size - 2)  # reserve 0=PAD, 1=UNK

word2idx = {"<PAD>": 0, "<UNK>": 1}
for word, _ in most_common:
    word2idx[word] = len(word2idx)

def encode_review(tokens, max_len=200):
    ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    if len(ids) < max_len:
        ids = ids + [word2idx["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

MAX_LEN = 200
x = [encode_review(tokens, MAX_LEN) for tokens in tokenized_reviews]
```

Every review becomes a fixed-length sequence of 200 token IDs — shorter reviews are padded with `0`, longer ones are truncated.

## Dataset & DataLoaders

```python
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

x = np.array(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

train_set = TensorDataset(
    torch.from_numpy(x_train).long(),      # token IDs, so long dtype
    torch.from_numpy(y_train.values).float()
)
test_set = TensorDataset(
    torch.from_numpy(x_test).long(),
    torch.from_numpy(y_test.values).float()
)

train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = DataLoader(test_set, shuffle=True, batch_size=64)
```

| Split | Shape |
|-------|-------|
| `x_train` | (39,665, 200) |
| `x_test` | (9,917, 200) |

## Model Architecture — Embedding + LSTM

```python
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)              # (batch, seq_len, embed_dim)
        out, (hidden, cell) = self.lstm(embedded)  # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])               # last timestep's hidden state
        return out                                  # raw logits — no sigmoid here
```

- `nn.Embedding` turns each token ID into a learned 128-dim vector, with `padding_idx=0` so padding tokens don't contribute meaningful gradients.
- `nn.LSTM` processes the sequence in order, letting the model learn dependencies between words (negation, intensifiers, context) that a bag-of-words representation would lose entirely.
- The final timestep's hidden state is passed through a linear layer to produce a single raw logit.

## Training

```python
vocab_size = len(word2idx)
model = SentimentLSTM(vocab_size)

criterion = nn.BCEWithLogitsLoss()   # operates directly on raw logits — no separate sigmoid needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(batch_x).squeeze(1)
        loss = criterion(outputs, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # prevents exploding gradients
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # per-epoch evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    test_acc = correct / total
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch+1}/{epochs} — avg loss = {avg_loss:.4f} — test acc = {test_acc:.4f}")
```

**Training log (loss → accuracy per epoch):**
```
Epoch 1/10  — avg loss = 0.6928 — test acc = 0.5045
Epoch 2/10  — avg loss = 0.6576 — test acc = 0.6117
...
Epoch 10/10 — avg loss = 0.0808 — test acc = 0.8615
```

## Evaluation & Plots

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, epochs + 1), train_losses, marker='o')
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Avg Loss")
axes[0].grid(True)

axes[1].plot(range(1, epochs + 1), test_accuracies, marker='o', color='green')
axes[1].set_title("Test Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Tech Stack

- Python
- PyTorch (Embedding, LSTM, training loop)
- scikit-learn (train/test split, label encoding)
- NLTK (tokenization, stopwords, stemming)
- pandas (data loading/cleaning)
- Matplotlib (loss/accuracy plots)
- kagglehub (dataset download)

## Project Structure

```
├── Sentiment_Analysis.ipynb   # Full pipeline: preprocessing → LSTM training → evaluation
└── README.md
```

The dataset is downloaded automatically at runtime via `kagglehub` — no manual download or local CSV needed.

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install torch scikit-learn nltk pandas matplotlib kagglehub
   ```
3. Open and run `Sentiment_Analysis.ipynb` top to bottom. The dataset downloads automatically on first run.

## Future Improvements

- Move training to GPU (`model.to(device)`, batches moved with `.to(device)`) — current run took ~1000s+ for epoch 1 on CPU before speeding up in later epochs.
- Add a validation split separate from the test set, plus early stopping.
- Try mean/max pooling over all LSTM timesteps instead of only the last one, especially for longer reviews.
- Experiment with bidirectional LSTM/GRU layers and stacking multiple LSTM layers.
- Add inference on custom/new reviews with a simple `predict_sentiment(text)` helper.

## License

This project is open source and available under the MIT License.
