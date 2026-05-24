# Query-Intent Classifier (Banking77)

Classifies short banking queries into one of 77 intents — a learned routing
layer that could decide how a RAG system handles a question before retrieval.

## Problem
Before a support system answers a customer, it has to know what the message is
*about*. This project frames that as a supervised text-classification problem:
given a message, predict its intent (1 of 77 categories).

## Approach
A model has two stages — how text becomes numbers (features), and how those
numbers map to a label (classifier). I held the data fixed and varied each stage
to measure its contribution, always comparing against a simple baseline first.

## Results (macro-F1 on a held-out test set)
| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression (baseline) | 0.845 | 0.836 |
| Sentence embeddings + Logistic Regression | 0.887 | 0.883 |
| Sentence embeddings + Linear SVM (best) | 0.909 | 0.909 |
| **Final model on untouched test set** | **0.929** | **0.929** |

Embeddings (all-MiniLM-L6-v2) gave the largest single gain (+0.047 over the
baseline with the classifier held fixed), confirming that meaning-based features
beat word-counts here. Swapping logistic regression for a linear SVM added a
smaller improvement. The final model was selected on a validation split, then
evaluated exactly once on the dataset's official test set.

## What I learned
The model's weakest intents were three transfer-related categories
(`declined_transfer`, `pending_transfer`, `balance_not_updated_after_bank_transfer`),
each around 0.72–0.75 accuracy. These are semantically near-identical — the kind
of distinction that would need a clarifying follow-up even for a human agent. The
errors clustered exactly where the categories genuinely overlap, which suggests
the model is learning meaning correctly rather than failing at random.

## Method notes (avoiding common pitfalls)
- Fit the vectorizer/encoder on the training data only (no leakage into test).
- Used stratified splits and report macro-F1, since the 77 classes are imbalanced.
- Tuned model choice on a validation split; touched the test set only once.

## Run it
Open `notebook.ipynb` in Google Colab and run the cells top to bottom.

![Confusion matrix](confusion_matrix.png)
