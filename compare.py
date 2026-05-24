import embedding_model
import baseline
import linear
import evaluate
base_acc = baseline.accuracy_score(baseline.quiz_topics,baseline.guesses)
base_f1 = baseline.f1_score(baseline.quiz_topics,baseline.guesses,average='macro')
embed_lr_acc = baseline.accuracy_score(baseline.quiz_topics,evaluate.guesses_embeddings)
embed_lr_f1 = baseline.f1_score(baseline.quiz_topics,evaluate.guesses_embeddings,average = 'macro')
embed_svc_acc = baseline.accuracy_score(baseline.quiz_topics,linear.guesses_svc)
embed_svc_f1 = baseline.f1_score(baseline.quiz_topics,linear.guesses_svc,average = 'macro')
print(f"{'Model':<35}{'Accuracy':<12}{'Fair score (macro-F1)'}")
print("-" * 70)
print(f"{'Baseline (TF-IDF + Logistic Regression)':<35}{round(base_acc,3):<12}{round(base_f1,3)}")
print(f"{'Embeddings (Logistic Regression)':<35}{round(embed_lr_acc,3):<12}{round(embed_lr_f1,3)}")
print(f"{'Embeddings (LinearSVC)':<35}{round(embed_svc_acc,3):<12}{round(embed_svc_f1,3)}")
print("-" * 70)


best_f1 = max(base_f1, embed_lr_f1, embed_svc_f1)


if best_f1 == embed_svc_f1:
    print(f"Winner: Embeddings (LinearSVC), by +{round(embed_svc_f1 - max(base_f1, embed_lr_f1), 3)} on the fair score.")
elif best_f1 == embed_lr_f1:
    print(f"Winner: Embeddings (Logistic Regression), by +{round(embed_lr_f1 - base_f1, 3)} on the fair score.")
else:
    print(f"Winner: Baseline (TF-IDF + Logistic Regression), by +{round(base_f1 - embed_lr_f1, 3)} on the fair score. (A real and interesting result!)")