import baseline 
import embedding_model
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score

model_svc = LinearSVC(max_iter=1000,random_state=42)
model_svc.fit(embedding_model.study_embedding,baseline.study_topics)
guesses_svc = model_svc.predict(embedding_model.quiz_embedding)
print('Accuracy with Embeddings(LinearSVC):', round(accuracy_score(baseline.quiz_topics,guesses_svc),3))
print('F1 score with Embeddings(LinearSVC):',round(f1_score(baseline.quiz_topics,guesses_svc,average='macro'),3))
