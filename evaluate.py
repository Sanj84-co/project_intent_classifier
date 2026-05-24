import baseline
import embedding_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
model_embeddings = LogisticRegression(max_iter=1000,solver = 'liblinear')
model_embeddings.fit(embedding_model.study_embedding,baseline.study_topics)
guesses_embeddings = model_embeddings.predict(embedding_model.quiz_embedding)
print('Accuracy with Embeddings(LogisticRegression):', round(accuracy_score(baseline.quiz_topics,guesses_embeddings),3))
print('F1 score with Embeddings(LogisticRegression):',round(f1_score(baseline.quiz_topics,guesses_embeddings,average='macro'),3))
