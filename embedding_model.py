import explore 
from sentence_transformers import SentenceTransformer
import baseline
da = explore.da
embedding_model = SentenceTransformer('all-MiniLM-l6-v2')
print('Generated embedding for study set')
study_embedding = embedding_model.encode(baseline.study_texts,show_progress_bar=True,normalize_embeddings=True)
print('Generating embedding for qui set')
quiz_embedding = embedding_model.encode(baseline.quiz_texts,show_progress_bar=True,normalize_embeddings=True)
print('Shape of study embedding:', study_embedding.shape)
print('Shape of quiz embedding:', quiz_embedding.shape)