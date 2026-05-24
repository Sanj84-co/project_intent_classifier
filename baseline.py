from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
import explore
da = explore.da
texts = list(da['train']['text'])
topics = list(da['train']['label'])

study_texts,quiz_texts,study_topics,quiz_topics = train_test_split(
    texts,topics,test_size=0.15,stratify=topics,random_state=42
)
counter = TfidfVectorizer(ngram_range=(1,2),min_df=2)
study_x = counter.fit_transform(study_texts)
quiz_x= counter.transform(quiz_texts)

model = LogisticRegression(max_iter=1000)
model.fit(study_x,study_topics)
guesses = model.predict(quiz_x)
print('Accuracy:',round(accuracy_score(quiz_topics,guesses),3))
print('Fair score:', round(f1_score(quiz_topics,guesses,average='macro'),3))