
from datasets import load_dataset#loads the dataset 
from collections import Counter # counts the items in a list
da = load_dataset('PolyAI/banking77')#donwloads banking 77 dataset from hugging face
print(da)#you print rhe dataset 
print(da['train'][0])# out of the training 
labels = da['train'].features['label'].names
print("Number of classes:", len(labels))
print("Example label:", labels[da["train"][0]["label"]])
counts = Counter(da['train']['label'])
print("Most common:", counts.most_common(3))
print("Least common:", counts.most_common()[-3:])
print('Messages to learn from:', len(da['train']))
print('Total topics:', len(labels))