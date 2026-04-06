import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score,precision_score,recall_score
train_data=[
    ("team win match","sports"),
    ("player score goal","sports"),
    ("government election vote","politics"),
    ("minister policy decision","politics")
]
class_counts=defaultdict(int)
word_counts=defaultdict(lambda:defaultdict(int))
vocab=set()
for text,label in train_data:
    class_counts[label]+=1
    words=text.split()
    for word in words:
        vocab.add(word)
        word_counts[label][word]+=1
total_docs=len(train_data)
def predict(text):
    words=text.split()
    best_class=None
    max_prob=-float('inf')
    for label in class_counts:
        log_prob=np.log(class_counts[label]/total_docs)
        total_words=sum(word_counts[label].values())
        for word in words:
            count=word_counts[label][word]
            word_prob=(count+1)/(total_words+len(vocab))
            log_prob+=np.log(word_prob)
        if log_prob>max_prob:
            max_prob=log_prob
            best_class=label
    return best_class
test_docs=[
    "team score win",
    "government policy",
    "player goal",
    "election vote"
]
actual_labels=["sports","politics","sports","politics"]
predicted_labels=[]
for doc in test_docs:
    pred=predict(doc)
    predicted_labels.append(pred)
    print(f" Document:{doc}->Predicted:{pred}")
accuracy = accuracy_score(actual_labels, predicted_labels)

precision = precision_score(actual_labels, predicted_labels, pos_label="sports")

recall = recall_score(actual_labels, predicted_labels, pos_label="sports")

print("\nAccuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)


        
