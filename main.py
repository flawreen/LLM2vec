from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from llm2vec import LLM2Vec
from transformers import AutoModel, AutoTokenizer

# Initializează tokenizer și model
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'

# Încarcă datele
data = pd.read_csv('./bold_response_LH.csv')
sentences = data['sentence']
columns_to_predict = [
    'lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb', 
    'lang_LH_MFG', 'lang_LH_PostTemp', 'lang_LH_netw'
]

# Discretizează coloanele de predicție
for column in columns_to_predict:
    data[column] = (data[column] > data[column].median()).astype(int)

# Generează vectori
llm2vec = LLM2Vec(model=model, tokenizer=tokenizer)

def encode_sentence(sentence):
    vector = llm2vec.encode(sentence)
    if len(vector.shape) == 2:
        vector = vector.detach().numpy()
        vector = np.mean(vector, axis=0)
    return vector

sentence_vectors = np.array([encode_sentence(sentence) for sentence in sentences])

# Initializează modelul de clasificare
model = LogisticRegression()

# DataFrame pentru salvarea predicțiilor
predictions_df = pd.DataFrame(sentences, columns=['sentence'])

# 5-fold cross-validation și salvarea predicțiilor
for column in columns_to_predict:
    labels = data[column]
    kf = KFold(n_splits=5)
    fold_predictions = []
    accuracies = []  # Mutat în interiorul buclei pentru fiecare coloană

    for train_index, test_index in kf.split(sentence_vectors):
        X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Salvăm predicțiile pentru fold-ul curent
        fold_predictions.extend(predictions)
        
        # Calculăm acuratețea pentru acest fold
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    # Adaugăm predicțiile pentru această coloană în DataFrame
    predictions_df[column] = fold_predictions
    print(f'Accuracy for predicting {column}: {np.mean(accuracies)}')

# Salvăm predicțiile într-un CSV
predictions_df.to_csv('predictions.csv', index=False)
print("Predicțiile au fost salvate în 'predictions.csv'.")
