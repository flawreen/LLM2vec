import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from llm2vec import LLM2Vec
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

# Defineste modelul si tokenizer-ul 
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configureaza tokenizer-ul sa foloseasca padding pe partea stanga (Important pentru LLM2Vec)
tokenizer.padding_side = 'left'

# Incarca datele
data = pd.read_csv('./bold_response_LH.csv')  # citirea datelor folosind pandas
sentences = data['sentence']  # Coloana cu propozitii

# Definirea coloanelor care vor fi prezise
columns_to_predict = [
    'lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb', 
    'lang_LH_MFG', 'lang_LH_PostTemp', 'lang_LH_netw'
]

# Initializeaza LLM2Vec
llm2vec = LLM2Vec(model=model, tokenizer=tokenizer)

# Dimensiunea fixa la care dorim sa completam sau trunchiem vectorii
fixed_dimension = 768

# Genereaza vectori pentru fiecare propozitie
sentence_vectors = []
for sentence in sentences:
    vector = llm2vec.encode(sentence)
    # Asigura-te ca vectorul are dimensiunea corecta
    if len(vector) > fixed_dimension:
        vector = vector[:fixed_dimension]  # Trunchiem vectorul
    elif len(vector) < fixed_dimension:
        # Daca vectorul este mai mic, completam cu 0
        vector = np.pad(vector, (0, fixed_dimension - len(vector)), 'constant')
    sentence_vectors.append(vector)

# Convertim lista de vectori intr-un numpy array omogen
#TODO: Eroare aici, trebuie sa se converteasca in numpy array
sentence_vectors = np.array(sentence_vectors)

# Initializeaza modelul de predictie
model = LogisticRegression()

# Realizeaza 5-fold cross-validation pentru fiecare coloana ce trebuie prezisa
for column in columns_to_predict:
    labels = data[column]  # Coloana de etichete pentru fiecare model
    kf = KFold(n_splits=5)
    accuracies = []

    for train_index, test_index in kf.split(sentence_vectors):
        X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    # Afiseaza rezultatele pentru fiecare coloana
    print(f'Accuracy for predicting {column}: {np.mean(accuracies)}')
