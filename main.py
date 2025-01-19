from transformers import AutoTokenizer, AutoModelForCausalLM
from llm2vec import LLM2Vec
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch

# Verifică disponibilitatea GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initializează tokenizer și modelul Hugging Face (pentru cuantizare)
model_name = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

# Cuantizează și salvează modelul

model =  AutoModelForCausalLM.from_pretrained(model_name)


# Folosim modelul cuantizat cu LLM2Vec

llm2vec = LLM2Vec(model=model, tokenizer=tokenizer, pooling_mode="mean")

# Încarcă datele
data = pd.read_csv('./bold_response_LH.csv')
sentences = data['sentence'].tolist()  # Convertește într-o listă
columns_to_predict = [
    'lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb',
    'lang_LH_MFG', 'lang_LH_PostTemp', 'lang_LH_netw'
]

# Normalizează coloanele de predicție
scaler = StandardScaler()
data[columns_to_predict] = scaler.fit_transform(data[columns_to_predict])

# Generează vectori folosind LLM2Vec (cu modelul cuantizat)
print("Encoding sentences using the quantized LLM2Vec model...")
sentence_vectors = llm2vec.encode(sentences, convert_to_numpy=True, device=device)

# Convertește vectorii într-un tensor de tip float64 pe GPU
sentence_vectors = torch.tensor(sentence_vectors, dtype=torch.float64).to(device)

# Verifică tipul și dispozitivul tensorilor
print(f"Tensor type: {sentence_vectors.dtype}")
print(f"Tensor device: {sentence_vectors.device}")

# DataFrame pentru salvarea predicțiilor
predictions_df = pd.DataFrame({'sentence': sentences})

# 5-fold cross-validation și salvarea predicțiilor
for column in columns_to_predict:
    labels = data[column]
    kf = KFold(n_splits=5)
    fold_predictions = []
    mse_values = []
    pearson_values = []
    alpha_values = [0.1, 1, 10, 100, 1000]  # Lista de valori alpha pe care vrem să le testăm

    best_alpha = None
    best_mse = float('inf')
    best_pearson = float('-inf')

    for alpha in alpha_values:
        fold_mse_values = []
        fold_pearson_values = []

        for train_index, test_index in kf.split(sentence_vectors):
            X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            # Antrenare și predicție cu fiecare valoare de alpha
            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(X_train.cpu().numpy(), y_train)
            predictions = ridge_model.predict(X_test.cpu().numpy())

            # Calculăm eroarea MSE
            mse = mean_squared_error(y_test, predictions)
            fold_mse_values.append(mse)

            # Calculăm coeficientul Pearson
            pearson_corr, _ = pearsonr(y_test, predictions)
            fold_pearson_values.append(pearson_corr)

        # Calculăm media valorilor pentru MSE și Pearson pentru fiecare alpha
        mean_mse = np.mean(fold_mse_values)
        mean_pearson = np.mean(fold_pearson_values)

        # Comparăm cu cele mai bune rezultate
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = alpha

        if mean_pearson > best_pearson:
            best_pearson = mean_pearson
            best_alpha = alpha

        # Salvăm rezultatele pentru această valoare de alpha
        mse_values.append(mean_mse)
        pearson_values.append(mean_pearson)

    # Afisăm cel mai bun alpha
    print(f"Best alpha for predicting {column}: {best_alpha}")
    print(f"Best MSE: {best_mse}")
    print(f"Best Pearson Correlation: {best_pearson}")

    # Re-antrenăm modelul cu cel mai bun alpha
    ridge_model = Ridge(alpha=best_alpha)
    fold_predictions = []

    for train_index, test_index in kf.split(sentence_vectors):
        X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        ridge_model.fit(X_train.cpu().numpy(), y_train)
        predictions = ridge_model.predict(X_test.cpu().numpy())
        fold_predictions.extend(predictions)

    # Adaugăm predicțiile pentru această coloană în DataFrame
    predictions_df[column] = fold_predictions

# Salvăm predicțiile într-un CSV
predictions_df.to_csv('predictions.csv', index=False)
print("Predicțiile au fost salvate în 'predictions.csv'.")


