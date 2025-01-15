from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import torch

# Check for CUDA availability
# torch_device = "cuda" if torch.cuda.is_available() else "cpu" ruleaza mai bine pe cpu din cauza preciziei numerice, o sa o fac sa fie mai apropiata in push-ul ulterior
# Load SentenceTransformer model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and fast model

# Load data
data = pd.read_csv('./bold_response_LH.csv')
sentences = data['sentence']
columns_to_predict = [
    'lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb',
    'lang_LH_MFG', 'lang_LH_PostTemp', 'lang_LH_netw'
]

# Drop rows with missing values
data.dropna(subset=columns_to_predict, inplace=True)

# Encode sentences using SentenceTransformers
sentence_vectors = np.array(sbert_model.encode(sentences, convert_to_numpy=True))

# Initialize Ridge model
ridge_model = Ridge(alpha=5.4)

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {col: {'mse': [], 'pearson': []} for col in columns_to_predict}

for train_index, test_index in kf.split(sentence_vectors):
    X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
    
    for column in columns_to_predict:
        y_train, y_test = data[column].iloc[train_index], data[column].iloc[test_index]
        
        # Train model
        ridge_model.fit(X_train, y_train)
        predictions = ridge_model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        pearson_corr = np.corrcoef(y_test, predictions)[0, 1]

        results[column]['mse'].append(mse)
        results[column]['pearson'].append(pearson_corr)

# Report results
for column in columns_to_predict:
    mean_mse = np.mean(results[column]['mse'])
    mean_pearson = np.mean(results[column]['pearson'])
    print(f"Column: {column}")
    print(f"  Mean MSE: {mean_mse:.4f}")
    print(f"  Mean Pearson: {mean_pearson:.4f}")
