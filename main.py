from transformers import AutoTokenizer, AutoModel
from llm2vec import LLM2Vec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer and Hugging Face model (BERT)
model_name = "bert-base-cased"  # Using BERT base uncased
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)

# Set padding token if it's not already defined
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Using BERT with LLM2Vec
llm2vec = LLM2Vec(model=model, tokenizer=tokenizer, pooling_mode="mean")

# Load the data
data = pd.read_csv('./bold_response_LH.csv')
sentences = data['sentence'].tolist()  # Convert to a list
columns_to_predict = [
    'lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb',
    'lang_LH_MFG', 'lang_LH_PostTemp', 'lang_LH_netw'
]

# Normalize the target columns
scaler = StandardScaler()
data[columns_to_predict] = scaler.fit_transform(data[columns_to_predict])

# Generate sentence embeddings using LLM2Vec (BERT model)
print("Encoding sentences using the BERT model...")
sentence_vectors = llm2vec.encode(sentences, convert_to_numpy=True, device=device)

# Convert vectors into float32 tensors on the GPU for better performance
sentence_vectors = torch.tensor(sentence_vectors, dtype=torch.float16).to(device)

# Check the tensor type and device
print(f"Tensor type: {sentence_vectors.dtype}")
print(f"Tensor device: {sentence_vectors.device}")

# DataFrame for saving predictions
predictions_df = pd.DataFrame({'sentence': sentences})

# 5-fold cross-validation and saving predictions
for column in columns_to_predict:
    labels = data[column]
    kf = KFold(n_splits=5)
    fold_predictions = []
    mse_values = []
    pearson_values = []
    accuracy_values = []

    best_mse = float('inf')
    best_pearson = float('-inf')
    best_accuracy = float('-inf')

    # Use a single value for n_estimators (50)
    n_estimators = 50

    fold_mse_values = []
    fold_pearson_values = []
    fold_accuracy_values = []

    for train_index, test_index in kf.split(sentence_vectors):
        X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        # Move data to CPU for GradientBoostingRegressor
        X_train_cpu = X_train.cpu().numpy()
        X_test_cpu = X_test.cpu().numpy()

        # Train and predict using GradientBoostingRegressor
        gb_model = GradientBoostingRegressor(n_estimators=n_estimators)
        gb_model.fit(X_train_cpu, y_train)
        predictions = gb_model.predict(X_test_cpu)

        # Calculate MSE
        mse = mean_squared_error(y_test, predictions)
        fold_mse_values.append(mse)

        # Calculate Pearson correlation
        pearson_corr, _ = pearsonr(y_test, predictions)
        fold_pearson_values.append(pearson_corr)

        # Calculate accuracy
        accuracy = accuracy_score(y_test.round(), predictions.round())  # Round predictions
        fold_accuracy_values.append(accuracy)

    # Calculate average MSE, Pearson, and accuracy
    mean_mse = np.mean(fold_mse_values)
    mean_pearson = np.mean(fold_pearson_values)
    mean_accuracy = np.mean(fold_accuracy_values)

    # Compare with the best results
    if mean_mse < best_mse:
        best_mse = mean_mse

    if mean_pearson > best_pearson:
        best_pearson = mean_pearson

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy

    # Save the results for this n_estimators value
    mse_values.append(mean_mse)
    pearson_values.append(mean_pearson)
    accuracy_values.append(mean_accuracy)

    # Print the best results
    print(f"Best MSE for predicting {column}: {best_mse}")
    print(f"Best Pearson Correlation: {best_pearson}")
    print(f"Best Accuracy: {best_accuracy}")

    # Retrain the model with the best n_estimators
    gb_model = GradientBoostingRegressor(n_estimators=n_estimators)
    fold_predictions = []

    for train_index, test_index in kf.split(sentence_vectors):
        X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        # Move data to CPU for GradientBoostingRegressor
        X_train_cpu = X_train.cpu().numpy()
        X_test_cpu = X_test.cpu().numpy()

        gb_model.fit(X_train_cpu, y_train)
        predictions = gb_model.predict(X_test_cpu)
        fold_predictions.extend(predictions)

    # Add predictions for this column to the DataFrame
    predictions_df[column] = fold_predictions

# Save predictions to a CSV
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")
