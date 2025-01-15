from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from llm2vec import LLM2Vec
from transformers import AutoModel, AutoTokenizer

# Initialize tokenizer and model
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'

# Load data
data = pd.read_csv('./bold_response_LH.csv')
sentences = data['sentence']
columns_to_predict = [
    'lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb',
    'lang_LH_MFG', 'lang_LH_PostTemp', 'lang_LH_netw'
]

# Normalize columns to predict
scaler = StandardScaler()
data[columns_to_predict] = scaler.fit_transform(data[columns_to_predict])

# Generate vectors
llm2vec = LLM2Vec(model=model, tokenizer=tokenizer)


def encode_sentence(sentence):
    vector = llm2vec.encode(sentence)
    if len(vector.shape) == 2:
        vector = vector.detach().numpy()
        vector = np.mean(vector, axis=0)
    return vector


sentence_vectors = np.array([encode_sentence(sentence) for sentence in sentences])

# Initialize Ridge model
model = Ridge(alpha=5.3)

# DataFrame for saving predictions
predictions_df = pd.DataFrame(sentences, columns=['sentence'])

# 5-fold cross-validation and saving predictions
for column in columns_to_predict:
    labels = data[column]
    kf = KFold(n_splits=5)
    fold_predictions = []
    mse_values = []
    pearson_values = []

    for train_index, test_index in kf.split(sentence_vectors):
        X_train, X_test = sentence_vectors[train_index], sentence_vectors[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Save predictions for the current fold
        fold_predictions.extend(predictions)

        # Calculate MSE for this fold
        mse = mean_squared_error(y_test, predictions)
        mse_values.append(mse)

        # Calculate Pearson correlation for this fold
        pearson_corr = np.corrcoef(y_test, predictions)[0, 1]
        pearson_values.append(pearson_corr)

    # Add predictions for this column to the DataFrame
    predictions_df[column] = fold_predictions
    print(f'Mean Squared Error for predicting {column}: {np.mean(mse_values)}')
    print(f'Pearson Correlation for predicting {column}: {np.mean(pearson_values)}')

# Save predictions to a CSV
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions have been saved in 'predictions.csv'.")
