import pandas as pd
import numpy as np
from llm2vec import LLM2Vec
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import torch

# Representational Similarity Analysis

data_path = "bold_response_LH.csv"
pred_path = "predictions.csv"
data = pd.read_csv(data_path)
pred = pd.read_csv(pred_path)
params = ['lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb', 'lang_LH_MFG', 'lang_LH_PostTemp']


def get_encoded_sentences():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "microsoft/deberta-v3-base"  # Folosim BERT base uncased
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm2vec = LLM2Vec(model=model, tokenizer=tokenizer, pooling_mode="mean")
    return llm2vec.encode(data['sentence'].to_list(), convert_to_numpy=True, device=device)


sentence_similarity = cosine_similarity(get_encoded_sentences())
brain_similarity = {}

for column in params:
    brain_values = data[column].values.reshape(-1, 1)
    brain_similarity[column] = cosine_similarity(brain_values)

rsa_results = {}
for column in params:
    sentence_sim_flat = sentence_similarity.flatten()
    brain_sim_flat = brain_similarity[column].flatten()

    rsa_correlation = np.corrcoef(sentence_sim_flat, brain_sim_flat)[0, 1]
    rsa_results[column] = rsa_correlation

print("Representational Similarity Analysis (RSA) Results:")
for column, rsa_corr in rsa_results.items():
    print(f"  {column}: {rsa_corr:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(list(rsa_results.keys()), list(rsa_results.values()), color='skyblue')
plt.xlabel('Brain Region Parameters', fontsize=14)
plt.ylabel('RSA Correlation', fontsize=14)
plt.title('Representational Similarity Analysis (RSA) Results', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()
