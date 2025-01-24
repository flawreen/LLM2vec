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

# Representational Similarity Analysis with scatter plots and regression
plt.figure(figsize=(15, 10))
for i, column in enumerate(params):
    sentence_sim_flat = sentence_similarity.flatten()
    brain_sim_flat = brain_similarity[column].flatten()

    # Pearson correlation
    rsa_correlation = np.corrcoef(sentence_sim_flat, brain_sim_flat)[0, 1]

    # Plot scatter with regression
    plt.subplot(2, 3, i + 1)
    sns.regplot(x=sentence_sim_flat, y=brain_sim_flat, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title(f"{column} (Pearson: {rsa_correlation:.4f})", fontsize=12)
    plt.xlabel("Sentence Similarity", fontsize=10)
    plt.ylabel("Brain Activation Similarity", fontsize=10)

plt.tight_layout()
plt.suptitle("RSA Scatter Plots with Regression Lines", fontsize=16, y=1.02)
plt.show()
