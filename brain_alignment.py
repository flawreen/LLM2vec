import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "bold_response_LH.csv"
data = pd.read_csv(file_path)
params = ['lang_LH_AntTemp', 'lang_LH_IFG', 'lang_LH_IFGorb', 'lang_LH_MFG', 'lang_LH_PostTemp']


def plot_similarity(data, column):
    if column in data.columns:
        values = data[column].values.reshape(-1, 1)
        similarity = cosine_similarity(values)

        plt.figure(figsize=(16, 10))
        sns.heatmap(similarity, cmap="coolwarm", annot=False, xticklabels=False, yticklabels=False)
        plt.title(f"Similaritate Cosinus între propoziții ({column})")
        plt.show()


for param in params:
    plot_similarity(data, param)
