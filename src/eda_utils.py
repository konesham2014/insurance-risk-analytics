import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    return pd.read_csv(path)

def summary_stats(df):
    return df.describe()

def missing_values(df):
    return df.isnull().sum()

def plot_hist(df, col):
    plt.figure()
    sns.histplot(df[col])
    plt.title(f"Distribution of {col}")
    plt.show()

def plot_box(df, col):
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()