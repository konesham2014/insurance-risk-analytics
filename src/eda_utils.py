import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_data(df):
    print(df.info())
    print(df.describe())


def check_missing(df):
    missing = df.isnull().sum()
    print(missing[missing > 0])


def calculate_loss_ratio(df):
    total_claims = df["TotalClaims"].sum()
    total_premium = df["TotalPremium"].sum()

    ratio = total_claims / total_premium

    print(f"Overall Loss Ratio: {ratio:.2f}")

    return ratio


def plot_histogram(df, column):
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_boxplot(df, column):
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


def correlation_heatmap(df, columns):
    plt.figure(figsize=(10,6))

    sns.heatmap(
        df[columns].corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Matrix")
    plt.show()