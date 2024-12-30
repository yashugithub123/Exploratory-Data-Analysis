import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
def load_data( file_path):
    return pd.read_csv(file_path)

# Perform EDA
def perform_eda(data):
    print("\nDataset Overview:")
    print(data.head())

    print("\nDataset Info:")
    print(data.info())

    print("\nSummary Statistics:")
    print(data.describe())

    print("\nMissing Values:")
    print(data.isnull().sum())

# Visualizations
def visualize_data(data):
    # Histograms
    data.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

    # Pairplot
    if len(data.select_dtypes(include=['float64', 'int64']).columns) <= 5:
        sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
        plt.suptitle("Pairplot of Features", y=1.02)
        plt.show()

    # Scatter Plots for relationships
    if 'target' in data.columns:
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            if column != 'target':
                plt.figure(figsize=(8, 5))
                sns.scatterplot(x=data[column], y=data['target'])
                plt.title(f"Scatter Plot: {column} vs Target")
                plt.show()

    # Boxplots for outlier detection
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        plt.show()

# Main function
def main():
    file_path = "diabetes_dataset.csv"  # Replace with your dataset path
    data = load_data("diabetes_dataset.csv")

    # Perform EDA
    perform_eda(data)

    # Visualize Data
    visualize_data(data)

if __name__ == "__main__":
    main()
