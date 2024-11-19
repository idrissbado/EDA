# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 06:58:33 2024

@author: Idriss Olivier BADO
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class EDA:
    def __init__(self, data):
        """
        Initialize the EDA class with the dataset.

        Parameters:
        data (DataFrame): The pandas DataFrame containing the data.
        """
        self.data = data

    def summary_statistics(self):
        """
        Generate summary statistics for the dataset.
        """
        print("Summary Statistics:")
        print(self.data.describe())
        print("\n")

    def missing_values(self):
        """
        Check for missing values in the dataset.
        """
        print("Missing Values:")
        print(self.data.isnull().sum())
        print("\n")

    def distribution_plots(self):
        """
        Plot the distribution of all numeric features in the dataset.
        """
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            plt.figure(figsize=(10, 4))
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()

    def correlation_heatmap(self):
        """
        Generate a correlation heatmap for the numeric features in the dataset.
        """
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()

    def categorical_analysis(self):
        """
        Generate a bar plot for each categorical feature in the dataset.
        """
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(x=self.data[column], palette='Set2')
            plt.title(f'Count of {column}')
            plt.xticks(rotation=45)
            plt.show()

    def pair_plot(self, target_column=None):
        """
        Generate a pair plot for the numeric features in the dataset.
        Optionally, include the target column for hue.

        Parameters:
        target_column (str): The column name for the target variable (optional).
        """
        if target_column:
            sns.pairplot(self.data, hue=target_column, diag_kind='kde')
        else:
            sns.pairplot(self.data, diag_kind='kde')
        plt.show()

    def box_plots(self, target_column):
        """
        Generate box plots for numeric features grouped by the target variable.

        Parameters:
        target_column (str): The column name for the target variable.
        """
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=self.data[target_column], y=self.data[column], palette='Set2')
            plt.title(f'Box Plot of {column} grouped by {target_column}')
            plt.show()

    def run_all(self, target_column=None):
        """
        Run all EDA functions: summary statistics, missing values, distributions, correlations,
        categorical analysis, pair plots, and box plots.

        Parameters:
        target_column (str): The column name for the target variable (optional).
        """
        self.summary_statistics()
        self.missing_values()
        self.distribution_plots()
        self.correlation_heatmap()
        self.categorical_analysis()
        if target_column:
            self.pair_plot(target_column)
            self.box_plots(target_column)
        else:
            self.pair_plot()

# Example Usage:
if __name__ == "__main__":
    # Load a sample dataset
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True).frame

    # Initialize the EDA class
    eda = EDA(data)

    # Run all EDA steps (replace 'target' with the appropriate column in your dataset)
    eda.run_all(target_column='target')
