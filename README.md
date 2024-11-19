# EDA: Exploratory Data Analysis Framework
# Overview
The EDA class provides an easy-to-use framework for performing comprehensive exploratory data analysis on a given dataset. It includes features for summarizing statistics, checking for missing values, visualizing distributions, and analyzing correlations and categorical features.

# Features
Summary Statistics: Provides descriptive statistics for all numeric features.
Missing Values Check: Identifies missing values in each column.
Distribution Plots: Visualizes the distribution of numeric features.
Correlation Heatmap: Displays correlations among numeric features using a heatmap.
Categorical Analysis: Generates bar plots for categorical features.
Pair Plots: Creates pairwise scatter plots of numeric features, with an optional target variable for hue.
Box Plots: Shows box plots of numeric features grouped by the target variable.
Run All: Executes all EDA functions in one command for a comprehensive analysis.
# Installation
Make sure you have the required libraries installed:

pip install pandas seaborn matplotlib scikit-learn numpy

# Usage
Here's how to use the EDA class in your Python script:


# Import libraries and load a sample dataset
from sklearn.datasets import load_breast_cancer
from EDA import EDA  # Assuming the EDA class is saved as EDA.py

# Load a sample dataset (Breast Cancer dataset from scikit-learn)
data = load_breast_cancer(as_frame=True).frame

# Initialize the EDA class with your data
eda = EDA(data)

# Run all EDA steps, specifying the target column for analysis
eda.run_all(target_column='target')
# Methods
summary_statistics()
# Description: Prints descriptive statistics of numeric columns.
missing_values()
# Description: Prints the count of missing values in each column.
distribution_plots()
# Description: Plots the distribution of each numeric feature, including a Kernel Density Estimate (KDE) curve.
correlation_heatmap()
Description: Generates a heatmap of the correlation matrix for numeric features.
categorical_analysis()
# Description: Creates bar plots for each categorical feature in the dataset.
pair_plot(target_column=None)
Description: Generates a pair plot for numeric features, using the target column for hue if specified.
# Parameters:
target_column (str, optional): The column name for the target variable.
box_plots(target_column)
Description: Plots box plots for numeric features, grouped by the target variable.
# Parameters:
target_column (str): The name of the target column.
run_all(target_column=None)
Description: Runs all the EDA methods. If a target column is provided, generates pair plots and box plots based on the target.
# Parameters:
target_column (str, optional): The column name for the target variable.
# Example Dataset
The example uses the Breast Cancer dataset from scikit-learn, which is ideal for demonstrating classification-related EDA.

# Author
Idriss Olivier BADO

# License
This project is open-source and available for modification and distribution. Use it as you see fit.


