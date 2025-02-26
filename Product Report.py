# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------

def perform_eda(file_path):
    df = pd.read_csv(file_path)
    
    # Summary statistics
    print(df.describe())
    
    # Visualizing revenue distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Revenue'], bins=20, kde=True)
    plt.title('Revenue Distribution')
    plt.show()
    
    return df
