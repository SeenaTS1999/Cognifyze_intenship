import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "Arial"

def load_and_preprocess(file_path):
    
    # Task 1: Data exploration and preprocessing
    
    df = pd.read_csv(file_path)
    
    print("--------------------------------------------------")
    print(" TASK 1: DATA EXPLORATION & PREPROCESSING")
    print("--------------------------------------------------")
    print(f"Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Missing values check
    missing_vals = df.isnull().sum()
    print("Missing values per column:\n", missing_vals[missing_vals > 0])
    
    # Fill missing values in Cuisines column
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    
    # Target variable distribution
    print("\nTarget variable ('aggregate rating') statistics:")
    print(df['Aggregate rating'].describe().round(2))
    
    # Save target distribution plot
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Aggregate rating'], bins=20, kde=True, color='#2b5c8f')
    plt.title('Distribution of aggregate ratings', fontsize=12, fontweight='bold')
    plt.xlabel('Aggregate Rating')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rating_distribution.png')
    plt.show()
    plt.close()
    
    return df

def perform_descriptive_analysis(df):
    
    # Task 2: Descriptive analysis
    
    print("\n------------------------------------------------")
    print(" TASK 2: DESCRIPTIVE ANALYSIS")
    print("---------------------------------------------------")
    
    num_cols = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
    print("Summary statistics for numerical columns:")
    print(df[num_cols].describe().round(2))
    
    print("\nTop 5 cities by restaurant count:")
    print(df['City'].value_counts().head(5))
    
    # Explode multi-cuisine values to count frequency correctly
    cuisines_series = df['Cuisines'].str.split(', ').explode()
    print("\nTop 5 most frequent cuisines:")
    print(cuisines_series.value_counts().head(5))

def perform_geospatial_analysis(df):
    
    # Task 3: Geospatial Analysis
    
    print("\n-------------------------------------------------")
    print(" TASK 3: GEOSPATIAL ANALYSIS")
    print("---------------------------------------------------")
    
    # Plot restaurant distribution by geography
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='Longitude', 
        y='Latitude', 
        hue='Aggregate rating', 
        palette='viridis', 
        alpha=0.6,
        size='Votes',
        sizes=(10, 200)
    )
    plt.title('Geographic distribution of restaurants by rating', fontsize=12, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('restaurant_geospatial_map.png')
    plt.show()
    plt.close()
    
    corr = df[['Longitude', 'Latitude', 'Aggregate rating']].corr()
    print("Correlation matrix (location vs rating):")
    print(corr.round(3))

if __name__ == "__main__":
    # Path setup for local environment
    dataset_path = r"C:\Users\Seena\Desktop\cognifyz_project\Dataset.csv"
    
    if os.path.exists(dataset_path):
        data = load_and_preprocess(dataset_path)
        perform_descriptive_analysis(data)
        perform_geospatial_analysis(data)
        print("\n[SUCCESS] Level 1 analysis completed successfully.")
    else:
        print(f"[ERROR] Dataset file not found at: {dataset_path}")