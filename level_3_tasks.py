import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_predictive_models(df):
    
    # Task 1: Predictive Modeling
    
    print("--------------------------------------------------")
    print(" TASK 1: PREDICTIVE MODELING")
    print("---------------------------------------------------")
    
    # Prepare features
    df['Name_Length'] = df['Restaurant Name'].apply(lambda x: len(str(x)))
    df['Address_Length'] = df['Address'].apply(lambda x: len(str(x)))
    df['Has_Table_Booking_Code'] = (df['Has Table booking'] == 'Yes').astype(int)
    df['Has_Online_Delivery_Code'] = (df['Has Online delivery'] == 'Yes').astype(int)
    
    feature_cols = [
        'Country Code', 'Price range', 'Votes', 
        'Average Cost for two', 'Has_Table_Booking_Code', 
        'Has_Online_Delivery_Code', 'Name_Length', 'Address_Length'
    ]
    
    X = df[feature_cols]
    y = df['Aggregate rating']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=8),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    print("Regression models performance metrics:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"\n> {name}:")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  R2 Score: {r2:.4f}")

def analyze_customer_preferences(df):
    """
    Task 2: Customer Preference Analysis
    """
    print("\n-------------------------------------------------")
    print(" TASK 2: CUSTOMER PREFERENCE ANALYSIS")
    print("---------------------------------------------------")
    
    exploded_df = df.assign(Cuisines=df['Cuisines'].str.split(', ')).explode('Cuisines')
    
    cuisine_votes = exploded_df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)
    print("Top 5 cuisines by total votes:")
    print(cuisine_votes.head(5))
    
    cuisine_counts = exploded_df['Cuisines'].value_counts()
    frequent_cuisines = cuisine_counts[cuisine_counts > 100].index
    
    cuisine_ratings = exploded_df[exploded_df['Cuisines'].isin(frequent_cuisines)] \
                        .groupby('Cuisines')['Aggregate rating'].mean() \
                        .sort_values(ascending=False)
                        
    print("\nHighest rated cuisines (min 100 restaurants):")
    print(cuisine_ratings.head(5).round(2))

def create_visualizations(df):
    
    # Task 3: Data Visualization
    
    print("*************************************************")
    print(" TASK 3: DATA VISUALIZATION")
    print("*****************************************************")
    
    # 1. Bar Plot
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x='Price range', y='Aggregate rating', palette='Blues_d')
    plt.title('Average Rating by Price Range', fontsize=12, fontweight='bold')
    plt.xlabel('Price Range')
    plt.ylabel('Mean Aggregate Rating')
    plt.tight_layout()
    plt.savefig('price_vs_rating.png')
    plt.show()
    plt.close()
    
    # 2. Scatter Plot
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x='Votes', y='Aggregate rating', alpha=0.5, color='#d95f02')
    plt.title('Votes vs Aggregate Rating', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Votes')
    plt.ylabel('Aggregate Rating')
    plt.tight_layout()
    plt.savefig('votes_vs_rating.png')
    plt.show()
    plt.close()
    
    print("Plots saved: 'price_vs_rating.png' and 'votes_vs_rating.png'.")

if __name__ == "__main__":
    dataset_path = r"C:\Users\Seena\Desktop\cognifyz_project\Dataset.csv"
    
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
        data['Cuisines'] = data['Cuisines'].fillna('Unknown')
        
        train_predictive_models(data)
        analyze_customer_preferences(data)
        create_visualizations(data)
        print("\n[SUCCESS] Level 3 analysis completed successfully.")
    else:
        print(f"[ERROR] Dataset file not found at: {dataset_path}")