import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_booking_and_delivery(df):
    
    # Task 1: Table Booking & Online Delivery Analysis
    
    print("--------------------------------------------------")
    print(" TASK 1: TABLE BOOKING & ONLINE DELIVERY ANALYSIS")
    print("--------------------------------------------------")
    
    total = len(df)
    tb_pct = (df['Has Table booking'] == 'Yes').sum() / total * 100
    od_pct = (df['Has Online delivery'] == 'Yes').sum() / total * 100
    
    print(f"Restaurants with table booking: {tb_pct:.2f}%")
    print(f"Restaurants with online delivery: {od_pct:.2f}%\n")
    
    tb_rating = df.groupby('Has Table booking')['Aggregate rating'].mean()
    print("Average rating by table booking availability:")
    print(tb_rating.round(2))
    
    od_price = pd.crosstab(df['Price range'], df['Has Online delivery'], normalize='index') * 100
    print("\nOnline belivery availability % /by Price Range:")
    print(od_price.round(2))

def analyze_price_ranges(df):
    
    # Task 2: Price Range Analysis
    
    print("-------------------------------------------------")
    print(" TASK 2: PRICE RANGE ANALYSIS")
    print("---------------------------------------------------")
    
    most_common = df['Price range'].mode()[0]
    print(f"Most common price range: {most_common}\n")
    
    avg_rating_price = df.groupby('Price range')['Aggregate rating'].mean()
    print("Average rating for each price range:")
    print(avg_rating_price.round(2))
    
    color_ratings = df.groupby('Rating color')['Aggregate rating'].mean().sort_values(ascending=False)
    print("Highest average rating by color code:")
    print(color_ratings.round(2))

def perform_feature_engineering(df):
    
    # Task 3: Feature Engineering
    
    print("------------------------------------------------")
    print(" TASK 3: FEATURE ENGINEERING")
    print("--------------------------------------------------")
    
    # Calculate character lengths
    df['Name_Length'] = df['Restaurant Name'].apply(lambda x: len(str(x)))
    df['Address_Length'] = df['Address'].apply(lambda x: len(str(x)))
    
    
    df['Has_Table_Booking_Code'] = (df['Has Table booking'] == 'Yes').astype(int)
    df['Has_Online_Delivery_Code'] = (df['Has Online delivery'] == 'Yes').astype(int)
    
    print("Sample of newly engineered features:")
    print(df[['Restaurant Name', 'Name_Length', 'Address_Length', 'Has_Table_Booking_Code', 'Has_Online_Delivery_Code']].head())
    
    return df

if __name__ == "__main__":
    dataset_path = r"C:\Users\Seena\Desktop\cognifyz_project\Dataset.csv"
    
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
        data['Cuisines'] = data['Cuisines'].fillna('Unknown')
        
        analyze_booking_and_delivery(data)
        analyze_price_ranges(data)
        data = perform_feature_engineering(data)
        print("\n[SUCCESS] Level 2 analysis completed successfully.")
    else:
        print(f"[ERROR] dataset file not found at: {dataset_path}")