import pandas as pd
from pathlib import Path
import os

def process_team_data(match_dir, team):
    
    file_path = match_dir / f"{team}.xlsx"
    if not file_path.exists():
        file_path = match_dir / f"{team}.csv"
    
    df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
    original_rows = len(df)
    
    df = df.dropna(subset=['ball_x', 'ball_y'])
    
    df = df[(df['ball_x'] <= 10000) & (df['ball_x'] >= -10000) & (df['ball_y'] <= 10000) & (df['ball_y'] >= -10000)]
    cleaned_rows = len(df)
    
    output_path = match_dir / f"{team}_cleaned{file_path.suffix}"
    df.to_csv(output_path, index=False) if file_path.suffix == '.csv' else df.to_excel(output_path, index=False)
    
    return df, original_rows, cleaned_rows

def clean_match_data(data_root='data', match_ids = [0, 1, 2, 3]):
    print("Starting data cleaning...")
    
    for match_id in match_ids:
        print(f"\nProcessing match {match_id}...")
        match_dir = Path(data_root) / f"match_{match_id}"
        
        _, home_original, home_cleaned = process_team_data(match_dir, "Home")
        print(f"Home data: {home_original} -> {home_cleaned} rows (removed {home_original - home_cleaned} rows)")
        
        _, away_original, away_cleaned = process_team_data(match_dir, "Away")
        print(f"Away data: {away_original} -> {away_cleaned} rows (removed {away_original - away_cleaned} rows)")
    
    print("\nData cleaning completed successfully!")

if __name__ == "__main__":
    clean_match_data() 
    df_home = pd.read_csv("data/match_4/Home.csv")
    df_away = pd.read_csv("data/match_4/Away.csv")
    df_home.to_csv("data/match_4/Home_cleaned.csv", index=False)
    df_away.to_csv("data/match_4/Away_cleaned.csv", index=False)
    
    