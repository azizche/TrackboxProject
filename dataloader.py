import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_ball_possession, MAX_X, MAX_Y, MIN_X, MIN_Y



class FootballTrackingDataset(Dataset):
    def __init__(self, match_ids, sequence_length=10,data_path='data'):
        self.match_ids = match_ids
        self.sequence_length = sequence_length
        self.data_path = data_path
        self.data= []
        for match_id in self.match_ids:
            home=[]
            away=[]
            match_path = Path(f'{self.data_path}/match_{match_id}')
            
            if (match_path / 'Home_cleaned.csv').exists():
                data_home = pd.read_csv(match_path / 'Home_cleaned.csv')
                data_away = pd.read_csv(match_path / 'Away_cleaned.csv')
            else:
                data_home = pd.read_excel(match_path / 'Home_cleaned.xlsx')
                data_away = pd.read_excel(match_path / 'Away_cleaned.xlsx')
            
            for _ , grp in data_home.groupby("IdPeriod"):
                home.append(grp)
            for _ , grp in data_away.groupby("IdPeriod"):
                away.append(grp)
            
            
            first_half = pd.merge(home[0],away[0],on="Time",how="inner",suffixes=("","_duplicate"))
            second_half = pd.merge(home[1],away[1],on="Time",how="inner",suffixes=("","_duplicate"))
            
            assert first_half.duplicated(subset=["Time"]).sum() == 0
            assert second_half.duplicated(subset=["Time"]).sum() == 0
            for col in first_half.columns:
                if col.endswith("_duplicate"):
                    first_half.drop(columns=[col],inplace=True)
            for col in second_half.columns:
                if col.endswith("_duplicate"):
                    second_half.drop(columns=[col],inplace=True)
            
            self.data.append(first_half)
            self.data.append(second_half)
            
        self.data_lengths = np.cumsum([len(data)-self.sequence_length for data in self.data])
        self.ball_col_names = ["ball_x", "ball_y"]
    
    def _handle_null_values(self, sequence):
       
   
        new_sequence = sequence.copy()
        new_column_order = []
        player_x_cols = [col for col in sequence.columns if (col.startswith('home') or col.startswith('away')) and col.endswith('_x')]
        player_y_cols = [col for col in sequence.columns if (col.startswith('home') or col.startswith('away')) and col.endswith('_y')]
        for x_col, y_col in zip(player_x_cols, player_y_cols):
            player_num = x_col.split('_')[0]+'_'+x_col.split('_')[1]
            presence_col = f'{player_num}_present'
            
            new_sequence[presence_col] = 1-((new_sequence[x_col].isna()) | (new_sequence[x_col]>10000.00 )| (new_sequence[y_col]>10000.00)).astype(int)
            
            new_sequence[x_col] = new_sequence[x_col].fillna(0)
            new_sequence[y_col] = new_sequence[y_col].fillna(0)
            new_sequence.loc[new_sequence[x_col]>10000.00, x_col] = 0
            new_sequence.loc[new_sequence[y_col]>10000.00, y_col] = 0
            
            new_column_order.extend([x_col, y_col, presence_col])
        
        remaining_cols = [col for col in sequence.columns if col not in new_column_order]
        new_column_order.extend(remaining_cols)
        
        new_sequence = new_sequence[new_column_order]
        
        return new_sequence
    
    def normalize_coordinates(self, sequence):
        
        player_x_cols = [col for col in sequence.columns if (col.startswith('home') or col.startswith('away')) and col.endswith('_x')]
        player_y_cols = [col for col in sequence.columns if (col.startswith('home') or col.startswith('away')) and col.endswith('_y')]
    
        for x_col in player_x_cols:
            
            sequence[x_col] = (sequence[x_col] / MAX_X)   
            
        for y_col in player_y_cols:
    
            sequence[y_col] = (sequence[y_col] / MAX_Y)
        if 'ball_x' in sequence.columns and 'ball_y' in sequence.columns:
           
            sequence['ball_x'] = (sequence['ball_x'] / MAX_X)
            sequence['ball_y'] = (sequence['ball_y'] / MAX_Y)
        
     
        return sequence
    
    def __len__(self):
        
        return self.data_lengths[-1]
    
    def __getitem__(self, index):
        for i, data_length in enumerate(self.data_lengths):
            if index < data_length:
                current_data = self.data[i]
                if i!=0:
                    index -= self.data_lengths[i-1]
                break
        
        
        start_idx = index 
        end_idx = start_idx + self.sequence_length
        
        sequence = current_data.iloc[start_idx:end_idx]
        
        sequence = self._handle_null_values(sequence)
        
        sequence = self.normalize_coordinates(sequence)
        player_col_names = [col for col in sequence.columns if col not in ["MatchId", "IdPeriod", "Time", "ball_x", "ball_y"]]
        features = sequence.loc[:, player_col_names].values
        
        if 'ball_x' in sequence.columns and 'ball_y' in sequence.columns:
            ball_features = sequence.loc[:, self.ball_col_names].values
            ball_features = torch.tensor(ball_features, dtype=torch.float32)
        else:
            ball_features = torch.empty((self.sequence_length, 2))
            
        features = torch.tensor(features, dtype=torch.float32)
        
        return features, ball_features
        


