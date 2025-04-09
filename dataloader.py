import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

class FootballTrackingDataset(Dataset):
    def __init__(
        self,
        data_root: str = "data",
        match_ids: List[int] = [0, 1, 2],
        sequence_length: int = 10,
        transform=None,
    ):
        """
        Args:
            data_root (str): Root directory containing match folders
            match_ids (List[int]): List of match IDs to include in the dataset
            sequence_length (int): Number of consecutive frames to use as input
            transform: Optional transform to be applied on a sample
        """
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load and concatenate all matching files
        self.data = []
        for match_id in match_ids:
            match_dir = self.data_root / f"match_{match_id}"
            
            # Load Home data
            home_path = match_dir / "Home.xlsx"
            if home_path.exists():
                home_df = pd.read_excel(home_path)
            else:
                home_path = match_dir / "Home.csv"
                home_df = pd.read_csv(home_path)
            self.data.append(home_df)
            
            # Load Away data
            away_path = match_dir / "Away.xlsx"
            if away_path.exists():
                away_df = pd.read_excel(away_path)
            else:
                away_path = match_dir / "Away.csv"
                away_df = pd.read_csv(away_path)
            self.data.append(away_df)
        
        self.data = pd.concat(self.data, ignore_index=True)
        
        
        self.player_x_cols = [col for col in self.data.columns if (col.startswith('home') or col.startswith('away')) and col.endswith('_x')]
        self.player_y_cols = [col for col in self.data.columns if (col.startswith('home') or col.startswith('away')) and col.endswith('_y')]
        self.label_cols = ['ball_x', 'ball_y'] if 'ball_x' in self.data.columns else None
        
        # Handle null values and add player presence indicators
        self._handle_null_values()
        # Clip and normalize coordinates
        self._clip_and_normalize_coordinates()
        
        self.feature_cols = [col for col in self.data.columns 
                           if col not in ['MatchId', 'IdPeriod', 'Time','ball_x', 'ball_y']]
        # Create sequences
        self.sequences = []
        # Group by IdPeriod instead of MatchId
        period_groups = self.data.groupby('IdPeriod')
        for _, period_data in period_groups:
            # Create sequences of consecutive frames without sorting
            for i in range(len(period_data) - sequence_length + 1):
                sequence = period_data.iloc[i:i + sequence_length]
                self.sequences.append(sequence)

    def _handle_null_values(self):
        """
        Handle null values in the data:
        - For player positions: fill with 0s and add presence indicators
        - For ball positions: forward fill
        """
        # Create presence indicators for each player and reorder columns
        new_column_order = []
        for x_col, y_col in zip(self.player_x_cols, self.player_y_cols):
            player_num = x_col.split('_')[0]+'_'+x_col.split('_')[1]
            presence_col = f'{player_num}_present'
            
            # Create presence indicator (1 if player is on pitch, 0 if not)
            self.data[presence_col] = (self.data[x_col].isna()).astype(int)
            
            # Fill null positions with 0s
            self.data[x_col] = self.data[x_col].fillna(0)
            self.data[y_col] = self.data[y_col].fillna(0)
            
            # Add columns to new order: x, y, presence
            new_column_order.extend([x_col, y_col, presence_col])
        
        # Add remaining columns (ball coordinates, MatchId, IdPeriod, Time)
        remaining_cols = [col for col in self.data.columns if col not in new_column_order]
        new_column_order.extend(remaining_cols)
        
        # Reorder columns
        self.data = self.data[new_column_order]
        
        # Handle ball positions with forward fill
        if 'ball_x' in self.data.columns and 'ball_y' in self.data.columns:
            self.data['ball_x'] = self.data['ball_x'].ffill()
            self.data['ball_y'] = self.data['ball_y'].ffill()
            
            # If there are still nulls at the start, backfill them
            self.data['ball_x'] = self.data['ball_x'].bfill()
            self.data['ball_y'] = self.data['ball_y'].bfill()
    def _clip_and_normalize_coordinates(self):
        """
        First clip coordinates to valid ranges, then normalize to [-1, 1].
        Ball coordinates are clipped to x:[-5390, 5260] and y:[-3870.00, 3740.00]
        Player coordinates are clipped to x:[-5830.00, 5770.00] and y:[-4020.00, 3960.00]
        """
        # Clip ball coordinates
        if 'ball_x' in self.data.columns and 'ball_y' in self.data.columns:
            self.data['ball_x'] = self.data['ball_x'].clip(-5390, 5260)
            self.data['ball_y'] = self.data['ball_y'].clip(-3870.00, 3740.00)
            
            # Normalize ball coordinates to [-1, 1]
            self.data['ball_x'] = (self.data['ball_x'] / 5390.0)
            self.data['ball_y'] = (self.data['ball_y'] / 3870.0)
        
     
        
        for x_col in self.player_x_cols:
            self.data[x_col] = self.data[x_col].clip(-5830.00, 5770.00)
            self.data[x_col] = (self.data[x_col] / 5830.00)
            
        for y_col in self.player_y_cols:
            self.data[y_col] = self.data[y_col].clip(-4020.00, 3960.00)
            self.data[y_col] = (self.data[y_col] / 4020.00)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Extract features (all positions)
        features = sequence[self.feature_cols].values
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        
        if self.transform:
            features = self.transform(features)
            
        # Get labels (ball coordinates)
        labels = sequence[self.label_cols].values if self.label_cols else torch.empty((0, 2),dtype=torch.float32)
        labels = torch.FloatTensor(labels)
        
        return features, labels



