import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloader import FootballTrackingDataset
from model import FootballTrackingLSTM
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from dotenv import load_dotenv
import neptune
from neptune.types import File
from lightning.pytorch.loggers import NeptuneLogger
from utils import MAX_X, MAX_Y, MIN_X, MIN_Y, get_pass_history
from utils import convert_relative_coordinates_to_real_coordinates

load_dotenv()

class FootballTrackingLightning(pl.LightningModule):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, learning_rate=1e-3, batch_size=32, sequence_length=10):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = FootballTrackingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def _calculate_euclidean_distance(self, predictions, labels):
        denorm_labels = labels.clone()
        denorm_predictions = predictions.clone()
        
        denorm_labels[:, :, 0] = denorm_labels[:, :, 0] * MAX_X
        denorm_labels[:, :, 1] = denorm_labels[:, :, 1] * MAX_Y
        denorm_predictions[:, :, 0] = denorm_predictions[:, :, 0] * MAX_X
        denorm_predictions[:, :, 1] = denorm_predictions[:, :, 1] * MAX_Y
        denorm_labels[:, :, 0], denorm_labels[:, :, 1] = convert_relative_coordinates_to_real_coordinates(denorm_labels[:, :, 0], denorm_labels[:, :, 1])
        denorm_predictions[:, :, 0], denorm_predictions[:, :, 1] = convert_relative_coordinates_to_real_coordinates(denorm_predictions[:, :, 0], denorm_predictions[:, :, 1])
        euclidean_distance = torch.sqrt(torch.sum((denorm_predictions - denorm_labels) ** 2, dim=2))
        return torch.mean(euclidean_distance)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.model(features)
        loss = self.criterion(predictions, labels)
        mean_distance = self._calculate_euclidean_distance(predictions, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_euclidean_distance', mean_distance, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.model(features)
        loss = self.criterion(predictions, labels)
        mean_distance = self._calculate_euclidean_distance(predictions, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_euclidean_distance', mean_distance, on_step=True, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    torch.set_float32_matmul_precision('medium')
    print(OmegaConf.to_yaml(cfg))
    neptune_logger = NeptuneLogger()
    os.makedirs("checkpoints", exist_ok=True)
    
    if cfg.mode == "train":
        train_dataset = FootballTrackingDataset(
            data_path="data",
            match_ids=[0,1,2],
            sequence_length=cfg.data.sequence_length
        )

        input_size = 84
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=30
        )

        val_dataset = FootballTrackingDataset(
            data_path="data",
            match_ids=[3],
            sequence_length=cfg.data.sequence_length
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=30
        )

        model = FootballTrackingLightning(
            input_size=input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            learning_rate=cfg.train.learning_rate,
            batch_size=cfg.train.batch_size,
            sequence_length=cfg.data.sequence_length
        )
        
        trainer = pl.Trainer(
            max_epochs=cfg.train.num_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=neptune_logger,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    dirpath="checkpoints",
                    filename='football_tracking-{epoch:02d}-{val_loss:.4f}',
                    save_top_k=1,
                    mode='min',
                ),
            ],
        )
        
        trainer.fit(model, train_loader,val_loader)
    
    elif cfg.mode == "predict":
        model = FootballTrackingLightning.load_from_checkpoint(cfg.chkpt_path)
        predict_dataset = FootballTrackingDataset(
            data_path="data",
            match_ids=[4],
            sequence_length=cfg.data.sequence_length
        )
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=30
        )
        
        predictions, labels = predict_match_4(
            model,
            predict_loader=predict_loader
        )

def predict_match_4(model, predict_loader):
    model.eval()
    all_ball_positions = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(predict_loader):
            features = features.to(device)
            predictions = model(features)
            predictions = predictions.cpu().numpy()
            if batch_idx == 0:
                all_predictions.append(predictions[0])
                all_predictions.append(predictions[1:,-1])
            else:
                last_prediction = predictions[:, -1]  
                all_predictions.append(last_prediction)
                
    all_predictions = np.concatenate(all_predictions, axis=0)
    print(all_predictions.shape)
    
    all_predictions[:, 0] = all_predictions[:, 0] * MAX_X
    all_predictions[:, 1] = all_predictions[:, 1] * MAX_Y
    
    results_df_away = pd.read_csv("data/match_4/Away_cleaned.csv")
    results_df_away["ball_x"] = all_predictions[:, 0]
    results_df_away["ball_y"] = all_predictions[:, 1]
    
    results_df_home = pd.read_csv("data/match_4/Home_cleaned.csv")
    results_df_home["ball_x"] = all_predictions[:, 0]
    results_df_home["ball_y"] = all_predictions[:, 1]
    
    results_df_home.to_csv("data/match_4/Home_cleaned_with_predictions.csv", index=False)
    results_df_away.to_csv("data/match_4/Away_cleaned_with_predictions.csv", index=False)
    print("Predictions saved to Home_cleaned_with_predictions.csv and Away_cleaned_with_predictions.csv")
    
    return all_predictions

if __name__ == "__main__":
    main()
