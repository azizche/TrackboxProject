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

# Load environment variables
load_dotenv()

class FootballTrackingLightning(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        sequence_length: int = 10
    ):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features, labels = batch
        
        predictions = self(features)
        loss = self.criterion(predictions, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        features, labels = batch
        predictions = self(features)
        loss = self.criterion(predictions, labels)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('medium')
    # Print configuration
    print(OmegaConf.to_yaml(cfg))
    neptune_logger = NeptuneLogger()
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = FootballTrackingDataset(
        data_root="data",
        match_ids=[0],
        sequence_length=cfg.data.sequence_length
    )

    # Calculate input size based on feature columns
    input_size = len(train_dataset.feature_cols)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4
    )

    val_dataset = FootballTrackingDataset(
        data_root="data",
        match_ids=[3],
        sequence_length=cfg.data.sequence_length
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = FootballTrackingLightning(
        input_size=input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        learning_rate=cfg.train.learning_rate,
        batch_size=cfg.train.batch_size,
        sequence_length=cfg.data.sequence_length
    )
    
    # Initialize trainer
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
    
    # Train the model
    trainer.fit(model, train_loader,val_loader)
    
    # Make predictions on match 4
    predictions, labels = predict_match_4(
        model,
        data_root="data",
        match_ids=[4],
        sequence_length=cfg.data.sequence_length,
        batch_size=cfg.train.batch_size,
        num_workers=4
    )



def predict_match_4(
    model: FootballTrackingLightning,
    data_root: str,
    match_ids: list,
    sequence_length: int,
    batch_size: int,
    num_workers: int
):
    # Load prediction data
    predict_dataset = FootballTrackingDataset(
        data_root=data_root,
        match_ids=match_ids,
        sequence_length=sequence_length
    )
    
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Store predictions
    all_predictions = []
    
    with torch.no_grad():
        for features, _ in predict_loader:
            predictions = model(features)
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'predicted_ball_x': predictions[:, -1, 0],  # Last time step predictions
        'predicted_ball_y': predictions[:, -1, 1],
    })
    
    results_df.to_csv('match_4_predictions.csv', index=False)
    print("Predictions saved to match_4_predictions.csv")
    
    return predictions

if __name__ == "__main__":
    main()
