import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.data.encoders import TorchNormalizer


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="/hkfs/work/workspace/scratch/bh6321-energy_challenge/data/", type=str)
    parser.add_argument("--save_dir", default="/hkfs/work/workspace/scratch/bh6321-E1/weights", help="saves the model, if path is provided")
    args = parser.parse_args()

    # load data
    train_data_file = os.path.join(args.data_dir, "train.csv")
    val_data_file = os.path.join(args.data_dir, "valid.csv")

    data = pd.read_csv(train_data_file)
    data["Time [s]"] = (pd.to_datetime(data["Time [s]"]).view(np.int64).to_numpy() / 10**9 / 3600).astype(int)
    data["Load [MWh]"] = data["Load [MWh]"]
    data["City"] = data.City.astype('category').cat.codes

    # define dataset
    max_encoder_length = 24*7 + 1
    max_prediction_length = 24*7

    # create validation and training dataset
    training = TimeSeriesDataSet(
        data,
        time_idx="Time [s]",
        target="Load [MWh]",
        group_ids=["City"],
        min_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["Load [MWh]"],
        target_normalizer=TorchNormalizer(),
    )

    val_data = pd.read_csv(val_data_file)
    val_data["Time [s]"] = (pd.to_datetime(val_data["Time [s]"]).view(np.int64).to_numpy() / 10 ** 9 / 3600).astype(int)
    val_data["Load [MWh]"] = val_data["Load [MWh]"]

    validation = TimeSeriesDataSet(
        val_data,
        time_idx="Time [s]",
        target="Load [MWh]",
        group_ids=["City"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["Load [MWh]"],
        target_normalizer=TorchNormalizer(),
    )

    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # define trainer with early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=2,
        gpus=1,
        gradient_clip_val=0.1,
        limit_train_batches=300,
        # callbacks=[lr_logger, early_stop_callback],
        # max_steps=5
    )

    # create the model
    tft = DeepAR.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        dropout=0.1,
        log_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit the model
    trainer.fit(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

    # save the feature_extractor_weights:
    state_dict = tft.state_dict()
    torch.save(state_dict, os.path.join(args.save_dir, f"complete_model.weights"))


if __name__ == "__main__":
    main()
