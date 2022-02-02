import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.data.encoders import TorchNormalizer
import pytorch_lightning as pl

def main():
    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str, default=".", help="Model weights path")
    parser.add_argument("--save_dir", type=str, help="Directory where weights and results are saved", default=".")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the data you want to predict",
        default="/hkfs/work/workspace/scratch/bh6321-energy_challenge/data",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    data_dir = args.data_dir
    # load data
    test_data_file = os.path.join(data_dir, "test.csv" )
    if not os.path.exists(test_data_file):
        test_data_file = os.path.join(data_dir, "valid.csv")

    data = pd.read_csv(test_data_file)
    data["Time [s]"] = (pd.to_datetime(data["Time [s]"]).view(np.int64).to_numpy() / 10**9 / 3600).astype(int)
    data["City"] = data.City.astype('category').cat.codes.astype(int)

    print("len(data) ", len(data) )

    # define dataset
    max_encoder_length = 24*7 + 1
    max_prediction_length = 24*7

    testing = TimeSeriesDataSet(
        data,
        time_idx="Time [s]",
        target="Load [MWh]",
        group_ids=["City"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["Load [MWh]"],
        target_normalizer=TorchNormalizer(),
    )

    # create the model
    tft = DeepAR.from_dataset(
        testing,
        learning_rate=0.03,
        hidden_size=32,
        dropout=0.0,
        log_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    tft.load_state_dict(torch.load(os.path.join(args.weights_path, "complete_model.weights")))
    tft.eval()

    trainer = pl.Trainer(
        max_epochs=2,
        gpus=1,
        gradient_clip_val=0.1,
        # limit_train_batches=300,
        # callbacks=[lr_logger, early_stop_callback],
        # max_steps=5
    )

    test_dataloader = testing.to_dataloader(train=False, batch_size=128, num_workers=0)
    print("len(test_dataloader): ", len(test_dataloader))
    predictions = []
    with torch.no_grad():
        for x, y in test_dataloader:
            prediction = tft(x)["prediction"][:, :, 0].detach().cpu().numpy()
            predictions.append(prediction)


    predictions = np.concatenate(predictions, axis=0)
    prediction_df = pd.DataFrame(predictions)
    result_path = os.path.join(save_dir, 'forecasts.csv')
    prediction_df.to_csv(result_path, header=False, index=False)
    print(f"Done! The result is saved in {result_path}")
    print(len(prediction_df))
    return prediction_df


if __name__ == "__main__":

    df = main()

