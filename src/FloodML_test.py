import FloodML_train_test
import ERA5_dataset
import torch
import random
import numpy as np
from queue import Queue


def test_FloodML():
    torch.cuda.manual_seed(123)
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    all_station_ids = sorted(open("../data/CAMELS_US/train_basins_ERA5.txt").read().splitlines())
    training_data = ERA5_dataset.Dataset_ERA5(
        dynamic_data_folder=ERA5_dataset.DYNAMIC_DATA_FOLDER_CARAVAN,
        static_data_folder=ERA5_dataset.STATIC_DATA_FOLDER,
        dynamic_attributes_names=[ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN[0]],
        static_attributes_names=ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
        train_start_date="01/10/1999",
        train_end_date="30/09/2008",
        validation_start_date="01/10/1989",
        validation_end_date="30/09/1999",
        test_start_date="01/10/1989",
        test_end_date="30/09/1999",
        stage="train",
        specific_model_type="LSTM",
        all_stations_ids=all_station_ids,
        sequence_length=270,
        discharge_str=ERA5_dataset.DISCHARGE_STR_CARAVAN,
        use_Caravan_dataset=True,
        create_new_files=True,
        sequence_length_spatial=7)
    test_data = ERA5_dataset.Dataset_ERA5(
        dynamic_data_folder=ERA5_dataset.DYNAMIC_DATA_FOLDER_CARAVAN,
        static_data_folder=ERA5_dataset.STATIC_DATA_FOLDER,
        dynamic_attributes_names=[ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN[0]],
        static_attributes_names=ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
        train_start_date="01/10/1999",
        train_end_date="30/09/2008",
        validation_start_date="01/10/1989",
        validation_end_date="30/09/1999",
        test_start_date="01/10/1989",
        test_end_date="30/09/1999",
        stage="validation",
        all_stations_ids=all_station_ids,
        sequence_length=270,
        discharge_str=ERA5_dataset.DISCHARGE_STR_CARAVAN,
        specific_model_type="LSTM",
        use_Caravan_dataset=True,
        y_std=training_data.y_std,
        y_mean=training_data.y_mean,
        x_means=training_data.x_means,
        x_stds=training_data.x_stds,
        create_new_files=True,
        sequence_length_spatial=7)
    nse_queue_single_pass = Queue()
    training_loss_queue_single_pass = Queue()
    queue_preds_dicts_ranks = Queue()
    FloodML_train_test.run_training_and_test(rank=0,
                                             world_size=1,
                                             learning_rate=5 * (10 ** -4),
                                             sequence_length=270,
                                             num_hidden_units=156,
                                             num_epochs=1,
                                             training_data=training_data,
                                             test_data=test_data,
                                             dropout=0,
                                             static_attributes_names=ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
                                             dynamic_attributes_names=[
                                                 ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN[0]],
                                             model_name="LSTM",
                                             nse_queue_single_pass=nse_queue_single_pass,
                                             training_loss_queue_single_pass=training_loss_queue_single_pass,
                                             queue_preds_dicts_ranks=queue_preds_dicts_ranks,
                                             calc_nse_interval=1,
                                             optim_name="Adam",
                                             num_workers_data_loader=0,
                                             profile_code=False,
                                             sequence_length_spatial=7)
    # nse_saved_list = []
    # while not nse_queue_single_pass.empty():
    #     nse_saved_list.append(nse_queue_single_pass.get())
    # np.save("./saved_nse_list_for_testing", nse_saved_list)
    nse_saved_list = np.load("./saved_nse_list_for_testing.npy")
    while not nse_queue_single_pass.empty():
        np.testing.assert_equal(nse_queue_single_pass.get(), nse_saved_list[-1], verbose=True)
        nse_saved_list = nse_saved_list[:-1]


if __name__ == "__main__":
    test_FloodML()
