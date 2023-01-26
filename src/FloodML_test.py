import pytest
import FloodML_train_test
import ERA5_dataset


def test_FloodML():
    training_data = ERA5_dataset.Dataset_ERA5(
        dynamic_data_folder=ERA5_dataset.DYNAMIC_DATA_FOLDER_CARAVAN,
        static_data_folder=ERA5_dataset.STATIC_DATA_FOLDER,
        dynamic_attributes_names=ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN,
        static_attributes_names=ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
        train_start_date="01/10/1999",
        train_end_date="30/09/2008",
        validation_start_date="01/10/1989",
        validation_end_date="30/09/1999",
        test_start_date="01/10/1989",
        test_end_date="30/09/1999",
        stage="train",
        specific_model_type="LSTM",
        all_stations_ids=all_station_ids_train,
        sequence_length=270,
        discharge_str=ERA5_dataset.DISCHARGE_STR_CARAVAN,
        use_Caravan_dataset=True
    )
    test_data = ERA5_dataset.Dataset_ERA5(
        dynamic_data_folder=ERA5_dataset.DYNAMIC_DATA_FOLDER_CARAVAN,
        static_data_folder=ERA5_dataset.STATIC_DATA_FOLDER,
        dynamic_attributes_names=ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN,
        static_attributes_names=ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
        train_start_date="01/10/1999",
        train_end_date="30/09/2008",
        validation_start_date="01/10/1989",
        validation_end_date="30/09/1999",
        test_start_date="01/10/1989",
        test_end_date="30/09/1999",
        stage="validation",
        all_stations_ids=all_station_ids_test,
        sequence_length=270,
        discharge_str=ERA5_dataset.DISCHARGE_STR_CARAVAN,
        specific_model_type="LSTM",
        use_Caravan_dataset=True,
        x_means=training_data.x_means,
        x_stds=training_data.x_stds,
        y_std=training_data.y_std,
        y_mean=training_data.y_mean
    )
    FloodML_train_test.run_training_and_test(rank=0,
                                             world_size=1,
                                             learning_rate=5 * (10 ** -4),
                                             sequence_length=270,
                                             num_hidden_units=156,
                                             num_epochs=1,
                                             training_data=training_data,
                                             test_data=test_data,
                                             dropout=0.25,
                                             static_attributes_names=ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
                                             dynamic_attributes_names=ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN,
                                             model_name="LSTM",
                                             calc_nse_interval=1,
                                             optim_name="Adam",
                                             num_workers_data_loader=2,
                                             profile_code=False)
