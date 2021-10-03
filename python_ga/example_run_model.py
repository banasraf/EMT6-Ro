import sys
import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer
from sklearn.preprocessing import StandardScaler

from utils import ConvertRepresentation, get_rand_population, read_config
from genetic_algorithm import new_genetic_algorithm

from emt6ro.simulation import load_state, load_parameters
from absl import app, flags
import wget
import importlib
import os

flags.DEFINE_string('model_path', 'models/model_1.ckpt', 'Relative path to model')
FLAGS = flags.FLAGS


class MockPredictionModel:
    """ Model for predicting output based on already trained ML model """
    TIMESERIES_PATH = 'models/dataset_time_series'
    
    def __init__(self, model):
        self.model = model
    
    def prepare_data(self, data):
        """ 
            Converting data from list of times and doses to DataFrame format
        
            parameters: self, list of  pairs (time, dose)
            
            returns: The same but in the same format as dataset
        """
        df = pd.DataFrame(columns=['Unnamed', 'time', 'dose', 'series', 'time_idx', 'is_target', 'target'])
        
        for series, protocol in enumerate(data):
            zeros_len = 20 - len(protocol)
            for i in range(zeros_len):
                df = df.append({
                       'Unnamed': i,
                       'time': 0,
                       'dose': 0,
                       'series': series,
                       'time_idx': i,
                       'is_target': 0,
                       'target': 0
                }, ignore_index = True)

            
            for i, (time, dose) in enumerate(protocol):
                df = df.append({
                       'Unnamed': zeros_len + i,
                       'time': time,
                       'dose': dose,
                       'series': series,
                       'time_idx': zeros_len + i,
                       'is_target': 0,
                       'target': 0
                }, ignore_index = True)

            df = df.append({
                       'Unnamed': 20,
                       'time': 0,
                       'dose': 0,
                       'series': series,
                       'time_idx': 20,
                       'is_target': 1,
                       'target': 0
                }, ignore_index = True)
                
        df['target'] = df['target'].astype(float)

        df['Unnamed'] = df['Unnamed'].astype(int)
        df['time_idx'] = df['time_idx'].astype(int)
        df['is_target'] = df['is_target'].astype(int)
        df['series'] = df['series'].astype(int)
        return df
    
    def predict(self, data):
        """ Transforms data and predicts output based on train model 
        
            Parameters: self, list of protocols
            
            Return: list of results for each protocol based on train model
        """
        print(data)
        self.save_time_series()
        dataset = self.prepare_data(data)
        
        time_series = TimeSeriesDataSet.load(self.TIMESERIES_PATH)
        validation = TimeSeriesDataSet.from_dataset(time_series, dataset)
        
        val_dataloader = validation.to_dataloader(train=False, num_workers=0)
    
        res = self.model.predict(val_dataloader)
#         print("wynik", res)
        res = np.array([int(x) for x in res])
        
        return res

    def save_time_series(self):
        """ Download preprocessing file and creates data in a format suited for temporal fusion """
        PREPROCESS_URL = 'https://raw.githubusercontent.com/AWarno/CancerOptimization/main/preprocess_data.py'
        FILE_PATH = 'data/preprocess_data.py'
        DATA_PATH = 'data/data.csv'
        FEATURES = ['dose', 'time']
        GROUP_ID = 'series'
        
        # Data file already exists so we don't need to generate it
        if os.path.isfile(DATA_PATH):
            return
        
        # Preprocessing file already exists so we don't need to download it again
        if not os.path.isfile(FILE_PATH):
            wget.download(PREPROCESS_URL, FILE_PATH)
        
        os.system('python ' + FILE_PATH)
        
        dataset = pd.read_csv(DATA_PATH)

        n = dataset[GROUP_ID].astype(int).max()

        dataset['target'] = dataset['target'].astype(float)

        dataset['time_idx'] = dataset['time_idx'].astype(int)

        training = TimeSeriesDataSet(
            dataset[dataset[GROUP_ID].apply(lambda x: int(x) < int(n * 0.7))],
            time_idx='time_idx',
            target='target',
            group_ids=[GROUP_ID],
            min_encoder_length=20,  
            max_encoder_length=20,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            variable_groups={},
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=['target'] + FEATURES,
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
            categorical_encoders={GROUP_ID: NaNLabelEncoder().fit(dataset.series)},
        )
        
        training.save(self.TIMESERIES_PATH)
        
    @staticmethod
    def test_model(model_path):
        """ Tests results of given model on dataset """
        DATA_PATH = 'data/data.csv'
        if not os.path.isfile(FILE_PATH):
            wget.download(PREPROCESS_URL, FILE_PATH)
        
        dataset = pd.read_csv(DATA_PATH)

        dataset['target'] = dataset['target'].astype(float)
        dataset['time_idx'] = dataset['time_idx'].astype(int)
        
        time_series = TimeSeriesDataSet.load('models/dataset_time_set')
        validation = TimeSeriesDataSet.from_dataset(time_series, dataset)
        
        all_dataloader = validation.to_dataloader(train=False, num_workers=0)
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)

        actuals = torch.cat([y[0] for (x, y) in iter(all_dataloader)])
        predictions = model.predict(all_dataloader)

        print(f'test mape is {((actuals - predictions).abs() / actuals).mean()}')

        print(f' max mape {max(((actuals - predictions).abs() / actuals))}')

        res = (actuals - predictions).abs() / actuals
        print(f' max 99 mape {np.quantile(res, .99)}')
#         print("wynik", res)
        res = np.array([int(x) for x in res])
    
def main(config_path: str, model_path: str):
    config_path = config_path
    config = read_config(config_path)
    config['config_path'] = config_path

    num_gpus = config['num_gpus']
    num_protocols = config['num_protocols']
    num_tests = config['num_tests']
    params = load_parameters("data/default-parameters.json")
    tumors = [load_state("data/tumor-lib/tumor-{}.txt".format(i), params) for i in range(1, 11)]
    model_temp = TemporalFusionTransformer.load_from_checkpoint(model_path)
    model = MockPredictionModel(model_temp)

    hour_steps = config['hour_steps']
    protocol_resolution = config['protocol_resolution']
    converter = ConvertRepresentation(hour_steps=hour_steps, protocol_resolution=protocol_resolution)

    pair_protocols = get_rand_population(
        num_protocols=config['num_protocols'],
        max_dose_value=config['max_dose_value'],
        time_interval_hours=config['time_interval_hours'])

    list_protocols = [
        converter.convert_pairs_to_list(protocol=protocol)
        for protocol in pair_protocols
    ]

    new_genetic_algorithm(population=list_protocols, model=model, config=config, converter=converter)

def dispatch(argv):
    if len(sys.argv) < 2:
        raise Exception(f'Please specify path to yaml config file.\n\nFor example:\n'
                        f'python python_ga/example_run_model.py '
                        f'python_ga/experiments_grid_search_retain_parent_protocols_inversed/grid_search__simple_selection__cross_two_points__mutate_time_dose_annealing__interval__dose_1.25__6h.yaml')

    config_path = sys.argv[1]
    model_path = FLAGS.model_path
    print(f'Using config file: {config_path}')

    
    main(config_path=config_path, model_path=model_path)

if __name__ == '__main__':
    app.run(dispatch)
