import sys
import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder

from utils import ConvertRepresentation, get_rand_population, read_config
from genetic_algorithm import new_genetic_algorithm

from emt6ro.simulation import load_state, load_parameters
from absl import app, flags

flags.DEFINE_string('model_path', '../CancerOptimization/Untitled/OLN-4171/checkpoints/epoch=3-step=543.ckpt', 'Relative path to model')
FLAGS = flags.FLAGS

DATA_PATH = '../CancerOptimization/data/data.csv' #path to preprocessed csv data
TARGET = 'target'
FEATURES = ['dose', 'time']
GROUP_ID = 'series'

class MockPredictionModel:
    def __init__(self, model):
        self.model = model
    
    def prepare_data(self, data):
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
                
        df[TARGET] = df[TARGET].astype(float)

        df['Unnamed'] = df['Unnamed'].astype(int)
        df['time_idx'] = df['time_idx'].astype(int)
        df['is_target'] = df['is_target'].astype(int)
        df['series'] = df['series'].astype(int)
        return df
    
    def predict(self, data):
        dataset = self.prepare_data(data)
       
        print(dataset.iloc[:30])            
        validation = TimeSeriesDataSet(
            dataset,
            time_idx="time_idx",
            target=TARGET,
            group_ids=[GROUP_ID],
            min_encoder_length=20,  
            max_encoder_length=20,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            variable_groups={},  # group of categorical variables can be treated as one variable
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[TARGET] + FEATURES,
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            categorical_encoders={GROUP_ID: NaNLabelEncoder().fit(dataset.series)},
        )
            
        validation = TimeSeriesDataSet.from_dataset(validation, dataset, predict=True, stop_randomization=True)
        
        validation_dataloader = validation.to_dataloader(train=False, num_workers=0)
        
        res = self.model.predict(validation_dataloader)
        print("wynik", res)
        res = np.array([int(x) for x in res])
        
        return res

def test_model():
    dataset = pd.read_csv(DATA_PATH)

    n = dataset[GROUP_ID].astype(int).max()

    dataset[TARGET] = dataset[TARGET].astype(float)

    dataset['time_idx'] = dataset['time_idx'].astype(int)

    print(dataset.iloc[:30])
    
#     dataset['target'] = 0
    
    training = TimeSeriesDataSet(
        dataset[dataset[GROUP_ID].apply(lambda x: int(x) < int(n * 0.7))],
        time_idx="time_idx",
        target=TARGET,
        group_ids=[GROUP_ID],
        min_encoder_length=20,  
        max_encoder_length=20,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[TARGET] + FEATURES,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        categorical_encoders={GROUP_ID: NaNLabelEncoder().fit(dataset.series)},
    )

#     validation_dataset = dataset[dataset[GROUP_ID].apply(lambda x: int(x) > int(n * 0.7) and int(x) < int(n * 0.9))]
#     validation = TimeSeriesDataSet.from_dataset(training, validation_dataset,
#                                             min_prediction_idx=0,
#                                             stop_randomization=True)
    val_dataloader = training.to_dataloader(train=False, num_workers=0)
    model_temp2 = TemporalFusionTransformer.load_from_checkpoint(FLAGS.model_path)
    
    
    actuals = torch.cat([y[0] for (x, y) in iter(test_dataloader)])
    
    print(model_temp2.predict(val_dataloader))
    print("actuals", actuals[:30])
    
def main(config_path: str, model_path: str):
    test_model()
    
    return
    
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
                        f'python python_ga/example_run.py '
                        f'python_ga/experiment_config/base__tournament__dose_value_time_value.yaml')

    config_path = sys.argv[1]
    model_path = FLAGS.model_path
    print(f'Using config file: {config_path}')

    
    main(config_path=config_path, model_path=model_path)

if __name__ == '__main__':
    app.run(dispatch)
