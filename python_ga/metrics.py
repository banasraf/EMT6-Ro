import logging

import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
import os

from datetime import datetime

from utils import resolve_saving_path, save_output


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s", level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger()


def collect_metrics(n_generation, pop_fitness, metrics=None):
    """
    Method for collecting metrics (average fitness and best fitness) for a given population
    :param n_generation: generation id
    :param pop_fitness: population with computed fitness function
    :param metrics: current metrics to which we append new metrics
    :return: metrics with added new metrics for the new population
    """
    best_fit = max(pop_fitness)
    avg_fit = np.mean(pop_fitness)
    data = pd.DataFrame([[n_generation, best_fit, avg_fit]], columns=['generation', 'best_fit', 'avg_fit'])
    return metrics.append(data, ignore_index=True)


def show_metrics(metrics, all_fitness, all_populations, config):
    """
    Method for showing best result and best individual
    :param metrics: values of metrics
    :param all_fitness: values of all fitnesses.
    :param all_populations: all populations.
    :param config: experiment configuration
    :return:
    """
    fit_idx = np.argsort(all_fitness[-1])[::-1]
    best_fit = all_populations[-1][fit_idx[0]]

    config['experiment_time'] = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    config['saving_path'] = resolve_saving_path(config=config)

    plot_saving_path = os.path.join(config['saving_path'], f'plot_fitness_{config["experiment_time"]}.png')

    plt.figure(figsize=(14, 7))
    plt.plot(metrics['generation'], metrics['best_fit'], label='Best fit')
    plt.plot(metrics['generation'], metrics['avg_fit'], label='Avg git')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness value')
    plt.title(f"SELECTION : {config['selection']['type']}; MUTATIONS : {', '.join(config['mutations'].keys())}")
    plt.suptitle(f"Experiment date and time: {config['experiment_time']}")
    plt.legend()
    plt.grid()
    plt.savefig(plot_saving_path)

    logger.info(f'best result: {max(all_fitness[-1])}')
    logger.info(f'best individual: {best_fit}')

    neptune.log_image('fitness_figure', plot_saving_path)


def save_metrics(metrics: pd.DataFrame, all_fitness: list, all_populations: list, config: dict):
    """
    Method for saving metrics
    :param metrics:         values of metrics
    :param all_fitness:     values of all fitnesses
    :param all_populations: all populations.
    :param config:          experiment configuration
    :return:
    """
    current_time = config["experiment_time"]

    save_output(file=metrics, file_name=f'results_metrics_{current_time}', extension='csv', config=config)
    save_output(file=config, file_name=f'config_{current_time}', extension='txt', config=config)
    if config['save_every_iteration']:
        save_output(file=all_fitness, file_name=f'fitness_all_{current_time}', extension='txt', config=config)
        save_output(file=all_populations, file_name=f'populations_all_{current_time}', extension='txt', config=config)
    if config['save_only_last_iteration'] and not config['save_every_iteration']:
        save_output(file=all_fitness[-1], file_name=f'fitness_{current_time}', extension='txt', config=config)
        save_output(file=all_populations[-1], file_name=f'population_{current_time}', extension='txt', config=config)
