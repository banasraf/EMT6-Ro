import logging
import random

import neptune
import numpy as np
import pandas as pd

from datetime import datetime
from itertools import cycle

from crossover import cross_one_point, cross_two_points, cross_uniform, normalized_crossover
from metrics import collect_metrics, save_metrics, show_metrics
from mutation import mutate_dose_value, mutate_split, mutate_swap, mutate_time_value
from selection import (round_select_n,
                       roulette_selection,
                       simple_selection,
                       tournament_selection_classic,
                       tournament_selection_tuned)
from utils import calculate_fitness, calculate_probability_annealing, store_fitness_and_populations


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s", level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger()


# ======================================================================================================================
# MUTATIONS
# ======================================================================================================================
def mutations(population, config, iteration):
    """
    Metoda sterująca przebiegiem mutacji w populacji. Dla każdego z wybranych typów mutacji wywoływana jest odpowiednia
    metoda.
    :param population:      list
    :param config:          dict
    :parma iteration:       int
    :return: population:    list
    """
    mutation = {
        'mutate_swap':         mutate_swap,
        'mutate_dose_value':   mutate_dose_value,
        'mutate_time_value':   mutate_time_value,
        'mutate_split':     mutate_split,
    }

    population_part_to_mutate = population[config['no_of_retaining_parents']:]
    for mut_type in list(config['mutations'].keys()):
        mutation_type = config['mutations'][mut_type]
        if mut_type in mutation.keys():
            if mutation_type['mut_prob'] == 'annealing' or \
                    mutation_type['mut_type'] == 'annealing':
                mutation_type['mut_type'] = 'annealing'
                mutation_type['mut_prob'] = calculate_probability_annealing(
                    iteration=iteration,
                    max_value=mutation_type['mut_prob_max'],
                    max_iter=config['max_iter'],
                )
            population_part_to_mutate = mutation[mut_type](population=population_part_to_mutate, config=config)
            logger.info(f'{mut_type} {[sum(pop) for pop in population_part_to_mutate]}')
        else:
            logger.warning(f'Mutation "{mut_type}" not supported')

    population[config['no_of_retaining_parents']:] = population_part_to_mutate
    return population


# ======================================================================================================================
# CROSSOVER
# ======================================================================================================================
def create_offspring(pop_size, selected_individuals, retaining_parents, config):
    """
    Metoda zarządzająca procesem krzyżowania:
    1. Obliczana jest liczba par potomków;
    2. Wyselekcjonowane osobniki przypisywane są do nowej populacjil
    3. Uruchamiana jest pętla, która dla każdej pary potomków:
        1) Wybiera rodziców;
        2) Uruchamia określoną metodę krzyżowania;
        3) Dodaje potomstwo do nowej populacji;
    Zwracana jest gotowa populacja nowego pokolenia;
    UWAGA: możliwe są dwa warianty wyboru rodziców:
    sekwencyjny - rodzice wybierani są kolejno z listy
    losowy - rodzice dobierani są losowo
    Aby wybrać jeden z tych wariantów należy odkomentować zaznaczone linie
    :param pop_size:                int
    :param selected_individuals:    list
    :param retaining_parents:       list
    :param config:                  dict
    :return: new_population:        list
    """
    cross = {'cross_uniform':               cross_uniform,
             'cross_one_point':             cross_one_point,
             'cross_two_points':            cross_two_points,
             'normalized_crossover':        normalized_crossover}
    offspring_pairs = int((pop_size - len(retaining_parents)) / 2)
    new_population = [genome for genome in retaining_parents]
    parent_index = cycle(range(len(selected_individuals)))

    for i in range(offspring_pairs):
        # Wybór sekwencyjny
        # x1 = selected_individuals[next(parent_index)]
        # x2 = selected_individuals[next(parent_index)]
        # Wybór losowy
        x1 = random.choice(selected_individuals)
        x2 = random.choice(selected_individuals)
        new_x1, new_x2 = cross[config['crossover']['type']](x1, x2, config)
        new_population.append(new_x1), new_population.append(new_x2)

    return new_population


# ======================================================================================================================
# SELECTION
# ======================================================================================================================
def next_generation(population, pop_fitness, config):
    """
    Tworzenie nowego pokolenia:
    1. Ustalana jest liczebniość osobników wybranych do krzyżowania;
    2. Wywoływana jest metoda selekcji, która zwraca poppulację wyselekcjonowanych osobników;
    3. Wywoływana jest metoda zarządzająca procesem krzyżowania, która konstrułuje populacje nowego poklenia;
    :param population:          list
    :param pop_fitness:         list
    :param config:              dict
    :return: new_population:    list
    """
    select_n = round_select_n(
        select_n=config['select_n'] - config['retained_parent_protocols'], pop_size=len(population))
    config['no_of_retaining_parents'] = round_select_n(
        select_n=config['retained_parent_protocols'], pop_size=len(population))

    selection = {
        'simple_selection':                 simple_selection,
        'tournament_selection_classic':     tournament_selection_classic,
        'tournament_selection_tuned':       tournament_selection_tuned,
        'roulette_selection':               roulette_selection,
    }
    retaining_parents = selection['simple_selection'](
        population=population,
        pop_fitness=pop_fitness,
        select_n=config['no_of_retaining_parents'],
        config=config['selection'],
    )
    selected_individuals = selection[config['selection']['type']](
        population=population,
        pop_fitness=pop_fitness,
        select_n=select_n,
        config=config['selection'],
    )
    new_population = create_offspring(
        pop_size=len(population),
        selected_individuals=selected_individuals,
        retaining_parents=retaining_parents,
        config=config,
    )

    return new_population


# ======================================================================================================================
# GENETIC ALGORITHM
# ======================================================================================================================
def new_genetic_algorithm(population, model, config, converter):
    """
    Główna metoda algorytmu - zawiera pętlę, która dla każdego pokolenia:
    1. Oblicza wartość fitness osobników w populacji;
    2. Przeprowadza proces krzyżowania i tworzy populację dla nowego pokolenia;
    3. Przeprowadza proces mutacji;
    :param population:  list
    :param model:       fitness model
    :param config:      dict
    :param converter:   representation converter
    """

    neptune.init('TensorCell/cancertreatment')
    neptune.create_experiment(name="Grid Search", params=config)
    neptune.append_tag('evaluate_protocols')
    neptune.append_tag('inversed')
    neptune.append_tag(config['selection']['type'])
    neptune.append_tag(config['crossover']['type'])
    neptune.append_tag(f"{int(config['time_interval_hours'])}h")
    for mutation_type in config['mutations'].keys():
        neptune.append_tag(mutation_type)
        neptune.append_tag(str(f"mut_proba {config['mutations'][mutation_type]['mut_prob']}"))
        if config['selection']['type'] != 'simple_selection' and config['selection']['type'] != 'roulette_selection':
            neptune.append_tag(str(f"select_proba {config['selection']['probability']}"))

    n_generation = 0

    metrics = pd.DataFrame(columns=['generation', 'best_fit', 'avg_fit'])

    logger.info('Initialize computation')
    
    date1 = datetime.now()
    paired_population = converter.convert_population_lists_to_pairs(protocols=population)
    pop_fitness = calculate_fitness(paired_population=paired_population, model=model)

    all_fitness, all_populations = store_fitness_and_populations(
        all_fitness=[],
        all_populations=[],
        fitness=pop_fitness,
        paired_population=paired_population,
    )
    logger.info(f'Initial fitness value calculated | Best fit: {max(pop_fitness)} '
                f'| For a starting protocol {paired_population[np.argmax(pop_fitness)]}')

    date2 = date1
    date1 = datetime.now()

    logger.info("Time: " + str(date1 - date2))

    while n_generation <= config['max_iter'] and max(pop_fitness) < config['stop_fitness']:
        n_generation += 1

        # nowe pokolenie
        population = next_generation(population=population, pop_fitness=pop_fitness, config=config)

        # mutacje
        population = mutations(population=population, config=config, iteration=n_generation)

        # population conversion
        paired_population = converter.convert_population_lists_to_pairs(protocols=population)

        # fitness
        pop_fitness = calculate_fitness(paired_population=paired_population, model=model)

        best_protocol = paired_population[np.argmax(pop_fitness)]
        metrics = collect_metrics(n_generation=n_generation, pop_fitness=pop_fitness, metrics=metrics)

        logger.info(f'Generation: {n_generation} | '
                    f'Best fit: {max(pop_fitness)} | '
                    f'For a protocol {best_protocol}')

        neptune.log_metric('iteration', n_generation)
        neptune.log_metric('best_fitness', max(pop_fitness))
        neptune.log_metric('avg_fitness', np.mean(pop_fitness))
        neptune.log_text('best_protocol', f'Protocol id: {np.argmax(pop_fitness)} | {best_protocol}')
        neptune.log_text('protocols', str({i: value for i, value in enumerate(paired_population)}))

        date2 = date1
        date1 = datetime.now()

        logger.info("Time: " + str(date1 - date2))
        
        all_fitness, all_populations = store_fitness_and_populations(
            all_fitness=all_fitness,
            all_populations=all_populations,
            fitness=pop_fitness,
            paired_population=paired_population,
        )

    show_metrics(metrics=metrics, all_fitness=all_fitness, all_populations=all_populations, config=config)
    save_metrics(metrics=metrics, all_fitness=all_fitness, all_populations=all_populations, config=config)
    neptune.stop()
