import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import neptune


from datetime import datetime
from itertools import cycle

from utils import save_output, resolve_saving_path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s", level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger()


# ======================================================================================================================
# METRICS
# ======================================================================================================================

def collect_metrics(n_generation, pop_fitness, metrics):
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


def store_fitness_and_populations(
        all_fitness: list,
        all_populations: list,
        fitness: np.ndarray,
        population: np.ndarray,
        converter,
    ):
    """
    Method for storing all the fitnesses and populations.
    :param all_fitness:     all fitness values.
    :param all_populations: all populations.
    :param fitness:         last fitness.
    :param population:      last population.
    :param converter:       representation converter
    :return: updated list of all populations
    """
    all_fitness.append(list(fitness))

    paired_population = converter.convert_population_lists_to_pairs(protocols=population)
    all_populations.append(paired_population)
    return all_fitness, all_populations
# ======================================================================================================================
# MUTATIONS
# ======================================================================================================================


def mutate_dose_value(population, config):
    """
    Mutacja zmian losowych dawki: badana jest każda dawka w genomie wszystkich osobników w populacji. Jeżeli wylosowana
    wartość p jest większa niż określone prawdopodobieństwo mutacji, wartość wybranej jest losowana. Zakres wartości
    dawki ograniczony jest między 0.25Gy a 10Gy z krokiem co 0.25Gy. Oba limity odnoszą sie do sumarycznej wartosci
    dawek dla calego genomu.
    :param population:
    :param config:
    :return:
    """
    mutation_config = config['mutations']['mutate_dose_value']
    max_dose_value = config['max_dose_value']

    max_dose = mutation_config['max_value']
    min_dose = mutation_config['min_value']
    step = mutation_config['step_value']
    for i, genome in enumerate(population):
        for gene_idx in range(len(genome)):
            p = np.random.uniform()
            if p < mutation_config['mut_prob']:
                limit_value = min(max_dose, max_dose - sum(genome))
                if limit_value > max_dose_value:
                    limit_value = max_dose_value
                if limit_value > min_dose:
                    new_dose_value = np.random.randint(0, int(round((limit_value - min_dose) / step))) * step + min_dose
                    genome[gene_idx] = new_dose_value
                elif limit_value > 0:
                    genome[gene_idx] = min_dose
                    break
                break
    return population


def mutate_time_value(population, config):
    """
    Mutacja zmian losowych czasu podania dawki
    :param population:
    :param config:
    :return:
    """
    mutation_config = config['mutations']['mutate_time_value']
    max_dose_value = config['max_dose_value']

    for i, genome in enumerate(population):
        for gene_idx in range(len(genome)):
            p = np.random.uniform()
            if p < mutation_config['mut_prob']:
                if genome[gene_idx] > 0:
                    new_dose_time = np.random.randint(len(genome))
                    new_dose_value = genome[new_dose_time] + genome[gene_idx]
                    dose_remainder = 0
                    if new_dose_value > max_dose_value:
                        dose_remainder = new_dose_value - max_dose_value
                        new_dose_value = max_dose_value

                    genome[new_dose_time] = new_dose_value
                    genome[gene_idx] = dose_remainder
    return population


def mutate_swap(population, config):
    """
    Mutacja wymiay: badany jest każdy gen w genomie wszystkich osobników w populacji. Jeżeli wylosowana wartość p jest
    większa niż określone prawdopodobieństwo mutacji, wybrany gen jest zamieniamy miejscami z losowo wybranym innym genem
    :param population:      list
    :param config:          dict
    :return: population:    list
    """
    mutation_config = config['mutations']['mutate_swap']

    length = len(population)
    width = len(population[0])
    for i in range(length):
        for j in range(width):
            p = np.random.uniform()
            if p < mutation_config['mut_prob']:
                k = np.random.randint(0, width-1)
                k = (j + k) % width
                population[i][j], population[i][k] = population[i][k], population[i][j]
    return population


def mutate_merge(population, config, max_dose=10):
    """
    Merging 2 doses (potentially zero) into one dose
    population - next population, array [population_size, element_size]
    """
    mutation_config = config['mutations']['mutate_merge']
    max_dose_value = config['max_dose_value']

    population = np.asarray(population)
    length = len(population[:, 1])
    width = len(population[1, :])
    for i in range(length):
        if np.random.uniform() < mutation_config['mut_prob']:
            while True:
                p = np.random.randint(0, width)
                q = np.random.randint(0, width)
                if (population[i, p] + population[i, q]) <= max_dose:
                    new_dose_value = population[i, p] + population[i, q]
                    dose_remainder = 0
                    if new_dose_value > max_dose_value:
                        dose_remainder = new_dose_value - max_dose_value
                        new_dose_value = max_dose_value

                    if np.random.uniform() < 0.5:
                        population[i, p] = new_dose_value
                        population[i, q] = dose_remainder
                        break
                    else:
                        population[i, q] = new_dose_value
                        population[i, p] = dose_remainder
                        break
    return population.tolist()


def mutate_split(population, config, max_dose=10, min_dose=0.25):
    """
    Splitting a non-zero dose (> 0.25Gy) into 2 doses.
    population - next population, array [population_size, element_size].
    """
    mutation_config = config['mutations']['mutate_merge']

    population = np.asarray(population)
    length = len(population[:, 1])
    width = len(population[1, :])
    for i in range(length):
        if np.random.uniform() < mutation_config['mut_prob']:
            while True:
                p = np.random.randint(0, width)
                if population[i, p] >= min_dose:
                    k = population[i, p] / min_dose
                    split = np.random.randint(0, k)
                    d1 = split * min_dose
                    d2 = population[i, p] - d1
                    population[i, p] = 0
                    while True:
                        p = np.random.randint(0, width)
                        if population[i, p] + d1 < max_dose:
                            population[i, p] = population[i, p] + d1
                            break
                    while True:
                        p = np.random.randint(0, width)
                        if population[i, p] + d2 < max_dose:
                            population[i, p] = population[i, p] + d2
                            break
                    break
    return population.tolist()


def mutations(population, config):
    """
    Metoda sterująca przebiegiem mutacji w populacji. Dla każdego z wybranych typów mutacji wywoływana jest odpowiednia
    metoda.
    :param population:      list
    :param config:          dict
    :return: population:    list
    """
    mutation = {
        'mutate_swap':         mutate_swap,
        'mutate_dose_value':   mutate_dose_value,
        'mutate_time_value':   mutate_time_value,
        'mutate_merge':     mutate_merge,
        'mutate_split':     mutate_split,
    }

    for mut_type in list(config['mutations'].keys()):
        population = mutation[mut_type](population=population, config=config)
        logger.info(f'{mut_type} {[sum(pop) for pop in population]}')

    return population


# ======================================================================================================================
# CREATE NEXT GENERATION
# ======================================================================================================================

# ==CROSSOVER===========================================================================================================


def crossroads_grouping():
    """
    Metoda łączy geny odpowiadające poszczególnym skrzyżowaniom w grupy, wzlędem ich sąsiedztwa.
    :return: list
    """
    group_1 = [6, 7, 8, 5, 20, 19, 0, 17]
    group_2 = [16, 1, 13, 9]
    group_3 = [10, 2, 3, 18]
    group_4 = [4, 15, 14, 11, 12]
    return [group_1, group_2, group_3, group_4]


def gene_group_replacement(x1, x2, group):
    """
    Metoda buduje genom dziecka przekazując mu geny z wyselekcjonowanej grupy od rodzica x2
    i pozostałe geny od rodzica x1.
    :param x1:      list
    :param x2:      list
    :param group:   list
    :return: new_x: list
    """
    new_x = [x2[idx] if idx in group else x1[idx] for idx, gene in enumerate(x1)]

    return new_x


def normalize_crossover(genome, config):
    step = config['step_value']
    max_value = config['max_value']
    while sum(genome) > max_value:
        top_args_indices = np.array(genome).argsort()[-len(genome):][::-1]
        for idx in top_args_indices:
            if genome[idx] < step:
                break
            genome[idx] -= step
            if sum(genome) <= max_value:
                break
    return genome


def normalized_crossover(x1, x2, config):
    """
    Znormalizowane krzyżowanie 2 genów: wybieramy indeksu genów z x2, krzyżujemy z x1, oba powstałe genomy normalizujemy,
    aby suma dawek nie przekroczyła limitu
    :param x1:                  list
    :param x2:                  list
    :return: new_x1, new_x1:    list, list
    """
    group = []
    for i in range(len(x1)):
        if np.random.randint(0, 2) == 0:
            group.append(i)

    new_x1 = gene_group_replacement(x1, x2, group)
    new_x2 = gene_group_replacement(x2, x1, group)

    new_x1 = normalize_crossover(new_x1, config)
    new_x2 = normalize_crossover(new_x2, config)

    return new_x1, new_x2


def cross_two_points(x1, x2, config):
    """
    Krzyżowanie w dwóch punktach: W chromosomie dziecka wybierane są dwa punkty. Geny pomiędzy tymi punktami pochodzą
    od drugiego rodzica, a pozostałe od rodzica pierwszego. Sytuacja jest odwrotna w przypadku drugiego dziecka.
    :param x1:                  list
    :param x2:                  list
    :return: new_x1, new_x1:    list, list
    """
    new_x1 = []
    new_x2 = []
    length = len(x1)
    lower_range = 1
    upper_range = length - 1
    if config['crossover']['cross_points_range_percentage']:
        lower_range = int(length * config['crossover']['cross_points_range_percentage'][0] / 100)
        middle_range = int(length * config['crossover']['cross_points_range_percentage'][1] / 100)
        upper_range = int(length * config['crossover']['cross_points_range_percentage'][2] / 100)

    cross_point_1 = np.random.randint(lower_range, middle_range)
    cross_point_2 = np.random.randint(cross_point_1, upper_range)
    new_x1[0:cross_point_1] = x1[0:cross_point_1]
    new_x1[cross_point_1:cross_point_2] = x2[cross_point_1:cross_point_2]
    new_x1[cross_point_2:length] = x1[cross_point_2:length]
    new_x2[0:cross_point_1] = x2[0:cross_point_1]
    new_x2[cross_point_1:cross_point_2] = x1[cross_point_1:cross_point_2]
    new_x2[cross_point_2:length] = x2[cross_point_2:length]

    new_x1 = normalize_crossover(new_x1, config)
    new_x2 = normalize_crossover(new_x2, config)

    return new_x1, new_x2


def cross_one_point(x1, x2, config):
    """
    Krzyżowanie w jednym punkcie: losowo wybierany jest jeden punkt w chromosomie dziecka. Wszystkie geny przed tym
    punktem pochodzą od pierwszego rodzica, a wszystkie geny za tym punktem pochodzą od rodzica drugiego. Sytuacja jest
    odwrotna w przypadku drugiego dziecka.
    :param x1:                  list
    :param x2:                  list
    :return: new_x1, new_x1:    list, list
    """
    new_x1 = []
    new_x2 = []
    length = len(x1)
    lower_range = 1
    upper_range = length
    if config['crossover']['cross_point_range_percentage']:
        lower_range = int(length * config['crossover']['cross_point_range_percentage'][0] / 100)
        upper_range = int(length * config['crossover']['cross_point_range_percentage'][1] / 100)

    cross_point = np.random.randint(lower_range, upper_range)
    new_x1[0:cross_point] = x1[0:cross_point]
    new_x1[cross_point:length] = x2[cross_point:length]
    new_x2[0:cross_point] = x2[0:cross_point]
    new_x2[cross_point:length] = x1[cross_point:length]

    new_x1 = normalize_crossover(new_x1, config)
    new_x2 = normalize_crossover(new_x2, config)

    return new_x1, new_x2


def cross_uniform(x1, x2, config):
    """
    Krzyżowanie jednorodne: każdy gen w chromosomach dzieci ma 50% szans na pochodzenie od pierwszego z rodziców i 50%
    na pochodzenie od drugiego rodzica. Jeśli pierwsze dziecko otrzymało gen pierwszego rodzica, to drugie dziecko musi
    otrzymać gen rodzica drugiego.
    :param x1:                  list
    :param x2:                  list
    :return: new_x1, new_x1:    list, list
    """
    new_x1 = []
    new_x2 = []
    for i in range(len(x1)):
        p = np.random.uniform()
        if p < 0.5:
            new_x1.append(x1[i])
            new_x2.append(x2[i])
        else:
            new_x1.append(x2[i])
            new_x2.append(x1[i])

    new_x1 = normalize_crossover(new_x1, config)
    new_x2 = normalize_crossover(new_x2, config)

    return new_x1, new_x2


def create_offspring(pop_size, selected_individuals, config):
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
    :param config:                  dict
    :return: new_population:        list
    """
    cross = {'cross_uniform':               cross_uniform,
             'cross_one_point':             cross_one_point,
             'cross_two_points':            cross_two_points,
             'normalized_crossover':        normalized_crossover}
    offspring_pairs = int((pop_size - len(selected_individuals)) / 2)
    new_population = [genome for genome in selected_individuals]
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

# ==SELECTION===========================================================================================================


def simple_selection(population, pop_fitness, select_n, config):
    """
    Metoda selekcji prostej:
    1. Indeksy osobników w populacji są sortowane rosnąco względem ich wyników w pop_fitness;
    2. select_n osobników z najlepszymi wynikami wybieranych jest do puli selekcji;
    :param population:  list
    :param pop_fitness: list
    :param select_n:    int
    :param config:      dict
    :return: list
    """
    best_index = np.argsort(pop_fitness)[::-1]
    return [population[i] for i in best_index][:select_n]


def tournament_selection_classic(population, pop_fitness, select_n, config):
    """
    Metoda selekcji tournament - klasyczna
    1. Wybieramy len(population / select_n) osobników
    2. Porównujemy dopasowanie len(population / select_n) osobników
    3. Wybieramy pojedynczego osobnika, gdzie prawdopobienstwa wybori wynoszą p dla najlepszego osobnika, p(1 - p)
     dla drugiego, p(1 - p)^2 dla trzeciego... wykorzystując roulette_selection bez rozpraszania osobników
    4. Powtarzamy aż uzyskania select_n wybranych osobników.
    :param population:  list
    :param pop_fitness: list
    :param select_n:    int
    :param config:      dict
    :return: list
    """
    def sort_candidates():
        descending_sorted_candidates_indices = np.argsort(pop_fitness[candidates_indices])[::-1]
        return [
            population[candidates_indices[i]]
            for i in descending_sorted_candidates_indices
        ]

    probability = config['probability']
    config_for_roulette = config.copy()
    config_for_roulette['candidates_dispersion'] = False

    selected = []
    for i in range(select_n):
        candidates_indices = []
        candidates_probabilities = []
        for j in range(round(len(population) / select_n)):
            candidates_indices.append(np.random.randint(0, len(population)))
            candidates_probabilities.append(probability * (1 - probability) ** j)

        sorted_candidates = sort_candidates()

        selected_candidate = roulette_selection(sorted_candidates, candidates_probabilities, 1, config_for_roulette)[0]
        selected.append(selected_candidate)
    return selected


def tournament_selection_tuned(population, pop_fitness, select_n, config):
    """
    Metoda selekcji tournament - zmodyfikowana
    1. Wyznaczamy k ilość kandydatów na osobnika
    2. Porównujemy dopasowanie kandydatów z najlepszym osobnikiem
    3. Wybieramy najlepszego osobnika z prawdopodobienstwem p, a z prawdopodobienstwem 1 - p uruchamiamy metodę selekcji
    ruletki do wyboru pojedynczego osobnika
    4. Powtarzamy aż uzyskania select_n osobników z prawdopodobienstwami p*(1-p), p*(1-p)^2...
    :param population:  list
    :param pop_fitness: list
    :param select_n:    int
    :param config:      dict
    :return: list
    """
    probability = config['probability']

    k = round(len(population) / select_n)
    selected = []
    for i in range(select_n):
        selection_probability = probability * (1 - probability) ** i
        idx_of_best = None
        for j in range(k):
            idx_of_candidate = np.random.randint(0, len(population))
            if idx_of_best is None or pop_fitness[idx_of_candidate] > pop_fitness[idx_of_best]:
                idx_of_best = idx_of_candidate
        if np.random.random() < selection_probability:
            selected.append(population[idx_of_best])
        else:
            selected.append(roulette_selection(population, pop_fitness, 1, config)[0])
    return selected


def roulette_selection(population, pop_fitness, select_n, config):
    """
    Metoda selekcji ruletki
    Minimalizujemy wartośc funkcji dopasowania, co jest podejściem odwrotnym do standardowego. By moc poprawnie
    zastosowac algorytm selekcji ruletki wykorzystujemy odwrocone wartosci funkcji.
    W celu rozproszenia kandydatow do selekcji, obliczamy rozstęp wartości do znormalizowania wartosci dopasowania.
    Pozwala to na lepszy wybor kandydatow, kiedy wartosci funkcji dopasowania osobnikow są duże ale ich wariancje małe

    1. Odwracamy wartosci funkcji dopasowania otrzymując 'inversed_fitness'
    2. Obliczamy 'fitness_dispersion' rozstęp wartość funkcji dopasowania osobników w populacji
    3. Obliczamy 'fitness_dispersion_ratio' stosunek wartości rozstępu do średniej wartości funkcji dopasowania
    4. Wyznaczamy znormalizowane wartosci funkcji dopasowania osobnikow 'dispersed_fitness' poprzez odjęcie od nich
    wartości najgorszego osobnika, pomnozonego przez 1 - 'fitness_dispersion_ratio'
    5. Obliczamy listę 'wheel_probability_list' będącą sumą skumulowaną 'dispersed_fitness' wszystkich osobników
    6. Losujemy 'select_n' osobników używając listy 'wheel_probability_list'.

    :param population:  list
    :param pop_fitness: list
    :param select_n:    int
    :param config:      dict
    :return: list
    """
    selected = []

    if 'candidates_dispersion' in config and config['candidates_dispersion']:
        fitness_dispersion = max(pop_fitness) - min(pop_fitness)
        fitness_dispersion_ratio = fitness_dispersion / np.mean(pop_fitness)
        fitness = pop_fitness - min(pop_fitness) * (1 - fitness_dispersion_ratio)
    else:
        fitness = pop_fitness
    wheel_probability_list = np.cumsum(fitness) / sum(fitness)

    for i in range(select_n):
        r = np.random.random()
        for j, proba in enumerate(wheel_probability_list):
            if r < proba:
                selected.append(population[j])
                break
    return selected


def round_select_n(select_n, pop_size):
    """
    Metoda odpowiada za obliczenie liczebności osobników, które zostaną wyselekcjonowane do krzyżowania i kolejnego
    pokolenia:
    1. Liczebność puli selekcji obliczana jest jako udział w całości populacji (select_n);
    2. Jeśli otrzymany wynik nie jest parzysty, przyjmowana jest najbliższa, mniejsza liczba parzysta;
    Rozwiązanie to ma na celu zapewnienie stabliności liczebności osobników w populacji, na przestrzeni kolejnych pokoleń.
    :param select_n:    int
    :param pop_size:    int
    :return: select_n:  int
    """
    select_n = round(pop_size * select_n)

    while select_n % 2 != 0:
        select_n -= 1

    return select_n


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
    select_n = round_select_n(select_n=config['select_n'], pop_size=len(population))

    selection = {
        'simple_selection':                 simple_selection,
        'tournament_selection_classic':     tournament_selection_classic,
        'tournament_selection_tuned':       tournament_selection_tuned,
        'roulette_selection':               roulette_selection,
    }
    selected_individuals = selection[config['selection']['type']](
        population=population,
        pop_fitness=pop_fitness,
        select_n=select_n,
        config=config['selection'],
    )
    new_population = create_offspring(
        pop_size=len(population),
        selected_individuals=selected_individuals,
        config=config,
    )

    return new_population

# ======================================================================================================================


def calculate_fitness(population, model, converter):
    """
    Metoda odpowiada za obliczenie wartości funkcji dopasowania dla osobników w populacji, przy użyciu wybranego modelu.
    Otrzymany wynik przekształcany jest do tablicy jednowymiarowej.
    :param population:      list
    :param model:           fitness model
    :param converter:       representation converter
    :return: pop_fitness:   list
    """
    paired_population = converter.convert_population_lists_to_pairs(protocols=population)

    pop_fitness = model.predict(paired_population)
    pop_fitness = pop_fitness.reshape(len(population))
    return pop_fitness


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

    #neptune.set_project('TensorCell/cancertreatment')
    neptune.init('TensorCell/cancertreatment')
    neptune.create_experiment(name="Grid Search", params=config)
    neptune.append_tag('grid_search')
    neptune.append_tag(config['selection']['type'])
    neptune.append_tag(config['crossover']['type'])
    for mutation_type in config['mutations'].keys():
        neptune.append_tag(mutation_type)

    n_generation = 0

    metrics = pd.DataFrame(columns=['generation', 'best_fit', 'avg_fit'])

    logger.info('Initialize computation')
    
    date1 = datetime.now()
    pop_fitness = calculate_fitness(population=population, model=model, converter=converter)

    all_fitness, all_populations = store_fitness_and_populations(
        all_fitness=[],
        all_populations=[],
        fitness=pop_fitness,
        population=population,
        converter=converter,
    )
    logger.info(f'Initial fitness value calculated | Best fit: {max(pop_fitness)} '
                f'| For a starting protocol {all_populations[-1][np.argmax(pop_fitness)]}')

    date2 = date1
    date1 = datetime.now()

    logger.info("Time: " + str(date1 - date2))

    while n_generation <= config['max_iter'] and max(pop_fitness) < config['stop_fitness']:

        # nowe pokolenie
        population = next_generation(population=population, pop_fitness=pop_fitness, config=config)

        # mutacje
        population = mutations(population=population, config=config)

        # fitness
        pop_fitness = calculate_fitness(population=population, model=model, converter=converter)
        metrics = collect_metrics(n_generation=n_generation, pop_fitness=pop_fitness, metrics=metrics)

        n_generation += 1

        best_protocol = all_populations[-1][np.argmax(pop_fitness)]

        paired_population = converter.convert_population_lists_to_pairs(protocols=population)

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
            population=population,
            converter=converter,
        )

    show_metrics(metrics=metrics, all_fitness=all_fitness, all_populations=all_populations, config=config)
    save_metrics(metrics=metrics, all_fitness=all_fitness, all_populations=all_populations, config=config)
    neptune.stop()


