import random
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    best_fit = min(pop_fitness)
    avg_fit = np.mean(pop_fitness)
    data = pd.DataFrame([[n_generation, best_fit, avg_fit]], columns=['generation', 'best_fit', 'avg_fit'])
    return metrics.append(data, ignore_index=True)


def show_metrics(metrics, pop_fitness, population):
    """
    Method for showing best result and best individual
    :param metrics:
    :param pop_fitness:
    :param population:
    :return:
    """
    fit_idx = np.argsort(pop_fitness)
    best_fit = population[fit_idx[0]]

    plt.plot(metrics['generation'], metrics['best_fit'])
    plt.xlabel('Pokolenia')
    plt.ylabel('Fitness')
    plt.show()

    print('best result: ', min(pop_fitness))
    print('best individual: ', best_fit)

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
    max_dose = config['max_value']
    min_dose = config['min_value']
    step = config['step_value']
    for i, genome in enumerate(population):
        for gene_idx in range(len(genome)):
            p = np.random.uniform()
            if p < config['mut_prob']:
                limit_value = min(max_dose, max_dose - sum(genome))
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
    for i, genome in enumerate(population):
        for gene_idx in range(len(genome)):
            p = np.random.uniform()
            if p < config['mut_prob']:
                if genome[gene_idx] > 0:
                    new_dose_time = np.random.randint(len(genome))
                    genome[new_dose_time] = genome[new_dose_time] + genome[gene_idx]
                    genome[gene_idx] = 0
    return population


def mutate_swap(population, config):
    """
    Mutacja wymiay: badany jest każdy gen w genomie wszystkich osobników w populacji. Jeżeli wylosowana wartość p jest
    większa niż określone prawdopodobieństwo mutacji, wybrany gen jest zamieniamy miejscami z losowo wybranym innym genem
    :param population:      list
    :param config:          dict
    :return: population:    list
    """
    length = len(population)
    width = len(population[0])
    for i in range(length):
        for j in range(width):
            p = np.random.uniform()
            if p < config['mut_prob']:
                k = np.random.randint(0, width-1)
                k = (j + k) % width
                population[i][j], population[i][k] = population[i][k], population[i][j]
    return population


def mutate_merge(population, config, max_dose=10):
    """
    Merging 2 doses (potentially zero) into one dose
    population - next population, array [population_size, element_size]
    """
    population = np.asarray(population)
    length = len(population[:, 1])
    width = len(population[1, :])
    for i in range(length):
        if np.random.uniform() < config['mut_prob']:
            while True:
                p = np.random.randint(0, width)
                q = np.random.randint(0, width)
                if (population[i, p] + population[i, q]) <= max_dose:
                    if np.random.uniform() < 0.5:
                        population[i, p] = population[i, p] + population[i, q]
                        population[i, q] = 0
                        break
                    else:
                        population[i, q] = population[i, p] + population[i, q]
                        population[i, p] = 0
                        break
    return population.tolist()


def mutate_split(population, config, max_dose=10, min_dose=0.25):
    """
    Splitting a non-zero dose (> 0.25Gy) into 2 doses.
    population - next population, array [population_size, element_size].
    """
    population = np.asarray(population)
    length = len(population[:, 1])
    width = len(population[1, :])
    for i in range(length):
        if np.random.uniform() < config['mut_prob']:
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
        'mut_swap':         mutate_swap,
        'mut_dose_value':   mutate_dose_value,
        'mut_time_value':   mutate_time_value,
        'mutate_merge':     mutate_merge,
        'mutate_split':     mutate_split,
    }

    for mut_type in list(config.keys()):
        population = mutation[mut_type](population, config[mut_type])
        print(mut_type, [sum(pop) for pop in population])

    return population


# ==NORMALIZATION=======================================================================================================


def round_dose(genome, min, step):
    """
    Metoda zaokrągla wartości dawek po normalizacji do 0.25Gy
    :param genome:          list
    :param min:             float
    :param step:            float
    :return: rounded_genome:list
    """
    inverse_step = int(1 / step)
    round_to_step = lambda genome: [round(value * inverse_step) / inverse_step for value in genome]
    rounded_genome = round_to_step(genome)
    return rounded_genome


def normalize(population, config):
    """
    Normalizacja wartości dawek po mutacjach. Mutacje mogą zmienić wartości dawek do wartości przekraczających ustalone
    ograniczenia. Normalizacja polega na porownaniu obecnej łącznej dawki z wartoscią maksymalną i zmniejszeniu
    poszczegolnych dawek proporcjonalnie do roznicy pomiedzy obiema wartosciami.

    W domyślnej konfiguracji maksymalna łączna dawka może wynosić 10Gy. Minimalna dawka to 0.25Gy,
    identycznie jak wartość o ktorą zwiekszamy dawkę.
    :param population:      list
    :oaram config:          dict
    :return: population:     list
    """
    normalization_factor = [
        np.clip(sum(genome) / config['normalization']['max_value'], 1, config['normalization']['max_value'])
        for genome in population
    ]

    normalized_population = [
        genome / normalization_factor[index]
        for index, genome in enumerate(population)
    ]

    rounded_population = [
        round_dose(genome=genome, min=config['normalization']['min_value'], step=config['normalization']['step_value'])
        for genome in normalized_population
    ]

    return rounded_population


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
    cross_point_1 = np.random.randint(1, length - 1)
    cross_point_2 = np.random.randint(cross_point_1, length)
    new_x1[0:cross_point_1] = x1[0:cross_point_1]
    new_x1[cross_point_1:cross_point_2] = x2[cross_point_1:cross_point_2]
    new_x1[cross_point_2:length] = x1[cross_point_2:length]
    new_x2[0:cross_point_1] = x2[0:cross_point_1]
    new_x2[cross_point_1:cross_point_2] = x1[cross_point_1:cross_point_2]
    new_x2[cross_point_2:length] = x2[cross_point_2:length]
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
    cross_point = np.random.randint(1, length)
    new_x1[0:cross_point] = x1[0:cross_point]
    new_x1[cross_point:length] = x2[cross_point:length]
    new_x2[0:cross_point] = x2[0:cross_point]
    new_x2[cross_point:length] = x1[cross_point:length]
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
        new_x1, new_x2 = cross[config['cross_type']](x1, x2, config)
        new_population.append(new_x1), new_population.append(new_x2)

    return new_population

# ==SELECTION===========================================================================================================


def simple_selection(population, pop_fitness, select_n):
    """
    Metoda selekcji prostej:
    1. Indeksy osobników w populacji są sortowane rosnąco względem ich wyników w pop_fitness;
    2. select_n osobników z najlepszymi wynikami wybieranych jest do puli selekcji;
    :param population:  list
    :param pop_fitness: list
    :param select_n:    int
    :return: list
    """
    best_index = np.argsort(pop_fitness)
    return [population[i] for i in best_index][:select_n]


def tournament_selection(population, pop_fitness, select_n):
    k = round(len(population) / select_n)
    selected = []
    for i in range(select_n):
        idx_of_best = None
        for j in range(k):
            idx_of_candidate = np.random.randint(0, len(population))
            if idx_of_best is None or pop_fitness[idx_of_candidate] < pop_fitness[idx_of_best]:
                idx_of_best = idx_of_candidate
        selected.append(population[idx_of_best])
    return selected


def roulette_selection(population, pop_fitness, select_n):
    total_fitness = sum(pop_fitness)
    probability_sum = 0
    selected = []
    wheel_probability_list = []

    for i, fitness in enumerate(pop_fitness):
        # (1 - p) from 1 to N = N - 1 -> (1 - p) / (N - 1)
        probability_sum += (1 - (fitness / total_fitness)) / (len(population) - 1)
        wheel_probability_list.append(probability_sum)

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
    select_n = round_select_n(config['select_n'], len(population))

    selection = {
        'simple_selection':         simple_selection,
        'tournament_selection':     tournament_selection,
        'roulette_selection':       roulette_selection,
    }
    selected_individuals = selection[config['selection']['type']](population, pop_fitness, select_n)

    new_population = create_offspring(len(population), selected_individuals, config)

    return new_population

# ======================================================================================================================


def calculate_fitness(population, model):
    """
    Metoda odpowiada za obliczenie wartości funkcji dopasowania dla osobników w populacji, przy użyciu wybranego modelu.
    Otrzymany wynik przekształcany jest do tablicy jednowymiarowej.
    :param population:      list
    :param model:           fitness model
    :return: pop_fitenss:   list
    """
    pop_fitenss = model.predict(population)
    pop_fitenss = pop_fitenss.reshape(len(population))
    return pop_fitenss


def new_genetic_algorithm(population, model, config):
    """
    Główna metoda algorytmu - zawiera pętlę, która dla każdego pokolenia:
    1. blicza wartość fitness osobników w populacji;
    2. Przeprowadza proces krzyżowania i tworzy populację dla nowego pokolenia;
    3. Przeprowadza proces mutacji;
    :param population:  list
    :param model:       fitness model
    :param config:      dict
    :return:
    """
    n_generation = 0

    metrics = pd.DataFrame(columns=['generation', 'best_fit', 'avg_fit'])

    pop_fitness = calculate_fitness(population, model)

    while n_generation <= config['max_iter'] and min(pop_fitness) > config['stop_fitness']:

        # nowe pokolenie
        population = next_generation(population, pop_fitness, config)

        # mutacje
        population = mutations(population, config['mutations'])

        # fitness
        pop_fitness = calculate_fitness(population, model)
        metrics = collect_metrics(n_generation, pop_fitness, metrics)

        n_generation += 1

        print('Generation: ', n_generation, '| Best fitt: ', min(pop_fitness))

    show_metrics(metrics, pop_fitness, population)
