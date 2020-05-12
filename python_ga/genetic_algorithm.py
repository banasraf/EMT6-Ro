import random
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================================================================
# METRICS
# ======================================================================================================================

def collect_metrics(n_generation, pop_fitness, metrics):
    best_fit = min(pop_fitness)
    avg_fit = np.mean(pop_fitness)
    data = pd.DataFrame([[n_generation, best_fit, avg_fit]], columns=['generation', 'best_fit', 'avg_fit'])
    return metrics.append(data, ignore_index=True)


def show_metrics(metrics, pop_fitness, population):
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


def mutate_random_change(population, config):
    """
    Mutacja zmian losowych: badany jest każdy gen w genomie wszystkich osobników w populacji. Jeżeli wylosowana wartość
    p jest większa niż określone prawdopodobieństwo mutacji, wartość wybranego genu jest losowana.
    :param population:
    :param config:
    :return:
    """
    for i, genome in enumerate(population):
        for gene_idx in range(len(genome)):
            p = np.random.uniform()
            if p < config['mut_prob']:
                genome[gene_idx] = np.random.randint(config['max_value'])

    return population


def mutate_swap(population, config):
    """
    Mutacja wymiay: badany jest każdy gen w genomie wszystkich osobników w populacji. Jeżeli wylosowana wartość p jest
    większa niż określone prawdopodobieństwo mutacji, wybrany gen jest zamieniamy miejscami z poprzedzającym go genem.
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
                population[i][j], population[i][j - 1] = population[i][j - 1], population[i][j]

    return population



'''
Merging 2 doses (potentially zero) into one dose
NextGeneration - next population, array [population_size, element_size]
'''
def mutate_merge(NextGeneration, config, max_dose=10):
    NextGeneration = np.asarray(NextGeneration)
    length = len(NextGeneration[:, 1])
    width = len(NextGeneration[1, :])
    for i in range(length):
        while True:
            p = np.random.randint(0,width)
            q = np.random.randint(0,width)
            if (NextGeneration[i,p] + NextGeneration[i,q]) <= max_dose:
                if np.random.uniform() < 0.5:
                    NextGeneration[i,p] = NextGeneration[i,p] + NextGeneration[i,q]
                    NextGeneration[i,q] = 0
                    break
                else:
                    NextGeneration[i,q] = NextGeneration[i,p] + NextGeneration[i,q]
                    NextGeneration[i,p] = 0
                    break
    return NextGeneration


'''
Splitting a non-zero dose (> 0.25Gy) into 2 doses.
NextGeneration - next population, array [population_size, element_size].

'''
def mutate_split(NextGeneration, config, max_dose=10, min_dose=0.5):
    NextGeneration = np.asarray(NextGeneration)
    length = len(NextGeneration[:, 1])
    width = len(NextGeneration[1, :])
    for i in range(length):
        while True:
            p = np.random.randint(0,width)
            if NextGeneration[i,p] > min_dose:
                k = NextGeneration[i,p] / min_dose
                split = np.random.randint(0, k)
                d1 = split * min_dose
                d2 = NextGeneration[i,p] - d1
                NextGeneration[i, p] = 0
                while True:
                    p = np.random.randint(0, width)
                    if NextGeneration[i,p] + d1 < max_dose:
                        NextGeneration[i,p] = NextGeneration[i,p] + d1
                        break
                while True:
                    p = np.random.randint(0, width)
                    if NextGeneration[i,p] + d2 < max_dose:
                        NextGeneration[i,p] = NextGeneration[i,p] + d2
                        break
                break
    return NextGeneration


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
        'mut_random':       mutate_random_change,
        'mutate_merge':     mutate_merge,
        'mutate_split':     mutate_split,
    }

    for mut_type in list(config.keys()):
        population = mutation[mut_type](population, config[mut_type])

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
    # {'min': 0.25, 'max': 10, 'step': 0.25}

    normalization_factor = [
        np.clip(sum(genome) / config['normalization']['max'], 1, config['normalization']['max'])
        for genome in population
    ]

    normalized_population = [
        genome / normalization_factor[index]
        for index, genome in enumerate(population)
    ]

    rounded_population = [
        round_dose(genome=genome, min=config['normalization']['min'], step=config['normalization']['step'])
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


def proximity_base_crossover(x1, x2):
    """
    Krzyżowanie oparte na bliskości skrzyżowań:
    1. Sąsiadujące skrzyżowania są grupowane;
    2. Dla każdego z dzieci wywoływana jest metoda przekazująca mu geny z wylosowanej grupy od jednego z rodziców
    i pozostałe geny od drugiego z rodziców
    :param x1:                  list
    :param x2:                  list
    :return: new_x1, new_x1:    list, list
    """
    g_index = np.random.randint(0, 4)

    group = crossroads_grouping()[g_index]

    new_x1 = gene_group_replacement(x1, x2, group)
    new_x2 = gene_group_replacement(x2, x1, group)

    return new_x1, new_x2


def cross_two_points(x1, x2):
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


def cross_one_point(x1, x2):
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


def cross_uniform(x1, x2):
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
             'proximity_base_crossover':    proximity_base_crossover}
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
        new_x1, new_x2 = cross[config['cross_type']](x1, x2)
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

    selection = {'simple_selection':    simple_selection}
    selected_individuals = selection[config['selection']['type']](population, pop_fitness, select_n)

    new_population = create_offspring(len(population), selected_individuals, config)

    return new_population

# ======================================================================================================================


def calculate_fitenss(population, model):
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

    pop_fitness = calculate_fitenss(population, model)

    while n_generation <= config['max_iter'] and min(pop_fitness) > config['stop_fitness']:

        # nowe pokolenie
        population = next_generation(population, pop_fitness, config)

        # mutacje
        population = mutations(population, config['mutations'])

        # normalizacja
        population = normalize(population, config)

        # fitness
        pop_fitness = calculate_fitenss(population, model)
        metrics = collect_metrics(n_generation, pop_fitness, metrics)

        n_generation += 1

        print('Generation: ', n_generation, '| Best fitt: ', min(pop_fitness))

    show_metrics(metrics, pop_fitness, population)
