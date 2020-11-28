import numpy as np


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
    best_index = np.argsort(pop_fitness)
    return [population[i] for i in best_index][:select_n]


def tournament_selection_classic(population, pop_fitness, select_n, config):
    """
    Metoda selekcji tournament - klasyczna
    1. Wybieramy len(population / select_n) osobników
    2. Porównujemy dopasowanie len(population / select_n) osobników
    3. Wybieramy pojedynczego osobnika, gdzie prawdopobienstwa wyboru wynoszą p dla najlepszego osobnika, p(1 - p)
     dla drugiego, p(1 - p)^2 dla trzeciego... wykorzystując roulette_selection bez rozpraszania osobników
    4. Powtarzamy aż uzyskania select_n wybranych osobników.
    :param population:  list
    :param pop_fitness: list
    :param select_n:    int
    :param config:      dict
    :return: list
    """
    def sort_candidates():
        sorted_candidates_indices = [
            candidates_indices[index]
            for index in np.argsort(pop_fitness[candidates_indices])[::-1]
        ]
        sorted_candidates = [population[index] for index in sorted_candidates_indices]
        return sorted_candidates, sorted_candidates_indices

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

        sorted_candidates, sorted_candidates_indices = sort_candidates()

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
