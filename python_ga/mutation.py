import numpy as np

from utils import refine_genome_around_cross_point_to_time_constraint


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
    interval_in_indices = int(2 * config['time_interval_hours'])

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
        population[i] = refine_genome_around_cross_point_to_time_constraint(
            genome=genome, interval_in_indices=interval_in_indices, config=config)
    return population


def mutate_time_value(population, config):
    """
    Mutacja zmian losowych czasu podania dawki
    :param population:
    :param config:
    :return:
    """
    interval_in_indices = int(2 * config['time_interval_hours'])

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
        population[i] = refine_genome_around_cross_point_to_time_constraint(
            genome=genome, interval_in_indices=interval_in_indices, config=config)
    return population


def mutate_swap(population, config):
    """
    Mutacja wymiany: badany jest każdy gen w genomie wszystkich osobników w populacji. Jeżeli wylosowana wartość p jest
    większa niż określone prawdopodobieństwo mutacji, wybrany gen jest zamieniamy miejscami z losowo wybranym innym genem
    :param population:      list
    :param config:          dict
    :return: population:    list
    """
    interval_in_indices = int(2 * config['time_interval_hours'])

    mutation_config = config['mutations']['mutate_swap']
    for i, genome in enumerate(population):
        for gene_idx in range(len(genome)):
            p = np.random.uniform()
            if p < mutation_config['mut_prob']:
                k = np.random.randint(len(genome) - 1)
                k = (gene_idx + k) % len(genome)
                genome[gene_idx], genome[k] = genome[k], genome[gene_idx]
        population[i] = refine_genome_around_cross_point_to_time_constraint(
            genome=population[i], interval_in_indices=interval_in_indices, config=config)
    return population


def mutate_split(population, config):
    """
    Splitting a non-zero dose (> 0.25Gy) into 2 doses.
    population - next population, array [population_size, element_size].
    """
    interval_in_indices = int(2 * config['time_interval_hours'])

    mutation_config = config['mutations']['mutate_split']
    min_dose = config['step_value']
    max_dose = config['max_dose_value']

    population = np.asarray(population)
    for i, genome in enumerate(population):
        if np.random.uniform() < mutation_config['mut_prob']:
            non_zero_dose_indices = np.nonzero(genome)[0]
            if non_zero_dose_indices.size:
                gene_idx = np.random.choice(non_zero_dose_indices)
                k = genome[gene_idx] / min_dose
                split = np.random.randint(0, k)
                d1 = split * min_dose
                d2 = genome[gene_idx] - d1
                for _ in range(len(population)):
                    new_gene_idx = np.random.randint(len(genome))
                    if genome[new_gene_idx] + d1 <= max_dose:
                        genome[new_gene_idx] = genome[new_gene_idx] + d1
                        break
                for _ in range(len(population)):
                    new_gene_idx = np.random.randint(len(genome))
                    if genome[new_gene_idx] + d2 <= max_dose:
                        genome[new_gene_idx] = genome[new_gene_idx] + d2
                        break
                genome[gene_idx] = 0
        population[i] = refine_genome_around_cross_point_to_time_constraint(
            genome=population[i], interval_in_indices=interval_in_indices, config=config)
    return population.tolist()
