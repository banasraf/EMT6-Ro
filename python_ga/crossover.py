import numpy as np

from utils import refine_genome_around_cross_point_to_time_constraint


def assign_dose_on_position_with_constraint(
    source_genome: np.ndarray,
    destination_genome: list,
    index: int,
    interval_in_indices: int,
    last_index_with_non_zero_dose: int = None,
):
    """
    Przypisuje niezerową dawkę w dozwolonym czasie z uwagi na obostrzenia czasowe.

    Do użytku z:
    * normalized crossover
    * uniform crossover
    """
    if source_genome[index] == 0:
        destination_genome.append(source_genome[index])
    else:
        if last_index_with_non_zero_dose is None:
            # if first non-zero dose
            destination_genome.append(source_genome[index])
            last_index_with_non_zero_dose = index
        elif last_index_with_non_zero_dose + interval_in_indices < index:
            # if non-zero dose after the time constraint
            destination_genome.append(source_genome[index])
            last_index_with_non_zero_dose = index
        else:
            # if non-zero dose but within the time constraint
            destination_genome.append(0)
    return last_index_with_non_zero_dose


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


def cross_one_point(x1, x2, config):
    """
    Krzyżowanie w jednym punkcie: losowo wybierany jest jeden punkt w chromosomie dziecka. Wszystkie geny przed tym
    punktem pochodzą od pierwszego rodzica, a wszystkie geny za tym punktem pochodzą od rodzica drugiego. Sytuacja jest
    odwrotna w przypadku drugiego dziecka.
    :param x1:                  list
    :param x2:                  list
    :return: new_x1, new_x1:    list, list
    """
    interval_in_indices = int(2 * config['time_interval_hours'])

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
    new_x2[0:cross_point] = x2[0:cross_point]

    new_x1[cross_point:length] = x2[cross_point:length]
    new_x2[cross_point:length] = x1[cross_point:length]

    new_x1 = refine_genome_around_cross_point_to_time_constraint(
        genome=new_x1, interval_in_indices=interval_in_indices, config=config)
    new_x2 = refine_genome_around_cross_point_to_time_constraint(
        genome=new_x2, interval_in_indices=interval_in_indices, config=config)

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
    interval_in_indices = int(2 * config['time_interval_hours'])

    new_x1 = []
    new_x2 = []
    length = len(x1)
    lower_range = 1
    upper_range = length - 1
    middle_range = (upper_range - lower_range) // 2
    if config['crossover']['cross_points_range_percentage']:
        lower_range = int(length * config['crossover']['cross_points_range_percentage'][0] / 100)
        middle_range = int(length * config['crossover']['cross_points_range_percentage'][1] / 100)
        upper_range = int(length * config['crossover']['cross_points_range_percentage'][2] / 100)

    cross_point_1 = np.random.randint(lower_range, middle_range)
    cross_point_2 = np.random.randint(cross_point_1, upper_range)

    new_x1[0:length] = x1[0:length]
    new_x2[0:length] = x2[0:length]

    new_x1[cross_point_1:cross_point_2] = x2[cross_point_1:cross_point_2]
    new_x2[cross_point_1:cross_point_2] = x1[cross_point_1:cross_point_2]

    new_x1 = refine_genome_around_cross_point_to_time_constraint(
        genome=new_x1, interval_in_indices=interval_in_indices, config=config)
    new_x2 = refine_genome_around_cross_point_to_time_constraint(
        genome=new_x2, interval_in_indices=interval_in_indices, config=config)

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
    interval_in_indices = int(2 * config['time_interval_hours'])

    new_x1 = []
    new_x2 = []
    last_index_with_non_zero_dose_x1 = None
    last_index_with_non_zero_dose_x2 = None
    for i in range(len(x1)):
        p = np.random.uniform()
        last_index_with_non_zero_dose_x1 = assign_dose_on_position_with_constraint(
            source_genome=x1 if p < 0.5 else x2,
            destination_genome=new_x1,
            index=i,
            interval_in_indices=interval_in_indices,
            last_index_with_non_zero_dose=last_index_with_non_zero_dose_x1,
        )
        last_index_with_non_zero_dose_x2 = assign_dose_on_position_with_constraint(
            source_genome=x2 if p < 0.5 else x1,
            destination_genome=new_x2,
            index=i,
            interval_in_indices=interval_in_indices,
            last_index_with_non_zero_dose=last_index_with_non_zero_dose_x2,
        )

    new_x1 = normalize_crossover(new_x1, config)
    new_x2 = normalize_crossover(new_x2, config)

    return new_x1, new_x2


def gene_group_replacement(x1, x2, group, config):
    """
    Metoda buduje genom dziecka przekazując mu geny z wyselekcjonowanej grupy od rodzica x2
    i pozostałe geny od rodzica x1.
    :param x1:      list
    :param x2:      list
    :param group:   list
    :return: new_x: list
    """
    interval_in_indices = int(2 * config['time_interval_hours'])

    new_x = []
    last_index_with_non_zero_dose = None
    for idx, gene in enumerate(x1):
        source_genome = x2 if idx in group else x1
        last_index_with_non_zero_dose = assign_dose_on_position_with_constraint(
            source_genome=source_genome,
            destination_genome=new_x,
            index=idx,
            interval_in_indices=interval_in_indices,
            last_index_with_non_zero_dose=last_index_with_non_zero_dose,
        )

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

    new_x1 = gene_group_replacement(x1, x2, group, config)
    new_x2 = gene_group_replacement(x2, x1, group, config)

    new_x1 = normalize_crossover(new_x1, config)
    new_x2 = normalize_crossover(new_x2, config)

    return new_x1, new_x2
