import numpy as np
import os
import yaml


# ==DOSE TIME CONSTRAINT UTILS==========================================================================================
def assign_dose_within_time_constraint(
    doses_time_steps: list,
    time_steps: np.ndarray = np.arange(0, 24*5*600, 300),
    time_interval_steps: int = 1800,
):
    """Obliczenie czasu dawki w przypadku obostrzeń czasowych."""
    allowed_time_steps = time_steps.copy()
    for dose_time_step in doses_time_steps:
        lower_constraint = max(dose_time_step - time_interval_steps, 0)
        upper_constraint = min(dose_time_step + time_interval_steps, max(time_steps))
        for time_step in allowed_time_steps:
            if lower_constraint <= time_step <= upper_constraint:
                allowed_time_steps = allowed_time_steps[allowed_time_steps != time_step]

    if len(allowed_time_steps) > 0:
        dose_time = np.random.choice(allowed_time_steps)
        return dose_time
    return None


def refine_genome_around_cross_point_to_time_constraint(genome: list, interval_in_indices: int, config: dict):
    """
    Zmiana istniejącego genomu do postaci zgodnej z narzuconymi obostrzeniami czasowymi dla niezerowych dawek.
    Szczególnie istotne przy mutacji, działa na zasadzie wywoływań rekurencyjnych,
    rozwiązujac problem dla każej z niezerowych dawek z osobna.

    Do użytku z:
    * crossover one point
    * crossover two points
    * mutations of all types
    """

    genome = np.array(genome)
    non_zero_dose_indices = np.argwhere(genome > 0)[:, 0]
    intervals_between_doses = np.diff(non_zero_dose_indices)
    if not (intervals_between_doses < interval_in_indices).any():
        # if time intervals between non-zero doses are above the threshold
        return list(genome)

    # relative positions of two non-zero doses which are too close to each other
    relative_positions_with_wrong_doses = np.argwhere(intervals_between_doses < interval_in_indices + 1)[:, 0]
    temporal_non_zero_dose_indices = np.delete(non_zero_dose_indices, relative_positions_with_wrong_doses)

    for relative_dose_position, _ in enumerate(relative_positions_with_wrong_doses):
        dose_time_step = assign_dose_within_time_constraint(
            doses_time_steps=temporal_non_zero_dose_indices * config['protocol_resolution'],
            time_interval_steps=int(config['time_interval_hours'] * 600),
        )
        current_dose_index = non_zero_dose_indices[relative_positions_with_wrong_doses[relative_dose_position]]
        if dose_time_step is not None:
            dose_time_step = int(dose_time_step / config['protocol_resolution'])
            genome[dose_time_step] = genome[current_dose_index]
            temporal_non_zero_dose_indices = np.append(temporal_non_zero_dose_indices, dose_time_step)
        genome[current_dose_index] = 0

    return list(genome)


# ==RANDOM PROTOCOL INITIALIZATION======================================================================================
def get_random_protocol(
    min_dose: float = 0.25,
    max_dose: float = 10.0,
    step_value: float = 0.25,
    max_dose_value: float = 2.5,
    time_steps: np.ndarray = None,
    time_interval_steps: int = 1800,
):
    """Tworzenie pojedynczego, losowego protokołu dla ekspermentu."""
    protocol_dose = 0
    max_protocol_value = max_dose_value
    protocol_pair = []
    doses_time_steps = []
    while protocol_dose < max_dose:
        if max_dose - protocol_dose < max_dose_value:
            max_protocol_value = max_dose - protocol_dose

        upper_value_boundary = int(round((max_protocol_value - min_dose) / step_value)) + 1
        new_dose_value = np.random.randint(0, upper_value_boundary) * step_value + min_dose

        dose_time_step = assign_dose_within_time_constraint(
            doses_time_steps=doses_time_steps,
            time_steps=time_steps,
            time_interval_steps=time_interval_steps,
        )
        if dose_time_step is not None:
            doses_time_steps.append(dose_time_step)
            protocol_pair.append((dose_time_step, new_dose_value))
            protocol_dose += new_dose_value
        else:
            break

    return protocol_pair


def get_rand_population(
        min_dose: float = 0.25,
        max_dose: float = 10.0,
        step_value: float = 0.25,
        num_protocols: int = 10,
        max_dose_value: float = 2.5,
        hour_steps: int = 600,
        protocol_resolution: int = 300,
        available_hours: int = 24 * 5,
        time_interval_hours: float = 3.0,
):
    """Tworzenie losowoej populacji protokołów dla eksperymentu."""
    time_steps = np.arange(0, available_hours * hour_steps, protocol_resolution)
    time_interval_steps = int(hour_steps * time_interval_hours)
    protocol_pairs = []
    for protocol in range(num_protocols):
        protocol_pair = get_random_protocol(
            min_dose=min_dose,
            max_dose=max_dose,
            step_value=step_value,
            max_dose_value=max_dose_value,
            time_steps=time_steps,
            time_interval_steps=time_interval_steps,
        )
        protocol_pairs.append(protocol_pair)

    return protocol_pairs


# ==REPRESENTATION CONVERTER============================================================================================
class ConvertRepresentation:
    """Konwerter reprezentacji protokołów."""
    NUMBER_OF_DAYS = 5

    def __init__(self, hour_steps, protocol_resolution):
        self.hour_steps = hour_steps
        self.protocol_resolution = protocol_resolution

    def calculate_number_of_steps(self):
        return self.NUMBER_OF_DAYS * 24 * self.hour_steps // self.protocol_resolution

    def convert_pairs_to_list(self, protocol):
        list_protocol = np.zeros(self.calculate_number_of_steps())
        for (timestep, value) in protocol:
            list_protocol[int(timestep // self.protocol_resolution)] = value
        return list_protocol

    def convert_list_to_pairs(self, protocol):
        non_zero_indices = [i for i, value in enumerate(protocol) if value != 0]
        pair_protocol = [
            tuple((index * self.protocol_resolution, protocol[index]))
            for index in non_zero_indices
        ]
        return pair_protocol

    def convert_population_lists_to_pairs(self, protocols):
        return [
            self.convert_list_to_pairs(protocol=protocol)
            for protocol in protocols
        ]


# ==PREDICTION MODEL WRAPPER============================================================================================
class ModelWrapper:
    def __init__(self, model, converter):
        self.model = model
        self.converter = converter

    def predict(self, population):
        return self.model.predict(population)


# ==OUTPUT SAVING=======================================================================================================
def save_output(file: any, file_name: str, extension: str = 'csv', config=None):
    """Zapis plików wyjściowych dla eksperymentu według rozszerzeń."""
    saving_directory = resolve_saving_path(config=config)
    saving_path = f'{saving_directory}/{file_name}.{extension}'

    if extension == 'csv':
        file.to_csv(saving_path, index=False)
    else:
        with open(saving_path, 'w') as f:
            f.write(str(file))


def resolve_saving_path(path: str = 'experiment_results', config=None):
    """Stworzenie ścieżki do zapisu plików wyjsciowych eksperymentu."""
    if config:
        sub_path = f'{os.path.basename(config["config_path"]).split(".")[0]}_{config["experiment_time"]}'
        path = os.path.join(path, sub_path)
    saving_directory = os.path.join(os.getcwd(), f'python_ga/{path}')

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    return saving_directory


# ==CONFIG FILE READER==================================================================================================
def read_config(config_path: str):
    """Odczyt konfiguracji z pliku Yaml."""
    with open(config_path, 'r') as file:
        config = yaml.load(file, yaml.FullLoader)
    return config


# ==PROBABILITY ANNEALING===============================================================================================
def calculate_probability_annealing(iteration, max_value=0.5, max_iter=100, eps=0.001, rounding_decimal=6):
    """Symulowanie wyżarzanie."""
    def formula(x):
        """Formuła do obliczenia prawdopodobieństw przy wyżarzaniu."""
        result = np.round((x ** 4) * max_value, rounding_decimal)
        return result if result >= eps else eps

    x = 1 - (iteration - 1) / max_iter
    return formula(x)


# ==FITNESS CALCULATION=================================================================================================
def calculate_fitness(paired_population, model):
    """
    Metoda odpowiada za obliczenie wartości funkcji dopasowania dla osobników w populacji, przy użyciu wybranego modelu.
    Otrzymany wynik przekształcany jest do tablicy jednowymiarowej.
    :param paired_population:       list
    :param model:                   fitness model
    :return: pop_fitness:           list
    :return: paired_population:     list
    """

    pop_fitness = model.predict(paired_population)
    pop_fitness = pop_fitness.reshape(len(paired_population))
    return pop_fitness


def store_fitness_and_populations(
        all_fitness: list, all_populations: list, fitness: np.ndarray, paired_population: np.ndarray):
    """
    Metoda do zapisu wszystkich wartości funkcji fitness oraz wszystkich populacji protokołów.
    :param all_fitness:             all fitness values.
    :param all_populations:         all populations.
    :param fitness:                 last fitness.
    :param paired_population:       last population in paired representation.
    :return: updated list of all populations
    """
    all_fitness.append(list(fitness))
    all_populations.append(paired_population)
    return all_fitness, all_populations
