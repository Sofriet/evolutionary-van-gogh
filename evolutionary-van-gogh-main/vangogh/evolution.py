import time

import numpy as np
from PIL import Image
from multiprocess import Pool, cpu_count

from vangogh import selection, variation
from vangogh.fitness import drawing_fitness_function, draw_voronoi_image
from vangogh.population import Population
from vangogh.util import NUM_VARIABLES_PER_POINT, IMAGE_SHRINK_SCALE, REFERENCE_IMAGE
import cv2
from skimage.color import rgb2gray
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Create grey-box initial genes
def generate_initial_seed_edgebased(reference_image, reference_points, num_points, seed=0):
    """
    Generate initial Voronoi points:
    - first reference_points come from edges
    - remaining points are random
    """

    np.random.seed(seed)  # Ensure deterministic output

    small_image = reference_image.copy()
    small_image.thumbnail((int(reference_image.width / IMAGE_SHRINK_SCALE),
                           int(reference_image.height / IMAGE_SHRINK_SCALE)))
    img_array = np.array(small_image)
    gray = rgb2gray(img_array)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
    ys, xs = np.where(edges > 0)

    total_available_edge_points = len(xs)
    actual_reference_points = min(reference_points, total_available_edge_points)

    # Sample edge points
    idx = np.random.choice(total_available_edge_points, size=actual_reference_points, replace=False)

    initial_points = []
    current_num_points = 0

    # Add edge-based points
    for i in range(actual_reference_points):
        x = xs[idx[i]]
        y = ys[idx[i]]
        color = img_array[y, x][:3]
        initial_points.extend([x, y, *color])
        current_num_points += 1

    # Add random points
    width, height = small_image.width, small_image.height
    while current_num_points < num_points:
        x_rand = np.random.randint(0, width)
        y_rand = np.random.randint(0, height)
        color_rand = img_array[y_rand, x_rand][:3]
        initial_points.extend([x_rand, y_rand, *color_rand])
        current_num_points += 1

    return np.array(initial_points).reshape(1, -1)  # shape: (1, genotype_length)

def generate_initial_seed_region_based(reference_image, reference_points, num_points, seed=0):
    np.random.seed(seed)  # <--- Ensure deterministic output

    small_image = reference_image.copy()
    small_image.thumbnail((int(reference_image.width / IMAGE_SHRINK_SCALE),
                           int(reference_image.height / IMAGE_SHRINK_SCALE)))
    img_array = np.array(small_image)
    height, width = img_array.shape[:2]

    # Compute grid dimensions based on number of points
    grid_size = max(int(np.sqrt(reference_points)), 1)

    if grid_size * grid_size == reference_points:
        x_cell = width // grid_size
    else:
        x_cell = width // (grid_size + 1)

    y_cell = height // grid_size

    points = []
    current_num_points = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if current_num_points >= reference_points:
                break

            # Define region boundaries
            x_start = j * x_cell
            x_end = (j + 1) * x_cell
            y_start = i * y_cell
            y_end = (i + 1) * y_cell

            # Ensure within bounds
            x_end = min(x_end, width)
            y_end = min(y_end, height)

            region = img_array[y_start:y_end, x_start:x_end]

            # Compute average color
            avg_color = region.reshape(-1, 3).mean(axis=0).astype(int)

            # Use center of the region for the point
            x_center = (x_start + x_end) // 2
            y_center = (y_start + y_end) // 2

            points.extend([x_center, y_center, *avg_color])
            current_num_points += 1

    # When reference_points isn't a perfect square
    diff_points = reference_points - current_num_points
    if diff_points > 0:
        y_new_cell = height // diff_points
        for i in range(diff_points):
            y_start = i * y_new_cell
            y_end = (i + 1) * y_new_cell
            y_end = min(y_end, height)

            x_start = grid_size * x_cell
            x_end = width

            #logger.info(f"Box: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}, width={width}, height={height}")

            region = img_array[y_start:y_end, x_start:x_end]

            avg_color = region.reshape(-1, 3).mean(axis=0).astype(int)

            x_center = (x_start + x_end) // 2
            y_center = (y_start + y_end) // 2

            points.extend([x_center, y_center, *avg_color])

            current_num_points += 1

    #logger.info(f"nrs: npoints={num_points}, current={current_num_points}, ref={reference_points}")

    width, height = small_image.width, small_image.height
    while current_num_points < num_points:
        x_rand = np.random.randint(0, width)
        y_rand = np.random.randint(0, height)
        color_rand = img_array[y_rand, x_rand][:3]
        points.extend([x_rand, y_rand, *color_rand])
        current_num_points += 1

    #logger.info(f"nrs: npoints={num_points}, current={current_num_points}, ref={reference_points}")
    #print(num_points, current_num_points, reference_points, flush=True)
    return np.array(points).reshape(1, -1)


class Evolution:
    def __init__(self,
                 num_points,
                 reference_image: Image,
                 alpha=0.35,
                 elite_frac=0,
                 elitist_set=None,
                 max_elitist_set_size=20,
                 evolution_type='p+o',
                 population_size=200,
                 generation_budget=-1,
                 evaluation_budget=-1,
                 crossover_method="ONE_POINT",
                 mutation_probability='inv_mutable_genotype_length',
                 num_features_mutation_strength=.5,
                 num_features_mutation_strength_decay=None,
                 num_features_mutation_strength_decay_generations=None,
                 selection_name='tournament_2',
                 initialization='GREYBOX',
                 noisy_evaluations=False,
                 verbose=False,
                 generation_reporter=None,
                 seed=0):

        self.reference_image: Image = reference_image.copy()
        self.reference_image.thumbnail((int(self.reference_image.width / IMAGE_SHRINK_SCALE),
                                        int(self.reference_image.height / IMAGE_SHRINK_SCALE)),
                                       Image.ANTIALIAS)
        self.reference_image_array = np.asarray(self.reference_image)

        num_variables = num_points * NUM_VARIABLES_PER_POINT
        feature_intervals = []
        for i in range(num_variables):
            if i % NUM_VARIABLES_PER_POINT == 0:  # X
                feature_intervals.append([0, self.reference_image.width])
            elif i % NUM_VARIABLES_PER_POINT == 1:  # Y
                feature_intervals.append([0, self.reference_image.height])
            else:  # color (RGBA)
                feature_intervals.append([0, 256])

        self.num_points = num_points
        self.feature_intervals = feature_intervals
        self.evolution_type = evolution_type
        self.population_size = population_size
        self.generation_budget = generation_budget
        self.evaluation_budget = evaluation_budget
        self.mutation_probability = mutation_probability
        self.num_features_mutation_strength = num_features_mutation_strength
        self.num_features_mutation_strength_decay = num_features_mutation_strength_decay
        self.num_features_mutation_strength_decay_generations = num_features_mutation_strength_decay_generations
        self.selection_name = selection_name
        self.noisy_evaluations = noisy_evaluations
        self.verbose = verbose
        self.generation_reporter = generation_reporter
        self.crossover_method = crossover_method
        self.num_evaluations = 0
        self.initialization = initialization
        self.alpha = alpha
        self.elite_frac = elite_frac
        self.elitist_set = elitist_set
        self.max_elitist_set_size = max_elitist_set_size

        np.random.seed(seed)
        self.seed = seed

        # set feature intervals to be a np.array
        if type(feature_intervals) != np.array:
            self.feature_intervals = np.array(feature_intervals, dtype=object)

        # check that tournament size is compatible
        if 'tournament' in selection_name:
            self.tournament_size = int(selection_name.split('_')[-1])
            if self.population_size % self.tournament_size != 0:
                raise ValueError('The population size must be a multiple of the tournament size')

        # set up population and elite
        self.genotype_length = len(feature_intervals)
        self.population = Population(self.population_size, self.genotype_length, self.initialization)
        self.elite = None
        self.elite_fitness = np.inf

        # set up mutation probability if set to default "inv_mutable_genotype_length"
        if mutation_probability == 'inv_genotype_length':
            self.mutation_probability = 1 / self.genotype_length
        elif mutation_probability == "inv_mutable_genotype_length":
            num_unmutable_features = 0
            self.mutation_probability = 1 / (self.genotype_length - num_unmutable_features)

            # incompatibilities
        if self.evolution_type == 'p+o' and self.noisy_evaluations:
            raise ValueError(
                "Using P+O is not compatible with noisy evaluations (you would need to re-evaluate the parents every generation, which is expensive)")
        elif 'age_reg' in self.evolution_type:
            print(
                "Warning: using noisy evaluations but age regularized evolution does not re-evaluate the entire population every generation")

    def __update_elite(self, population):
        best_fitness_idx = np.argmin(population.fitnesses)
        best_fitness = population.fitnesses[best_fitness_idx]
        if self.noisy_evaluations or best_fitness < self.elite_fitness:
            self.elite = population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

    def __update_elitist_set(self):
        """Updates the elitist set by adding/replacing elites if they are better than the worst in the set."""
        if self.elitist_set is None:
            self.elitist_set = Population(1, self.genotype_length, "N/A")
            self.elitist_set.genes = np.array([self.elite])
            self.elitist_set.fitnesses = np.array([self.elite_fitness])
            return

        # check if new elite is unique
        is_different = not any(np.array_equal(self.elite, elite) for elite in self.elitist_set.genes)
        if not is_different:
            return
        
        worst_fitness_in_set = np.max(self.elitist_set.fitnesses)
        is_better_than_worst = self.elite_fitness < worst_fitness_in_set

        # elitist set not full
        if len(self.elitist_set.fitnesses) < self.max_elitist_set_size:
            new_genes = np.vstack([self.elitist_set.genes, self.elite])
            new_fitnesses = np.concatenate([self.elitist_set.fitnesses, [self.elite_fitness]])
        
        # elitist full
        elif is_better_than_worst:
            worst_index = np.argmax(self.elitist_set.fitnesses)
            new_genes = self.elitist_set.genes.copy()
            new_fitnesses = self.elitist_set.fitnesses.copy()
            new_genes[worst_index] = self.elite
            new_fitnesses[worst_index] = self.elite_fitness
        
        # set is full and new elite is worse than worst of set
        else:
            return

        self.elitist_set.genes = new_genes
        self.elitist_set.fitnesses = new_fitnesses

    def __classic_generation(self, merge_parent_offspring=False):
        # create offspring population
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()
        # variation
        offspring.genes = variation.crossover(offspring.genes, self.crossover_method)
        offspring.genes = variation.mutate(offspring.genes, self.feature_intervals,
                                           mutation_probability=self.mutation_probability,
                                           num_features_mutation_strength=self.num_features_mutation_strength)
        # evaluate offspring
        offspring.fitnesses = drawing_fitness_function(offspring.genes,
                                                       self.reference_image)
        self.num_evaluations += len(offspring.genes)

        self.__update_elite(offspring)

        if 'sus-elitist-select' in self.selection_name:
            self.__update_elitist_set()

        # selection
        if merge_parent_offspring:
            # p+o mode
            self.population.stack(offspring)
        else:
            # just replace the entire thing
            self.population = offspring

        self.population = selection.select(population=self.population,
                                            selection_size=self.population_size, 
                                            alpha=self.alpha if 'sus' in self.selection_name else None, 
                                            elite_frac=self.elite_frac if 'sus-select' in self.selection_name else None,
                                            selection_name=self.selection_name, 
                                            elitist_set=self.elitist_set if 'sus-elitist-select' in self.selection_name else None)
    
    
    def __ssga_generation(self):
        """
        Perform one steady‐state update:
        pick two parents from self.population
        produce exactly one child via crossover+mutation
        evaluate that child
        pick one individual (victim) in self.population to replace
        if child is better, overwrite the victim
        """
        
        parent1_pop = selection.select(self.population, 1, selection_name=self.selection_name)
        parent2_pop = selection.select(self.population, 1, selection_name=self.selection_name)
        p1 = parent1_pop.genes[0, :].copy()
        p2 = parent2_pop.genes[0, :].copy()

        parents_array = np.vstack([p1, p2])  # shape = (2, genotype_length)
        children = variation.crossover(parents_array, self.crossover_method)
        child = children[0].copy()  # take the first of the two offspring

        # Now mutate that single child (shape => (1, genotype_length)), then unpack back to 1‐D
        child = variation.mutate(child[None, :],
                                self.feature_intervals,
                                mutation_probability=self.mutation_probability,
                                num_features_mutation_strength=self.num_features_mutation_strength)[0]

        child_fitness = drawing_fitness_function(child[None, :], self.reference_image)[0]
        self.num_evaluations += 1

        # Update the global elite if needed
        if child_fitness < self.elite_fitness or self.elite is None:
            self.elite = child.copy()
            self.elite_fitness = child_fitness

        # For a simple “replace‐worst” rule:
        victim_idx = np.argmax(self.population.fitnesses)

        # replace if child better
        if child_fitness < self.population.fitnesses[victim_idx]:
            self.population.genes[victim_idx, :] = child
            self.population.fitnesses[victim_idx] = child_fitness


    def run(self, custom_initial_genes=None):
        data = []

        self.population.initialize(self.feature_intervals, custom_genes=custom_initial_genes)

        self.population.fitnesses = drawing_fitness_function(self.population.genes,
                                                             self.reference_image)
        self.num_evaluations = len(self.population.genes)

        best_fitness_idx = np.argmin(self.population.fitnesses)
        best_fitness = self.population.fitnesses[best_fitness_idx]
        if best_fitness > self.elite_fitness:
            self.elite = self.population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

        start_time_seconds = time.time()

        # run generation_budget
        i_gen = 0
        while True:
            if self.num_features_mutation_strength_decay_generations is not None:
                if i_gen in self.num_features_mutation_strength_decay_generations:
                    self.num_features_mutation_strength *= self.num_features_mutation_strength_decay

            if self.evolution_type == 'classic':
                self.__classic_generation(merge_parent_offspring=False)
            elif self.evolution_type == 'p+o':
                self.__classic_generation(merge_parent_offspring=True)
            elif self.evolution_type == 'ssga':
                steps_per_gen = self.population_size// 2
                # “take” steps_per_gen microevaluations before we call that a generation
                for _ in range(steps_per_gen):
                    self.__ssga_generation()
            
            else:
                raise ValueError('unknown evolution type:', self.evolution_type)

            # generation terminated
            i_gen += 1
            avg_fitness = np.mean(self.population.fitnesses)
            if self.verbose:
                print('generation:', i_gen, 'best fitness:', self.elite_fitness, 'avg. fitness:',
                      avg_fitness)

            data.append({"num-generations": i_gen,
                         "num-evaluations": self.num_evaluations,
                         "time-elapsed": time.time() - start_time_seconds,
                         "best-fitness": self.elite_fitness,
                         'avg-fitness': np.mean(self.population.fitnesses),
                         'selection-name':self.selection_name,
                         'num-features-mutation-strength': self.num_features_mutation_strength,
                         "crossover-method": self.crossover_method,
                         "population-size": self.population_size, "num-points": self.num_points,
                         "initialization": self.initialization,
                         "seed": self.seed})
            
            if self.selection_name == 'sus-select':
                data.append({
                    "elite-frac": self.elite_frac,
                    "alpha": self.alpha,
                })
            if self.selection_name == 'sus-elitist-select':
                data.append({
                    "elitist_size": self.max_elitist_set_size,
                    "alpha": self.alpha,
                })

            if self.generation_reporter is not None:
                self.generation_reporter(
                    {"num-generations": i_gen, "num-evaluations": self.num_evaluations,
                     "time-elapsed": time.time() - start_time_seconds}, self)

            if 0 < self.generation_budget <= i_gen:
                break
            if 0 < self.evaluation_budget <= self.num_evaluations:
                break

            # check if evolution should terminate because optimum reached or population converged
            if self.population.is_converged():
                break

        draw_voronoi_image(self.elite, self.reference_image.width, self.reference_image.height,
                           scale=IMAGE_SHRINK_SCALE) \
            .save(
            f"./img/van_gogh_final_{self.seed}_{self.population_size}_{self.crossover_method}_{self.num_points}_{self.initialization}_{self.generation_budget}.png")
            # f"./final_img/van_gogh_final_seed{self.seed}_sel{self.selection_name}_mutS{self.num_features_mutation_strength}_{self.population_size}_{self.crossover_method}_{self.num_points}_{self.initialization}_{self.generation_budget}.png")
        return data


if __name__ == '__main__':
    evo = Evolution(100,
                    1.0,
                    0.7,
                    REFERENCE_IMAGE,
                    elitist_set=None,
                    max_elitist_set_size=80,
                    evolution_type='p+o',
                    population_size=100,
                    generation_budget=300,
                    crossover_method='ONE_POINT',
                    initialization='GREYBOX',
                    num_features_mutation_strength=.5,
                    num_features_mutation_strength_decay=None,
                    num_features_mutation_strength_decay_generations=None,
                    selection_name='tournament_4',
                    noisy_evaluations=False,
                    verbose=False)
    evo.run()
