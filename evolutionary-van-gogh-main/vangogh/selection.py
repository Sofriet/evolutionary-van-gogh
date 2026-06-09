import numpy as np

from vangogh.population import Population


def select(population, selection_size, elite_frac, alpha, elitist_set, selection_name='tournament_4'):
    if 'tournament' in selection_name:
        tournament_size = int(selection_name.split('_')[-1])
        return tournament_select(population, selection_size, tournament_size)
    if 'truncation' in selection_name:
        trunc_percent = float(selection_name.split('_')[-1])
        return truncation_select(population, selection_size, trunc_percent)
    if 'sus-select' in selection_name:
        return sus_select(population, selection_size, elite_frac, alpha)
    if 'sus-elitist-select' in selection_name:
        return sus_select_elitist_track(population, selection_size, alpha, elitist_set)
    else:
        raise ValueError('Invalid selection name:', selection_name)


def one_tournament_round(population, tournament_size, return_winner_index=False):
    rand_perm = np.random.permutation(len(population.fitnesses))
    competing_fitnesses = population.fitnesses[rand_perm[:tournament_size]]
    winning_index = rand_perm[np.argmin(competing_fitnesses)]
    if return_winner_index:
        return winning_index
    else:
        return {
            'genotype': population.genes[winning_index, :],
            'fitness': population.fitnesses[winning_index],
        }


def tournament_select(population, selection_size, tournament_size=4):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")

    n = len(population.fitnesses)
    num_selected_per_iteration = n // tournament_size
    num_parses = selection_size // num_selected_per_iteration

    for i in range(num_parses):
        # shuffle
        population.shuffle()

        winning_indices = np.argmin(population.fitnesses.squeeze().reshape((-1, tournament_size)),
                                    axis=1)
        winning_indices += np.arange(0, n, tournament_size)

        selected.genes[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration,
        :] = population.genes[winning_indices, :]
        selected.fitnesses[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration] = \
        population.fitnesses[winning_indices]
    
    return selected

def sus_select(population, selection_size, elite_frac, alpha):
    fitnesses = population.fitnesses.squeeze()
    elite_size = int(elite_frac * selection_size)
    elite_indices = np.argpartition(fitnesses, elite_size)[:elite_size]
    
    # select the rest with sus
    non_elite_mask = np.ones(len(fitnesses), dtype=bool)
    non_elite_mask[elite_indices] = False
    non_elite_fitness = fitnesses[non_elite_mask]
    
    # normalize and weight non-elites
    scaled = (non_elite_fitness - np.min(non_elite_fitness)) / (np.ptp(non_elite_fitness) + 1e-10)
    # inverse scaling
    weights = (1.0 / (scaled + 1e-10)) ** alpha
    weights /= np.sum(weights)
    
    pointers = np.random.uniform(0, 1/(selection_size - elite_size)) + \
               np.arange(selection_size - elite_size)/(selection_size - elite_size)
    non_elite_selected = np.searchsorted(np.cumsum(weights), pointers)
    
    all_indices = np.concatenate([elite_indices, np.where(non_elite_mask)[0][non_elite_selected]])
    selected = Population(selection_size, population.genes.shape[1], "N/A")
    selected.genes = population.genes[all_indices, :]
    selected.fitnesses = population.fitnesses[all_indices]
    return selected


def truncation_select(population: Population, selection_size,trunc_percent=70):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")
    n = len(population.fitnesses)

    # sort and get the winning idx
    sorted_indices = np.argsort(population.fitnesses)
    cutoff = max(int(n*trunc_percent/100),1)
    winning_indices = sorted_indices[:cutoff]

    # Find the elite individuals
    elite_genes = population.genes[winning_indices, :] # genes are 2D array
    elite_fitnesses = population.fitnesses[winning_indices]

    # select randomly among elites
    for i in range(selection_size):
        selected_idx = np.random.randint(0, cutoff)
        selected.genes[i,:] = elite_genes[selected_idx,:]
        selected.fitnesses[i] = elite_fitnesses[selected_idx]

def sus_select_elitist_track(population, selection_size, alpha, elitist_set):

    fitnesses = population.fitnesses.squeeze()
    combined_pop = Population(population.genes.shape[0] + elitist_set.genes.shape[0], 
                             population.genes.shape[1], "N/A")
    combined_pop.genes = np.vstack([population.genes, elitist_set.genes])
    combined_pop.fitnesses = np.concatenate([population.fitnesses, elitist_set.fitnesses])
    combined_fitness = combined_pop.fitnesses.squeeze()

    scaled = (combined_fitness - np.min(combined_fitness)) / (np.ptp(combined_fitness) + 1e-10)
    weights = (1.0 / (scaled + 1e-10)) ** alpha
    weights /= np.sum(weights)

    pointers = np.random.uniform(0, 1/selection_size) + np.arange(selection_size)/selection_size
    selected_indices = np.searchsorted(np.cumsum(weights), pointers)
    
    selected = Population(selection_size, population.genes.shape[1], "N/A")
    selected.genes = combined_pop.genes[selected_indices, :]
    selected.fitnesses = combined_pop.fitnesses[selected_indices]
    return selected