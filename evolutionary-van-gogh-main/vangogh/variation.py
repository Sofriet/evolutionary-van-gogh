import numpy as np
import re

BLOCK_SIZE = 5


def crossover(genes, method="ONE_POINT"):
    # print(genes.shape)
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        offspring = np.zeros(shape=genes.shape, dtype=int)

        for i in range(len(genes)):
            offspring[i, :] = np.where(np.arange(genes.shape[1]) <= crossover_points[i], parents_1[i, :],
                                       parents_2[i, :])

    elif method == "ONE_POINT_BLOCK":
        crossover_points = np.random.randint(0, int(genes.shape[1] / BLOCK_SIZE), size=genes.shape[0]) * BLOCK_SIZE
        offspring = np.zeros(shape=genes.shape, dtype=int)

        for i in range(len(genes)):
            offspring[i, :] = np.where(np.arange(genes.shape[1]) < crossover_points[i], parents_1[i, :],
                                       parents_2[i, :])

    elif method == "UNIFORM":
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(len(genes)):
            for j in range(genes.shape[1]):
                r = np.random.rand()
                if r < 0.5:
                    offspring[i, j] = parents_1[i, j]
                else:
                    offspring[i, j] = parents_2[i, j]

    elif method == "UNIFORM_BLOCK":
        block_mask = np.random.rand(genes.shape[0], int(genes.shape[1] / BLOCK_SIZE)) < 0.5
        rand_mask = np.repeat(block_mask, BLOCK_SIZE, axis=1)[:, :genes.shape[1]]
        offspring = np.where(rand_mask, parents_1, parents_2)

    elif re.match(r"\d+_POINT$", method):  # Match on things like 7_point
        n = int(re.match(r"(\d+)_POINT", method).group(1))  # And then get the integer 7
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(len(genes)):
            points = sorted(np.random.choice(range(1, genes.shape[1]), size=n, replace=False))
            points = [0] + points + [genes.shape[1] - 1]
            first_parent = True
            for j in range(n + 1):
                if first_parent:
                    offspring[i, points[j]: points[j + 1]] = parents_1[i, points[j]: points[j + 1]]
                else:
                    offspring[i, points[j]: points[j + 1]] = parents_2[i, points[j]: points[j + 1]]
                first_parent = not first_parent

    elif re.match(r"\d+_POINT_BLOCK$", method):
        n = int(re.match(r"(\d+)_POINT_BLOCK", method).group(1))
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(len(genes)):
            points = sorted(np.random.choice(range(1, int(genes.shape[1] / BLOCK_SIZE)), size=n, replace=False))
            points = [0] + points + [int(genes.shape[1] / BLOCK_SIZE) - 1]
            first_parent = True
            for j in range(n + 1):
                if first_parent:
                    offspring[i, points[j] * 5: points[j + 1] * 5] = parents_1[i, points[j] * 5: points[j + 1] * 5]
                else:
                    offspring[i, points[j] * 5: points[j + 1] * 5] = parents_2[i, points[j] * 5: points[j + 1] * 5]
                first_parent = not first_parent

    else:
        raise Exception("Unknown crossover method")

    return offspring


def mutate(genes, feature_intervals,
           mutation_probability=0.1, num_features_mutation_strength=0.05):
    mask_mut = np.random.choice([True, False], size=genes.shape,
                                p=[mutation_probability, 1 - mutation_probability])

    mutations = generate_plausible_mutations(genes, feature_intervals,
                                             num_features_mutation_strength)

    offspring = np.where(mask_mut, mutations, genes)

    return offspring


def generate_plausible_mutations(genes, feature_intervals,
                                 num_features_mutation_strength=0.25):
    mutations = np.zeros(shape=genes.shape)

    for i in range(genes.shape[1]):
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2

        mutations[:, i] = range_num * np.random.uniform(low=low, high=high,
                                                        size=mutations.shape[0])
        mutations[:, i] += genes[:, i]

        # Fix out-of-range
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i][1],
                                   feature_intervals[i][1], mutations[:, i])
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i][0],
                                   feature_intervals[i][0], mutations[:, i])

    mutations = mutations.astype(int)
    return mutations
