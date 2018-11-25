import itertools
import random
from copy import deepcopy
from typing import List

import numpy as np
import tqdm

from qap_lab.source.data_utils import Problem


class Chromosome:
    """
    Permutation vector wrapper
    """

    def __init__(self, gene_list: List[int], _problem: Problem):
        self.genome = gene_list
        self.flow_matrix = _problem.flow_matrix
        self.dist_matrix = _problem.distance_matrix
        self.fitness = self.calc_fitness()
        # print(self.fitness)

    def calc_fitness(self) -> float:
        n = len(self.genome)
        return sum([self.flow_matrix[i][j] * self.dist_matrix[self.genome[i]][self.genome[j]]
                    for i in range(n)
                    for j in range(n)])

    def swap_mutation(self):
        pass

    def scrumble_mutation(self):
        pass


class Population:

    def __init__(self, _chromosome_list: List[Chromosome], _problem: Problem):
        self.chromosome_list = _chromosome_list
        self.problem = _problem
        self.rolling_wheel_prob = self.calc_rolling_wheel_prob()

    def breeding(self) -> None:
        """
        Population evolving process
        """
        pass

    def get_best_chromosome(self) -> Chromosome:
        return min(self.chromosome_list, key=lambda x: x.fitness)

    def calc_rolling_wheel_prob(self):
        """
        Calculate inverted related fitness for minimization task
        :return:
        """
        _sum: float = sum([1 / chrom.fitness for chrom in self.chromosome_list])
        logits: list = [1 / chrom.fitness for chrom in self.chromosome_list]
        e_x: np.array = np.exp(np.array(logits) - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def select_n_chromosomes(self, n: int) -> List[Chromosome]:
        selected_index = np.random.choice(len(self.chromosome_list), n, p=self.rolling_wheel_prob)
        return [self.chromosome_list[i] for i in selected_index]


class GeneticAlgorithmSolver:
    def __init__(self, _problem: Problem):
        self.problem = _problem
        self.population_size = _problem.problem_size * 10
        self.selection_size = int(self.population_size * 0.3)
        self.population = self.generate_initial_population()

    def generate_initial_population(self) -> Population:
        _chromo_list = []
        genome = list(range(self.problem.problem_size))
        for _ in range(self.population_size):
            _chromo_list.append(Chromosome(deepcopy(genome), self.problem))
            random.shuffle(genome)
        return Population(_chromo_list, self.problem)

    def selection(self, population: Population) -> List[Chromosome]:
        """
        Rolling wheel selection
        :param population:
        :return:
        """
        return population.select_n_chromosomes(self.selection_size)

    def ordered_crossover(self, chrom_1: Chromosome, chrom_2: Chromosome) -> Chromosome:
        _ub = len(chrom_1.genome) - 1
        start_index = np.random.randint(0, _ub)
        end_index = start_index + np.random.randint(1, _ub - start_index + 1)
        alpha_genome = chrom_1.genome[start_index:end_index]
        beta_genome = [gen for gen in chrom_2.genome if gen not in alpha_genome]
        resulted_genome = beta_genome[:start_index] + alpha_genome + beta_genome[start_index:]
        return Chromosome(resulted_genome, self.problem)

    def reproduction(self, parents: List[Chromosome], n: int) -> List[Chromosome]:

        pairs_universe: List[(Chromosome, Chromosome)] = [(ch_1, ch_2) for (ch_1, ch_2) in
                                                          itertools.product(parents, parents) if ch_1 != ch_2]
        pair_sample = [pairs_universe[i] for i in
                       np.random.choice(len(pairs_universe), n, p=[1 / len(pairs_universe)] * len(pairs_universe))]
        child_list = []
        for parent_1, parent_2 in pair_sample:
            child_list.append(self.ordered_crossover(parent_1, parent_2))
        return child_list

    def solve(self) -> Chromosome:
        current_best: Chromosome = self.population.get_best_chromosome()
        # is_not_ready = True
        for _ in tqdm.tqdm(range(10000)):
            parents: List[Chromosome] = self.selection(self.population)
            childes: List[Chromosome] = self.reproduction(parents, self.population_size - self.selection_size)
            self.population = self.mutation(childes, parents)
            cand_best: Chromosome = self.population.get_best_chromosome()
            if cand_best.fitness < current_best.fitness:
                current_best = cand_best
                print('Best update: {0}'.format(current_best.fitness))
        return current_best

    def mutation(self, _childes: List[Chromosome], _parents: List[Chromosome]):
        def _mutate(_chromosome: Chromosome) -> Chromosome:
            def _swap(_chromosome, a_ind, b_ind) -> Chromosome:
                _genome = _chromosome.genome
                _genome[a_ind], _genome[b_ind] = _genome[b_ind], _genome[a_ind]
                return Chromosome(_genome, self.problem)

            def _scramble(_chromosome, a_ind, b_ind) -> Chromosome:
                _genome = _chromosome.genome
                _buff = deepcopy(_genome[a_ind:b_ind])
                random.shuffle(_buff)
                _genome[a_ind:b_ind] = _buff
                return Chromosome(_genome, self.problem)

            _ub = len(_chromosome.genome) - 1
            start_index = np.random.randint(0, _ub)
            end_index = start_index + np.random.randint(1, _ub - start_index + 1)
            if random.uniform(0, 1) > 0.5:
                return _swap(_chromosome, start_index, end_index)
            else:
                return _scramble(_chromosome, start_index, end_index)

        unmutated_population = _childes + _parents
        threshold = random.uniform(1 / self.population_size, 1 / self.problem.problem_size)
        resulted_population = []
        for chromosome in unmutated_population:
            if random.uniform(0, 1) > threshold:
                resulted_population.append(chromosome)
            else:
                resulted_population.append(_mutate(chromosome))
        return Population(resulted_population, self.problem)


if __name__ == "__main__":
    tai_problem = Problem("../data/tai20a")
    genetic_solver = GeneticAlgorithmSolver(tai_problem)
    print("Final solution score: {0}".format(genetic_solver.solve().fitness))
