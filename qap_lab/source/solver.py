import itertools
import os
import random
from copy import deepcopy
from typing import List

import numpy as np
import tqdm

from qap_lab.source.data_utils import Problem
from tqdm import trange

import json
import os
from multiprocessing import Pool


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
        return sum([self.dist_matrix[i][j] * self.flow_matrix[self.genome[i]][self.genome[j]]
                    for i in range(n)
                    for j in range(n)])

    def swap_mutation(self):
        pass

    def scrumble_mutation(self):
        pass

    def persist(self, dir, problem_name):
        with open(os.path.join(dir, problem_name), 'w') as f:
            f.write(" ".join([str(it + 1) for it in self.genome]))


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
        _sum = sum([1 / chrom.fitness for chrom in self.chromosome_list])
        probs = [(1 / chrom.fitness) / _sum for chrom in self.chromosome_list]
        return probs

    def select_n_chromosomes(self, n: int) -> List[Chromosome]:
        selected_index = np.random.choice(len(self.chromosome_list), n, p=self.rolling_wheel_prob)
        return [self.chromosome_list[i] for i in selected_index]


class GeneticAlgorithmSolver:
    def __init__(self, _problem: Problem):
        self.problem = _problem
        self.population_size = 100
        self.selection_size = int(self.population_size * 0.3)
        self.population = self.generate_initial_population()

    def generate_initial_population(self) -> Population:
        _chromo_list = set()
        genome = list(range(self.problem.problem_size))
        while len(_chromo_list) != self.population_size:
            # for _ in range(self.population_size):
            _chromo_list.add(Chromosome(deepcopy(genome), self.problem))
            random.shuffle(genome)
        return Population(list(_chromo_list), self.problem)

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

        pairs_universe: List[(Chromosome, Chromosome)] = [(ch_1, ch_2) for ch_1 in parents for ch_2 in parents
                                                          if ch_1 != ch_2]
        # pair_sample = [pairs_universe[i] for i in
        #               ]
        child_list = set()
        # for parent_1, parent_2 in pair_sample:
        while len(child_list) != n:
            parent_1, parent_2 = pairs_universe[
                np.random.choice(len(pairs_universe), n, p=[1 / len(pairs_universe)] * len(pairs_universe))[0]]
            child_list.add(self.ordered_crossover(parent_1, parent_2))
        return list(child_list)

    def solve(self) -> Chromosome:
        current_best: Chromosome = self.population.get_best_chromosome()
        # t = trange(100, desc='Solving')
        for _ in range(25000):
            # avg_fitness = np.average([it.fitness for it in self.population.chromosome_list])
            # t.set_description('Solving (avg fitness=%g)' % avg_fitness)
            parents = self.selection(self.population)
            childes = self.reproduction(parents, self.population_size - self.selection_size)
            self.population = self.mutation(childes, parents)
            cand_best: Chromosome = self.population.get_best_chromosome()
            if cand_best.fitness < current_best.fitness:
                current_best = cand_best
                # print('Best update: {0}'.format(current_best.fitness))
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


def main(path):
    problem_name = path.split("/")[-1].split(".")[0]
    tai_problem = Problem(path)
    genetic_solver = GeneticAlgorithmSolver(tai_problem)
    solution = genetic_solver.solve()
    result_dict = json.load(open("../data/best_results.json"))
    if result_dict[problem_name] > solution.fitness or result_dict[problem_name] == 0:
        solution.persist(os.path.dirname(path), problem_name + ".sol")
        print("Improvement in {0} problem!".format(problem_name))
        result_dict[problem_name] = int(solution.fitness)
    print("Problem {0} finished!".format(problem_name))
    json.dump(result_dict, open("../data/best_results.json", 'w'), indent=2)


if __name__ == "__main__":
    root_dir = '../data'
    problem_path_list = [os.path.join(root_dir, it) for it in os.listdir(root_dir) if
                         not (it.endswith(".json") or it.endswith(".sol"))] * 10
    with Pool(processes=4) as pool:
        for res in pool.imap_unordered(main, problem_path_list):
            pass
