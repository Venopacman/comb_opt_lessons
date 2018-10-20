import itertools
import random
from typing import List

import utils as ut


class VariableNeighborhoodSolver:
    def __init__(self, machine_part_matrix):
        self.machine_part_matrix = machine_part_matrix

    def _get_init_solution(self):
        n, m = self.machine_part_matrix.shape
        return ut.Solution(
            [ut.Cell(set([it for it in range(n)]),
                     set([ik for ik in range(m)]),
                     self.machine_part_matrix)],
            self.machine_part_matrix)

    def split_cell(self, _cell: ut.Cell, machine_subset: set, parts_subset: set):
        cell_1 = ut.Cell(machine_subset, parts_subset, self.machine_part_matrix)
        cell_2 = ut.Cell(_cell.machines_set - machine_subset,
                         _cell.parts_set - parts_subset,
                         self.machine_part_matrix)
        return [cell_1, cell_2]

    def merge_cells(self, _cell_list: List[ut.Cell]):
        machine_set = set()
        machine_set.union([_cell.machines_set for _cell in _cell_list])
        parts_set = set()
        parts_set.union([_cell.parts_set for _cell in _cell_list])
        resulted_cell = ut.Cell(machine_set, parts_set, self.machine_part_matrix)
        return resulted_cell

    def shake_by_split(self, solution: ut.Solution):
        split_neighborhood = []
        for ind, cell in enumerate(solution.cell_list):
            if len(cell.parts_set) == 1 or len(cell.machines_set) == 1:
                continue
            m_comb_len = random.randint(1, min(len(cell.machines_set) - 1, 4))
            p_comb_len = random.randint(1, min(len(cell.parts_set) - 1, 4))
            machine_set_combinations = [it for it in itertools.combinations(cell.machines_set, m_comb_len)]
            part_set_combinations = [it for it in itertools.combinations(cell.parts_set, p_comb_len)]
            # print(m_comb_len, p_comb_len)
            _machines_subset = machine_set_combinations[random.randint(0, len(machine_set_combinations)-1)]
            _parts_subset = part_set_combinations[random.randint(0, len(part_set_combinations)-1)]
            # for _machines_subset, _parts_subset in itertools.product(machine_set_combinations, part_set_combinations):
            buff_cell_list = solution.cell_list[:ind] + solution.cell_list[ind + 1:]
            buff_cell_list.extend(self.split_cell(cell, set(_machines_subset), set(_parts_subset)))
            split_neighborhood.append(ut.Solution(buff_cell_list, self.machine_part_matrix))
        return split_neighborhood

    def get_shaking_neighborhood(self, solution: ut.Solution):
        return sorted(self.shake_by_split(solution), key=lambda x: x.efficacy, reverse=True)

    def general_vns_process(self, iter_criteria):
        curr_solution = self._get_init_solution()
        for _ in range(iter_criteria):
            cand_solution = self.get_shaking_neighborhood(curr_solution)[0]
            if cand_solution.efficacy >= curr_solution.efficacy:
                curr_solution = cand_solution
        return curr_solution
