from typing import List, Set

import numpy as np


class Cell:
    """
    Cell wrapper for set of parts and set of machines
    """

    def __init__(self, machines_set: Set[int], parts_set: Set[int], machine_part_matrix: np.array):
        self.machine_part_matrix = machine_part_matrix
        self.parts_set = parts_set
        self.machines_set = machines_set
        self.number_of_ones = self._get_number_of_ones()
        self.number_of_zeroes = self._get_number_of_zeroes()

    def _get_number_of_ones(self):
        return self.machine_part_matrix[list(self.machines_set), :][:, list(self.parts_set)].sum()

    def _get_number_of_zeroes(self):
        return (1 - self.machine_part_matrix[list(self.machines_set), :][:, list(self.parts_set)]).sum()


class Solution:
    """
    List of cells wrapper with implicit feasibility check
    """

    def __init__(self, cell_list: List[Cell], machine_part_matrix: np.array):
        self.machine_part_matrix = machine_part_matrix
        self.cell_list = cell_list
        self.is_feasible = self.check_feasibility()
        self.efficacy = self.get_efficacy()

    def get_efficacy(self):
        return self.get_number_of_ones_in_cells() / (
                self.machine_part_matrix.sum() + self.get_number_of_zeroes_in_cells())

    def get_number_of_ones_in_cells(self):
        return sum([cell_.number_of_ones for cell_ in self.cell_list])

    def get_number_of_zeroes_in_cells(self):
        return sum([cell_.number_of_zeroes for cell_ in self.cell_list])

    def check_feasibility(self):
        flag = True
        # Each cluster must contain at least 1 machine and 1 part.
        flag &= all([len(_cell.parts_set) > 0 and len(_cell.machines_set) > 0 for _cell in self.cell_list])
        # Each machine must be assigned to exactly 1 cluster.
        flag &= all([len(self.cell_list[i].machines_set & self.cell_list[j].machines_set) == 0
                     for i in range(len(self.cell_list))
                     for j in range(i, len(self.cell_list))])
        # Each part must be assigned to exactly 1 cluster.
        flag &= all([len(self.cell_list[i].parts_set & self.cell_list[j].parts_set) == 0
                     for i in range(len(self.cell_list))
                     for j in range(i, len(self.cell_list))])
        return flag


def read_cfp_problem(file_path: str):
    """

    :param file_path:
    :return:
    """

    def _generate_problem_matrix(_machine_to_part_dict, _part_to_machine_dict):
        n = len(_machine_to_part_dict)
        m = len(_part_to_machine_dict)
        result_matrix = np.zeros((n, m), dtype=np.int_)
        for machine_key in _machine_to_part_dict:
            for part_key in _machine_to_part_dict[machine_key]:
                result_matrix[int(machine_key) - 1, int(part_key) - 1] = 1
        return result_matrix

    machine_to_part_dict = dict()
    part_to_machine_dict = dict()
    with open(file_path) as f:
        f_list = f.read().split('\n')
        for row in f_list:
            row_list = row.split(" ")
            machine_id = int(row_list[0])
            machine_to_part_dict[machine_id] = [int(it) for it in row_list[1:]]
            for part in row_list[1:]:
                if part not in part_to_machine_dict:
                    part_to_machine_dict[part] = [machine_id]
                else:
                    part_to_machine_dict[part].append(machine_id)
    return _generate_problem_matrix(machine_to_part_dict, part_to_machine_dict)
