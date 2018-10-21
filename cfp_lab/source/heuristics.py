import itertools
import random
from copy import deepcopy
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

    def shake_by_split(self, solution: ut.Solution):
        """

        :param solution:
        :return:
        """

        def split_cell(_cell: ut.Cell, machine_subset: set, parts_subset: set):
            cell_1 = ut.Cell(machine_subset, parts_subset, self.machine_part_matrix)
            cell_2 = ut.Cell(_cell.machines_set - machine_subset,
                             _cell.parts_set - parts_subset,
                             self.machine_part_matrix)
            return [cell_1, cell_2]

        def _get_combinations_sample(_coll):

            def _get_sample(_coll):
                return random.sample(_coll, min(10, len(_coll)))

            def _get_combinations(_coll):
                def _get_comb_size(_coll):
                    return random.randrange(1, min(len(_coll), 5))

                return [it for it in itertools.combinations(_coll, _get_comb_size(_coll))]

            return _get_sample(_get_combinations(_coll))

        split_neighborhood = []
        for ind, cell in enumerate(solution.cell_list):
            if len(cell.parts_set) == 1 or len(cell.machines_set) == 1:
                continue
            _machines_comb_sample = _get_combinations_sample(cell.machines_set)
            _parts_comb_sample = _get_combinations_sample(cell.parts_set)
            for _machines_subset, _parts_subset in itertools.product(_machines_comb_sample, _parts_comb_sample):
                buff_cell_list = solution.cell_list[:ind] + solution.cell_list[ind + 1:]
                buff_cell_list.extend(split_cell(cell, set(_machines_subset), set(_parts_subset)))
                split_neighborhood.append(ut.Solution(buff_cell_list, self.machine_part_matrix))
        return split_neighborhood

    def shake_by_merge(self, solution: ut.Solution):
        """

        :param solution:
        :return:
        """

        def merge_cells(_cell_list: List[ut.Cell]):
            machine_set = set()
            parts_set = set()
            for _cell in _cell_list:
                machine_set |= _cell.machines_set
                parts_set |= _cell.parts_set
            return ut.Cell(machine_set, parts_set, self.machine_part_matrix)

        def _get_pairs_sample(_cell_list):
            def _get_sample(_coll):
                return random.sample(_coll, min(10, len(_coll)))

            def _get_cell_index_pairs(_cell_list):
                return [(i, j) for i in range(len(_cell_list)) for j in range(i, len(_cell_list)) if i != j]

            return _get_sample(_get_cell_index_pairs(_cell_list))

        if len(solution.cell_list) == 1:
            return [solution]
        merge_neighborhood = []
        for cell_id_1, cell_id_2 in _get_pairs_sample(solution.cell_list):
            new_cell_list = deepcopy(solution.cell_list)
            if cell_id_1 < cell_id_2:
                cell_2 = new_cell_list.pop(cell_id_2)
                cell_1 = new_cell_list.pop(cell_id_1)
            else:
                cell_1 = new_cell_list.pop(cell_id_1)
                cell_2 = new_cell_list.pop(cell_id_2)

            new_cell_list.append(merge_cells([cell_1, cell_2]))
            merge_neighborhood.append(ut.Solution(new_cell_list, self.machine_part_matrix))
        return merge_neighborhood

    def relocate_machine(self, init_solution: ut.Solution):
        """

        :param init_solution:
        :return:
        """

        def _get_index_pairs(_coll):
            return [(i, j) for i in range(len(_coll)) for j in range(i, len(_coll)) if i != j]

        def _relocate(_cell_list: List[ut.Cell], from_cell_id: int, to_cell_id: int):
            if from_cell_id < to_cell_id:
                cell_to = _cell_list.pop(to_cell_id)
                cell_from = _cell_list.pop(from_cell_id)
            else:
                cell_from = _cell_list.pop(from_cell_id)
                cell_to = _cell_list.pop(to_cell_id)
            _result_collection = []
            if len(cell_from.machines_set) == 1:
                return _result_collection

            for machine_traveller in cell_from.machines_set:
                _list = deepcopy(_cell_list)
                _list.append(ut.Cell(cell_from.machines_set - {machine_traveller},
                                     cell_from.parts_set, self.machine_part_matrix))
                _list.append(ut.Cell(cell_to.machines_set | {machine_traveller},
                                     cell_to.parts_set, self.machine_part_matrix))
                _result_collection.append(ut.Solution(_list, self.machine_part_matrix))
            return _result_collection

        neighborhood: List[ut.Solution] = [init_solution]
        for cell_id_1, cell_id_2 in _get_index_pairs(init_solution.cell_list):
            neighborhood.extend(_relocate(deepcopy(init_solution.cell_list), cell_id_1, cell_id_2))
            neighborhood.extend(_relocate(deepcopy(init_solution.cell_list), cell_id_2, cell_id_1))
        return sorted(neighborhood, key=lambda x: x.efficacy, reverse=True)[0]

    def relocate_part(self, init_solution: ut.Solution):
        """

        :param init_solution:
        :return:
        """

        def _get_index_pairs(_coll):
            return [(i, j) for i in range(len(_coll)) for j in range(i, len(_coll)) if i != j]

        def _relocate(_cell_list: List[ut.Cell], from_cell_id: int, to_cell_id: int):
            if from_cell_id < to_cell_id:
                cell_to = _cell_list.pop(to_cell_id)
                cell_from = _cell_list.pop(from_cell_id)
            else:
                cell_from = _cell_list.pop(from_cell_id)
                cell_to = _cell_list.pop(to_cell_id)
            _result_collection = []
            if len(cell_from.parts_set) == 1:
                return _result_collection
            for part_traveller in cell_from.parts_set:
                _list = deepcopy(_cell_list)
                _list.append(ut.Cell(cell_from.machines_set, cell_from.parts_set - {part_traveller},
                                     self.machine_part_matrix))
                _list.append(ut.Cell(cell_to.machines_set, cell_to.parts_set | {part_traveller},
                                     self.machine_part_matrix))
                _result_collection.append(ut.Solution(_list, self.machine_part_matrix))
            return _result_collection

        neighborhood: List[ut.Solution] = [init_solution]
        for cell_id_1, cell_id_2 in _get_index_pairs(init_solution.cell_list):
            neighborhood.extend(_relocate(deepcopy(init_solution.cell_list), cell_id_1, cell_id_2))
            neighborhood.extend(_relocate(deepcopy(init_solution.cell_list), cell_id_2, cell_id_1))
        return sorted(neighborhood, key=lambda x: x.efficacy, reverse=True)[0]

    def variable_neighborhood_descent(self, init_solution: ut.Solution):
        vnd_family = [self.relocate_machine, self.relocate_part]  # , self.machine_swap, self.part_swap]
        vnd_id = 0
        current_best_solution = deepcopy(init_solution)
        while vnd_id < len(vnd_family):
            cand_solution = vnd_family[vnd_id](current_best_solution)
            if cand_solution.efficacy > current_best_solution.efficacy:
                current_best_solution = cand_solution
                vnd_id = 0
            else:
                vnd_id += 1
        return current_best_solution

    def general_vns_process(self, iter_criteria):
        curr_solution = self._get_init_solution()
        shake_family = [self.shake_by_split, self.shake_by_merge]
        shake_id = 0
        while shake_id < iter_criteria:
            # print(shake_family[shake_id])
            cand_solution = random.sample(shake_family[shake_id % len(shake_family)](curr_solution), 1)[0]
            local_opt_solution = self.variable_neighborhood_descent(cand_solution)
            if local_opt_solution.efficacy > curr_solution.efficacy:
                # print(local_opt_solution.efficacy)
                curr_solution = local_opt_solution
                shake_id = 0
            else:
                shake_id += 1

        return curr_solution
