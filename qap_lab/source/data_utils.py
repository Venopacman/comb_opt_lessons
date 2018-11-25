from typing import List
import re

import numpy as np


def get_problem_size(problem_path: str) -> int:
    return int(open(problem_path).read().split("\n")[0])


class Problem:
    def __init__(self, problem_path: str):
        self.problem_size = get_problem_size(problem_path)
        self.flow_matrix, self.distance_matrix = self.read_problem(problem_path)

    def read_problem(self, problem_path: str) -> (np.array, np.array):
        def _form_matrix(data_list: List[str]):
            result_matrix = np.zeros([self.problem_size] * 2, dtype=np.int_)
            for i, row in enumerate(data_list):
                for j, elem in enumerate(re.split("\s+", row.strip())):
                    result_matrix[i][j] = int(elem)
            return result_matrix
        with open(problem_path) as f:
            data = f.read().split("\n")
            dist_list = data[1:self.problem_size + 1]
            flow_list = data[self.problem_size + 2:]
        return _form_matrix(flow_list), _form_matrix(dist_list)


if __name__ == "__main__":
    problem = Problem("../data/tai20a")
    print(problem.distance_matrix)
    print("")
    print(problem.flow_matrix)
