from heuristics import VariableNeighborhoodSolver
from utils import Cell, Solution, read_cfp_problem

if __name__ == "__main__":
    solver = VariableNeighborhoodSolver(read_cfp_problem('../data/30x90.txt'))
    cell_ = Cell({0, 1}, {0, 89}, solver.machine_part_matrix)
    solution = Solution([cell_], solver.machine_part_matrix)
    print(solution.efficacy)
    # print(solver.machine_part_matrix.shape)
