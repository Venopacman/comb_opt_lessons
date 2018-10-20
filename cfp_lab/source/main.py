import os
from multiprocessing import Pool

import heuristics as he
from utils import read_cfp_problem, Solution


def save_solution(solution: Solution, sol_path):
    machine_dict = dict()
    part_dict = dict()
    for ind, cell in enumerate(solution.cell_list):
        for part in cell.parts_set:
            part_dict[part] = ind
        for machine in cell.machines_set:
            machine_dict[machine] = ind
    with open(sol_path, "w", encoding="utf-8") as f:
        f.write(
            " ".join(sorted(["m{0}_{1}".format(it + 1, machine_dict[it]) for it in machine_dict],
                            key=lambda x: int(x.split("_")[0].replace("m", "")))) + "\n")
        f.write(" ".join(sorted(["p{0}_{1}".format(it + 1, part_dict[it]) for it in part_dict],
                                key=lambda x: int(x.split("_")[0].replace("p", "")))) + "\n")


def main(path):
    solver = he.VariableNeighborhoodSolver(read_cfp_problem(path))
    init_solution = solver._get_init_solution()
    print(init_solution.efficacy)
    global_solution = solver.general_vns_process(1000)
    # print(len(neighborhood))
    save_solution(global_solution, "/".join(path.split("/")[:-1] + [path.split("/")[-1].split(".")[0] + ".sol"]))


if __name__ == "__main__":
    root_dir = '../data'
    problem_path_list = [os.path.join(root_dir, it) for it in os.listdir(root_dir) if it.endswith('.txt')]

    # for problem in problem_path_list:
    #     main(problem)
    with Pool(processes=4) as pool:
        for res in pool.imap_unordered(main, problem_path_list):
            pass
