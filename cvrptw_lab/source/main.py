import graph_utils as gu
import heuristics as he
from multiprocessing import Pool
import os
import time


def save_solution(solution, sol_path):
    with open(sol_path, "w", encoding="utf-8") as f:
        for route in solution:
            f.write(" ".join([" ".join([str(it['node']), str(it['begin_t'])]) for it in route]) + "\n")


def main(graph_path):
    reader = gu.GraphReader(graph_path)
    graph = reader.create_route_graph()
    solver = he.HeuristicSolver(graph, reader.vehicle_info, deltas=[0.95, 0, 0.05])
    init_solution = solver.get_initial_solution()
    local_solution = solver.local_search(init_solution)
    history = [local_solution.copy()]

    timeout = time.time() + 60 * 120  # 120 minutes from now
    for _ in range(1000):
        perturbed_solution = solver.perturbation(local_solution.copy())
        local_solution = solver.local_search(perturbed_solution.copy())
        history.append((local_solution.copy()))
        if time.time() > timeout:
            break
    # TODO fix strange bug with incorrect depo begin_time
    global_solution = min(history,
                          key=lambda x: solver.calculate_solution_target_metric(
                              [solver.recalc_b_time_from_index(it, -1) for it in x],
                              False))
    global_solution = [solver.recalc_b_time_from_index(route, -1) for route in global_solution]
    metric = solver.calculate_solution_target_metric(global_solution, False)
    print("Global solution score for {0} graph: ".format(graph_path), metric)
    gu.plot_routes(graph, global_solution, graph_path.replace(".txt", ".png").replace(".TXT", ".png"), metric)
    save_solution(global_solution,
                  graph_path.replace(".txt", ".sol").replace(".TXT", ".sol"))


if __name__ == "__main__":
    root_dir = "../data/instances"
    bonus_dir = os.path.join(root_dir, "bonus")
    problem_path_list = [os.path.join(root_dir, it) for it in os.listdir(root_dir) if
                         it.endswith(".txt") or it.endswith(".TXT")] + [os.path.join(bonus_dir, it) for it in
                                                                        os.listdir(bonus_dir)]
    with Pool(processes=4) as pool:
        for res in pool.imap_unordered(main, problem_path_list):
            pass
    # for path in problem_path_list[:1]:
    #     main(path)
