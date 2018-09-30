import graph_utils as gu
import heuristics as he
import tqdm
if __name__ == "__main__":
    reader = gu.GraphReader("../data/instances/R202.txt")
    G = reader.create_route_graph()
    solver = he.HeuristicSolver(G, reader.vehicle_info, deltas=[0.95, 0, 0.05])
    init_solution = solver.get_initial_solution()
    local_solution = solver.local_search(init_solution)
    print("Init solution score: ", solver.calculate_solution_target_metric(local_solution, False))
    # gu.plot_routes(G, local_solution)
    history = [local_solution.copy()]

    for i in tqdm.tqdm(range(5)):
        perturbed_solution = solver.perturbation(local_solution.copy())
        local_solution = solver.local_search(perturbed_solution.copy())
        history.append((local_solution.copy()))
    global_solution = min(history, key=lambda x: solver.calculate_solution_target_metric(x, False))
    print("Global solution score: ", solver.calculate_solution_target_metric(global_solution, False))
    gu.plot_routes(G, global_solution)
