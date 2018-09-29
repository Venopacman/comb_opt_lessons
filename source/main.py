import graph_utils as gu
import heuristics as he

if __name__ == "__main__":
    reader = gu.GraphReader("../data/instances/C108.txt")
    G = reader.create_route_graph()
    solver = he.HeuristicSolver(G, reader.vehicle_info, deltas=[0.95, 0, 0.05])
    init_solution = solver.get_initial_solution()
    local_solution = solver.local_search(init_solution)
    print(solver.calculate_solution_target_metric(local_solution, False), solver.calculate_solution_target_metric(init_solution, False))
    gu.plot_routes(G, local_solution)
