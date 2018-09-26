import graph_utils as gu
import heuristics as he
if __name__ == "__main__":
    reader = gu.GraphReader("../data/instances/C108.txt")
    G = reader.create_route_graph()
    solver = he.HeuristicSolver(G, reader.vehicle_info)
    solution_list = solver.get_initial_solution()
    print(len(solution_list))
    # for route in solution_list:
    #     print(route)
    # gu.plot_graph_nodes(G)
    gu.plot_routes(G, solution_list)