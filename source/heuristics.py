import networkx as nx


def calc_begin_time(feasible_ready_time, prev_begin_time, prev_service_time, travel_time):
    """
    Calculate time at which we can service feasible customer
    :param feasible_ready_time: 
    :param prev_begin_time: 
    :param prev_service_time: 
    :param travel_time: 
    :return: 
    """
    return max([feasible_ready_time, prev_begin_time + prev_service_time + travel_time])


def calc_cost_metric(deltas, travel_time, feasible_ready_time, feasible_due_time, prev_begin_time, prev_service_time):
    """
    Calculate cost function between tail and candidate nodes(customers) for current route

    :param deltas: weight hyperparameters, sum should be equal 1
    :param travel_time: distance between nodes
    :param feasible_ready_time: candidate node ready time
    :param feasible_due_time: candidate node due time
    :param prev_begin_time: tail node service start-time
    :param prev_service_time: tail node service process time
    :return:
    """

    def get_time_diff():
        return calc_begin_time(feasible_ready_time, prev_begin_time, prev_service_time, travel_time) - (
                prev_begin_time + prev_service_time)

    def calc_urgency():
        return feasible_due_time - (prev_begin_time + prev_service_time + travel_time)

    assert sum(deltas) == 1
    assert all([it >= 0 for it in deltas])
    return deltas[0] * travel_time + deltas[1] * get_time_diff() + deltas[2] * calc_urgency()


def _get_route_with_neighbour_heuristic(graph: nx.Graph, deltas, vehicle_capacity, depo_due_time):
    def is_route_valid(_route):
        if (sum([it['demand'] for it in _route]) <= vehicle_capacity) & (
                _route[-1]['begin_t'] <= depo_due_time):
            return True
        else:
            return False

    filtered_graph = graph.copy()
    route = [{'node': 0, 'begin_t': 0, **graph.nodes[0]},
             {'node': 0, 'begin_t': 0, **graph.nodes[0]}]
    while True:
        # TODO implement case with small capacity and big next_stop node demand
        cand_dict = dict()
        for cand_node, cand_data in list(filtered_graph.nodes(data=True))[1:]:
            cand_dict[cand_node] = calc_cost_metric(deltas,
                                                    graph[route[-2]['node']][cand_node]['time'],
                                                    cand_data['ready_t'], cand_data['due_t'],
                                                    route[-2]['begin_t'], route[-2]['service_t'])
        # TODO workaround, logic fix needed
        if len(cand_dict.keys()) == 0:
            break
        next_stop_node = min(set(cand_dict.keys()), key=cand_dict.get)
        next_stop_begin_time = calc_begin_time(graph.nodes[next_stop_node]['ready_t'],
                                               route[-2]['begin_t'],
                                               route[-2]['service_t'],
                                               graph[route[-2]['node']][next_stop_node]['time'])
        depot_arrival_time = calc_begin_time(graph.nodes[0]['ready_t'],
                                             next_stop_begin_time,
                                             graph.nodes[next_stop_node]['service_t'],
                                             graph[0][next_stop_node]['time'])
        feasible_route = route[:-1] + [
            {'node': next_stop_node,
             'begin_t': next_stop_begin_time,
             **graph.nodes[next_stop_node]}] + [
                             {'node': 0,
                              'begin_t': depot_arrival_time,
                              **graph.nodes[0]}]
        if is_route_valid(feasible_route):
            route = feasible_route
            filtered_graph.remove_node(next_stop_node)
            # print(next_stop_node)
        else:
            break

        # assert is_route_valid(route)
    return route, filtered_graph


class HeuristicSolver:
    def __init__(self, route_graph: nx.Graph, vehicle_info, deltas=(0.8, 0.0, 0.2)):
        self.route_graph = route_graph
        self.deltas = deltas
        self.route_list = []
        self.vehicle_amount = vehicle_info['amount']
        self.vehicle_capacity = vehicle_info['capacity']
        self.global_due_time = route_graph.nodes[0]['due_t']

    def get_initial_solution(self):
        route_list = []
        init_graph = self.route_graph.copy()
        while len(init_graph.nodes) > 1:
            # print(len(init_graph.nodes))
            new_route, init_graph = _get_route_with_neighbour_heuristic(init_graph, self.deltas, self.vehicle_capacity,
                                                                        self.global_due_time)
            route_list.append(new_route)
        return route_list
