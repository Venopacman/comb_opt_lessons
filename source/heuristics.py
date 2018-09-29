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


def _get_route_with_neighbour_heuristic(graph: nx.Graph, deltas, vehicle_capacity):
    """
    Generate init solution according to Marius M. Solomon
    A Time-Oriented, Nearest-Neighbor Heuristic algorithm
    :param graph:
    :param deltas:
    :param vehicle_capacity:
    :return:
    """

    def is_route_valid_for_nn(_route):
        if (sum([it['demand'] for it in _route]) <= vehicle_capacity) & (
                all([_route[key]['begin_t'] <= _route[key]['due_t'] for key in [-1, -2]])):
            return True
        else:
            return False

    filtered_graph = graph.copy()
    result_graph = graph.copy()
    route = [{'node': 0, 'begin_t': 0, **graph.nodes[0]},
             {'node': 0, 'begin_t': 0, **graph.nodes[0]}]
    condition = True
    while condition:
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
        if is_route_valid_for_nn(feasible_route):
            route = feasible_route
            filtered_graph.remove_node(next_stop_node)
            # print(next_stop_node)
        else:
            filtered_graph.remove_node(next_stop_node)
            continue

        condition = len(cand_dict.keys()) != 0
        # assert is_route_valid(route)
    result_graph.remove_nodes_from([it['node'] for it in route[1:-1]])
    return route, result_graph


class HeuristicSolver:
    def __init__(self, route_graph: nx.Graph, vehicle_info, deltas=(0.8, 0.0, 0.2)):
        self.route_graph = route_graph
        self.deltas = deltas
        self.route_list = []
        self.vehicle_amount = vehicle_info['amount']
        self.vehicle_capacity = vehicle_info['capacity']
        self.global_due_time = route_graph.nodes[0]['due_t']

    def calculate_solution_target_metric(self, solution, with_vehicle_amount=True):
        result = sum([sum([self.route_graph[route[ind - 1]['node']][route[ind]['node']]['time']
                           for ind in range(1, len(route))]) for route in solution])
        return result + (len(solution) / self.vehicle_amount) if with_vehicle_amount else result

    def get_initial_solution(self):
        route_list = []
        init_graph = self.route_graph.copy()
        while len(init_graph.nodes) > 1:
            new_route, init_graph = _get_route_with_neighbour_heuristic(init_graph, self.deltas, self.vehicle_capacity)
            route_list.append(new_route)
        return route_list

    def get_route_with_reversed_segment(self, _route, i, j):
        # print("[DEBUG] node_tuple: ", _route[i - 1], _route[i], _route[j - 1], _route[j])
        resulted_route = _route[:i - 1] + [it for it in reversed(_route[i:j - 1])] + _route[j - 1:]
        for ind in range(len(resulted_route)):
            if ind < i:
                continue
            else:
                # TODO check push-forward factor to make it faster (avoid redundant calculations)
                resulted_route[ind]['begin_t'] = calc_begin_time(resulted_route[ind]['ready_t'],
                                                                 resulted_route[ind - 1]['begin_t'],
                                                                 resulted_route[ind - 1]['service_t'],
                                                                 self.route_graph[resulted_route[ind]['node']][
                                                                     resulted_route[ind - 1]['node']]['time'])
        return resulted_route

    def local_search(self, init_solution):
        """

        :param init_solution:
        :return:
        """

        def is_route_valid(_route):
            return sum([it['demand'] for it in _route]) <= self.vehicle_capacity & all(
                [_route[ind]['begin_t'] <= _route[ind]['due_t'] for ind in range(len(_route))])

        def _reverse_segment_if_feasible_and_better(_route, i, j):
            """
            Route [...(i-1)-i...(j-1)-j...]
            If reversing route [j:(i-1)] *(equally [i:(j-1)])* would make the route shorter
                -- then change route in appropriate way.
            :param _route:
            :param i:
            :param j:
            :return:
            """

            candidate_route = None
            a, b, c, d = _route[i - 1]['node'], _route[i]['node'], _route[j - 1]['node'], _route[j]['node']
            d0 = self.route_graph[a][b]['time'] + self.route_graph[c][d]['time']
            d1 = self.route_graph[a][c]['time'] + self.route_graph[b][d]['time']
            if d0 > d1:
                candidate_route = self.get_route_with_reversed_segment(_route, i, j)
                is_change_feasible = is_route_valid(candidate_route)
            else:
                is_change_feasible = False

            if is_change_feasible:
                return candidate_route

            else:
                return _route

        def two_opt(_route):
            """
            Iterative improvement based on 2 edges exchange
            :param _route:
            :return:
            """

            def all_segments(n):
                """
                Generate all pair nodes combinations
                :param n:
                :return:
                """
                return [(i, j % n)
                        for i in range(1, n)
                        for j in range(i + 2, n)]

            for a, b in all_segments(len(_route)):
                print(a, b)
                print(_route[a])
                print(_route[b])
                reversed_route = _reverse_segment_if_feasible_and_better(_route, a, b)
                if reversed_route != _route:
                    print("[DEBUG] route {0} \n updated to: \n {1}".format(_route, reversed_route))
                    return two_opt(reversed_route)
            return _route

        upgraded_route = []
        for route in init_solution:
            upgraded_route.append(two_opt(route))
        return upgraded_route
