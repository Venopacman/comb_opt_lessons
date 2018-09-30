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
    assert travel_time > 0
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


class HeuristicSolver:
    def __init__(self, route_graph: nx.Graph, vehicle_info, deltas=(0.8, 0.0, 0.2)):
        """

        :param route_graph:
        :param vehicle_info:
        :param deltas:
        """
        self.route_graph = route_graph
        self.deltas = deltas
        self.route_list = []
        self.vehicle_amount = vehicle_info['amount']
        self.vehicle_capacity = vehicle_info['capacity']
        self.global_due_time = route_graph.nodes[0]['due_t']

    def is_route_valid(self, _route):
        if sum([it['demand'] for it in _route]) <= self.vehicle_capacity:
            if all([_route[ind]['begin_t'] <= _route[ind]['due_t'] for ind in range(len(_route))]):
                return True
            else:
                # print("Не убираемся по времени")
                return False
        else:
            # print("Не убираемся по capacity")
            return False

    def calculate_solution_target_metric(self, solution, with_vehicle_amount=True):
        result = sum([sum([self.route_graph[route[ind - 1]['node']][route[ind]['node']]['time']
                           for ind in range(1, len(route))]) for route in solution])
        return result + (len(solution) / self.vehicle_amount) if with_vehicle_amount else result

    def get_initial_solution(self):
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
                if sum([it['demand'] for it in _route]) <= vehicle_capacity:
                    if all([_route[key]['begin_t'] <= _route[key]['due_t'] for key in [-1, -2]]):
                        return True
                    else:
                        return False
                else:
                    return False

            filtered_graph = graph.copy()
            result_graph = graph.copy()
            route = [{'node': 0, 'begin_t': 0, **graph.nodes[0].copy()},
                     {'node': 0, 'begin_t': 0, **graph.nodes[0].copy()}]
            condition = True
            while condition:
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
                     **graph.nodes[next_stop_node]},
                    {'node': 0,
                     'begin_t': depot_arrival_time,
                     **graph.nodes[0]}]
                feasible_route = self.recalc_b_time_from_index(feasible_route, -1)

                if is_route_valid_for_nn(feasible_route):
                    route = feasible_route
                    filtered_graph.remove_node(next_stop_node)
                else:
                    filtered_graph.remove_node(next_stop_node)  # TODO it may break all ?
                    continue

                condition = len(cand_dict.keys()) != 0
            result_graph.remove_nodes_from([it['node'] for it in route[1:-1]])
            return route, result_graph

        route_list = []
        init_graph = self.route_graph.copy()
        while len(init_graph.nodes) > 1:
            new_route, init_graph = _get_route_with_neighbour_heuristic(init_graph, self.deltas, self.vehicle_capacity)
            route_list.append(new_route)
        return route_list

    def recalc_b_time_from_index(self, _route, _ind):
        for _i in range(1, len(_route)):
            if _i < _ind:
                continue
            else:
                # TODO check push-forward factor to make it faster (avoid redundant calculations)
                _route[_i]['begin_t'] = calc_begin_time(_route[_i]['ready_t'],
                                                        _route[_i - 1]['begin_t'],
                                                        _route[_i - 1]['service_t'],
                                                        self.route_graph[_route[_i]['node']][
                                                            _route[_i - 1]['node']]['time'])
        return _route

    def do_reverse_sub_route(self, _route, i, j):
        resulted_route = self.recalc_b_time_from_index(
            _route[:i] + [it for it in reversed(_route[i:j])] + _route[j:], -1)
        return resulted_route

    def local_search(self, init_solution):

        """
        Perform local search
        by applying 2-opt heuristic for every route independently
        :param init_solution:
        :return: locally best solution
        """

        def two_opt(_route):
            """
            Iterative improvement based on 2 edges exchange
            :param _route:
            :return:
            """

            def _reverse_segment_if_feasible_and_better(_route, _i, _j):
                """
                Route [...(i-1)-i...(_j-1)-_j...]
                If reversing route [_j:(i-1)] *(equally [i:(_j-1)])* would make the route shorter
                    -- then change route in appropriate way.
                :param _route:
                :param _i:
                :param _j:
                :return:
                """

                candidate_route = None
                a, b, c, d = _route[_i - 1]['node'], _route[_i]['node'], _route[_j - 1]['node'], _route[_j]['node']
                d0 = self.route_graph[a][b]['time'] + self.route_graph[c][d]['time']
                d1 = self.route_graph[a][c]['time'] + self.route_graph[b][d]['time']
                if d0 > d1:
                    # print("d0>d1")
                    # TODO it may be better, check feasibility before do_reverse
                    candidate_route = self.do_reverse_sub_route(_route, _i, _j)
                    is_change_feasible = self.is_route_valid(candidate_route)
                else:
                    is_change_feasible = False

                return candidate_route if is_change_feasible else _route

            def all_segments(n):
                """
                Generate all pair nodes combinations
                :param n:
                :return:
                """
                return [(_i, _j)
                        for _i in range(1, n)
                        for _j in range(_i + 2, n)]

            for i, j in all_segments(len(_route)):
                reversed_route = _reverse_segment_if_feasible_and_better(_route, i, j)
                if reversed_route != _route:
                    # print("[DEBUG] route {0} \n updated to: \n {1}".format(_route, reversed_route))
                    return two_opt(reversed_route.copy())
            return _route

        minimum_local_solution = []
        for route in init_solution:
            minimum_local_solution.append(two_opt(route))
        return minimum_local_solution

    def perturbation(self, local_solution):
        """

        :param local_solution:
        :return:
        """

        def do_relocate(_route_from: list, _route_to: list, i, j):
            """
            Relocate operator
            move node i from _route_a to _route_to between j and (j-1) nodes and return pair of two new routes
            :param _route_from:
            :param _route_to:
            :param i: i-th node in _route_a
            :param j: j-th node in _route_to
            :return: estimation of target metric delta and perturbation feasibility flag
            """
            _route_to = _route_to[:j] + [_route_from.pop(i)] + _route_to[j:]

            if len(_route_from) > 2:
                _route_from = self.recalc_b_time_from_index(_route_from, i - 1)
            else:
                _route_from = []
            _route_to = self.recalc_b_time_from_index(_route_to, j - 1)

            return _route_from, _route_to

        def estimate_relocate_op(_route_from, _route_to, i, j):
            """
            Relocate node i from _route_from to _route_to between j and (j-1) nodes and estimate target metric delta
            :param _route_from:
            :param _route_to:
            :param i: i-th node in _route_from
            :param j: j-th node in _route_to
            :return: estimation of target metric delta and perturbation feasibility flag
            """
            new_route_from, new_route_to = do_relocate(_route_from.copy(), _route_to.copy(), i, j)
            old_sub_solution_cost = self.calculate_solution_target_metric([_route_from, _route_to], False)
            new_sub_solution_cost = self.calculate_solution_target_metric([new_route_from, new_route_to], False)
            _delta = old_sub_solution_cost - new_sub_solution_cost
            feasibility_flag = self.is_route_valid(new_route_from) & self.is_route_valid(new_route_to) if len(
                new_route_from) > 2 else self.is_route_valid(new_route_to)
            return _delta, feasibility_flag, (new_route_from, new_route_to)

        def do_exchange(_route_a: list, _route_b: list, a_node_ind, b_node_ind):
            """

            :param _route_a:
            :param _route_b:
            :param a_node_ind:
            :param b_node_ind:
            :return:
            """
            _route_a[a_node_ind], _route_b[b_node_ind] = _route_b[b_node_ind], _route_a[a_node_ind]
            _route_b = self.recalc_b_time_from_index(_route_b, b_node_ind - 1)
            _route_a = self.recalc_b_time_from_index(_route_a, a_node_ind - 1)
            return _route_a, _route_b

        def estimate_exchange_op(_route_a, _route_b, a_node_ind, b_node_ind):
            """

            :param _route_a:
            :param _route_b:
            :param a_node_ind:
            :param b_node_ind:
            :return:
            """
            new_route_a, new_route_b = do_exchange(_route_a.copy(), _route_b.copy(), a_node_ind, b_node_ind)
            old_sub_solution_cost = self.calculate_solution_target_metric([_route_a, _route_b], False)
            new_sub_solution_cost = self.calculate_solution_target_metric([new_route_a, new_route_b], False)
            _delta = old_sub_solution_cost - new_sub_solution_cost
            feasibility_flag = self.is_route_valid(new_route_a) & self.is_route_valid(new_route_b)
            return _delta, feasibility_flag, (new_route_a, new_route_b)

        def do_cross(_route_a, _route_b, a_node_ind, b_node_ind):
            _route_a, _route_b = _route_a[:a_node_ind + 1] + _route_b[b_node_ind + 1:], \
                                 _route_b[:b_node_ind + 1] + _route_a[a_node_ind + 1:]
            _route_b = self.recalc_b_time_from_index(_route_b, b_node_ind - 1)
            _route_a = self.recalc_b_time_from_index(_route_a, a_node_ind - 1)
            return _route_a, _route_b

        def estimate_cross_op(_route_a, _route_b, a_node_ind, b_node_ind):
            """

            :param _route_a:
            :param _route_b:
            :param a_node_ind:
            :param b_node_ind:
            :return:
            """
            new_route_a, new_route_b = do_cross(_route_a.copy(), _route_b.copy(), a_node_ind, b_node_ind)
            old_sub_solution_cost = self.calculate_solution_target_metric([_route_a, _route_b], False)
            new_sub_solution_cost = self.calculate_solution_target_metric([new_route_a, new_route_b], False)
            _delta = old_sub_solution_cost - new_sub_solution_cost
            feasibility_flag = self.is_route_valid(new_route_a) & self.is_route_valid(new_route_b)
            return _delta, feasibility_flag, (new_route_a, new_route_b)

        def generate_route_pairs(n):
            return [(i, j) for i in range(n) for j in range(n) if i != j]

        def generate_node_pairs_within_two_routes(n, m):
            """

            :param n: len of first route
            :param m: len of second route
            :return:
            """
            return [(i, j) for i in range(1, n - 1) for j in range(1, m - 1)]

        feasible_relocate_collection = [it for it in [
            (estimate_relocate_op(local_solution[ind_a], local_solution[ind_b], i, j), (ind_a, ind_b))
            for ind_a, ind_b in generate_route_pairs(len(local_solution))
            for i, j in generate_node_pairs_within_two_routes(len(local_solution[ind_a]),
                                                              len(local_solution[ind_b]))] if it[0][1]]
        feasible_exchange_collection = [it for it in [
            (estimate_exchange_op(local_solution[ind_a], local_solution[ind_b], i, j), (ind_a, ind_b))
            for ind_a, ind_b in generate_route_pairs(len(local_solution))
            for i, j in generate_node_pairs_within_two_routes(len(local_solution[ind_a]),
                                                              len(local_solution[ind_b]))] if it[0][1]]
        feasible_cross_colection = [it for it in [
            (estimate_cross_op(local_solution[ind_a], local_solution[ind_b], i, j), (ind_a, ind_b))
            for ind_a, ind_b in generate_route_pairs(len(local_solution))
            for i, j in generate_node_pairs_within_two_routes(len(local_solution[ind_a]),
                                                              len(local_solution[ind_b]))] if it[0][1]]

        perturbation_collection = sorted(feasible_exchange_collection
                                         + feasible_relocate_collection
                                         + feasible_cross_colection,
                                         key=lambda x: x[0][0], reverse=True)

        perturbed_solution = local_solution.copy()
        perturbed_route_list = []
        for (_, _, route_tuple), ind_tuple in perturbation_collection:
            if all([ind not in perturbed_route_list for ind in ind_tuple]):
                perturbed_solution[ind_tuple[0]], perturbed_solution[ind_tuple[1]] = route_tuple
                perturbed_route_list.append(ind_tuple[0])
                perturbed_route_list.append(ind_tuple[1])
            else:
                if len(perturbed_route_list) == len(perturbed_solution):
                    break
        # print(perturbed_route_list)
        perturbed_solution = [it for it in perturbed_solution if len(it) > 2]
        return perturbed_solution
