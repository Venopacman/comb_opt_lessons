import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
import re


class GraphReader:
    def __init__(self, graph_file_path):
        def _parse_cust_info_from_str(input_str):
            assert len(re.findall("\d+", input_str)) == len(customer_info_keys)
            return dict(zip(customer_info_keys, [int(it) for it in re.findall("\d+", input_str)]))

        def _parse_vehicle_info_from_str(input_str):
            assert len(re.findall("\d+", input_str)) == len(vehicle_info_keys)
            return dict(zip(vehicle_info_keys, [int(it) for it in re.findall("\d+", input_str)]))

        with open(graph_file_path) as f:
            file_row_list = f.read().split("\n")
            customer_info_keys = ['cust_n', 'x', 'y', 'demand', 'ready_t', 'due_t', 'service_t']
            vehicle_info_keys = ['amount', 'capacity']
            self.graph_name = file_row_list[0]
            self.vehicle_info = _parse_vehicle_info_from_str(file_row_list[4])
            self.customers_info_list = [_parse_cust_info_from_str(it) for it in file_row_list[9:-1]]

    def create_route_graph(self):
        _G = nx.Graph()
        for _cust_info in self.customers_info_list:
            _attr_dict = _cust_info.copy()
            _attr_dict.pop('cust_n')
            _G.add_node(_cust_info['cust_n'], **_attr_dict)
            for _node, _node_data in _G.nodes(data=True):
                if _node != _cust_info['cust_n']:
                    _G.add_edge(_node,
                                _cust_info['cust_n'],
                                time=distance.euclidean((_node_data['x'], _node_data['y']),
                                                        (_cust_info['x'], _cust_info['y'])))
        return _G


def print_graph_info(graph: GraphReader):
    for key in graph.__dict__:
        if key == "route_graph":
            print(key, graph.__dict__[key].edges(data=True))
        else:
            print(key, graph.__dict__[key])


def plot_graph_nodes(graph: nx.Graph):
    pos = dict(zip(graph.nodes, zip(*[[it[1] for it in graph.nodes.data(key)] for key in ("x", "y")])))
    nx.draw_networkx_nodes(graph, pos=pos, node_size=50, node_color=["r"] + ["g"] * (len(pos) - 1))
    plt.show()


def plot_routes(graph: nx.Graph, route_list):
    pos = dict(zip(graph.nodes, zip(*[[it[1] for it in graph.nodes.data(key)] for key in ("x", "y")])))
    color_map = plt.cm.get_cmap('cubehelix')
    # print(color_map)
    nx.draw_networkx_nodes(graph, nodelist=[0], pos=pos, node_size=60, node_color=(0.5, 0.5, 0.5))
    for i, route in enumerate(route_list):
        color = color_map((i + 3) / (len(route_list) + 5))
        # print(color[:-1])
        # print(color)
        nx.draw_networkx_nodes(graph, nodelist=[it['node'] for it in route[1:-1]], pos=pos, node_size=50,
                               node_color=[color[:-1]] * len(route[1:-1]))
        for ind in range(1, len(route)):
            nx.draw_networkx_edges(graph, pos=pos, edgelist=[(route[ind]['node'], route[ind - 1]['node'])],
                                   edge_color=[color])
    plt.show()
    # print(route_list)
