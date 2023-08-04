import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


def nx_plot(
        graph=None,
        nodes=None,
        nodes_pos=None,
        nodes_args=None,
        nodes_labels=None,
        nodes_labels_pos=None,
        nodes_labels_args=None,
        edges=None,
        edges_pos=None,
        edges_args=None,
        edges_labels=None,
        edges_labels_pos=None,
        edges_labels_args=None,
        ax=None):
    """More complete version of`networkx` plotting. 

    For formatting please refer to:
    * `nx.draw_networkx_nodes`
    * `nx.draw_networkx_edges`
    * `nx.draw_networkx_edge_labels`
    * `matplotlib.pyplot.annotate`

    TODO: 
        * document arguments
        * add examples

    Notes:
    * Alpha for individual edges: set alpha in color using `edge_color`

    * NetworkX = 2.6.3
        * Self-loops are plotted all of a sudden but their formatting doesn't work as expected. 
          Luckily this does not influence non-self-loops:
          Link: https://github.com/networkx/networkx/issues/5106
    * NetworkX < 2.4:
        * Weirdness to make **edges transparent**:
            Set edge alpha to a array or a function (value does not matter),
            and then the set edge colors to RGBA:
            `edges_args=dict(edge_color=lambda g,src,dst,d: return (1,0,0,d['alpha']), alpha=lambda g,src,dst,d: 1)`
        * When using matplotlib.pyplot.subplots networkx adjusted axes by calling `plt.tick_params` causing unwanted
            behavior. This was fixed in networkx 2.4.

    Parameters
    ----------
    graph : [type]
        [description]
    nodes : [type]
        [description]
    nodes_pos : [type]
        [description]
    nodes_args : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels_pos : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels_args : [type], optional
        [description] (the default is None, which [default_description])
    edges : [type], optional
        [description] (the default is None, which [default_description])
    edges_pos : [type], optional
        [description] (the default is None, which [default_description])
    edges_args : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels_pos : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels_args : [type], optional
        [description] (the default is None, which [default_description])
    
    Returns
    -------
    [type]
        [description]
    """

    # get axis
    if ax is None:
        ax = plt.gca()

    # init graph

    if graph is None or isinstance(graph, str):
        if graph == "bi" or graph == "di":
            g = nx.DiGraph()
        else:
            g = nx.Graph()
    else:
        g = graph

    # init nodes

    if graph is None and nodes is None:
        if nodes_pos is not None:
            if isinstance(nodes_pos, dict):
                nodes = nodes_pos.keys()
            else: 
                nodes = np.arange(nodes_pos.shape[0])
        else:
            raise Exception("Either `graph` or `nodes` must be given. Both were `None`.")

    if nodes is not None:
        if isinstance(nodes, dict):
            nodes = [(k, v) for k, v in nodes.items()]
        g.add_nodes_from(nodes)

    # init edges

    if edges is not None:
        if isinstance(edges, dict):
            edges = [(*k, v) for k, v in edges.items()]
        g.add_edges_from(edges)

    # init positions

    def init_pos(pos):
        if pos is None:
            return nodes_pos
        elif callable(pos):
            return {n: pos(g, n, d) for n, d in g.nodes(data=True)}
        elif isinstance(pos, dict):
            return pos
        else:
            return {n: p for n, p in zip(g.nodes(), pos)}

    nodes_pos = init_pos(nodes_pos)
    nodes_labels_pos = init_pos(nodes_labels_pos)
    edges_pos = init_pos(edges_pos)
    edges_labels_pos = init_pos(edges_labels_pos)

    # init labels

    def init_nodes_labels(labels):
        if callable(labels):
            return {n: labels(g, n, d) for n, d in g.nodes(data=True)}
        else:
            return labels
    nodes_labels = init_nodes_labels(nodes_labels)

    def init_edges_labels(labels):
        if callable(labels):
            tmp = {(src, dst): labels(g, src, dst, d) for src, dst, d in g.edges(data=True)}
            tmp = {k: v for k, v in tmp.items() if v is not None}  # filter "None" labels
            return tmp
        else:
            return labels
    edges_labels = init_edges_labels(edges_labels)

    # init layout arguments

    def init_node_args(args):
        if args is None:
            args = {}
        else:
            args = args.copy()
            for k, v in args.items():
                if callable(v):
                    args[k] = [v(g, n, d) for n, d in g.nodes(data=True)]
        if "ax" not in args:
            args["ax"] = ax
        return args

    nodes_args = init_node_args(nodes_args)
    nodes_labels_args = init_node_args(nodes_labels_args)

    def init_edges_args(args):
        if args is None:
            args = {}
        else:
            args = args.copy()
            for k, v in args.items():
                if callable(v):
                    args[k] = [v(g, src, dst, d) for src, dst, d in g.edges(data=True)]
        if "ax" not in args:
            args["ax"] = ax
        return args

    edges_args = init_edges_args(edges_args)
    edges_labels_args = init_edges_args(edges_labels_args)

    # draw nodes (allow for several of shapes for nodes)
    if "node_shape" in nodes_args and type(nodes_args["node_shape"]) is list:

        shapes = list(zip(range(len(g.nodes())), nodes_args["node_shape"]))
        unique_shapes = np.unique(nodes_args["node_shape"])

        for shape in unique_shapes:

            shape_idx = [i for i, s in shapes if s == shape]

            nodes = list(g.nodes())
            nodelist = [nodes[i] for i in shape_idx]

            shape_args = nodes_args.copy()
            del shape_args["node_shape"]

            for arg, _ in shape_args.items():
                if type(shape_args[arg]) is list:
                    shape_args[arg] = [shape_args[arg][i] for i in shape_idx]

            nx.draw_networkx_nodes(g, nodes_pos, nodelist=nodelist, node_shape=shape, **shape_args)
    else:
        nx.draw_networkx_nodes(g, nodes_pos, **nodes_args)

    # draw edges
    # print(g.nodes(data=True))
    # print(g.edges(data=True))
    # print(edges_args)
    nx.draw_networkx_edges(g, nodes_pos, **edges_args)

    # draw node labels
    if nodes_labels is not None:

        # rename args for compatibility to `nx.draw_networkx_labels`
        args = nodes_labels_args.copy()
        del args["ax"]
        for original_key, new_key in {
                "font_size": "fontsize",
                "font_color": "color",
                "font_family": "family",
                "font_weight": "weight"}.items():
            if original_key in args:
                args[new_key] = args[original_key]
                del args[original_key]

        # check if we have list args
        list_args = []
        for arg, value in list(args.items()):
            if isinstance(value, list) and len(value) == len(g.nodes):
                list_args.append((arg, value))
                del args[arg]

        for i, node in enumerate(g.nodes):
            ax.annotate(nodes_labels[node], nodes_labels_pos[node], **args, **{a: v[i] for a, v in list_args})

    # draw edge labels
    if edges_labels is not None:
        nx.draw_networkx_edge_labels(g, pos=edges_labels_pos, edge_labels=edges_labels, **edges_labels_args)

    return g
