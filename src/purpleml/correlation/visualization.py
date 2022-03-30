import numpy as np
import pandas as pd
from purpleml.correlation.base import normalize_coordinates


def print_correlation_df(selected_correlations):
    
    from IPython.core.display import display
    
    colored_correlations = selected_correlations.style\
        .applymap(correlation_coloring, subset=['correlation'])\
        .applymap(pvalue_coloring, subset=['p'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(colored_correlations)


def plot_graph(G, coordinates=None, labels='node_names', norm_coordinates=True, tooltips=None):
    
    import networkx as nx
    from bokeh.models import ColumnDataSource , LabelSet
    from bokeh.models import Plot, StaticLayoutProvider, Range1d, MultiLine, Circle, HoverTool, BoxZoomTool, ResetTool, TapTool, WheelZoomTool, PanTool
    from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes

    if tooltips is None:
        tooltips = [("index", "@index")]
        
    # show with Bokeh
    plot = Plot(plot_width=900, 
                plot_height=650,
                x_range=Range1d(-1.1, 1.1), 
                y_range=Range1d(-1.1, 1.1))
    
    # tools
    node_hover_tool = HoverTool(tooltips=tooltips)
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(),  WheelZoomTool(), TapTool(), PanTool())
    
    # renderer
    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
    
    # layout
    if coordinates is None:
        print("Layout: Spring layout")
        coordinates = nx.spring_layout(G)
    if type(coordinates) is str:
        print("Layout: Spring layout, weights: ", coordinates)
        coordinates = nx.spring_layout(G, weight=coordinates)
        
    if norm_coordinates:
        coordinates = normalize_coordinates(coordinates)
    graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=coordinates)

    # styling fun
    graph_renderer.node_renderer.glyph = Circle(size="style_node_size", fill_color="style_node_color", line_width=0, line_alpha=0)
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="style_edge_color", line_alpha=0.5, line_width=4)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color="style_edge_color", line_alpha=0.8, line_width=7)
    
    # selection stuff
    #graph_renderer.inspection_policy = EdgesAndLinkedNodes()
    #graph_renderer.inspection_policy = NodesAndLinkedEdges()
    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()

    # add renderer
    plot.renderers.append(graph_renderer)

    # labels
    if labels is 'node_names':
        labels = coordinates
    
    if labels is not None:   
        source = ColumnDataSource({
            'x': [xy[0] for n, xy in labels.items()],
            'y':[xy[1] for n, xy in labels.items()],
            'label': [n for n, xy in labels.items()]})
        labels = LabelSet(x='x', y='y', text='label', source=source)
        plot.renderers.append(labels)
    
    return plot


def correlation_coloring(val):
    color = "lightgrey"
    if val >= 0.3:
        color = "orange"
    if val >= 0.5:
        color = "red"
    if val <= -0.3:
        color = "blue"
    if val <= -0.5:
        color = "purple"
    return color


def pvalue_coloring(val):
    weight = "none"
    if val <= 0.05:
        weight = "bold"
    return 'font-weight: %s' % weight

def normalize_coordinates(coordinates):
    nf = abs(np.array(list(coordinates.values()))).max(axis=0)
    return { n: xy / nf for n, xy in coordinates.items() }

