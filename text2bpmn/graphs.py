import yaml
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from text2bpmn.nodes import make_node
from text2bpmn.nodes import *

# ------------------------------------------------------------
# Create Graphs
# ------------------------------------------------------------

def build_graphs_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    graphs = {}
    for name, graph in data["graphs"].items():
        if graph["type"] == "dynamic":
            graphs[name] = build_dynamic_graph(graph)
        elif graph["type"] == "nested":
            graphs[name] = build_nested_graph(graph)
        elif graph["type"] == "explicit":
            graphs[name] = build_explicit_graph(graph)
        elif graph["type"] == "explicit_tools":
            # TODO: Implement explicit tools graph
            pass
    return graphs

# ------------------------------------------------------------
# Graph Factory
# ------------------------------------------------------------ 

def build_dynamic_graph(config:dict):
    builder = StateGraph(MessagesState)
    for node in config["nodes"]:
        builder.add_node(make_node(node))
    for edge in config["edges"]:
        builder.add_edge(edge["from"], edge["to"])
    return builder.compile()

def build_nested_graph(config:dict):
    subgraphs = {}
    for team_name, team_cfg in config["subgraphs"].items():
        builder = StateGraph(MessagesState)
        for node in team_cfg["nodes"]:
            builder.add_node(make_node(node))
        for edge in team_cfg["edges"]:
            builder.add_edge(edge["from"], edge["to"])
        subgraphs[f"{team_name}_graph"] = builder.compile()

    builder = StateGraph(MessagesState)
    sup = config["supervisor"]
    builder.add_node(make_node(sup))
    for name, graph in subgraphs.items():
        builder.add_node(name, graph)
    for edge in sup["edges"]:
        builder.add_edge(edge["from"], edge["to"])
    return builder.compile()

def build_explicit_graph(config):
    builder = StateGraph(MessagesState)
    for node_dic in config["nodes"]:
        name, node = make_node(node_dic)
        builder.add_node(name, node)
    for edge in config["edges"]:
        from_ege = edge["from"]
        to_edge = edge["to"]
        if from_ege == "__START__":
            from_ege = START
        if to_edge == "__END__":
            to_edge = END
        builder.add_edge(from_ege, to_edge)
    return builder.compile()

if __name__ == "__main__":
    graphs = build_graphs_from_yaml("configs/graphs.yaml")
    print()