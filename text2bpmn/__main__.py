from text2bpmn.graphs import build_graphs_from_yaml
from langchain_core.messages import HumanMessage

GRAPH_CONFIG_PATH = "configs/graphs.yaml"

def main():
    graphs = build_graphs_from_yaml(GRAPH_CONFIG_PATH)
    for _, graph in graphs.items():
        graph.invoke({"messages": [HumanMessage(content="Making coffee involves boiling water, adding coffee grounds, and pouring the brewed coffee into a cup.")]})


if __name__ == "__main__":
    main()
