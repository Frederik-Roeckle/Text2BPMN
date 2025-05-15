from langgraph.graph import StateGraph, MessagesState, START, END
from agents import NormalAgent
from config import get_model
from agents import NormalAgent
from langsmith import traceable


@traceable
def build_graph():
    """
    Build as graph that uses only one agent to create the output.
    """

    builder = StateGraph(MessagesState)

    # Define nodes
    
    # text2bpmn = NormalAgent(
    #     model=get_model(),
    #     system_message="data/promts/base_promt.txt",
    #     few_shot_examples="data/examples/few_shot_examples.json",
    # )

    extract = NormalAgent(
        model=get_model(),
        system_message="data/promts/extraction_prompt.txt",
        step="extract"
    )

    create_xml = NormalAgent(
        model=get_model(),
        system_message="data/promts/create_xml_prompt.txt",
        step="xml"
    )

    #builder.add_node("text2bpmn", text2bpmn.invoke)
    builder.add_node("extract", extract.invoke)
    builder.add_node("create_xml", create_xml.invoke)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "create_xml")
    builder.add_edge("create_xml", END)

    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({"messages": [""]})