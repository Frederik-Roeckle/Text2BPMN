from langgraph.graph import StateGraph, MessagesState, START, END
from text2bpmn.agents import NormalAgent
from text2bpmn.config import get_model


def build_graph():
    """
    Build as graph that uses only one agent to create the output.
    """

    builder = StateGraph(MessagesState)

    # Define nodes
    text2bpmn = NormalAgent(
        model=get_model(),
        system_message="data/promts/base_promt.txt",
        few_shot_examples="data/examples/few_shot_examples.json",
    )

    builder.add_node("text2bpmn", text2bpmn.invoke)

    builder.add_edge(START, "text2bpmn")
    builder.add_edge("text2bpmn", END)

    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({"messages": ["Making a reservation involves calling the restaurant, providing the date and time, and confirming the reservation."]})