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

    baseline = NormalAgent(
        model=get_model(),
        system_message="data/promts/baseline_prompt_few.txt",
        few_shot_examples="data/examples/five_shot_examples.json",
        step="baseline"
    )

    #builder.add_node("text2bpmn", text2bpmn.invoke)
    builder.add_node("baseline", baseline.invoke)

    builder.add_edge(START, "baseline")
    builder.add_edge("baseline", END)

    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({"messages": [""]})