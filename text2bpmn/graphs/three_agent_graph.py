from text2bpmn.agents import NormalAgent, EvaluatorAgent
from langgraph.graph import StateGraph, START, MessagesState
from langchain.schema import AIMessage
from text2bpmn.config import get_model, set_model
from text2bpmn.models import OpenAILLM
from typing import Annotated
from langgraph.graph import END
from langsmith import traceable


def read_txt_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

# @traceable
# def supervisor_fn(state):
#     last_message = state["messages"][-1]
#     content = last_message.content if hasattr(last_message, "content") else str(last_message)

#     if "IS_VALID" in content:
#         return {"messages": [AIMessage(content="Process completed successfully.")], "next": END}
#     else:
#         return {"messages": [AIMessage(content="Retrying XML creation.")], "next": "xml_Agent"}

# -----------------------------
# Main graph builder
# -----------------------------
@traceable
def build_graph():
    # Agents
    
    prompt_extract = read_txt_file("data/promts/extraction_prompt.txt")
    prompt_xml = read_txt_file("data/promts/create_xml_prompt.txt")
    prompt_validate = read_txt_file("data/promts/validate_prompt.txt")
    prompt_supervisor = read_txt_file("data/promts/supervisor_prompt.txt")

    extractAgent  = NormalAgent(
        model=get_model(),
        system_message=prompt_extract,
        #few_shot_examples="",
        step="extractAgent"
    )

    xml_Agent = NormalAgent(
        model=get_model(),
        system_message=prompt_xml,
        #few_shot_examples="data/examples/xml_example.txt",
        step="xml_Agent"
    )

    validate_Agent = NormalAgent(
        model=get_model(),
        system_message=prompt_validate,
        #few_shot_examples="data/examples/few_shot_examples.txt",
        step="validate_Agent"
    )

    supervisor = EvaluatorAgent(
        model=get_model(),
        system_message=prompt_supervisor,
        step="supervisor"
    )

    

    #supervisor_agent = supervisor_fn


   
    # Build the full graph
    graph = (
        StateGraph(MessagesState)
        .add_node("xml_Agent", xml_Agent.invoke)
        .add_node("validate_Agent", validate_Agent.invoke)
        .add_node("extractAgent", extractAgent.invoke)
        .add_node("supervisor", supervisor.invoke)

        .add_edge(START, "extractAgent")
        .add_edge("extractAgent", "xml_Agent")
        .add_edge("xml_Agent", "validate_Agent")
        .add_edge("validate_Agent", "supervisor")
        .add_edge("validate_Agent", END)
        .compile()
    )

    return graph



# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({"messages": [""]})
    

