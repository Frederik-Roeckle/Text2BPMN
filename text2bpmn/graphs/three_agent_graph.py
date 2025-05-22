from text2bpmn.agents import NormalAgent, EvaluatorAgent
from langgraph.graph import StateGraph, START, MessagesState
from text2bpmn.config import get_model, set_model
from text2bpmn.models import OpenAILLM
from typing import Annotated
from langgraph.graph import END
from langsmith import traceable
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langgraph.types import Command



def read_txt_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

# @traceable


def only_last_message(agent):
    def wrapper(state):
        # Extract the last AI message content
        last_ai_msg = next(
            (msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)),
            None
        )
        if not last_ai_msg:
            raise ValueError("No AIMessage found in state['messages'].")

        # Construct new minimal message list
        system_msg_content = getattr(agent, "system_message", None)
        if system_msg_content is None:
            raise ValueError("Agent has no system_message attribute")

        # ⚠️ Modify state in-place to preserve other keys like supervisor_runs
        state["messages"] = [
            SystemMessage(content=system_msg_content),
            HumanMessage(content=last_ai_msg.content)
        ]

        return agent.invoke(state)
    
    # Optional: preserve metadata for graph visualization/debugging
    wrapper.system_message = getattr(agent, "system_message", None)
    wrapper.step = getattr(agent, "step", None)

    return wrapper





# -----------------------------
# Main graph builder
# -----------------------------
@traceable
def build_graph():
    # Agents
    
    prompt_extract = read_txt_file("data/promts/extraction_prompt.txt")
    prompt_xml = read_txt_file("data/promts/create_xml_prompt.txt")
    prompt_xml_few = read_txt_file("data/promts/create_xml_prompt_few.txt")
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
        system_message=prompt_xml_few,
        few_shot_examples="data/examples/five_shot_examples.json",
        step="xml_Agent"
    )

    validate_Agent = NormalAgent(
        model=get_model(),
        system_message=prompt_validate,
        #few_shot_examples="data/examples/few_shot_examples.txt",
        step="validate_Agent"
    )
    #




    supervisor = EvaluatorAgent(
        model=get_model(),
        system_message=prompt_supervisor,
        step="supervisor",
         max_runs=2
    )

   

    

    # Build the full graph
    graph = (
        StateGraph(MessagesState)
    .add_node("extractAgent", extractAgent.invoke)
    .add_node("xml_Agent", only_last_message(xml_Agent))
    .add_node("validate_Agent", only_last_message(validate_Agent))
    .add_node("supervisor",    supervisor.invoke)

    .add_edge(START, "extractAgent")
    .add_edge("extractAgent", "xml_Agent")
    .add_edge("xml_Agent", "validate_Agent")
    .add_edge("validate_Agent", "supervisor")
    .add_edge("supervisor", END)
    .compile()
    )

    return graph



# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({"messages": [""]})
    

