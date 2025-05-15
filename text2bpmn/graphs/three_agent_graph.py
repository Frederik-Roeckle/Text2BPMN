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

        # Construct new messages list:
        # 1. Agent's system prompt (assuming it is accessible)
        # 2. Last AI message content as HumanMessage
        system_msg_content = getattr(agent, "system_message", None)
        if system_msg_content is None:
            raise ValueError("Agent has no system_message attribute")

        new_messages = [
            SystemMessage(content=system_msg_content),
            HumanMessage(content=last_ai_msg.content)
        ]

        # Replace the state's messages with this minimal set
        state["messages"] = new_messages

        # Now invoke the agent with this reduced state
        return agent.invoke(state)
    return wrapper



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
    #




    supervisor = EvaluatorAgent(
        model=get_model(),
        system_message=prompt_supervisor,
        step="supervisor"
    )

    def supervisor_with_limit(agent_fn, max_runs: int = 2):
        def wrapper(state):
            # count how many times we've hit supervisor
            runs = state.get("supervisor_runs", 0) + 1
            state["supervisor_runs"] = runs
            print(f"🔁 Supervisor run #{runs}")

            if runs > max_runs:
                print("🛑 Max supervisor runs exceeded → ending process.")
                last_msg = state["messages"][-1]
                return Command(
                    update={"messages": [last_msg]},
                    goto=END
                )

            # already invoked in agent_fn, just pass through
            return agent_fn(state)

        # propagate metadata
        wrapper.system_message = getattr(agent_fn, "system_message", None)
        wrapper.step = getattr(agent_fn, "step", None)
        return wrapper

    only_super = only_last_message(supervisor)
    limited_supervisor = supervisor_with_limit(only_super, max_runs=2)

    # Build the full graph
    graph = (
        StateGraph(MessagesState)
    .add_node("extractAgent", extractAgent.invoke)
    .add_node("xml_Agent", only_last_message(xml_Agent))
    .add_node("validate_Agent", only_last_message(validate_Agent))
    .add_node("supervisor",    limited_supervisor)

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
    

