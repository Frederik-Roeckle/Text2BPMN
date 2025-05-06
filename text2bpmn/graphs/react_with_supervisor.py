from langgraph.graph import StateGraph, START, MessagesState
from text2bpmn.config import get_model, set_model
from text2bpmn.models import OpenAILLM
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END
from langchain_core.tools import tool, InjectedToolCallId


# -----------------------------
# Handoff tool logic
# -----------------------------
def create_handoff_tool(
   *, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool


# -----------------------------
# Main graph builder
# -----------------------------
def build_graph():
    # Agents
    

    extractAgent = create_react_agent(
        name="extractAgent",
        model=get_model(),
        tools=[],
        prompt=(
            "Extract all relevant BPMN elements, including:\n"
            "- Participants as Pools and Lanes (departments, roles, teams, etc.)\n"
            "- Activities, such as Task, Subprocess, Call Activity, Event Subprocess, or Transaction\n"
            "- Events, including Start Events, Intermediate Events, and End Events\n"
            "- Gateways, such as Exclusive, Inclusive, Parallel, or Event-based\n"
            "- Determine sequence flows between the elements, including any conditional logic (e.g., 'if approved').\n"
            "- Assign elements to lanes if multiple participants are involved. If not specified, assume a single pool with one default lane."
        )
    )

    xmlAgent = create_handoff_tool(
        agent_name="xml_Agent",
        description="Assign task to a xml agent."
    )

    validateAgent = create_handoff_tool(
        agent_name="validate_Agent",
        description="Assign task to a validate agent."
    )

    xml_Agent = create_react_agent(
    name="xml_Agent",
    model=get_model(),
    tools=[],
    prompt="You are an agent that creates a BPMN XML from extracted BPMN elements."
    )   

    validate_Agent = create_react_agent(
    name="validate_Agent",
    model=get_model(),
    tools=[],
    prompt="You are an agent that validates a BPMN XML for correctness and completeness. When an xml is valid return ***IS_VALID*** as the last word in your message after the xml."
    )

    # Supervisor creation
    supervisor_agent = create_react_agent(
    model=get_model(),
    tools=[xmlAgent, validateAgent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "1. An agent that creates a BPMN XML from extracted steps.\n"
        "2. An agent that validates a BPMN XML.\n"
        "You receive the output of the xml agent and give the whole xml to the validator.\n"
        "If the validator is OK with the result and returned IS_VALID, hand it over to the user. This is the stop condition\n"
        "If the validator agent is not OK with the result, hand the output of the validator over to the xml creation again.\n"
        "Always pass the full message received from one agent to the next agent without modification."

    ),
    name="supervisor",
    )

    # Build the full graph
    graph = (
        StateGraph(MessagesState)
        .add_node(supervisor_agent, destinations=("xml_Agent", "validate_Agent", END))
        .add_node(xml_Agent)
        .add_node(validate_Agent)
        .add_node(extractAgent)
        .add_edge(START, "extractAgent")
        .add_edge("extractAgent", "supervisor")
        .add_edge("xml_Agent", "supervisor")
        .add_edge("validate_Agent", "supervisor")
        .compile()
    )



    return graph



# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    set_model(OpenAILLM(model="gpt-4.1-mini",temperature=0))

    graph = build_graph()
    graph.invoke({"messages": [""]})
    

