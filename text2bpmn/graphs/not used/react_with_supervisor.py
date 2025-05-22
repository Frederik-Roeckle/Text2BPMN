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
        print(f"[HANDOFF DEBUG] Transferring to {agent_name} with tool_call_id: {tool_call_id}")
        print(f"[HANDOFF DEBUG] Current state has {len(state['messages'])} messages")
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        updated_state = {**state, "messages": state["messages"] + [tool_message]}
            
        print(f"[HANDOFF DEBUG] After update, state has {len(updated_state['messages'])} messages")
        print(f"[HANDOFF DEBUG] Issuing Command to go to {agent_name}")
        return Command(
            goto=agent_name,  
            update=updated_state,  
            graph=Command.PARENT,  
        )

    return handoff_tool

def read_txt_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

# -----------------------------
# Main graph builder
# -----------------------------
def build_graph():
    # Agents
    
    prompt_extarct = read_txt_file("data/promts/extraction_prompt.txt")
    prompt_xml = read_txt_file("data/promts/create_xml_prompt.txt")
    prompt_validate = read_txt_file("data/promts/validate_prompt.txt")

    extractAgent = create_react_agent(
        name="extractAgent",
        model=get_model(),
        tools=[],
        prompt=prompt_extarct
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
    prompt=prompt_xml,
    )   

    validate_Agent = create_react_agent(
    name="validate_Agent",
    model=get_model(),
    tools=[],
    prompt=prompt_validate,
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
    debug=True 
    )

    # Build the full graph
    graph = (
        StateGraph(MessagesState)
        .add_node(supervisor_agent, destinations=("xml_Agent", "validate_Agent", END))
        .add_node(xml_Agent)
        .add_node(validate_Agent)
        .add_node(extractAgent)
        .add_edge(START, "extractAgent")
        .add_edge("extractAgent", "xml_Agent")
        .add_edge("xml_Agent", "validate_Agent")
        .add_edge("validate_Agent", "supervisor")
        .compile()
    )



    return graph



# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({"messages": [""]})
    

