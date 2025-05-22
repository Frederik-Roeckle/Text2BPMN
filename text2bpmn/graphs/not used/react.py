from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import convert_to_messages
from text2bpmn.config import get_model, set_model
from text2bpmn.models import OpenAILLM
from text2bpmn.agents import NormalAgent
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.graph import END



# -----------------------------
# Pretty printing
# -----------------------------
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:\n"
        if is_subgraph:
            update_label = "\t" + update_label
        print(update_label)

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


# -----------------------------
# Handoff tool logic
# -----------------------------
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


# -----------------------------
# Main graph builder
# -----------------------------
def build_graph():
    # Agents
    

    extractAgent = create_handoff_tool(
        agent_name="extractAgent",
        description=(
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
        agent_name="xmlAgent",
        description="You are an agent that creates a BPMN XML from extracted steps."
    )

    validateAgent = create_handoff_tool(
        agent_name="validateAgent",
        description="You are an agent that validates a BPMN XML."
    )

    # Supervisor creation
    supervisor_agent = create_react_agent(
    model=get_model(),
    agents=[extractAgent, xmlAgent, validateAgent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "1. An agent that creates a BPMN XML from extracted steps.\n"
        "2. An agent that validates a BPMN XML.\n"
        "If the validator agent is not OK with the result, hand it over to the xml creation again.\n"
        "Do not do this more than 3 times.\n"
        "If the validator agent is OK with the result, hand it over to the user.\n"
 
    ),
    name="supervisor",
    )

    # Build the full graph
    graph = (
        StateGraph(MessagesState)
        .add_node(supervisor_agent, END)
        .add_node(extractAgent)
        .add_node(xmlAgent)
        .add_node(validateAgent)
        .add_edge(START, "extractAgent")
        .add_edge("extractAgent", "xml_Agent")
        .add_edge("xmlAgent", "supervisor")
        .add_edge("validateAgent", "supervisor")

       
        .compile()
    )

    for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "fThe process starts when a customer submits an order. The system then checks the inventory for item availability. If the items are in stock, they are packed and prepared for shipment, after which the order is shipped to the customer.",
            }
        ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)

    final_message_history = chunk["supervisor"]["messages"]
    for message in final_message_history:
        message.pretty_print()

    return graph


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    set_model(OpenAILLM(model="gpt-4.1-mini",temperature=0))

    graph = build_graph()
    

