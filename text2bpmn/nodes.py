import yaml
from langgraph.prebuilt import create_react_agent
from text2bpmn.models import OpenAILLM, MistralLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from text2bpmn.utils import load_data
from text2bpmn.utils import get_model_from_config
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from typing import Annotated, Literal
from text2bpmn.utils import get_model

AGENT_CONFIG_PATH = "configs/agents.yaml"


# ---------------------------------------------------------
# Create Nodes
# ---------------------------------------------------------


def make_node(node: dict):
    if node["kind"] == "command":
        return make_command_node(**node)
    elif node["kind"] == "react":
        return make_react_node(**node)
    elif node["kind"] == "normal":
        return make_normal_agent(**node)
    else:
        raise ValueError(f"Unknown node kind: {node['kind']}")


# ------------------------------------------------------------
# Agent Factory
# ------------------------------------------------------------


def make_normal_agent(
    name: str,
    model_name: str = "mistral",
    system_message: str = None,
    few_shot_examples: str = None,
    routes: list[str] = None,
    tools: list = None,
    kind: str = None,
    invoke_message: str = None
):
    """
    Creates a node for a normal agent with the given name and default model.
    An example how such an angent can be used can found here: https://langchain-ai.github.io/langgraph/concepts/multi_agent/#custom-multi-agent-workflow
    """

    def node(state):
        model = get_model(model_name)
        if tools:
            model.bind_tools(tools)

        messages = []
        if system_message:
            promt = load_data(system_message)
            messages.append(SystemMessage(content=promt))

        if few_shot_examples:
            for ex in load_data(few_shot_examples):
                messages.append(HumanMessage(content=ex["human"]))
                messages.append(AIMessage(content=ex["ai"]))

        messages += state["messages"]
        if invoke_message:
            messages.append(HumanMessage(content=invoke_message))
        response = model.invoke(messages)
        return {"messages": [response]}

    return name, node


# TODO refactor
def make_command_agent(
    agent_name: str, possible_routes: list[str], default_model="mistral"
):
    """
    Creates a command agent that can route to different nodes based on the command.
    An exmaple how this agent is used can be found here: https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor
    """
    routes_literal = Literal[tuple(r if r != "END" else "END" for r in possible_routes)]

    def agent(state: Annotated[dict, InjectedState]) -> Command[routes_literal]:
        cfg = load_data(AGENT_CONFIG_PATH).get("agents", {})
        agent_cfg = cfg.get(agent_name, {})

        model_name = agent_cfg.get("model", default_model)
        model = get_model(model_name)

        messages = []
        if "system_message" in agent_cfg:
            promt = load_data(agent_cfg["system_message"])
            messages.append(SystemMessage(content=promt))

        if "few_shot_examples" in agent_cfg:
            for ex in load_data(agent_cfg["few_shot_examples"]):
                messages.append(HumanMessage(content=ex["human"]))
                messages.append(AIMessage(content=ex["ai"]))

        messages += state["messages"]
        response = model.invoke({"messages": messages})
        return Command(
            goto=response.get("next_agent", "END"),
            update={"messages": [response["content"]]},
        )

    return agent


# TODO: Refactor
def make_react_agent(agent_name: str, tools: list[str], default_model="mistral"):
    cfg = load_data(AGENT_CONFIG_PATH).get("agents", {})
    agent_cfg = cfg.get(agent_name, {})

    model_name = agent_cfg.get("model", default_model)
    model = get_model(model_name)

    if "system_message" in agent_cfg:
        promt = load_data(agent_cfg["system_message"])
        return create_react_agent(model, tools, prompt=SystemMessage(content=promt))
    else:
        return create_react_agent(model, tools)
