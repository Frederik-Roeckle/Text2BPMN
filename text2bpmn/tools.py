
from langchain_core.tools import tool
from text2bpmn.agents import NormalAgent
from langgraph.prebuilt import InjectedState
from typing import Annotated
from text2bpmn.config import get_model

# ----------------------------------------------------
# Create a feedback agent as a tool

@tool
def fedback_agent(state: Annotated[dict, InjectedState]):
        """
        Provide feedback on the given process description.
        """
        feedback_agent_object = NormalAgent(model=get_model(),system_message="data/promts/feedback_promt.txt")
        response = feedback_agent_object.invoke(state["messages"][-1])
        return response.content

# ----------------------------------------------------

