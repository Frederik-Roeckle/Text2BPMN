from langgraph.prebuilt import create_react_agent
from text2bpmn.tools import fedback_agent
from text2bpmn.config import get_model
from text2bpmn.utils import load_data


# ---------------------------------------
# Implementation of the ReAct agent. See also this paper for the concept: https://arxiv.org/abs/2210.03629
# However, this implementation is not exactly the same as the paper: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation
# ---------------------------------------

# TODO: Output is noot a XML!!!
def build_graph():

    tools = [fedback_agent]
    # the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
    # that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
    supervisor = create_react_agent(get_model(), tools, prompt=load_data("data/promts/base_promt.txt")) #TODO: Check Promt for the supevisor
    return supervisor

if __name__ == "__main__":
    from text2bpmn.config import set_model
    from models import MistralLLM
    set_model(MistralLLM())

    graph = build_graph()
    graph.invoke({"messages": ["Making a reservation involves calling the restaurant, providing the date and time, and confirming the reservation."]})