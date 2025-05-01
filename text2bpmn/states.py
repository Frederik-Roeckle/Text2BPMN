from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage

class ExampleState(TypedDict):
    """
    This is an example of a state. Same as langgraph.graph.MessagesState.
    """
    messages: Annotated[list[AnyMessage], add_messages]


