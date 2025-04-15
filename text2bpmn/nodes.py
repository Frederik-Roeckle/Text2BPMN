from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from text2bpmn.utils import load_data
from langchain_core.language_models.chat_models import BaseChatModel
from abc import ABC, abstractmethod

class BaseNode(ABC):
    @abstractmethod
    def invoke(self, state):
        raise NotImplementedError("Subclasses must implement the invoke method.")

class NormalNode(BaseNode):
    def __init__(self, model: BaseChatModel, system_message=None, few_shot_examples=None, invoke_message=None, tools=None):
        self.model = model
        self.system_message = system_message
        self.few_shot_examples = few_shot_examples
        self.invoke_message = invoke_message
        self.tools = tools

    def invoke(self, state):
        messages = []
        if self.tools:
            self.model.bind_tools(self.tools)
        if self.system_message:
            prompt = load_data(self.system_message)
            messages.append(SystemMessage(content=prompt))

        if self.few_shot_examples:
            for ex in load_data(self.few_shot_examples):
                messages.append(HumanMessage(content=ex["human"]))
                messages.append(AIMessage(content=ex["ai"]))

        messages += state["messages"]

        if self.invoke_message:
            messages.append(HumanMessage(content=self.invoke_message))

        response = self.model.invoke(messages)
        return {"messages": [response]}

