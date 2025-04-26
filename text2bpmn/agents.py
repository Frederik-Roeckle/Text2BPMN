from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import load_data
from langchain_core.language_models.chat_models import BaseChatModel
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)

class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel, system_message=None, few_shot_examples=None, invoke_message=None, tools=None):
        self.model = model
        self.system_message = system_message
        self.few_shot_examples = few_shot_examples
        self.invoke_message = invoke_message
        self.tools = tools
        self.start_messages = []
    
    def add_tools(self):
        if self.tools:
            self.model.bind_tools(self.tools)
    
    def add_system_message(self):
        if self.system_message:
            self.start_messages.append(SystemMessage(content=load_data(self.system_message)))

    def add_few_shot_examples(self):
        if self.few_shot_examples:
            for ex in load_data(self.few_shot_examples):
                self.start_messages.append(HumanMessage(content=ex["human"]))
                self.start_messages.append(AIMessage(content=ex["ai"]))
    def add_invoke_message(self):
        if self.invoke_message:
            self.start_messages.append(HumanMessage(content=self.invoke_message))

    @abstractmethod
    def invoke(self, state):
        pass

class NormalAgent(BaseAgent):
    def __init__(self, model: BaseChatModel, system_message=None, few_shot_examples=None, invoke_message=None, tools=None):
        super().__init__(model, system_message, few_shot_examples, invoke_message, tools)

    def invoke(self, state):
        self.add_tools()
        self.add_system_message()
        self.add_few_shot_examples()
        self.start_messages += state["messages"]
        self.add_invoke_message()

        response = self.model.invoke(self.start_messages)
        logging.info(f"Response: {response}")
        return {"messages": [response]}
    

    
class FeedbackAgent(BaseAgent):
    def __init__(self, model: BaseChatModel, system_message=None, few_shot_examples=None, invoke_message=None, tools=None):
        super().__init__(model, system_message, few_shot_examples, invoke_message, tools)

    def invoke(self, state):
        self.add_tools()
        self.add_system_message()
        self.add_few_shot_examples()
        

