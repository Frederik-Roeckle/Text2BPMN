from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from text2bpmn.utils import load_data
from langchain_core.language_models.chat_models import BaseChatModel
from abc import ABC, abstractmethod
import logging
from typing import Literal
from langgraph.types import Command
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from text2bpmn.formats import EvaluatorResult

logging.basicConfig(level=logging.INFO)

class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel, system_message=None, few_shot_examples=None, invoke_message=None, tools=None,step=None):
        self.model = model
        self.system_message = system_message
        self.few_shot_examples = few_shot_examples
        self.invoke_message = invoke_message
        self.tools = tools
        self.start_messages = []
        self.step = step
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
    def __init__(self, model: BaseChatModel, system_message=None, few_shot_examples=None, invoke_message=None, tools=None,step=None):
        super().__init__(model, system_message, few_shot_examples, invoke_message, tools, step)

    def invoke(self, state):
        self.add_tools()
        self.add_system_message()
        self.add_few_shot_examples()
        self.start_messages += state["messages"]
        self.add_invoke_message()
        response = self.model.invoke(self.start_messages)
        logging.info(f"Response: {response}")
        with open(f"{self.step}.txt", "w") as file:
            file.write(response.content)
        return {"messages": [response]}
    

    
class FeedbackAgent(BaseAgent):
    def __init__(self, model: BaseChatModel, system_message=None, few_shot_examples=None, invoke_message=None, tools=None):
        super().__init__(model, system_message, few_shot_examples, invoke_message, tools)

    def invoke(self, state):
        self.add_tools()
        self.add_system_message()
        self.add_few_shot_examples()




class EvaluatorAgent(BaseAgent):
    def __init__(self, model, system_message=None, few_shot_examples=None, invoke_message=None, tools=None, step=None):
        super().__init__(model, system_message, few_shot_examples, invoke_message, tools, step)
        self.parser = PydanticOutputParser(pydantic_object=EvaluatorResult)

        # Load prompt template and inject parser instructions
        raw_prompt = load_data(system_message)
        self.prompt = PromptTemplate.from_template(raw_prompt).partial(
            format_instructions=self.parser.get_format_instructions()
        )

    def invoke(self, state) -> Command[Literal["extract", "create_xml", "__end__"]]:
        self.start_messages = []
        self.add_few_shot_examples()
        self.add_invoke_message()

        # Format prompt and get model output
        formatted_prompt = self.prompt.format()
        messages = self.start_messages + [HumanMessage(content=formatted_prompt)] + state["messages"] # Note that the order of the promt is changed compared to the Base angent. The instruction to check the given xml is put at the end to emphazise it.
        response = self.model.invoke(messages)

        # Parse and validate output
        parsed = self.parser.parse(response.content)

        # Add model's reasoning to message history
        state["messages"].append(AIMessage(content=f"{parsed.reason} (Routing to: {parsed.next_node})"))

        return Command(
            update={"messages": state["messages"]},
            goto="__end__" if parsed.next_node == "end" else parsed.next_node
        )

