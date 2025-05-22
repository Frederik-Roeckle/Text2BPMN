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


global global_int
global_int = 0

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
            if self.system_message.endswith(".txt") or self.system_message.endswith(".json"):  # file path
                content = load_data(self.system_message)
            else:
                content = self.system_message  # use directly
            self.start_messages.append(SystemMessage(content=content))

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
        self.start_messages = []

        self.add_tools()
        self.add_system_message()
        self.add_few_shot_examples()

        if state["messages"]:
            self.start_messages.append(state["messages"][-1])

        self.add_invoke_message()
        #logging.info(f"Sending the following messages to the model: {self.start_messages}")

        response = self.model.invoke(self.start_messages)
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
    def __init__(self, model, system_message=None, few_shot_examples=None,
                 invoke_message=None, tools=None, step=None, max_runs: int = 3):
        super().__init__(model, system_message, few_shot_examples, invoke_message, tools, step)
        self.parser = PydanticOutputParser(pydantic_object=EvaluatorResult)
        self.max_runs = max_runs

        # Load prompt template and inject parser instructions
        if self.system_message:
            if system_message.endswith(".txt") or system_message.endswith(".json"):
                raw_prompt = load_data(system_message)
            else:
                raw_prompt = system_message
        self.prompt = PromptTemplate.from_template(raw_prompt).partial(
            format_instructions=self.parser.get_format_instructions()
        )

    def invoke(self, state) -> Command[Literal["xml_Agent", "__end__"]]:
        # 1) bump our own run count in state
        global global_int
        global_int += 1
       
        print(f"🔁 Supervisor run #{global_int}")

        # 2) if we exceed max, terminate
        if global_int > self.max_runs:
            print("🛑 Max supervisor runs exceeded → ending process.")
            last_msg = state["messages"][-2]
            global_int = 0
            return Command(
                update={"messages": [last_msg]},
                goto="__end__"
            )

        # 3) convert AIMessage to HumanMessage if needed
        latest_message = state["messages"][-1]
        if isinstance(latest_message, AIMessage):
            logging.warning("Latest message is an AIMessage. Converting to HumanMessage to satisfy Gemini format.")
            latest_message = HumanMessage(content=latest_message.content)

        # 4) prepare messages and invoke model
        formatted_prompt = self.prompt.format()
        messages = [SystemMessage(content=formatted_prompt), latest_message]

        response = self.model.invoke(messages)
        parsed = self.parser.parse(response.content)

        # 5) return Command with updated state
        return Command(
            update={"messages": [latest_message]},
            goto="__end__" if parsed.next_node == "end" else parsed.next_node
        )
