import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from rendering import render_BPMN

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_api_key,
)
messages = [
    SystemMessage(content="You are an expert process modeller, which creates BPMN 2.0 xml from structured process info. Return **only** valid BPMN 2.0 XML—no markdown fences, no commentary. Include all relevant elements of a bpmn, such as a pool"),
    HumanMessage(content="""{
  "process_name": "Request Approval Process",
  "pools": [
    {
      "name": "Organization",
      "lanes": [
        {
          "name": "User Lane",
          "elements": ["start_event_user_request"]
        },
        {
          "name": "Employee Lane",
          "elements": ["task_review_request", "exclusive_gateway_approval"]
        },
        {
          "name": "Manager Lane",
          "elements": ["task_final_approval"]
        }
      ]
    }
  ],
  "elements": [
    {
      "id": "start_event_user_request",
      "type": "StartEvent",
      "name": "User Submits Request"
    },
    {
      "id": "task_review_request",
      "type": "Task",
      "name": "Review Request"
    },
    {
      "id": "exclusive_gateway_approval",
      "type": "ExclusiveGateway",
      "name": "Request Approved?"
    },
    {
      "id": "task_final_approval",
      "type": "Task",
      "name": "Final Approval by Manager"
    },
    {
      "id": "end_event_rejected",
      "type": "EndEvent",
      "name": "Request Rejected"
    }
  ],
  "sequence_flows": [
    {
      "source": "start_event_user_request",
      "target": "task_review_request"
    },
    {
      "source": "task_review_request",
      "target": "exclusive_gateway_approval"
    },
    {
      "source": "exclusive_gateway_approval",
      "target": "task_final_approval",
      "condition": "Approved"
    },
    {
      "source": "exclusive_gateway_approval",
      "target": "end_event_rejected",
      "condition": "Not Approved"
    }
  ]
}""")
]
response_msg = llm.invoke(messages)
bpmn_xml = response_msg.content.strip()
if bpmn_xml.startswith("```"):
    # Some models still wrap code; strip it out.
    bpmn_xml = bpmn_xml.split("```")[1].strip()

file_path = "simple_test.bpmn"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(bpmn_xml)

render_BPMN("simple_test.bpmn", "simpletest.png")