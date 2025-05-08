from graphs import baseline,react,two_agent_graph
from utils import render_BPMN
import json
#from models import MistralLLM
from models import OpenAILLM
from models import GeminiLLM
import config

GRAPH_MAP = {
    "base_line": baseline.build_graph,
    "two_agent_graph": two_agent_graph.build_graph,
    "react": react.build_graph
}


def main():
    #config.set_model(OpenAILLM(model="gpt-4.1-mini",temperature=0))
    config.set_model(GeminiLLM(model="gemini-2.5-pro-preview-05-06", temperature=0))

    #input_path = "data/test_cases/example_test_case.jsonl"
    input_path = "data/test_cases/wu_wien.json"

    #output_path = "data/test_cases/example_test_case_with_answers.jsonl"
    #output_path = "data/bpmn/baseline.bpmn"
    
    # TODO: Define logic that the output is appended to the original file

    
    # Uncomment the following lines to process the input file with each graph

    # for graph_name, graph in GRAPH_MAP.items():
    #     print(f"Processing graph: {graph_name}")
    #     modified_lines = []
    #     # Read the input lines
    #     with open(input_path, "r") as f:
    #         for line in f:
    #             as_dict = json.loads(line)
    #             result = graph().invoke({"messages": as_dict["text"]})
    #             as_dict[f"result_{graph_name}"] = result["messages"][-1].content
    #             modified_lines.append(as_dict)

    #     # Write the modified lines back
    #     with open(output_path, "w") as f:
    #         for item in modified_lines:
    #             f.write(json.dumps(item) + "\n")

    

    # Process the input file with the base_line graph
    print("Processing graph: baseline")    

    with open(input_path, 'r') as file:
        data = json.load(file)
    modified_lines = []
    # Read the input lines
    #with open(input_path, "r") as f:
    #    file_content = f.read().strip()
    for i in range(len(data)):
        print(f"Processing process description {i+1} of {len(data)}")
        result = GRAPH_MAP["base_line"]().invoke({"messages": data[i]['text']})
        final_message = result["messages"][-1].content
    # Write the modified lines back as plain text
        output_path = f"data/bpmn/baseline_{i}.bpmn"
        with open(output_path, "w") as f:
            f.write(final_message + "\n")
    
        print(f"Results written to {output_path}")
        render_BPMN(f"data/bpmn/baseline_{i}.bpmn",f"data/img/{i}.png")


if __name__ == "__main__":
    main()
