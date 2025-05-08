from text2bpmn.graphs import baseline, react, two_agent_graph, three_agent_graph
from text2bpmn.utils import render_BPMN
from text2bpmn.models import OpenAILLM, MistralLLM
from text2bpmn.config import set_model

GRAPH_MAP = {
    "base_line": baseline.build_graph,
    "react": react.build_graph,
    "two_agent_graph": two_agent_graph.build_graph,
    "three_agent_graph": three_agent_graph.build_graph
}

temp_selected_graph = "three_agent_graph"

def main():
    set_model(OpenAILLM(model="gpt-4o-mini",temperature=0))
    #set_model(MistralLLM())

    #input_path = "data/test_cases/example_test_case.jsonl"
    input_path = "data/test_cases/short_bpmn_process.txt"

    #output_path = "data/test_cases/example_test_case_with_answers.jsonl"
    output_path = "data/test_cases/baseline.bpmn"
    
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
    print("Processing graph: base_line")    

    
    modified_lines = []
    # Read the input lines
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()  # Remove any leading/trailing whitespace
            if not line:  # Skip empty lines
                continue
            result = GRAPH_MAP[temp_selected_graph]().invoke({"messages": line})
            modified_lines.append(result["messages"][-1].content)
    
    # Write the modified lines back as plain text
    with open(output_path, "w") as f:
        for item in modified_lines:
            f.write(item + "\n")
    
    print(f"Results written to {output_path}")
    render_BPMN("data/test_cases/baseline.bpmn","data/img/baseline.png")


if __name__ == "__main__":
    main()
