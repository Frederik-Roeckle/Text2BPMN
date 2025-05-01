from text2bpmn.graphs import base_line
from text2bpmn.graphs import react
import json
from text2bpmn.models import MistralLLM
from text2bpmn import config

GRAPH_MAP = {
    "base_line": base_line.build_graph,
    "react": react.build_graph
}


def main():
    config.set_model(MistralLLM())

    input_path = "data/test_cases/example_test_case.jsonl"
    output_path = "data/test_cases/example_test_case_with_answers.jsonl"
    # TODO: Define logic that the output is appended to the original file

    for graph_name, graph in GRAPH_MAP.items():
        print(f"Processing graph: {graph_name}")
        modified_lines = []
        # Read the input lines
        with open(input_path, "r") as f:
            for line in f:
                as_dict = json.loads(line)
                result = graph().invoke({"messages": as_dict["text"]})
                as_dict[f"result_{graph_name}"] = result["messages"][-1].content
                modified_lines.append(as_dict)

        # Write the modified lines back
        with open(output_path, "w") as f:
            for item in modified_lines:
                f.write(json.dumps(item) + "\n")

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
