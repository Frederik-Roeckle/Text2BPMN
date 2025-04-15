import argparse
from text2bpmn.graphs import simple_graph
import json

GRAPH_MAP = {
    "simple_graph": simple_graph.build_graph,
}

def main():
    parser = argparse.ArgumentParser(description="Run a specific graph.")
    parser.add_argument("graph", choices=GRAPH_MAP.keys(), help="The graph to run.")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(["simple_graph"])  # default to simple_graph if no argument is provided
        print("No graph specified. Defaulting to 'simple_graph'.")

    input_path = "data/test_cases/example_test_case.jsonl"
    output_path = "data/test_cases/example_test_case_with_answers.jsonl"

    modified_lines = []

    # Read the input lines
    with open(input_path, "r") as f:
        for line in f:
            as_dict = json.loads(line)
            graph = GRAPH_MAP[args.graph]()
            result = graph.invoke({"messages": as_dict["text"]})
            as_dict["result_llm"] = result["messages"][-1].content
            modified_lines.append(as_dict)

    # Write the modified lines back
    with open(output_path, "w") as f:
        for item in modified_lines:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
