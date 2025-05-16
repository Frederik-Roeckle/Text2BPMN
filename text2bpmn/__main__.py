from text2bpmn.graphs import baseline
from text2bpmn.graphs import react_with_supervisor
from text2bpmn.graphs import three_agent_graph


import json
#from models import MistralLLM
from text2bpmn.models import OpenAILLM, GeminiLLM
import text2bpmn.config as config
from langchain_core.messages import convert_to_messages
from langchain_community.callbacks import get_openai_callback
from text2bpmn.utils import render_BPMN  
from langsmith.wrappers import wrap_openai
from langsmith import traceable



# -----------------------------
# Pretty printing
# -----------------------------
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        if node_update is None or "messages" not in node_update:
            print(f"Update from node {node_name}: No messages.\n")
            continue

        update_label = f"Update from node {node_name}:\n"
        if is_subgraph:
            update_label = "\t" + update_label
        print(update_label)

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


GRAPH_MAP = {
    "base_line": baseline.build_graph,
    "react": react_with_supervisor.build_graph,
    "supervisor": three_agent_graph.build_graph
}

@traceable
def main():
    #config.set_model(OpenAILLM(model="gpt-4.1-mini",temperature=0))
    config.set_model(GeminiLLM(model="gemini-2.5-pro-preview-05-06", temperature=0))


    #input_path = "data/test_cases/example_test_case.jsonl"
    input_path = "data/test_cases/wu_wien.json"

    #output_path = "data/test_cases/example_test_case_with_answers.jsonl"
    
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
    print("Processing graph: three_agent_graph")    

    #
    
    with open(input_path, 'r') as file:
         data = json.load(file)
    modified_lines = []
            # Read the input lines
            #with open(input_path, "r") as f:
            #    file_content = f.read().strip()
    for i, item in enumerate(data):
        print(f"Processing process description {i+1} of {len(data)}")

        input_message = {"role": "user", "content": item["text"]}
        print(f"Input message: {input_message}")

        try:
            graph = GRAPH_MAP["supervisor"]()

            final_chunk = None
            with get_openai_callback() as cb:
                for chunk in graph.stream({"messages": [input_message]}):
                    pretty_print_messages(chunk, last_message=True)
                    final_chunk = chunk

            print(f"[TOKENS] Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens}, Total: {cb.total_tokens}")
        
            final_messages = final_chunk["supervisor"]["messages"]
            modified_lines.append(final_messages[-1].content)
        
            final_message = final_messages[-1].content
            # Write the modified lines back as plain text
            output_path = f"data/bpmn/three_agent_1shot{data[i]['id']}.bpmn"
            with open(output_path, "w") as f:
                f.write(final_message + "\n")
        
            print(f"Results written to {output_path}")
            render_BPMN(f"data/bpmn/three_agent_1shot{data[i]['id']}.bpmn", f"data/img/three_agent_1shot{data[i]['id']}.png")
        except Exception as e:    
            print(f"❌ Exception occurred while processing item {i+1} (ID: {item.get('id', 'unknown')}): {e}")
            continue
    # # # Process the input file with the base_line graph
    # print("Processing graph: base_line")    

    
    # modified_lines = []
    # # Read the input lines
    # with open(input_path, "r") as f:

    #     for line in f:
    #         line = line.strip()  # Remove any leading/trailing whitespace
    #         if not line:  # Skip empty lines
    #             continue
    #         result = GRAPH_MAP["base_line"]().invoke({"messages": line})
    #         modified_lines.append(result["messages"][-1].content)
    
    # Write the modified lines back as plain text
   #with open(output_path, "w") as f:
    #    for item in modified_lines:
    #       f.write(item + "\n")




if __name__ == "__main__":
    main()
