from text2bpmn.graphs import four_agent_graph, two_agent_graph, baseline
import json
from text2bpmn.models import OpenAILLM, GeminiLLM
import text2bpmn.config as config
from langchain_core.messages import convert_to_messages
from langchain_community.callbacks import get_openai_callback
from text2bpmn.utils import render_BPMN  
from langsmith import traceable
import argparse


GRAPH_MAP = {
    "base_line": baseline.build_graph,
    "two_agent_graph": two_agent_graph.build_graph,
    "four_agent_graph": four_agent_graph.build_graph
}

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


@traceable
@traceable
def main():
    parser = argparse.ArgumentParser(description="Transform a natural language description to BPMN diagram.")
    parser.add_argument("-g", "--graph", required=True, help="name of the graph to run. Chose from {'base_line', 'two_agent_graph', 'four_agent_graph', 'all'}")
    parser.add_argument("-m", "--model", help="Model supplier. Chose from {'google', 'openAI'}.")
    parser.add_argument("-i", "--input", help="Path to the input data set.")
    
    args = parser.parse_args()

    # Set model
    if args.model == "google":
        config.set_model(GeminiLLM(model="gemini-2.5-pro-preview-05-06", temperature=0))
    elif args.model == "openAI":
        config.set_model(OpenAILLM(model="gpt-4.1-mini",temperature=0))
    else:
        # Set fall back model:
        config.set_model(OpenAILLM(model="gpt-4.1-mini",temperature=0))

    input_path = args.input
    
    
    if args.graph == "all":
        for graph_name, graph_method in GRAPH_MAP.items():
            with open(input_path, 'r') as file:
                data = json.load(file)
            modified_lines = []
            for i, item in enumerate(data):
                print(f"Processing process description {i+1} of {len(data)}")

                input_message = {"role": "user", "content": item["text"]}
                print(f"Input message: {input_message}")

                try:
                    graph = graph_method()

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
                    output_path = f"data/bpmn/{graph_name}_{data[i]['id']}.bpmn"
                    with open(output_path, "w") as f:
                        f.write(final_message + "\n")
                
                    print(f"Results written to {output_path}")
                    render_BPMN(output_path, f"data/img/{graph_name}_{data[i]['id']}.png")
                except Exception as e:    
                    print(f"❌ Exception occurred while processing item {i+1} (ID: {item.get('id', 'unknown')}): {e}")
                    continue
    else:
        graph_name = args.graph
        graph_method = GRAPH_MAP[args.graph]

        with open(input_path, 'r') as file:
            data = json.load(file)
        modified_lines = []
        for i, item in enumerate(data):
            print(f"Processing process description {i+1} of {len(data)}")

            input_message = {"role": "user", "content": item["text"]}
            print(f"Input message: {input_message}")

            try:
                graph = graph_method()

                final_chunk = None
                with get_openai_callback() as cb:
                    for chunk in graph.stream({"messages": [input_message]}):
                        pretty_print_messages(chunk, last_message=True)
                        final_chunk = chunk

                print(f"[TOKENS] Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens}, Total: {cb.total_tokens}")
            
                if args.graph == "four_agent_graph":
                    final_messages = final_chunk["supervisor"]["messages"]
                elif args.graph == "base_line":
                    final_messages = final_chunk["baseline"]["messages"]
                elif args.graph == "two_agent_graph":
                    final_messages = final_chunk["create_xml"]["messages"]
                else:
                    raise Exception("Not specified graphed used. Specify graph and corresponding last edge in __main__.py")

                modified_lines.append(final_messages[-1].content)
            
                final_message = final_messages[-1].content
                # Write the modified lines back as plain text
                output_path = f"data/bpmn/{graph_name}_{data[i]['id']}.bpmn"
                with open(output_path, "w") as f:
                    f.write(final_message + "\n")
            
                print(f"Results written to {output_path}")
                try:
                    render_BPMN(output_path, f"data/img/{graph_name}_{data[i]['id']}.png")
                except Exception as e:
                    print(f"Exception accured druing redering of the XML. Please check if the graph returned a valid BPMN XML: {e}")
            except Exception as e:    
                print(f"❌ Exception occurred while processing item {i+1} (ID: {item.get('id', 'unknown')}): {e}")
                continue

if __name__ == "__main__":
    main()
