import json
import yaml
import importlib
from langgraph.prebuilt import create_react_agent

MODEL_MODULE = "text2bpmn.models"

def load_data(path: str):
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".txt"):
        with open(path, "r") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format. Use .json, .yaml or .txt")

def get_model(model_name: str):
    class_name = model_name.capitalize() + "LLM"  
    module = importlib.import_module(MODEL_MODULE)
    cls = getattr(module, class_name)
    return cls()

def get_model_from_config(agent_config_path:str, agent_name:str, default_model:"mistral"):
    cfg = load_data(agent_config_path).get("agents", {})
    agent_cfg = cfg.get(agent_name, {})    
    model_name = agent_cfg.get("model", default_model)
    return get_model(model_name)



# Example usage
if __name__ == "__main__":
    supervisor = build_graph("configs/graphs.yaml", "react_agent_graph")
    print(supervisor)