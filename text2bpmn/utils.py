import json
import yaml
import importlib
import subprocess


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



def render_BPMN(bpmn_name,img_name):
    subprocess.run(["node", "text2bpmn/convert.js", bpmn_name, img_name],
                check=True)