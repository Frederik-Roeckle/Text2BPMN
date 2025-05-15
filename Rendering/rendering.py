import subprocess

def render_BPMN(bpmn_name,img_name):
    subprocess.run(["node", "Rendering/convert.js", bpmn_name, img_name],
                check=True)

render_BPMN("data/bpmn/three_agent_6.bpmn", "data/img/three_agent_6.png")