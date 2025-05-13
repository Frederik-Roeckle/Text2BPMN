import subprocess

def render_BPMN(bpmn_name,img_name):
    subprocess.run(["node", "Rendering/convert.js", bpmn_name, img_name],
                check=True)

render_BPMN("data/bpmn/baseline_0.bpmn", "test_final1.png")