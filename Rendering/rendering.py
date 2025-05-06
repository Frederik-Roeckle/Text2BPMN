import subprocess

def render_BPMN(bpmn_name,img_name):
    subprocess.run(["node", "Rendering/convert.js", bpmn_name, img_name],
                check=True)

render_BPMN("Rendering/simple_test.bpmn", "test1.png")