import subprocess

def render_BPMN(bpmn_name,img_name):
    subprocess.run(["node", "convert.js", bpmn_name, img_name],
                check=True)
