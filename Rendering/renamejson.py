import json

xml = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<semantic:definitions id="_1275940554887" targetNamespace="http://www.trisotech.com/definitions/_1275940554887" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:di="http://www.omg.org/spec/DD/20100524/DI" xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" xmlns:dc="http://www.omg.org/spec/DD/20100524/DC" xmlns:semantic="http://www.omg.org/spec/BPMN/20100524/MODEL">
    <semantic:message id="_1275940554905"/>
    <semantic:process isExecutable="false" id="_6-1">
        <semantic:laneSet id="ls_6-69">
            <semantic:lane name="Account Manager" id="_6-77">
                <semantic:flowNodeRef>_6-81</semantic:flowNodeRef>
                <semantic:flowNodeRef>_6-94</semantic:flowNodeRef>
                <semantic:flowNodeRef>_6-196</semantic:flowNodeRef>
                <semantic:flowNodeRef>_6-313</semantic:flowNodeRef>
                <semantic:flowNodeRef>_6-423</semantic:flowNodeRef>
            </semantic:lane>
            <semantic:lane name="1st level support" id="_6-79">
                <semantic:flowNodeRef>_6-333</semantic:flowNodeRef>
                <semantic:flowNodeRef>_6-263</semantic:flowNodeRef>
            </semantic:lane>
            <semantic:lane name="2nd level support" id="_6-61">
                <semantic:flowNodeRef>_6-63</semantic:flowNodeRef>
                <semantic:flowNodeRef>_6-289</semantic:flowNodeRef>
            </semantic:lane>
            <semantic:lane name="Software development" id="_6-287">
                <semantic:flowNodeRef>_6-190</semantic:flowNodeRef>
            </semantic:lane>
        </semantic:laneSet>
        <semantic:startEvent name="question received" id="_6-81">
            <semantic:outgoing>_6-1027</semantic:outgoing>
            <semantic:messageEventDefinition messageRef="_1275940554905"/>
        </semantic:startEvent>
        <semantic:task completionQuantity="1" isForCompensation="false" startQuantity="1" name="handle question" id="_6-94">
            <semantic:incoming>_6-1027</semantic:incoming>
            <semantic:outgoing>_6-1029</semantic:outgoing>
        </semantic:task>
        <semantic:exclusiveGateway gatewayDirection="Unspecified" name="can handle myself?" id="_6-196">
            <semantic:incoming>_6-1029</semantic:incoming>
            <semantic:outgoing>_6-1033</semantic:outgoing>
            <semantic:outgoing>_6-1039</semantic:outgoing>
        </semantic:exclusiveGateway>
        <semantic:task completionQuantity="1" isForCompensation="false" startQuantity="1" name="Explain solution" id="_6-313">
            <semantic:incoming>_6-415</semantic:incoming>
            <semantic:incoming>_6-417</semantic:incoming>
            <semantic:incoming>_6-419</semantic:incoming>
            <semantic:incoming>_6-1033</semantic:incoming>
            <semantic:outgoing>_6-437</semantic:outgoing>
        </semantic:task>
        <semantic:endEvent name="" id="_6-423">
            <semantic:incoming>_6-437</semantic:incoming>
        </semantic:endEvent>
        <semantic:task completionQuantity="1" isForCompensation="false" startQuantity="1" name="Handle 1st level issue" id="_6-333">
            <semantic:incoming>_6-1039</semantic:incoming>
            <semantic:outgoing>_6-1041</semantic:outgoing>
        </semantic:task>
        <semantic:exclusiveGateway gatewayDirection="Unspecified" name="Finished?" id="_6-263">
            <semantic:incoming>_6-1041</semantic:incoming>
            <semantic:outgoing>_6-285</semantic:outgoing>
            <semantic:outgoing>_6-415</semantic:outgoing>
        </semantic:exclusiveGateway>
        <semantic:task completionQuantity="1" isForCompensation="false" startQuantity="1" name="Handle 2nd level issue" id="_6-63">
            <semantic:incoming>_6-285</semantic:incoming>
            <semantic:outgoing>_6-421</semantic:outgoing>
        </semantic:task>
        <semantic:exclusiveGateway gatewayDirection="Unspecified" name="Unsure?" id="_6-289">
            <semantic:incoming>_6-421</semantic:incoming>
            <semantic:outgoing>_6-311</semantic:outgoing>
            <semantic:outgoing>_6-417</semantic:outgoing>
        </semantic:exclusiveGateway>
        <semantic:task completionQuantity="1" isForCompensation="false" startQuantity="1" name="Provide feedback" id="_6-190">
            <semantic:incoming>_6-311</semantic:incoming>
            <semantic:outgoing>_6-419</semantic:outgoing>
        </semantic:task>
        <semantic:sequenceFlow sourceRef="_6-263" targetRef="_6-63" name="no" id="_6-285"/>
        <semantic:sequenceFlow sourceRef="_6-289" targetRef="_6-190" name="Yes" id="_6-311"/>
        <semantic:sequenceFlow sourceRef="_6-263" targetRef="_6-313" name="Yes" id="_6-415"/>
        <semantic:sequenceFlow sourceRef="_6-289" targetRef="_6-313" name="No" id="_6-417"/>
        <semantic:sequenceFlow sourceRef="_6-190" targetRef="_6-313" name="" id="_6-419"/>
        <semantic:sequenceFlow sourceRef="_6-63" targetRef="_6-289" name="" id="_6-421"/>
        <semantic:sequenceFlow sourceRef="_6-313" targetRef="_6-423" name="" id="_6-437"/>
        <semantic:sequenceFlow sourceRef="_6-81" targetRef="_6-94" name="" id="_6-1027"/>
        <semantic:sequenceFlow sourceRef="_6-94" targetRef="_6-196" name="" id="_6-1029"/>
        <semantic:sequenceFlow sourceRef="_6-196" targetRef="_6-313" name="Yes" id="_6-1033"/>
        <semantic:sequenceFlow sourceRef="_6-196" targetRef="_6-333" name="No" id="_6-1039"/>
        <semantic:sequenceFlow sourceRef="_6-333" targetRef="_6-263" name="" id="_6-1041"/>
        <semantic:textAnnotation id="_6-1361">
            <semantic:text>Sometimes opinion of development is needed</semantic:text>
        </semantic:textAnnotation>
        <semantic:association associationDirection="None" sourceRef="_6-1361" targetRef="_6-289" id="_6-1364"/>
    </semantic:process>
    <semantic:collaboration id="C1275940555293">
        <semantic:participant name="VIP customer" id="_6-53"/>
        <semantic:participant name="Software Company" processRef="_6-1" id="_6-69"/>
        <semantic:messageFlow name="" sourceRef="_6-53" targetRef="_6-81" id="_6-1355"/>
        <semantic:messageFlow name="" sourceRef="_6-313" targetRef="_6-53" id="_6-1359"/>
    </semantic:collaboration>
    <bpmndi:BPMNDiagram documentation="" id="Trisotech.Visio-_6" name="Untitled Diagram" resolution="96.00000267028808">
        <bpmndi:BPMNPlane bpmnElement="C1275940555293">
            <bpmndi:BPMNShape bpmnElement="_6-53" isHorizontal="true" id="Trisotech.Visio__6-53">
                <dc:Bounds height="108.0" width="1035.0" x="9.0" y="24.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-69" isHorizontal="true" id="Trisotech.Visio__6-69">
                <dc:Bounds height="576.0" width="1035.0" x="9.0" y="180.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-77" isHorizontal="true" id="Trisotech.Visio__6__6-77">
                <dc:Bounds height="144.0" width="1005.0" x="39.0" y="180.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-79" isHorizontal="true" id="Trisotech.Visio__6__6-79">
                <dc:Bounds height="144.0" width="1005.0" x="39.0" y="324.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-61" isHorizontal="true" id="Trisotech.Visio__6__6-61">
                <dc:Bounds height="144.0" width="1005.0" x="39.0" y="468.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-287" isHorizontal="true" id="Trisotech.Visio__6__6-287">
                <dc:Bounds height="144.0" width="1005.0" x="39.0" y="612.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-81" id="Trisotech.Visio__6__6-81">
                <dc:Bounds height="30.0" width="30.0" x="76.0" y="237.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-94" id="Trisotech.Visio__6__6-94">
                <dc:Bounds height="68.0" width="83.0" x="138.0" y="218.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-196" isMarkerVisible="false" id="Trisotech.Visio__6__6-196">
                <dc:Bounds height="42.0" width="42.0" x="277.0" y="231.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-313" id="Trisotech.Visio__6__6-313">
                <dc:Bounds height="68.0" width="83.0" x="843.0" y="218.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-423" id="Trisotech.Visio__6__6-423">
                <dc:Bounds height="32.0" width="32.0" x="968.0" y="236.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-333" id="Trisotech.Visio__6__6-333">
                <dc:Bounds height="68.0" width="83.0" x="336.0" y="348.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-263" isMarkerVisible="false" id="Trisotech.Visio__6__6-263">
                <dc:Bounds height="42.0" width="42.0" x="453.0" y="361.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-63" id="Trisotech.Visio__6__6-63">
                <dc:Bounds height="68.0" width="83.0" x="530.0" y="510.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-289" isMarkerVisible="false" id="Trisotech.Visio__6__6-289">
                <dc:Bounds height="42.0" width="42.0" x="642.0" y="523.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-190" id="Trisotech.Visio__6__6-190">
                <dc:Bounds height="68.0" width="83.0" x="708.0" y="650.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNShape bpmnElement="_6-1361" id="Trisotech.Visio__6__6-1361">
                <dc:Bounds height="49.0" width="108.0" x="544.0" y="655.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNShape>
            <bpmndi:BPMNEdge bpmnElement="_6-311" id="Trisotech.Visio__6__6-311">
                <di:waypoint x="663.0" y="565.0"/>
                <di:waypoint x="663.0" y="684.0"/>
                <di:waypoint x="708.0" y="684.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1355" id="Trisotech.Visio__6__6-1355">
                <di:waypoint x="92.0" y="132.0"/>
                <di:waypoint x="92.0" y="237.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-421" id="Trisotech.Visio__6__6-421">
                <di:waypoint x="613.0" y="544.0"/>
                <di:waypoint x="642.0" y="544.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-285" id="Trisotech.Visio__6__6-285">
                <di:waypoint x="474.0" y="403.0"/>
                <di:waypoint x="474.0" y="544.0"/>
                <di:waypoint x="530.0" y="544.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-417" id="Trisotech.Visio__6__6-417">
                <di:waypoint x="684.0" y="544.0"/>
                <di:waypoint x="726.0" y="544.0"/>
                <di:waypoint x="726.0" y="252.0"/>
                <di:waypoint x="843.0" y="252.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1029" id="Trisotech.Visio__6__6-1029">
                <di:waypoint x="222.0" y="252.0"/>
                <di:waypoint x="277.0" y="252.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1359" id="Trisotech.Visio__6__6-1359">
                <di:waypoint x="885.0" y="218.0"/>
                <di:waypoint x="884.0" y="132.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1364" id="Trisotech.Visio__6__6-1364">
                <di:waypoint x="544.0" y="655.0"/>
                <di:waypoint x="653.0" y="554.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1039" id="Trisotech.Visio__6__6-1039">
                <di:waypoint x="298.0" y="273.0"/>
                <di:waypoint x="298.0" y="382.0"/>
                <di:waypoint x="336.0" y="382.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-415" id="Trisotech.Visio__6__6-415">
                <di:waypoint x="495.0" y="382.0"/>
                <di:waypoint x="537.0" y="382.0"/>
                <di:waypoint x="537.0" y="252.0"/>
                <di:waypoint x="843.0" y="252.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1027" id="Trisotech.Visio__6__6-1027">
                <di:waypoint x="107.0" y="252.0"/>
                <di:waypoint x="125.0" y="252.0"/>
                <di:waypoint x="138.0" y="252.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-419" id="Trisotech.Visio__6__6-419">
                <di:waypoint x="791.0" y="684.0"/>
                <di:waypoint x="809.0" y="684.0"/>
                <di:waypoint x="809.0" y="252.0"/>
                <di:waypoint x="843.0" y="252.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1033" id="Trisotech.Visio__6__6-1033">
                <di:waypoint x="318.0" y="252.0"/>
                <di:waypoint x="548.0" y="252.0"/>
                <di:waypoint x="843.0" y="252.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-437" id="Trisotech.Visio__6__6-437">
                <di:waypoint x="926.0" y="252.0"/>
                <di:waypoint x="944.0" y="252.0"/>
                <di:waypoint x="968.0" y="252.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
            <bpmndi:BPMNEdge bpmnElement="_6-1041" id="Trisotech.Visio__6__6-1041">
                <di:waypoint x="419.0" y="382.0"/>
                <di:waypoint x="437.0" y="382.0"/>
                <di:waypoint x="453.0" y="382.0"/>
                <bpmndi:BPMNLabel/>
            </bpmndi:BPMNEdge>
        </bpmndi:BPMNPlane>
    </bpmndi:BPMNDiagram>
</semantic:definitions>"""


# Dump with indentation:
json_output = json.dumps({"xml": xml}, indent=4)

with open("output.json", "w") as f:
    f.write(json_output)
