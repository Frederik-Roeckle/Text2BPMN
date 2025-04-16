# Text2BPMN

Multi-Agentic System for the creation of BPMN models and visuaizations from process text descriptions

# Setup
## Enviroment Variables

Create a .env file in the root directory of the project and add the following variables:

**If you use the mistral API, you need to add the following variable:**
```
MISTRAL_API_KEY=<y<our_mistral_api_key>
```

**if you use the openAI API, you need to add the following variable:**

```
OPENAI_API_KEY=<your_openai_api_key>
```

## Install Dependencies

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Install text2bpmn

To install the text2bpmn package in editable mode, run the following command:

```bash
pip install -e .
```

## Run the Project
### Configute the input data

The input data is a jsonl file with the following structure:
```json
{
    "text": "Your process description here"
}
```

The input data is located in the data/test_cases/example_test_case.jsonl file.

### Configute which graphs should be run

To configure which graphs should be run, you need to modify the GRAPH_MAP dictionary in the text2bpmn/__main__.py file.

'### Run the Project
To run the project install the text2bpmn package first and run the following command:

```bash
text2bpmn
```

The output will be saved in the data/test_cases/example_test_case_with_answers.jsonl file.