# Text2BPMN

Multi-Agentic System for the creation of BPMN models and visuaizations from process text descriptions

# Setup

1. Create and activate a virtual environment
2. Install python dependencies from the repository root with

```bash
pip install -r requirements
```
3. Install Java Script dependencies from the repository root with

```bash
npm install bpmn-to-image
```

4. Install text2bpmn package from the repository root with:
    
```bash
pip install -e .
```

5. See usage instruction:  [Environment Variables](#-environment-variables)
6. Set up git lfs for you global user account by running 

```
git lfs install
```
# Usage

The `text2bpmn` package can be used from the command line.

## Command Syntax

```bash
text2bpmn -g <GRAPH> -m <MODEL_PROVIDER> -i <INPUT_FILE>
```

| Argument        | Required | Description                                                                                       |
| --------------- | -------- | ------------------------------------------------------------------------------------------------- |
| `-g`, `--graph` | ✅        | Name of the graph strategy. Options: `base_line`, `two_agent_graph`, `four_agent_graph`, or `all` |
| `-m`, `--model` | ❌        | Language model provider. Options: `openAI`, `google`. Defaults to `openAI`.                       |
| `-i`, `--input` | ❌        | Path to a JSON input file. Each item must include a `text` and `id` field.                        |

## Example Usage

Run with the baseline graph using OpenAI:

```bash
text2bpmn -g base_line -m openAI -i data/test_cases/wu_wien_easy.json
```

Run all graph strategies using Google's Gemini model:

```bash
text2bpm -g all -m google -i data/test_cases/wu_wien_easy.json
```
## Output

For each process description in the input:

- A .bpmn file is written to: `data/bpmn/<graph_name>_<id>.bpmn`

-  A .png diagram is rendered to: `data/img/<graph_name>_<id>.png`

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

**if you use the Gemini models from the Google API, you need to add the following variabke:**
```
GOOGLE_API_KEY=<your_google_api_key>
```