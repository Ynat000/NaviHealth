# Traditional Keyword Search 

## Prerequisites
Python +3.10 and Pipenv Installed 

## Setup
0. Ensure you are in the correct folder! 
1. Install Dependencies - `pipenv install`
2. Install Development Dependencies - `pipenv install --dev`
3. Install New Package - `pipenv install <package-name>`
4. Install New Dev Package - `pipenv install <package-name> --dev`
4. Run the main script - `pipenv run python main.py`

## Project Structure
```
.
├── main.py           # Main entry point
├── Pipfile           # Project dependencies
├── Pipfile.lock      # Locked dependency versions
└── README.md         
```

## Ollama
Ollama is the easiest way to automate your work using open models, while keeping your data safe.
1. Install - `brew install ollama`
2. Pull Model - `ollama pull llama3.2:3b`