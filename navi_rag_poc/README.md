# Traditional Keyword Search 

## Prerequisites
Python +3.10 and Pipenv Installed 

## Setup
0. Ensure you are in the correct folder! 
1. Install Dependencies - `pipenv install`
2. Install Development Dependencies - `pipenv install --dev`
3. Install New Package - `pipenv install <package-name>`
4. Install New Dev Package - `pipenv install <package-name> --dev`
4. Run the app script - `pipenv run python app.py`

## Project Structure
```
.
├── src/              # Backend Code
├── app.py            # Backend Flask Application
├── Pipfile           # Project dependencies
├── Pipfile.lock      # Locked dependency versions
└── README.md         
```

## App Assumptions 
This app assumes there is an existing frontend that will call it's post request
This app assumes there is a Weaviate Vector Database with connected Embedding Model. 
