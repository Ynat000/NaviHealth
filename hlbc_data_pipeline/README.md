# HLBC Data Pipeline

HLBC Data Pipeline is a modular Python pipeline for downloading, cleaning, and converting HealthLink BC (HLBC) web pages. It is set up with Pipenv for dependency and virtual environment management.

## Pipeline Goals

This pipeline will:
1. Crawl and save all HLBC file metadata as a timestamped CSV.
2. Download all HLBC HTML files listed in the latest crawl.
3. Clean the downloaded HTML files to remove unnecessary markup.
4. Convert the cleaned HTML files to Markdown format.

## Prerequisites

1. Python 3.10+ installed.
2. Pipenv installed.

## Setup

1. Install Pipenv (if needed):
   ```bash
   pip install pipenv
   ```
2. Install project dependencies and create the virtual environment:
   ```bash
   pipenv install
   ```

## Usage

Run a Python script inside the Pipenv environment:

```bash
pipenv run python main.py
```

