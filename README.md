# AI Search Tool for Monster Hunter Wilds
**Note: This project is still under development.**


## Overview

This small project is a search tool for Monster Hunter Wilds that uses AI to answer questions about the game. Source of information is the [Monster Hunter Wiki](https://monsterhunter.fandom.com/wiki/Monster_Hunter_Wiki).


**Note: This repository is intended for practice and experimentation**

## Features

- Search the Monster Hunter Wiki for information about the game.
- Use the LangChain framework and RAG technique  with OpenAI API to answer questions about the game.
- *TBD* Create a user interface to ask questions and get answers.

## Technologies

- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)

## Setup

1. Clone the repository
2. Install the dependencies (see requirements.txt, and [LangChain](https://python.langchain.com/docs/get_started/install))
```bash
pip install -r requirements.txt
```

3. Create a `.env` file
```txt
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<your_langsmith_api_key>"
LANGSMITH_PROJECT="<your_project_name>"
OPENAI_API_KEY="<your_openai_api_key>"
```

4. Retrieve the data from the Monster Hunter Wiki
    - Get the HTML pages from the Monster Hunter Wiki
```bash
python get_source.py
```
5. Run the script
- for one-time use:
```bash
python rag.py
```
- for interactive use:
```bash
python rag_chain.py
```
