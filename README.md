# NLP2SQLAPP

# 🧠 Natural Language to SQL (RAG + Ollama + MySQL)

A Streamlit app that allows users to ask questions in natural language, and get SQL results using Ollama LLM, FAISS-based retrieval, and MySQL.

## Features

- **LLM-powered** SQL query generation (via Ollama's LLaMA 3.2 3B)
- **RAG pipeline** using FAISS + SentenceTransformer
- **MySQL database** support
- **Streamlit interface** for interactive usage

## Demo

Ask questions like:
- "List all employees in the HR department"
- "Show all orders placed in March 2024"
- "Find all products with price above 1000"

## Database Schema Setup

You must create a MySQL database `company_db` and the following tables:

### Create 4 tables 'department', `employees`, 'orders' and 'products'. 

Populate tables with sample data if needed for testing.


## Set Up Python Environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

## Run Ollama LLM

Make sure Ollama is downloaded and installed and the llama3.2:3b model is pulled:
--bash--
ollama run llama3.2:3b
