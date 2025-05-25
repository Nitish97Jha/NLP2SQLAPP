import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import re
import os

# Disable file watcher errors (fix for Windows and torch)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Database and model config
DB_URI = "mysql+mysqlconnector://username:password@localhost/company_db"  # Update with Your DataBase Username and password
OLLAMA_MODEL = "llama3.2:3b"

# Example queries (RAG context)
examples = [
    "List all employees in the HR department",
    "Show all orders placed in March 2024",
    "Get the average salary of employees",
    "Find all products with price above 1000",
    "Who placed the highest order?"
]

# Create FAISS index with example embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
example_embeddings = embedder.encode(examples, convert_to_numpy=True)
dim = example_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(example_embeddings)
id_to_text = {i: examples[i] for i in range(len(examples))}

def get_similar_examples(query, k=1):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, k)
    return [id_to_text[i] for i in indices[0]]

def clean_sql(raw_sql: str) -> str:
    """
    Cleans LLM output by removing markdown-style formatting like ```sql ... ```
    """
    return re.sub(r"```(?:sql)?\s*([\s\S]*?)\s*```", r"\1", raw_sql.strip(), flags=re.IGNORECASE)

# Streamlit UI setup
st.set_page_config(page_title="RAG + MySQL + LLM", page_icon="🧠")
st.title("🧠 Natural Language SQL Assistant")

user_query = st.text_input("Ask your question:")

if st.button("Search"):
    if not user_query:
        st.warning("Please enter your question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Connect to the database
                db = SQLDatabase.from_uri(DB_URI)

                schema_str = db.get_table_info()

                # Load the LLM
                llm = Ollama(model=OLLAMA_MODEL)

                # Retrieve similar example(s) for RAG
                similar_examples = get_similar_examples(user_query)
                context = "\n".join(similar_examples)
                final_input = f"{context}\n\n{user_query}"

                # Strict SQL-only prompt
                custom_prompt = PromptTemplate.from_template("""
You are a MySQL expert. Given a natural language question, generate a valid and executable MySQL query based on the given database.
⚠️ ONLY return the SQL statement.
🚫 DO NOT include:
- markdown (e.g., ```sql)
- comments
- explanations
- any text before or after

Schema: {schema}
                                                                   
Input: {input}
SQL:
""")
                final_prompt = custom_prompt.partial(schema=schema_str)

                # Build the SQLDatabaseChain
                chain = SQLDatabaseChain.from_llm(
                    llm=llm,
                    db=db,
                    prompt=final_prompt,
                    input_key="input", 
                    return_intermediate_steps=True,
                    verbose=True
                )

                # Call with correct variable
                result = chain.invoke({'input':final_input})
                sql_query = result['result']
                sql_query = clean_sql(sql_query)

                # Show generated SQL
                st.success("✅ SQL generated successfully.")
                st.subheader("🗃️ Generated SQL:")
                st.code(sql_query, language="sql")

                # Execute the SQL query
                engine = create_engine(DB_URI)
                with engine.connect() as conn:
                    rows = conn.execute(text(sql_query))
                    df = pd.DataFrame(rows.fetchall(), columns=rows.keys())

                # Display result
                st.subheader("📊 Query Result:")
                st.dataframe(df)

            except Exception as e:
                st.error(f"❌ Error: {e}")
