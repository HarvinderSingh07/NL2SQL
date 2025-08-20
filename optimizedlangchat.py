import streamlit as st
import pandas as pd
import langchain
import langchain_community
import langchain_core
import langchain_ollama
import pyodbc
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import re

load_dotenv()

st.set_page_config(
    page_title="SQL Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stChatMessage {
        background-color: inherit;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .sql-query {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    .answer-section {
        background-color: #fafafa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# Initialize database connection
@st.cache_resource
def init_database():
    try:
        conn_string = os.getenv('ConnectionString')
        if not conn_string:
            st.error("Database connection string not found in environment variables")
            return None
        db = SQLDatabase.from_uri(f"mssql+pyodbc:///?odbc_connect={conn_string}")
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def init_llm(model_name, temperature=0.1):
    try:
        if model_name.startswith('gpt'):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif model_name.startswith('claude'):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, temperature=temperature)
        else:
            # Default to Ollama
            return ChatOllama(model=model_name, temperature=temperature)
    except Exception as e:
        st.error(f"Failed to initialize {model_name}: {str(e)}")
        return ChatOllama(model='qwen3:14b', temperature=temperature)

def get_query_prompt_template():
    system_message = """
You are an expert {dialect} database query generator with advanced reasoning capabilities. Before writing any SQL query, you MUST follow this systematic approach:

**STEP 1: ANALYZE THE QUESTION AND IDENTIFY REQUIREMENTS**
- Read the user's question carefully and identify what information they're seeking
- Determine what type of data they need and any specific filters, conditions, or calculations required
- Note any aggregations, groupings, or sorting requirements

**STEP 2: INTELLIGENT TABLE SELECTION**
Based on the question context, choose the appropriate table(s):
- Analyze table names to understand their purpose and domain
- Match question keywords with table names or related concepts
- For multi-table queries → Identify relationships and JOIN appropriately

**STEP 3: SMART COLUMN IDENTIFICATION**
Carefully examine the available columns in each table and select the most appropriate ones:
- Match question keywords with column names (exact or semantic matching)
- Look for identifier columns (IDs, codes, references)
- Find descriptive columns (names, descriptions, titles)
- Identify quantitative columns (amounts, quantities, counts, values)
- Locate temporal columns (dates, timestamps, periods)
- Consider categorical columns (status, type, category)

**STEP 4: COLUMN NAME MATCHING LOGIC**
When searching for specific data:
- Pay attention to column names with spaces - they MUST be quoted with square brackets
- Be precise with column names - use exact spelling and casing as shown in table schema

**STEP 5: WRITE QUERY FOLLOWING {dialect} BEST PRACTICES**

**LIMIT AND PAGINATION RULES:**
- NEVER use LIMIT keyword (not supported in MSSQL)
- ALWAYS use TOP({top_k}) after SELECT clause for limiting results: SELECT TOP({top_k}) * FROM table
- For pagination, use OFFSET and FETCH: ORDER BY column OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY

**Available Tables and Schema:**
{table_info}

Now create a syntactically correct MSSQL query following this complete analysis process.
Only use the TOP({top_k}) clause if you need to limit the number of examples you retrieve, and only if the user does not specify a limit.
"""
    
    user_prompt = "Question: {input}"
    
    return ChatPromptTemplate([("system", system_message), ("user", user_prompt)])

def write_query(state: State, llm, db):
    """Generate SQL query to fetch information."""
    query_prompt_template = get_query_prompt_template()
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 20,
        "table_info": db.get_table_info(),
        "input": state["question"],
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State, db):
    """Execute SQL query."""
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(state["query"])
        return {"result": result}
    except Exception as e:
        return {"result": f"Error executing query: {str(e)}"}

def generate_answer(state: State, llm):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, provide a clear and concise answer to the user question. "
        "Focus on the key insights from the data.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def parse_sql_result_to_dataframe(result_str):
    """Parse SQL result string to pandas DataFrame."""
    try:
        # Try to extract data from the result string
        lines = result_str.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Look for table-like structure
        header_line = None
        data_lines = []
        
        for i, line in enumerate(lines):
            if '|' in line and not line.strip().startswith('|'):
                # This might be a header or data line
                if header_line is None:
                    header_line = line
                else:
                    data_lines.append(line)
        
        if header_line and data_lines:
            # Parse header
            headers = [col.strip() for col in header_line.split('|') if col.strip()]
            
            # Parse data
            data = []
            for line in data_lines:
                if '|' in line:
                    row = [col.strip() for col in line.split('|') if col.strip()]
                    if len(row) == len(headers):
                        data.append(row)
            
            if data:
                return pd.DataFrame(data, columns=headers)
    
    except Exception as e:
        st.error(f"Error parsing result: {str(e)}")
    
    return None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'db' not in st.session_state:
    st.session_state.db = init_database()

# Sidebar for model selection and settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    model_options = {
        "Qwen3 8B (Ollama)": "qwen3:8b",
        "Qwen2.5 7B (Ollama)": "qwen2.5:7b", 
        "Llama 3.1 8B (Ollama)": "llama3.1:8b",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
        "Claude 3 Sonnet": "claude-3-sonnet-20240229",
        "Claude 3 Haiku": "claude-3-haiku-20240307"
    }
    
    selected_model_name = st.selectbox(
        "🤖 Select LLM Model",
        options=list(model_options.keys()),
        index=0
    )
    
    selected_model = model_options[selected_model_name]
    
    # Temperature setting
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower values make responses more focused and deterministic"
    )
    
    # Display database info
    if st.session_state.db:
        st.success("✅ Database Connected")
        with st.expander("📊 Database Info"):
            st.write(f"**Dialect:** {st.session_state.db.dialect}")
            tables = st.session_state.db.get_usable_table_names()
            st.write(f"**Tables:** {', '.join(tables)}")
    else:
        st.error("❌ Database Not Connected")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("💬 SQL Database Chatbot")
st.markdown("Ask questions about your database and get SQL queries with results!")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {message['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Assistant** (using {message.get('model', 'Unknown Model')}):")
            
            # Display SQL Query
            if 'query' in message:
                st.markdown("**🔍 Generated SQL Query:**")
                st.code(message['query'], language='sql')
            
            # Display Results
            if 'result' in message:
                st.markdown("**📊 Query Results:**")
                
                # Try to parse as DataFrame
                df = parse_sql_result_to_dataframe(message['result'])
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    # Show raw result if parsing fails
                    with st.expander("Raw Results"):
                        st.text(message['result'])
            
            # Display Answer
            if 'answer' in message:
                st.markdown("**💡 Answer:**")
                st.markdown(message['answer'])

# Chat input
user_question = st.chat_input("Ask a question about your database...")

if user_question and st.session_state.db:
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": user_question
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_question}")
    
    # Process with LLM
    with st.chat_message("assistant"):
        with st.spinner(f"Processing with {selected_model_name}..."):
            try:
                # Initialize LLM
                llm = init_llm(selected_model, temperature)
                
                # Create state
                state = State(
                    question=user_question,
                    query="",
                    result="",
                    answer=""
                )
                
                # Step 1: Generate query
                st.markdown("**🔍 Generated SQL Query:**")
                query_result = write_query(state, llm, st.session_state.db)
                state.update(query_result)
                st.code(state["query"], language='sql')
                
                # Step 2: Execute query
                st.markdown("**📊 Query Results:**")
                exec_result = execute_query(state, st.session_state.db)
                state.update(exec_result)
                
                # Try to display as DataFrame
                df = parse_sql_result_to_dataframe(state["result"])
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    with st.expander("Raw Results"):
                        st.text(state["result"])
                
                # Step 3: Generate answer
                st.markdown("**💡 Answer:**")
                answer_result = generate_answer(state, llm)
                state.update(answer_result)
                st.markdown(state["answer"])
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "model": selected_model_name,
                    "query": state["query"],
                    "result": state["result"],
                    "answer": state["answer"]
                })
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "model": selected_model_name,
                    "answer": f"Sorry, I encountered an error: {str(e)}"
                })

elif user_question and not st.session_state.db:
    st.error("Please check your database connection before asking questions.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "💡 Tip: Ask specific questions about your data for better results"
    "</div>", 
    unsafe_allow_html=True
)