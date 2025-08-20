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
from typing_extensions import Annotated
from IPython.display import Image, display

from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

from langchain_community.utilities import SQLDatabase

conn_string = os.getenv('ConnectionString')
db = SQLDatabase.from_uri(f"mssql+pyodbc:///?odbc_connect={conn_string}")
print(db.dialect)
print(db.get_usable_table_names())

from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

from langchain_ollama import ChatOllama
llm = ChatOllama(model='qwen3:14b', temperature=1)

from langchain_core.prompts import ChatPromptTemplate

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
- ALWAYS use TOP({top_k}) after SELECT clause for limiting results: SELECT TOP{top_k} * FROM table
- For pagination, use OFFSET and FETCH: ORDER BY column OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY

**DATE AND TIME FUNCTIONS:**
- Use YEAR(date_column) to extract year
- Use MONTH(date_column) to extract month  
- Use DAY(date_column) to extract day
- Use DATEPART(datepart, date) for specific parts: DATEPART(quarter, date_column)
- Use DATEDIFF(datepart, startdate, enddate) for date differences
- Use GETDATE() for current date/time (not NOW() or CURRENT_DATE)
- Use DATEADD(datepart, number, date) to add/subtract dates
- For date formatting use FORMAT(date, 'yyyy-MM-dd') or CONVERT functions

**STRING FUNCTIONS:**
- Use LEN(string) instead of LENGTH(string)
- Use CHARINDEX(substring, string) instead of INSTR() or LOCATE()
- Use SUBSTRING(string, start, length) for string extraction
- Use LTRIM() and RTRIM() for trimming spaces
- Use CONCAT() or + operator for string concatenation

**CONDITIONAL LOGIC:**
- Use CASE WHEN ... THEN ... ELSE ... END for conditional logic
- Use IIF(condition, true_value, false_value) for simple conditions
- Use ISNULL(column, default_value) instead of IFNULL() or COALESCE()

**AGGREGATE AND WINDOW FUNCTIONS:**
- Use ROW_NUMBER() OVER (ORDER BY column) for row numbering
- Use RANK() and DENSE_RANK() for ranking
- Always include ORDER BY with window functions
- Use PARTITION BY in window functions when grouping is needed

**JOIN AND SUBQUERY BEST PRACTICES:**
- Always use explicit JOIN syntax (INNER JOIN, LEFT JOIN, etc.)
- Use table aliases for better readability
- Join on appropriate relationship columns (foreign keys, common identifiers)
- Prefer EXISTS over IN for subqueries when checking existence

**DATA TYPE HANDLING:**
- Use CAST(value AS datatype) or CONVERT(datatype, value) for type conversion
- Handle FLOAT columns appropriately for quantities and values
- Use NVARCHAR for text comparisons

**PERFORMANCE OPTIMIZATION:**
- Always include WHERE clauses when possible to limit result sets
- Use indexed columns in WHERE and JOIN conditions
- Avoid SELECT * unless absolutely necessary
- Use appropriate JOIN types based on data relationships

**IDENTIFIER QUOTING:**
- ALWAYS use square brackets [column_name] for ALL column and table names
- This is CRITICAL for columns with spaces or special characters

**NULL HANDLING:**
- Use IS NULL and IS NOT NULL for null checks
- Never use = NULL or != NULL
- Use ISNULL() function to handle null values in calculations 

**STEP 6: VALIDATION CHECKLIST**
Before finalizing the query, verify:
✓ Correct table name(s) selected based on question context
✓ All column names are exact matches from the schema
✓ All identifiers use square brackets
✓ Appropriate WHERE conditions for filtering
✓ Proper JOIN conditions if multiple tables
✓ MSSQL-specific syntax used throughout
✓ Query answers the original question completely

**Available Tables and Schema:**
{table_info}

Now create a syntactically correct MSSQL query following this complete analysis process.
Only use the TOP({top_k}) clause if you need to limit the number of examples you retrieve, and only if the user does not specify a limit.
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


checkpoint = MemorySaver()

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile(checkpointer=checkpoint)

config = {"configurable": {"thread_id": "1"}}

for step in graph.stream(
    {"question": "Show me monthly sales in 2014?"},
    config,
    stream_mode="updates",):
    print(step)


