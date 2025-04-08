# Databricks notebook source
packages: str = (
    "langgraph "
    "langchain "
    "databricks-langchain "
    "pydantic "
    "sql-formatter "
    "python-dotenv "
    "rich "
)

%pip install --quiet --upgrade {packages}
%restart_python

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version

pip_requirements: Sequence[str] = (
    f"langgraph=={version('langgraph')}",
    f"langchain=={version('langchain')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"pydantic=={version('pydantic')}",
    f"sql-formatter=={version('sql-formatter')}",
)

print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


# COMMAND ----------

from typing import (
  Annotated, 
  Sequence, 
  TypedDict, 
  Tuple, 
  List,
  Optional,
)
from enum import Enum

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
  BaseMessage, 
  SystemMessage, 
  HumanMessage
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.graph.graph import Graph, CompiledGraph, END
from langgraph.graph.state import StateGraph, CompiledStateGraph

from databricks_langchain import ChatDatabricks

from pydantic import BaseModel, Field

import sqlite3

import mlflow



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
   
    commented_ddl: List[str] = None
    table_name: str                # Testing Purposes
    columns: List[Tuple[str, str]] # Testing Purposes

    is_valid: bool
    validation_message: Optional[str]
    retry_count: int



system_message: BaseMessage = SystemMessage(
    content="""
    You are an AI assistant that generates SQL DDL statements. 
    Given a table name and column definitions, construct a CREATE TABLE statement.
    """
)

def llm_from(config: RunnableConfig) -> LanguageModelLike:
  endpoint: str = config.get("configurable", {}).get("endpoint", "databricks-meta-llama-3-3-70b-instruct")
  llm: LanguageModelLike = ChatDatabricks(endpoint=endpoint, temperature=0.1)
  return llm


@mlflow.trace
def generate_ddl(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    print(f"generate_ddl: state={state}")
    
    table_name: str = state["table_name"]
    columns: List[Tuple[str, str]] = state["columns"]

    columns_ddl: str = ",".join([f"{col_name} {col_type}" for col_name, col_type in columns])
    ddl: str = f"""
    CREATE TABLE {table_name} (
        {columns_ddl}
    );
    """.strip()
    return {"commented_ddl": ddl}
  

def max_retries_from(config: RunnableConfig) -> int:
  DEFAULT_MAX_RETRIES: int = 3
  return config.get("configurable", {}).get("max_retries", DEFAULT_MAX_RETRIES)


class FixSqlResponse(BaseModel):
    fixed_sql: str = Field(..., description="The corrected SQL statement")


@mlflow.trace
def validate_sql(state: AgentState, config: RunnableConfig) -> dict[str, str]:

    is_valid = True
    validation_message: Optional[str] = None
    try:
        with sqlite3.connect(":memory:") as conn: 
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute(state["commented_ddl"])
            conn.rollback()                 
    except sqlite3.Error as e:
        is_valid = False
        validation_message = str(e)
        
    return {
        "is_valid": is_valid,
        "validation_message": validation_message
    }


@mlflow.trace
def fix_sql(state: AgentState, config: RunnableConfig) -> dict:
    """
    If SQL validation fails, use LLM to fix the SQL based on error message.
    """
    print(f"fix_sql: Attempting to fix invalid SQL (retry {state['retry_count']})")
    
    llm: LanguageModelLike = llm_from(config)
    
    fix_message: BaseMessage = HumanMessage(
        content=f"""
        The SQL DDL I generated has an error:
        
        ```sql
        {state["commented_ddl"]}
        ```
        
        Error message: {state["validation_message"]}
        
        Please fix the SQL DDL for table {state["table_name"]} with columns {state["columns"]}.
        Return only the corrected SQL DDL statement with no additional explanation.
        """
    )

    messages: Sequence[BaseMessage] = [system_message, fix_message]
    llm_with_tools: LanguageModelLike = llm.with_structured_output(FixSqlResponse)
    response: FixSqlResponse = llm_with_tools.invoke(messages)
    
    retry_count = state["retry_count"] + 1
    
    return {
        "commented_ddl": response.fixed_sql,
        "retry_count": retry_count
    }


@mlflow.trace
def ddl_router(state: AgentState, config: RunnableConfig) -> str:
    """
    Determine next step based on SQL validation results and retry count.
    """
    max_retries: int = max_retries_from(config)
    
    if state["is_valid"]:
        return END
    elif state["retry_count"] >= max_retries:
        return "max_retries_reached"
    else:
        return "fix_sql"


@mlflow.trace
def handle_max_retries(state: AgentState, config: RunnableConfig) -> dict:

    print(f"handle_max_retries: Maximum retry attempts ({state['retry_count']}) reached")
    
    return {
        "validation_message": f"Failed to generate valid SQL after {state['retry_count']} attempts. Last error: {state['validation_message']}"
    }



graph: Graph = StateGraph(AgentState)

graph.add_node("generate_ddl", generate_ddl)
graph.add_node("validate_sql", validate_sql)
graph.add_node("fix_sql", fix_sql)
graph.add_node("handle_max_retries", handle_max_retries)
graph.add_conditional_edges(
  "validate_sql",
  ddl_router,
  {
      "fix_sql": "fix_sql",
      END: END, # Terminate state
      "max_retries_reached": "handle_max_retries"
  }
) 
graph.add_edge("fix_sql", "validate_sql")  # Revalidate after fixing
graph.add_edge("handle_max_retries", END)  # Terminal state


graph.add_edge("generate_ddl", "validate_sql")

graph.set_entry_point("generate_ddl")
graph.set_finish_point("generate_ddl")


app: CompiledGraph = graph.compile()


# COMMAND ----------

from IPython.display import Image, display

display(Image(app.get_graph(xray=True).draw_mermaid_png())) 

# COMMAND ----------

from langgraph.pregel.io import AddableValuesDict


config = {
  "configurable": {
    "thread_id": "1",
    "endpoint": "databricks-meta-llama-3-3-70b-instruct", # Configure the LLM endpoint
    "max_retries": 3,  # Number of retries to attempt to generate valid SQL
  }
}

initial_state: AgentState = AgentState(
    messages=[],
    table_name="employees",
    columns=[
      ("id", "INT PRIY KEY"), 
      ("name", "VARCHAR(100)"), 
      ("age", "IN")                   # Invalid SQL
    ],
    retry_count=0
)


response: AddableValuesDict = app.invoke(input=initial_state, config=config)
response

# COMMAND ----------

from rich import print
from sql_formatter.core import format_sql


ddl: str = format_sql(response["commented_ddl"])
print(ddl)
