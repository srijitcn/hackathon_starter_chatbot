# Databricks notebook source
# MAGIC %run ./utils/init

# COMMAND ----------

import os

os.environ["BRAVE_API_KEY"] = dbutils.secrets.get("multi_agent","web_search_api_key")
os.environ["VECTOR_SEARCH_PAT"] = dbutils.secrets.get("multi_agent","pat")
os.environ["WORKSPACE_URL"] = db_host_url
os.environ["RAG_AGENT_CONFIG_FILE"] = "config/rag_agent_config.yaml"
os.environ["HELPER_AGENT_CONFIG_FILE"] = "config/helper_agent_config.yaml"
os.environ["GENIE_AGENT_CONFIG_FILE"] = "config/genie_agent_config.yaml"

# COMMAND ----------

import mlflow

from agents.rag_agent import rag_chain, rag_config
from agents.helper_agent import helper_chain, helper_config
from agents.genie_agent import genie_agent, genie_config

from pydantic import BaseModel
from typing import Literal, TypedDict, Annotated, List
from typing_extensions import TypedDict

import functools
import operator
from typing import Sequence, Annotated

from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage

from langchain_core.runnables import RunnableLambda

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent



# COMMAND ----------

class AgentState(TypedDict):
    input_messages: list[AnyMessage]
    context : Annotated[list[AnyMessage], operator.add]
    rag_output:str
    genie_output:str
    plan: str
    response: str
    genie_count: int
    max_genie_count: int

# COMMAND ----------

model = ChatDatabricks(endpoint="srijit-nair-openai-o1")


# COMMAND ----------

def invoke_rag_node(state:AgentState):
    input_messages = state["input_messages"]
    rag_output = rag_chain.invoke(input_messages)
    return {
        "rag_output": rag_output,
    }
