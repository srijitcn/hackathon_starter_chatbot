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

#from agents.rag_agent import rag_chain, rag_config
from agents.helper_agent import helper_chain, helper_config
from agents.genie_agent import genie_chain, genie_config

from pydantic import BaseModel
from typing import Literal, TypedDict, Annotated, List
from typing_extensions import TypedDict

import functools
import operator
from typing import Sequence, Annotated

from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
import logging


def log_print(msg):
    logging.warning(f"=====> {msg}")


# COMMAND ----------

class AgentState(TypedDict):
    question:str
    next_agent: str
    next_question: str
    agent_responses : Annotated[list[str], operator.add] 
    response: str
    num_attempts: int
    max_attempts: int 


# COMMAND ----------

class MultiAgent:
  
  PLAN_PROMPT = """
  Given the question and information given below, find the next agent to invoke from the list of agents.
  You MUST use only the below information, nothing else. Dont assume anything.
  ##Information
  {past_agent_responses_from_state}

  You must use only the agents listed below to find any extra information that is needed. 
  You have the following agents you can use:

  ##List of Agents:
  {agent_names_for_prompt}
  
  Only respond with the agent_name and question in the given output format.
  ##Output format: 
  agent_name:question

  If the result is ready and no more action is needed respond with just the word DONE
  ##Question: {question}
  """

  FINAL_MESSAGE_PROMPT = """
  Provide a final answer to the user explaining the answer in a profesional way. 
  You must use only the below information to answer the question, nothing else.
  If there is an error_response, send just the error response back to the user.
  You MUST use only the below information to answer the question, nothing else:
  ##Information
  {past_agent_responses_from_state}
  ##Question: {question}
  """
  
  def __init__(self, available_agents:dict, model:ChatDatabricks):
    self.available_agents = available_agents
    self.model = model
    self.build_state_graph()

  def build_state_graph(self):
    log_print("Building State Graph")
    builder = StateGraph(AgentState)
    builder.add_node("find_next_action", self.find_next_action)    
    builder.add_node("take_action", self.take_action)
    builder.add_node("set_max_tries_exceeded_message", self.set_max_tries_exceeded_message)
    builder.add_node("get_user_response", self.get_user_response)

    builder.set_entry_point("find_next_action")
    builder.add_conditional_edges("find_next_action",
                                  self.decide_next_step,
                                  {
                                    "USER_RESPONSE" : "get_user_response",
                                    "MAX_TRIES" : "set_max_tries_exceeded_message",
                                    "NEXT_ACTION" : "take_action" 
                                  })
    
    builder.add_edge("take_action", "find_next_action")
    builder.add_edge("set_max_tries_exceeded_message","get_user_response")
    builder.add_edge("get_user_response", END)
    self.graph = builder.compile()    

  def format_agent_names_for_prompt(self) -> str:    
    log_print("format_agent_names_for_prompt")
    return "\n".join([f"- {agent_name}: {agent_details['usage']}" 
                      for agent_name, agent_details in self.available_agents.items()])

  def get_past_agent_responses_from_state(self,past_responses : [str]) -> str:
    log_print("get_past_agent_responses_from_state")
    if past_responses is None or len(past_responses) == 0:
      return " - None"
    else:
      return "\n".join([ f" - {agent_response}"  for agent_response in past_responses ])
  
  def find_next_action(self, state:AgentState) -> dict:
    log_print("------------------------------------")
    log_print("find_next_action")
    log_print(f"Past responses: {state['agent_responses']}")

    question = state["question"]
    
    prompt = self.PLAN_PROMPT.format(
      agent_names_for_prompt=self.format_agent_names_for_prompt(),
      past_agent_responses_from_state=self.get_past_agent_responses_from_state(state["agent_responses"]),
      question=question
    )    
    response = self.model.invoke(prompt).content.strip().split(":")
    return {
      "next_agent": response[0].strip(),
      "next_question": response[1].strip() if len(response) > 1 else None,
      "num_attempts" : state["num_attempts"] + 1
    }
  
  def is_actions_complete(self, state:AgentState) -> bool:
    log_print("is_actions_complete")
    return state["next_agent"] == "DONE"
  
  def is_max_tries_exceeded(self, state:AgentState) -> bool:
    log_print("is_max_tries_exceeded")
    return state["num_attempts"] >= state["max_attempts"]
  
  def decide_next_step(self, state:AgentState) -> str :
    log_print("decide_next_step")

    next = ""
    if self.is_actions_complete(state):
      next = "USER_RESPONSE"
    elif self.is_max_tries_exceeded(state):
      next = "MAX_TRIES"
    else:
      next = "NEXT_ACTION"

    log_print(f"next:{next}")
    return next

  def set_max_tries_exceeded_message(self, state:AgentState) -> dict:
    log_print("set_max_tries_exceeded_message")
    return {
      "agent_responses" : [ f"- error_response: Number of attempts to find answer exceeded max attempts of {state['max_attempts']}" ]
    }

  def take_action(self, state:AgentState) -> dict:
    log_print("take_action")

    question = state["question"]
    next_agent = self.available_agents[state["next_agent"]]["chain"]
    next_question = state["next_question"] if state["next_question"] is not None else question

    log_print(f"next_agent:{state['next_agent']}")
    log_print(f"next_question:{next_question}")

    response = next_agent.invoke({"messages":[HumanMessage(content=next_question)] })
    print("RESPONSE=>> ",response)
    return {
      "agent_responses" : [ f"- {next_question} :{response['messages'][-1].content.strip()}" ]
    }

  def get_user_response(self, state:AgentState) -> AIMessage:
    log_print("get_user_response")
    prompt = self.FINAL_MESSAGE_PROMPT.format(
      past_agent_responses_from_state=self.get_past_agent_responses_from_state(state["agent_responses"]),
      question=state["question"]
    )    
    response = self.model.invoke(prompt).content
    return {
      "response": response
    }

  


# COMMAND ----------

model = ChatDatabricks(endpoint="srijit_nair_openai")

available_agents = {
  #"covid_rag_agent" : {
  #  "chain": covid_rag_chain,
  #  "usage": "Can be used for questions related to names and descriptions of publications on covid trials."
  #  },
  "covid_genie_agent" : {
    "chain": genie_chain,
    "usage": "Can be used for questions specific to statistics and information about COVID clinical trials."
    },
  "helper_agent" : {
    "chain": helper_chain,
    "usage": "Can be used for questions related to finding general information, current date, for performing web search and executing mathematical calculations"
    }
}

# COMMAND ----------

multi_agent = MultiAgent(available_agents, model)
multi_agent.graph.invoke({"question": "How many covid trials happened last year?",
                          "num_attempts": 0,
                          "max_attempts": 10
                          })


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

genie_chain.invoke({"messages":[{"content": "How many covid trials completed after last comet seen", "role": "user"}] })

# COMMAND ----------

md_str = "|    |   completed_trials_2024 |\n|---:|------------------------:|\n|  0 |                      84 |"

md_str.count("|")

# COMMAND ----------

import csv
import json

def genie_markdown_tbl_to_json(markdown_table: str) -> str :
  lines = markdown_table.split("\n")
  dict_reader = csv.DictReader(lines, delimiter="|")
  data = []
  # skip first row, i.e. the row between the header and data
  for row in list(dict_reader)[1:]:
    #strip spaces and ignore columns without name
    r = {k.strip(): v.strip() for k, v in row.items() if k.strip() != ""}
    data.append(r)

  return (json.dumps(data).replace("\n",""))

print(genie_markdown_tbl_to_json("|    |   completed_trials_2024 |\n|---:|------------------------:|\n|  0 |                      84 |"))

# COMMAND ----------

helper_chain.invoke({"messages":[HumanMessage(content="what is the current year?")]})

# COMMAND ----------


