import os
import mlflow

from agents.helper_agent import helper_chain, helper_config
from agents.genie_agent import genie_agent, genie_config
from agents.rag_agent import rag_chain, rag_config

from pydantic import BaseModel
from typing import Literal
from typing_extensions import TypedDict

import functools
import operator
from typing import Sequence, Annotated

from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from databricks_langchain.genie import GenieAgent

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

import logging

def log_print(msg):
    logging.warning(f"=====> {msg}")

#this config file will be used for dev and test
#when the model is logged, the config file will be overwritten
multi_agent_config = mlflow.models.ModelConfig(development_config=os.environ["MULTI_AGENT_CONFIG_FILE"])

class AgentState(TypedDict):
    question:str
    next_agent: str
    next_question: str
    agent_responses : Annotated[list[str], operator.add] 
    response: str
    num_attempts: int
    max_attempts: int

multi_agent_llm_config = multi_agent_config.get("multi_agent_llm_config")

multi_agent_llm = ChatDatabricks(
    endpoint=multi_agent_llm_config.get("llm_endpoint_name"),
    extra_params=multi_agent_llm_config.get("llm_parameters"),
)

class MultiAgent:
  
  PLAN_PROMPT = """
  Given the question and information given below, find the next agent to invoke from the list of agents.
  If all information is available and no more agents need to be invoked, you MUST respond with just the word DONE.
  You can break the question into smaller sub-questions and identify the next step.
  You MUST use only the below information, nothing else. Don't assume anything.
  * Question: {question}

  * Information
  {past_agent_responses_from_state}
  
  You can only pick one of the following agents:
  * List of Agents:
  {agent_names_for_prompt}
  
  Only respond with the agent_name and question in the given output format.
  * Output format: 
  agent_name:question
  
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
    return "\n".join([f"- {agent_name}: {agent_details['usage']}" 
                      for agent_name, agent_details in self.available_agents.items()])

  def get_past_agent_responses_from_state(self,past_responses : [str]) -> str:
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
    return state["next_agent"] == "DONE"
  
  def is_max_tries_exceeded(self, state:AgentState) -> bool:
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
    log_print(f"RESPONSE=>> {response}")
    
    return {
      "agent_responses" : [ f"{next_question} :{response['messages'][-1].content.strip()}" ]
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
        

def get_final_message(resp):
    log_print(resp)
    return resp.get('response', 'No response provided by the multi agent.')

def convert_chatcompletion_to_invoke_format(input_data):
    """
    Convert an input in ChatCompletion format to the format expected by
    multi_agent.graph.

    Expected ChatCompletion input format:
      {
          "messages": [
              {"role": "user", "content": "How many covid trials was completed in capital of france?"},
              ...  # optionally, more messages
          ],
          "num_attempts": <int>,
          "max_attempts": <int>
      }

    Converted payload for multi_agent.graph:
      {
          "question": "How many covid trials was completed in capital of france?",
          "num_attempts": <int>,
          "max_attempts": <int>
      }
    """
    payload = []
    user_question = None
    if (isinstance(input_data, dict)):
      payload = input_data.get("messages", [])
    elif (isinstance(input_data, list)):
      payload = input_data
    elif (isinstance(input_data, HumanMessage)):
      payload = [input_data]

    for msg in payload:
      if isinstance(msg, HumanMessage):
        user_question = msg.content
        break
      elif isinstance(msg, dict) and msg["role"]=="user":
        user_question = msg.get("content")
        break
  
    if not user_question:
        raise ValueError("No user message found in input messages.")
    
    return {
      "question": user_question,
      "num_attempts": input_data.get("num_attempts", 0),
      "max_attempts": input_data.get("max_attempts", 3)
    }

available_agents = {
  "covid_rag_agent" : {
    "chain": rag_chain,
    "usage": "Can be used for questions related to names and descriptions of publications about covid trials."
    },
  "covid_genie_agent" : {
    "chain": genie_agent,
    "usage": "Can be used for questions specific to statistics and information about COVID clinical trials."
    },
  "helper_agent" : {
    "chain": helper_chain,
    "usage": "Can be used for questions related to finding general information, current date, for performing web search and executing mathematical calculations"
    }
}

multi_agent = MultiAgent(available_agents, multi_agent_llm)

# parse the output from the graph to get the final message, and then format into ChatCompletions
graph_with_parser = RunnableLambda(convert_chatcompletion_to_invoke_format) | multi_agent.graph | RunnableLambda(get_final_message)

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=graph_with_parser)
