# Databricks notebook source
from langchain.chat_models import ChatDatabricks

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Union
import mlflow


db_host_name = spark.conf.get('spark.databricks.workspaceUrl')
db_host_url = f"https://{db_host_name}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Tools
# MAGIC
# MAGIC Tool consists of several components:
# MAGIC
# MAGIC * A python function that implements the logic for the tool
# MAGIC * `name` (`str`), is required and must be unique within a set of tools provided to an agent
# MAGIC * `description` (`str`), is optional but recommended, as it is used by an agent to determine tool use
# MAGIC * `args_schema` (`Pydantic BaseModel`), is optional but recommended, can be used to provide more information (e.g., few-shot examples) or validation for expected parameters.
# MAGIC
# MAGIC In this exercise, we will use LangChain library to create tools. There are several ways to create tool using LangChain. We will cover few semantics here

# COMMAND ----------

# MAGIC %md
# MAGIC ####Math Tool
# MAGIC Let us build a Math tool that can perform mathematical operations from natural language input.
# MAGIC
# MAGIC We will do it by subclassing the BaseTool class. This provides maximal control over the tool definition, but is a bit more work.

# COMMAND ----------

#define the input data class
class MathToolInput(BaseModel):
    question: str = Field(description="Sentence containing a math question")

#define the tool class
class MathTool(BaseTool):
    name : str = "MathTool"
    description : str = "Useful for performing mathematical operations given a text"
    args_schema : Type[BaseModel] = MathToolInput
    
    chat_model_endpoint_name:str = None
    chat_model: ChatDatabricks = None
    llm_prompt: ChatPromptTemplate = None
    llm_chain: LLMChain = None

    max_tokens:int = 200
    temperature : float =0.01

    prompt:str = "You have to convert the input text to an equivalent mathematical expression in Python programming language. \
        If you dont know, return NA \
        Only respond with a valid math expression and nothing else. Do not include explanation in the response. \
        Few examples are given below \
        Input: add 4 and 5 \
        Response:4+5 \
        Input: 2 raised to the power of 3 and then multiplied by 2 \
        Response: (2^3)*2 \
        Input: sum all numbers from 1 to 10 \
        Response:sum(range(1, 11)) \
        Input Text: {question}"

    def __init__(self, chat_model_endpoint_name : str):
        super().__init__()
        self.chat_model_endpoint_name = chat_model_endpoint_name
        self.chat_model = ChatDatabricks(endpoint=self.chat_model_endpoint_name, 
                                   max_tokens = self.max_tokens, 
                                   temperature=self.temperature)
        self.llm_prompt = ChatPromptTemplate.from_template(self.prompt)
        self.llm_chain = LLMChain(
          llm = self.chat_model,
          prompt = self.llm_prompt
        )
    
    #Implement the tool logic
    @mlflow.trace(name="MathTool_solve", span_type="func")
    def solve(self, question:str) -> str:         
        math_expr_response = self.llm_chain.run(question=question)
        print(f"Converted math expression is:>{math_expr_response}<")
        if not math_expr_response.startswith("NA"):
            return str(eval(math_expr_response))
        else: 
            raise Exception("Cannot formulate a math expression ")        
    
    #override the _run method so that it integrates well with orchestrator
    def _run(self, question:str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.solve(question)
    
    #override the _arun method so that orchestrator can run it asynchronously
    def _arun(self, question:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self.solve(question)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Web Search tool
# MAGIC
# MAGIC Let us create another tool to search Web for information. This time let's use the `from_function` method of `StructuredTool` class of LangChain api. 
# MAGIC

# COMMAND ----------

from langchain_community.tools import BraveSearch
import json 
from bs4 import BeautifulSoup

class WebSearchToolInput(BaseModel):
    question: str = Field(description="Sentence containing a general question for web search")

class WebSearchTool:
    name : str = "WebSearchTool"
    description : str = "Useful for performing general search on Internet and return a sigle string result"
    args_schema : Type[BaseModel] = WebSearchToolInput
    api_key_secret_scope:str = None
    api_key_secret_key:str = None
        
    def __init__(self, api_key_secret_scope:str, api_key_secret_key:str):
        super().__init__()
        self.api_key_secret_scope = api_key_secret_scope
        self.api_key_secret_key = api_key_secret_key

    @mlflow.trace(name="WebSearchTool_search", span_type="func")
    def search(self, question:str) -> str: 
        #####################################################################
        ##    PLEASE USE DATABRICKS SECRETS FOR PASSING SENSITIVE INFORMATION
        #Ideally we would be using secrets to set tokens and other sensitive information
        #For the sake of this demo, we will give you a short lived token 
        # searcher = BraveSearch.from_api_key(api_key=dbutils.secrets.get(
        #                                         scope = self.api_key_secret_scope,
        #                                         key = self.api_key_secret_key),
        #                                     search_kwargs={"count": 1})

        #Create a Brave AI Api key from https://api.search.brave.com/
        searcher = BraveSearch.from_api_key(api_key="BSAi72gapKd22Y6aX3_krqpv9Cx3QAr",                                            
                                            search_kwargs={"count": 1})
        

        html_text = json.loads(searcher.run(question))[0]["snippet"]
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    
    def get_tool(self):
        return StructuredTool.from_function(func=self.search,
                                            name=self.name,
                                            description=self.description,
                                            args_schema=self.args_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Today Tool
# MAGIC Lets us now create a simple tool, that returns current date. 
# MAGIC
# MAGIC For this purpose we will simply define a python function and decorate it with `@tool` decorator. This is the most simplest way to create a tool

# COMMAND ----------

from datetime import datetime

#Since a tool need to have an input, we will provide a dummy input even if we wont use it

@tool
@mlflow.trace(name="get_todays_date", span_type="func")
def get_todays_date(unnecessary:str):
  """This tool will return the current date in the format of Month, Day, Year"""
  return datetime.now().strftime("%B %d, %Y")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Agent and Agent Executor
# MAGIC
# MAGIC AgentExecutor allows agents to take actions based on their inputs, using tools to generate observations and iterate. It provides a flexible and customizable way to run an agent, allowing users to specify the tools and memory to be used.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####ReACT Framework
# MAGIC
# MAGIC The ReACT (Reasoning and Action) is a framework aimed at enhancing the capabilities of LLMs. It does this by merging two key functions: reasoning and action.
# MAGIC
# MAGIC Reasoning: At its core, the model leverages the advanced reasoning capabilities of LLMs, which can analyze complex information, draw inferences, and understand context. This enables the model to evaluate various situations and determine the best course of action based on available data.
# MAGIC
# MAGIC Action: In addition to reasoning, the ReACT model incorporates the ability to take concrete actions. This means that once the model has analyzed a situation, it can perform specific tasks or respond in a way that directly impacts the environment or user.
# MAGIC
# MAGIC Integration: By combining these two elements, the ReACT model can not only process and understand information but also act upon it. This leads to a more sophisticated system capable of managing complex interactions, such as responding to user queries, adjusting to changing circumstances, and keeping track of ongoing scenarios.
# MAGIC
# MAGIC Overall, the ReACT model represents a significant advancement in creating intelligent systems that can reason and act in a cohesive manner, making them more effective in real-world applications.
# MAGIC
# MAGIC This is achieved by prompting the model to decompose the question into smaller pieces and solve individually.
# MAGIC
# MAGIC #####Let's build a ReACT Agent using LangChain api

# COMMAND ----------

from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain import hub

#Get a repbuilt react prompt from langchain hub
prompt = hub.pull("hwchase17/react")

#lets use the llama 3.1 405b model for our Agent
agent_model_endpoint = "databricks-meta-llama-3-1-405b-instruct"

#instantiate the tools
math_tool_model_endpoint =  "databricks-meta-llama-3-1-405b-instruct"
math_tool = MathTool(chat_model_endpoint_name=math_tool_model_endpoint)

web_tool = WebSearchTool(api_key_secret_scope="<your_secret_scope>",
                            api_key_secret_key="<your_secret_key>"
                        ).get_tool()

#define the tools that is available to the agent
tools = [web_tool, math_tool, get_todays_date]

#define the model class for agent that uses the endpoint
agent_chat_model = ChatDatabricks(
            endpoint=agent_model_endpoint,
            max_tokens = 2000,
            temperature= 0.1)

#create a react agent
agent = create_react_agent(agent_chat_model,
                           tools,
                           prompt = prompt
        )

agent_executor = AgentExecutor(agent=agent, 
                                tools=tools,
                                handle_parsing_errors=True,
                                verbose=True,
                                max_iterations=10)

# COMMAND ----------

#lets create a chain to make our agent executor compatible with ChatRequest
#this makes it deployable to an endpoint and make it available in playground
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter

def extract_user_query_string(input_messages:[dict])->str:
    return {"input" : input_messages[-1]["content"]}

def output_extractor(agent_output:dict)->str:
  return agent_output["output"]

chain = (itemgetter("messages")
         | RunnableLambda(extract_user_query_string)
         | agent_executor
         | RunnableLambda(output_extractor) 
         | StrOutputParser()
)


# COMMAND ----------

mlflow.models.set_model(model=chain)
