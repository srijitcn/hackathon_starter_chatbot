# Databricks notebook source
# MAGIC %pip install -q mlflow==2.16.2 langchain==0.3.0 langchain-community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.chat_models import ChatDatabricks

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Union
import mlflow

# Method to reuse local session for token, this is not a production approach for managing authentication
db_host_name = spark.conf.get('spark.databricks.workspaceUrl')
db_host_url = f"https://{db_host_name}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Tools

# COMMAND ----------

# MAGIC %md
# MAGIC ######Math Tool

# COMMAND ----------

class MathToolInput(BaseModel):
    question: str = Field(description="Sentence containing a math question")

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
        self.llm_chain = LLMChain(llm = self.chat_model,
                                  prompt = self.llm_prompt)
        # TODO test and adopt langchain prefered convention
        #self.llm_chain = self.llm_prompt | self.chat_model
    
    @mlflow.trace(name="solve", span_type="func")
    def solve(self, question:str) -> str:         
        math_expr_response = self.llm_chain.run(question=question)
        print(f"Converted math expression is:>{math_expr_response}<")
        if not math_expr_response.startswith("NA"):
            return str(eval(math_expr_response))
        else: 
            raise Exception("Cannot formulate a math expression ")        
    
    def _run(self, question:str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.solve(question)
    
    def _arun(self, question:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self.solve(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ######Test Math Tool

# COMMAND ----------

model_endpoint =  "databricks-meta-llama-3-1-405b-instruct"

math_tool = MathTool(chat_model_endpoint_name=model_endpoint)
print(math_tool.run("what added to 8 is 15"))
print("\n---")
print(math_tool.run("what is 2.1 ^ 3.1"))


# COMMAND ----------

# MAGIC %md
# MAGIC ###### Web Search tool

# COMMAND ----------

from langchain_community.tools import BraveSearch
import json 
from bs4 import BeautifulSoup

class WebSearchToolInput(BaseModel):
    question: str = Field(description="Sentence containing a general question for web search")

class WebSearchTool(BaseTool):
    name : str = "WebSearchTool"
    description : str = "Useful for performing general search on Internet and return a sigle string result"
    args_schema : Type[BaseModel] = WebSearchToolInput
    api_key_secret_scope:str = None
    api_key_secret_key:str = None
        
    def __init__(self, api_key_secret_scope:str, api_key_secret_key:str):
        super().__init__()
        self.api_key_secret_scope = api_key_secret_scope
        self.api_key_secret_key = api_key_secret_key

    @mlflow.trace(name="search", span_type="func")
    def search(self, question:str) -> str: 
        searcher = BraveSearch.from_api_key(api_key=dbutils.secrets.get(
                                                scope = self.api_key_secret_scope,
                                                key = self.api_key_secret_key),
                                            search_kwargs={"count": 1})
        
        html_text = json.loads(searcher.run(question))[0]["snippet"]
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    
    def _run(self, question:str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.search(question)
    
    def _arun(self, question:str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self.search(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ######Test Web Search Tool

# COMMAND ----------


web_tool = WebSearchTool(api_key_secret_scope="biomed_genai",
                         api_key_secret_key="brave_search_key")

print(web_tool.run("Whats the capital of France?"))
print("\\n===============\\n")
print(web_tool.run("In WNBA who is the best rookie?"))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Agent

# COMMAND ----------

from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain import hub

prompt = hub.pull("hwchase17/react")

agent_model_endpoint = "databricks-meta-llama-3-1-405b-instruct"

tools = [web_tool, math_tool]

agent_chat_model = ChatDatabricks(
            endpoint=agent_model_endpoint,
            max_tokens = 2000,
            temperature= 0.1)

agent = create_react_agent(agent_chat_model,
                           tools,
                           prompt = prompt
        )
        
agent_executor = AgentExecutor(agent=agent, 
                                tools=tools,
                                handle_parsing_errors=True,
                                verbose=True,
                                max_iterations=5)

# COMMAND ----------

agent_executor.invoke({"input": "How big is africa?"})

# COMMAND ----------

agent_executor.invoke({"input": "How many florida land areas fit in India?"})

# COMMAND ----------

agent_executor.invoke({"input": "What is the largest state east of the Mississippi River?"}
                      

# COMMAND ----------

agent_executor.invoke({"input": "What is the sum of the area of Florida and Ohio?"})
