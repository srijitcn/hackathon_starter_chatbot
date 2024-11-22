import mlflow
import json 
import os
from bs4 import BeautifulSoup
from datetime import datetime


from langchain.chat_models import ChatDatabricks
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)
from langchain_community.tools import BraveSearch
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from mlflow.langchain.output_parsers import ChatCompletionsOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Union

#this config file will be used for dev and test
#when the model is logged, the config file will be overwritten
helper_chain_config = mlflow.models.ModelConfig(development_config="config/helper_agent_config.yaml")

###################################
# Math Tool
# Let us build a Math tool that can perform mathematical operations from natural language input.

# We will do it by subclassing the BaseTool class. This provides maximal control over the tool definition, but is a bit more work.

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
      

###################################
#### Web Search tool
# Let us create another tool to search Web for information. 
# This time let's use the `from_function` method of `StructuredTool` class of LangChain api. 

class WebSearchToolInput(BaseModel):
    question: str = Field(description="Sentence containing a general question for web search")

class WebSearchTool:
    name : str = "WebSearchTool"
    description : str = "Useful for performing general search on Internet and return a sigle string result"
    args_schema : Type[BaseModel] = WebSearchToolInput
        
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    @mlflow.trace(name="WebSearchTool_search", span_type="func")
    def search(self, question:str) -> str: 
        #####################################################################
        #Create a Brave AI Api key from https://api.search.brave.com/
        searcher = BraveSearch.from_api_key(api_key=self.api_key,
                                            search_kwargs={"count": 1})
        

        html_text = json.loads(searcher.run(question))[0]["snippet"]
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    
    def get_tool(self):
        return StructuredTool.from_function(func=self.search,
                                            name=self.name,
                                            description=self.description,
                                            args_schema=self.args_schema)

###################################
# Today Tool
# Lets us now create a simple tool, that returns current date.
# For this purpose we will simply define a python function and decorate it with @tool decorator.
# This is the most simplest way to create a tool
#Since a tool need to have an input, we will provide a dummy input even if we wont use it
@tool
@mlflow.trace(name="get_todays_date", span_type="func")
def get_todays_date(unnecessary:str):
  """This tool will return the current date in the format of Month, Day, Year"""
  return datetime.now().strftime("%B %d, %Y")

###################################
# Create Agent and Agent Executor
# AgentExecutor allows agents to take actions based on their inputs, using tools to generate observations and iterate. 
# It provides a flexible and customizable way to run an agent, allowing users to specify the tools and memory to be used.

#instantiate the tools
math_tool_model_endpoint =  helper_chain_config.get("math_tool").get("llm_endpoint_name")
math_tool = MathTool(chat_model_endpoint_name=math_tool_model_endpoint)

api_key_env_var = helper_chain_config.get("web_search_tool").get("api_key_environment_var").upper()
web_tool = WebSearchTool(api_key=os.environ[api_key_env_var]).get_tool()


#agent

#lets use the llama 3.1 405b model for our Agent
helper_agent_model_config = helper_chain_config.get("helper_agent_llm_config")
#define the model class for agent that uses the endpoint
agent_chat_model = ChatDatabricks(
    endpoint=helper_agent_model_config.get("llm_endpoint_name"),
    extra_params=helper_agent_model_config.get("llm_parameters"),
)

#define the tools that is available to the agent
tools = [web_tool, math_tool, get_todays_date]

#Get a prebuilt react prompt from langchain hub
prompt = hub.pull("hwchase17/react")

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

#lets create a chain to make our agent executor compatible with ChatRequest
#this makes it deployable to an endpoint and make it available in playground
def extract_user_query_string(input_messages:[dict])->str:
    return {"input" : input_messages[-1]["content"]}

def output_extractor(agent_output:dict)->str:
  return agent_output["output"]

helper_chain = (itemgetter("messages")
         | RunnableLambda(extract_user_query_string)
         | agent_executor
         | RunnableLambda(output_extractor) 
         | StrOutputParser()
)


