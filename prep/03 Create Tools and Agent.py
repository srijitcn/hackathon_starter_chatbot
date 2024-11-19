# Databricks notebook source
# MAGIC %md
# MAGIC # Create Tools and Agent

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

# MAGIC %md
# MAGIC ### Utility Classes and Functions 

# COMMAND ----------

import mlflow
import mlflow.deployments
import os
import pandas as pd
import requests
import json
import logging

from typing import Optional, Type, List, Union

from pydantic import BaseModel, Field

from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)
from langchain.chat_models import ChatDatabricks
from langchain.llms import Databricks
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.documents.base import Document
from langchain.output_parsers import PydanticOutputParser

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex

# COMMAND ----------

#a utility function to use logging api instead of print
def log_print(msg):
    logging.warning(f"=====> {msg}")

#a dataclass for retriever configuration
class RetrieverConfig(BaseModel):
    """A data class for passing around vector index configuration"""
    vector_search_endpoint_name:str
    vector_index_name:str
    vector_index_id_column:str
    num_documents_to_retrieve:int = 1
    retrieve_columns:List[str]


#a base class for all tools. simplifies structured tool creation
class BaseToolBuilder:
    name:str = None
    description:str = None
    args_schema : Type[BaseModel] = None
    def execute(self, **kwargs):
        raise NotImplementedError("Please Implement this method")
        
    def get(self):
        return StructuredTool.from_function(func=self.execute,
                                            name=self.name,
                                            description=self.description,
                                            args_schema=self.args_schema)

#a utility method to build a LLM chain from databricks endpoint
def build_api_chain(model_endpoint_name:str,
                    prompt_template:str,
                    qa_chain:bool=False,
                    max_tokens:int=500,
                    temperature:float=0.01):
    
    client = mlflow.deployments.get_deploy_client("databricks")
    endpoint_details = [ep for ep in client.list_endpoints() if ep["name"]==model_endpoint_name]
    if len(endpoint_details)>0:
      endpoint_detail = endpoint_details[0]
      endpoint_type = endpoint_detail["task"]

      if endpoint_type.endswith("chat"):
        llm_model = ChatDatabricks(endpoint=model_endpoint_name, max_tokens = max_tokens, temperature=temperature)
        llm_prompt = ChatPromptTemplate.from_template(prompt_template)

      elif endpoint_type.endswith("completions"):
        llm_model = Databricks(endpoint_name=model_endpoint_name, 
                               model_kwargs={"max_tokens": max_tokens,
                                             "temperature":temperature})
        llm_prompt = PromptTemplate.from_template(prompt_template)
      else:
        raise Exception(f"Endpoint {model_endpoint_name} not compatible ")

      if qa_chain:
        return create_stuff_documents_chain(llm=llm_model, prompt=llm_prompt)
      else:
        return LLMChain(
          llm = llm_model,
          prompt = llm_prompt
        )
      
    else:
      raise Exception(f"Endpoint {model_endpoint_name} not available ")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Classifier Tool

# COMMAND ----------

class TextClassifierInput(BaseModel):
    """Data class for tool input"""
    text: str = Field(description="Text to be classified")

class TextClassifier(BaseToolBuilder):
    """A tool to classify a text into categories"""
    name : str = "TextClassifier"
    description : str = "useful for classifying a given text into categories"
    args_schema : Type[BaseModel] = TextClassifierInput
    model_endpoint_name:str = None
    categories_and_description:dict = None
    category_str: str = ""

    prompt:str = "Classify the given text into one of below the categories. \
        {categories}\
        Only respond with a single word which is the category code. \
        Do not include any other  details in response.\
        Text: {text}"

    def __init__(self, model_endpoint_name : str, categories_and_description : dict[str:str]):
        super().__init__()
        self.model_endpoint_name = model_endpoint_name
        self.categories_and_description = categories_and_description
        self.category_str = "\n".join([ f"{c}:{self.categories_and_description[c]}" for c in self.categories_and_description])
    
    @mlflow.trace(name="get_text_category", span_type="func")
    def execute(self, text:str) -> str: 
        chain = build_api_chain(self.model_endpoint_name, self.prompt)
        category = chain.run(categories=self.category_str, text=text)
        return category

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAG Tool
# MAGIC A tool that helps you perform Retrieval Augmented Generation based on a query

# COMMAND ----------

class DBVectorIndexRetriever():    
    """A retriever class to do Databricks Vector Index Search"""
    retriever_config: RetrieverConfig = None
    vector_index: VectorSearchIndex = None

    def __init__(self, retriever_config: RetrieverConfig):
        super().__init__()
        self.retriever_config = retriever_config
        
        vsc = VectorSearchClient(disable_notice=True)
        
        self.vector_index = vsc.get_index(endpoint_name=self.retriever_config.vector_search_endpoint_name,
                                          index_name=self.retriever_config.vector_index_name)

    @mlflow.trace(name="retrieve", span_type="func")
    def retrieve(self, query:str, filters:dict = {}):
        query_results = self.vector_index.similarity_search(
            query_text=query,
            filters=filters,
            columns=self.retriever_config.retrieve_columns,
            num_results=self.retriever_config.num_documents_to_retrieve)
        
        return query_results


class RAGInput(BaseModel):
    """Data class for tool input"""
    filters : dict = Field(description="This is a python dictionary with key value pairs as additional filters to be applied to the search")
    query: str = Field(description="Query for which the documents need to be retrieved")

#This class need to be extended appropriately for different applications
class BaseRAG(BaseToolBuilder):
    """Tool class implementing a simple Retrieval Augmented Generation pipeline"""
    #implementing class must define
    #name : str = give proper name
    
    #implementing class must define
    #description : str = Give a proper description
    
    #implementing class must define
    #args_schema : Type[BaseModel] = provide the base class for inputs

    model_endpoint_name:str = None
    retriever_config: RetrieverConfig = None    
    retrieved_documents:List[Document] = None

    def __init__(self,
                 model_endpoint_name : str,
                 prompt:str,
                 retriever_config: RetrieverConfig):
        super().__init__()
        self.model_endpoint_name = model_endpoint_name
        self.prompt = prompt
        self.retriever_config = retriever_config
    
    def get_prompt_variables(self) -> dict:
      raise NotImplementedError("This method should be implemented by child class")

    @mlflow.trace(name="base_rag_execute", span_type="func")
    def execute(self, query:str, filters:dict={}) -> str:

        retriever = DBVectorIndexRetriever(self.retriever_config)        
        self.retrieved_documents = None
        query_results = retriever.retrieve(query=query, filters=filters)
        
        if query_results["result"]["row_count"] > 0:
            context_docs = []
            for data in query_results["result"]["data_array"]:              
              record_str = ""
              for i in range(len(self.retriever_config.retrieve_columns)):
                col_name = self.retriever_config.retrieve_columns[i]
                col_data = data[i]
                record_str += f"{col_name}: {col_data if col_data else 'No Data'} \n"
              
              context_docs.append(Document(page_content=record_str))

            #save the records for evaluation
            self.retrieved_documents = context_docs

            qa_chain = build_api_chain(model_endpoint_name=self.model_endpoint_name,
                                       prompt_template=self.prompt,
                                       qa_chain=True)
            
            prompt_variables = self.get_prompt_variables()
            prompt_variables.update({"context": context_docs})
            prompt_variables.update({"question": query})
            
            answer = qa_chain.invoke(prompt_variables)
            return answer
        else:
            return "Sorry no records were found related to the query that satisfies the given filters."

# COMMAND ----------

# MAGIC %md
# MAGIC ####Usage

# COMMAND ----------

#implement a RAG to answer question on Covid Trials based on a query on title
class CovidTrialTitleRAG(BaseRAG):
  name : str = "CovidTrialTitleRAG"
  description : str = "Retrieves Covid Trial documents based on title"
  args_schema : Type[BaseModel] = RAGInput
  prompt:str = "Answer the question based on the text available in the context.\
        Always append all the URL of the documents in the end as separate bulleted list. \n\
        Question: {question} \n\
        Context:{context}"

  num_documents:int = 3

  def __init__(self, model_endpoint_name : str, retriever_config: RetrieverConfig):
    super().__init__(model_endpoint_name=model_endpoint_name,
                      prompt=self.prompt,
                      retriever_config=retriever_config)
    
  def get_prompt_variables(self) -> dict:
    #since we are not setting any more variables in the prompt other than context, we can return an empty dict
    return {}
      

# COMMAND ----------

# MAGIC %md
# MAGIC ### AI/BI Genie Tool
# MAGIC A tool that can be customized to interact with a AI/BI Genie room

# COMMAND ----------

from databricks.sdk.service.dashboards import GenieAPI
from databricks.sdk import WorkspaceClient


class GenieAPIWrapper:
    def __init__(
        self,
        space_id,
        encountered_error_user_message: str = "I encountered an error trying to answer your question, please try again.",
    ):
        self.space_id = space_id

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        workspace_client = WorkspaceClient()
        self._genie_client = workspace_client.genie
        self.encountered_error_user_message = encountered_error_user_message

        # We build the GenieResponse throughout this wrapper's logic since you must poll for the result & the results come back from multiple polling requests.
        self.genie_result = GenieResponse()

    @mlflow.trace()
    def start_conversation(self, content):
        resp = self._genie_client._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def create_message(self, conversation_id, content):
        resp = self._genie_client._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def poll_for_result(self, conversation_id, message_id):
        @mlflow.trace()
        def poll_result():
            iteration_count = 0
            while True and iteration_count < MAX_ITERATIONS:
                # try:  # genie API randomly crashes with BadRequest: Message <id> does not have a query statementId.  This is instead caught in the Agent itself to capture all unknown exceptions from the API wrapper.
                iteration_count += 1
                logging.debug(
                    f"Polling for result {message_id} {conversation_id} iteration {iteration_count}"
                )
                resp = self._genie_client._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}",
                    headers=self.headers,
                )
                logging.debug(f"Genie polling response: {resp}")
                if resp["status"] == "EXECUTING_QUERY":
                    with mlflow.start_span(name="get_sql_query") as span:
                        query_result = next(
                            r for r in resp["attachments"] if "query" in r
                        )["query"]
                        span.set_inputs(resp)
                        self.genie_result.sql_query = query_result.get("query")
                        self.genie_result.response = query_result.get("description")
                        span.set_outputs(
                            {
                                "sql_query": self.genie_result.sql_query,
                                "response": self.genie_result.response,
                            }
                        )
                    return poll_query_results()
                elif resp["status"] == "COMPLETED":
                    """
                    Genie didn't run a query, returned a question or comment to the user
                    """
                    with mlflow.start_span(name="get_genie_response") as span:
                        logging.debug(f"Genie polling returned {resp}")
                        span.set_inputs(resp)
                        # Get first attachment from array safely
                        first_attachment = (
                            resp.get("attachments", [])[0]
                            if resp.get("attachments")
                            else None
                        )
                        if first_attachment:
                            # TODO: we shouldn't need this logic, but it's here to handle a bug in the Genie API where sometimes you get COMPLETED before EXECUTING_QUERY is returned.
                            if "text" in first_attachment:
                                # genie didn't run a query, just returned a question or comment to the user
                                response = first_attachment["text"]["content"]
                                self.genie_result.response = response
                                span.set_outputs(
                                    {"response": self.genie_result.response}
                                )
                                return asdict(self.genie_result)
                            elif "query" in first_attachment:
                                # genie ran a query, get the results
                                response = first_attachment["query"]["description"]
                                self.genie_result.sql_query = first_attachment["query"][
                                    "query"
                                ]
                                self.genie_result.response = first_attachment["query"][
                                    "description"
                                ]
                                span.set_outputs(
                                    {
                                        "sql_query": self.genie_result.sql_query,
                                        "response": self.genie_result.response,
                                    }
                                )
                                return poll_query_results()
                            else:
                                # unknown state, assume an error state
                                self.genie_result.response = (
                                    self.encountered_error_user_message
                                )
                                span.set_outputs(
                                    {"response": self.genie_result.response}
                                )
                                return asdict(self.genie_result)
                        else:
                            # no response, must be an error state
                            self.genie_result.response = (
                                self.encountered_error_user_message
                            )
                            span.set_outputs({"response": self.genie_result.response})
                            return asdict(self.genie_result)

                elif resp["status"] == "FAILED":
                    """
                    Genie failed
                    """
                    self.genie_result.response = self.encountered_error_user_message
                    return asdict(self.genie_result)
                else:
                    logging.debug(f"Waiting...: {resp['status']}")
                    time.sleep(1)
                # except Exception as e:  # hack per above
                #     logging.error(
                #         f"Error polling for result: {e}, in polling iteration {iteration_count} of {MAX_ITERATIONS}"
                #     )
                #     print(iteration_count)
                #     continue

        @mlflow.trace()
        def poll_query_results():
            iteration_count = 0
            while True and iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self._genie_client._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}/query-result",
                    headers=self.headers,
                )["statement_response"]

                state = resp["status"]["state"]
                if state == "SUCCEEDED":
                    with mlflow.start_span(name="get_sql_query_results") as span:
                        span.set_inputs(resp)
                        data_table_as_md = _parse_query_result(resp)
                        self.genie_result.data_table = data_table_as_md
                        span.set_outputs(self.genie_result.data_table)
                    return asdict(self.genie_result)
                elif state == "RUNNING" or state == "PENDING":
                    logging.debug("Waiting for query result...")
                    time.sleep(1)
                else:
                    logging.debug(f"No query result: {resp['state']}")
                    return None

        return poll_result()

    @mlflow.trace(span_type="AGENT", name="genie")
    def ask_question(self, question):
        self.genie_result = GenieResponse()
        resp = self.start_conversation(question)
        return self.poll_for_result(resp["conversation_id"], resp["message_id"])

# COMMAND ----------

w.apps.list()


# COMMAND ----------

a = w.genie

# COMMAND ----------

w.genie.

# COMMAND ----------

w.genie.start_conversation("01efa5d3862c1317969a569e3cdfcb1c","How many trials are in Recruiting status")

# COMMAND ----------

w.genie.start_conversation_and_wait("01efa5d3862c1317969a569e3cdfcb1c","How many trials are in Recruiting status")

# COMMAND ----------


