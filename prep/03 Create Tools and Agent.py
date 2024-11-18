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

# COMMAND ----------

class VectorIndexRetriever():    
    """A retriever class to do Databricks Vector Index Search"""
    retriever_config: RetrieverConfig = None
    vector_index: VectorSearchIndex = None

    def __init__(self, retriever_config: RetrieverConfig):
        super().__init__()
        self.retriever_config = retriever_config
        
        vsc = VectorSearchClient()
        
        self.vector_index = vsc.get_index(endpoint_name=self.retriever_config.vector_search_endpoint_name,
                                          index_name=self.retriever_config.vector_index_name)

    @mlflow.trace(name="get_benefit_retriever", span_type="func")
    def get_benefits(self, client_id:str, question:str):
        query_results = self.vector_index.similarity_search(
            query_text=question,
            filters={"client":client_id},
            columns=self.retriever_config.retrieve_columns,
            num_results=1)
        
        return query_results


class BenefitsRAGInput(BaseModel):
    """Data class for tool input"""
    client_id : str = Field(description="Client ID for which the benefits need to be retrieved")
    question: str = Field(description="Question for which the benefits need to be retrieved")

class Benefit(BaseModel):
    """Data class for tool output"""
    text:str = Field(description="Full text as provided in the context as-is without changing anything")
    in_network_copay:float = Field(description="In Network copay amount. Set to -1 if not covered or has coinsurance")
    in_network_coinsurance:float= Field(description="In Network coinsurance amount without the % sign. Set to -1 if not covered or has copay")
    out_network_copay:float = Field(description="Out of Network copay amount. Set to -1 if not covered or has coinsurance")
    out_network_coinsurance:float = Field(description="Out of Network coinsurance amount without the % sign. Set to -1 if not covered or has copay")
    
class BenefitsRAG(BaseCareCostToolBuilder):
    """Tool class implementing the benefits retriever"""
    name : str = "BenefitsRAG"
    description : str = "useful for retrieving benefits from a vector search index in json format"
    args_schema : Type[BaseModel] = BenefitsRAGInput
    model_endpoint_name:str = None
    retriever_config: RetrieverConfig = None    
    retrieved_documents:List[Document] = None
    prompt_coverage_qa:str = "Get the member medical coverage benefits from the input sentence at the end:\
        The output should only contain the formatted JSON instance that conforms to the JSON schema below.\
        Do not provide any extra information other than the json object.\
        {pydantic_parser_format_instruction}\
        Input Sentence:{context}"
    

    def __init__(self,
                 model_endpoint_name : str,
                 retriever_config: RetrieverConfig):
        super().__init__()
        self.model_endpoint_name = model_endpoint_name
        self.retriever_config = retriever_config
        
    @mlflow.trace(name="get_benefits", span_type="func")
    def execute(self, client_id:str, question:str) -> str:

        retriever = BenefitsRetriever(self.retriever_config)        
        self.retrieved_documents = None
        query_results = retriever.get_benefits(client_id, question)
        
        if query_results["result"]["row_count"] > 0:
            coverage_records = [Document(page_content=data[1]) for data in query_results["result"]["data_array"]]
            #save the records for evaluation
            self.retrieved_documents = coverage_records

            qa_chain = build_api_chain(model_endpoint_name=self.model_endpoint_name,
                                       prompt_template=self.prompt_coverage_qa,
                                       qa_chain=True)
            parser = PydanticOutputParser(pydantic_object=Benefit)

            answer = qa_chain.invoke({"context": coverage_records,
                               "pydantic_parser_format_instruction": parser.get_format_instructions()})
            return answer.replace('`','')# Benefit.model_validate_json(answer)
        else:
            raise Exception("No coverage found")
