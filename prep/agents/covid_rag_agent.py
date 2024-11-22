import mlflow
import os
from pydantic import BaseModel, Field
from typing import Optional, Type, List, Union

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex

from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import DatabricksVectorSearch

from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

from operator import itemgetter

def format_query_results_documents(documents):
  context_docs = []
  for document in documents:
    record_str = f"Title: {document.page_content}, Interventions: {document.metadata['interventions']}, URL: {document.metadata['url']}, NCT Number: {document.metadata['nct_number']}"
    context_docs.append(record_str)

  return "\n\n".join(context_docs)


mlflow.langchain.autolog()

#Read config file
#this config file will be used for dev and test
#when the model is logged, the config file will be overwritten
rag_chain_config = mlflow.models.ModelConfig(development_config="config/rag_agent_config.yaml")

#########################
#Create vector store retriever

#Connect to the Vector Search Index
retriever_config=rag_chain_config.get("retriever_config")
client_id_environment_var = retriever_config.get("client_id_environment_var").upper()
client_secret_environment_var = retriever_config.get("client_secret_environment_var").upper()
workspace_url_environment_var = retriever_config.get("workspace_url_environment_var").upper()

vs_client = VectorSearchClient(
    workspace_url=os.getenv(workspace_url_environment_var),
    service_principal_client_id=os.getenv(client_id_environment_var),
    service_principal_client_secret=os.getenv(client_secret_environment_var),
)
vs_index = vs_client.get_index(
    endpoint_name=retriever_config.get("vector_search_endpoint_name"),
    index_name=retriever_config.get("vector_search_index"),
)
vector_search_schema = retriever_config.get("schema")

# Turn the Vector Search index into a LangChain retriever
db_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=retriever_config["text_column"],
    columns=retriever_config["retrieve_columns"]
).as_retriever(search_kwargs=retriever_config.get("parameters"))


#########################
#Define prompt

model_config = rag_chain_config.get("rag_agent_llm_config") 

prompt = ChatPromptTemplate.from_messages(
  [  
      ("system", model_config.get("llm_prompt_template")), 
      ("user", "{query}") 
  ]
)

#########################
#Define model

model = ChatDatabricks(
    endpoint=model_config.get("llm_endpoint_name"),
    extra_params=model_config.get("llm_parameters"),
)

#########################
#Create chain
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


covid_rag_chain = (
      {
        "query": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages") 
                   | RunnableLambda(extract_user_query_string)
                   | db_retriever
                   | RunnableLambda(format_query_results_documents)
                   | StrOutputParser()
      } 
      | prompt
      | model
      | StrOutputParser()  
  )

