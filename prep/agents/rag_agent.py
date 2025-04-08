import mlflow
import os
import json

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.utils import CredentialStrategy
from databricks.vector_search.index import VectorSearchIndex

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks
from langchain.vectorstores import DatabricksVectorSearch
from langchain_core.messages import AIMessage, HumanMessage
from operator import itemgetter
from mlflow.models import ModelConfig


mlflow.langchain.autolog()

rag_config = mlflow.models.ModelConfig(development_config=os.environ.get("RAG_AGENT_CONFIG_FILE")).get("rag_agent_config")

retriever_config=rag_config.get("retriever_config")

vs_client = VectorSearchClient(); #credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS)

vs_index = vs_client.get_index(
    endpoint_name=retriever_config.get("vector_search_endpoint_name"),
    index_name=retriever_config.get("vector_search_index"),
)

db_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=retriever_config["text_column"],
    columns=retriever_config["retrieve_columns"]
).as_retriever(search_kwargs=retriever_config.get("parameters"))

model = ChatDatabricks(
  endpoint=rag_config.get("llm_endpoint_name"),
  extra_params=rag_config.get("llm_parameters"),
)

prompt = ChatPromptTemplate.from_messages(
  [  
      ("system", rag_config.get("llm_prompt_template")), 
      ("user", "{query}") 
  ]
)

def format_query_results_documents(documents):
  context_docs = []
  for document in documents:
    record_str = json.dumps({"Title": document.page_content, 
                  "Interventions": document.metadata['interventions'],
                  "URL": document.metadata['url'],
                  "NCT_Number": document.metadata['nct_number']
                })
    context_docs.append(record_str)

  return "\n\n".join(context_docs)

def extract_user_query_string(input_data):
    payload = []
    user_question = ""
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
    
    return {"query" : user_question}

def output_extractor(agent_output:dict)->str:
  model_response = agent_output.get("model_response",{}).content
  if "Irrelevant Question" in model_response:
    return {"messages":[AIMessage(content="Irrelevant Question")]}
  else:
    supporting_documents = agent_output.get("supporting_documents","")
    return {"messages":[AIMessage(f"{model_response}\n\nReference Links:\n{supporting_documents}")]}

rag_chain = (
      RunnableLambda(extract_user_query_string)
      | RunnablePassthrough.assign(supporting_documents= itemgetter("query") | db_retriever | RunnableLambda(format_query_results_documents) | StrOutputParser())
      | RunnablePassthrough.assign(prompt=prompt)
      | RunnablePassthrough.assign(model_response= itemgetter("prompt") | model)
      | RunnableLambda(output_extractor)
)
  
## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=rag_chain)

