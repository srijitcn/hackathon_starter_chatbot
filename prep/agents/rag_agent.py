import mlflow
import os

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_databricks import ChatDatabricks
from langchain.vectorstores import DatabricksVectorSearch

from operator import itemgetter

from mlflow.models import ModelConfig

mlflow.langchain.autolog()

rag_config = mlflow.models.ModelConfig(development_config=os.environ.get("RAG_AGENT_CONFIG_FILE"))
retriever_config=rag_config.get("retriever_config")

vs_client = VectorSearchClient()
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
    record_str = f"Title: {document.page_content}, Interventions: {document.metadata['interventions']}, URL: {document.metadata['url']}, NCT Number: {document.metadata['nct_number']}"
    context_docs.append(record_str)

  return "\n\n".join(context_docs)

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


rag_chain = (
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
## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=rag_chain)

