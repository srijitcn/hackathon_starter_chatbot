import mlflow
import os
import csv
import json
from operator import itemgetter

from databricks_langchain.genie import GenieAgent
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

genie_config = mlflow.models.ModelConfig(development_config=os.environ.get("GENIE_AGENT_CONFIG_FILE")).get("genie_agent_config")

def genie_markdown_tbl_to_json(markdown_table: str) -> str :
  lines = markdown_table.split("\n")
  dict_reader = csv.DictReader(lines, delimiter="|")
  data = []
  # skip first row, i.e. the row between the header and data
  for row in list(dict_reader)[1:]:
    #strip spaces and ignore columns without name
    print(row)
    r = {k.strip(): v.strip() for k, v in row.items() if k.strip() != ""}
    data.append(r)

  return (json.dumps(data).replace("\n",""))


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
    return user_question
  
def question_formatter(agent_input) -> str:
  user_question = extract_user_query_string(agent_input)
  new_question = f"Explain your answer in few words. {user_question}"
  return {"messages": [HumanMessage(content=new_question)]}

def output_extractor(agent_output:dict)->str:
  agent_output_str = agent_output["messages"][-1].content
  
  if agent_output_str.count("|") > 1:
    return_str = genie_markdown_tbl_to_json(agent_output_str)
  else:
    return_str = agent_output_str

  return return_str

def output_formatter(response_str:str) -> dict:
  return {"messages": [AIMessage(content=f"Answer is : {response_str}") ]}

#set authentication to on behalf user
#user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
user_client=WorkspaceClient(
        host=os.getenv("DATABRICKS_HOST"),
        token=os.getenv("DATABRICKS_TOKEN"),
    )

#Create a genie agent
genie_space_id =  genie_config.get("genie_space_id")
genie_agent = GenieAgent(genie_space_id=genie_space_id,
                         genie_agent_name="GenieAgent",
                         client=user_client,
                         description="An agent to query Genie Database and answer queries related to COVID Trials")


genie_chain = (
           RunnableLambda(question_formatter)
         | genie_agent
         | RunnableLambda(output_extractor) 
         | RunnableLambda(output_formatter)
)

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=genie_chain)