import mlflow
import os
import csv
import json
from databricks_langchain.genie import GenieAgent
from operator import itemgetter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

genie_config = mlflow.models.ModelConfig(development_config=os.environ.get("GENIE_AGENT_CONFIG_FILE"))

def genie_markdown_tbl_to_json(markdown_table: str) -> str :
  lines = markdown_table.split("\n")
  dict_reader = csv.DictReader(lines, delimiter="|")
  data = []
  # skip first row, i.e. the row between the header and data
  for row in list(dict_reader)[1:]:
    #strip spaces and ignore columns without name
    r = {k.strip(): v.strip() for k, v in row.items() if k.strip() != ""}
    data.append(r)

  return (json.dumps(data).replace("\n",""))

def question_formatter(agent_input:dict) -> str:
  new_question = f"Do not respond in markdown. Respond in few words in simple text {agent_input['messages'][-1].content}"
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


#Create a genie agent
genie_space_id =  genie_config.get("genie_space_id")
genie_agent = GenieAgent(genie_space_id=genie_space_id,
                         genie_agent_name="GenieAgent",
                         description="An agent to query Genie Database and answer queries related to COVID Trials")


genie_chain = (
           RunnableLambda(question_formatter)
         | genie_agent
         | RunnableLambda(output_extractor) 
         | RunnableLambda(output_formatter)
)

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=genie_chain)