import os
import mlflow

from agents.helper_agent import helper_chain, helper_chain_config
from agents.covid_rag_agent import covid_rag_chain, rag_chain_config

from pydantic import BaseModel
from typing import Literal
from typing_extensions import TypedDict

import functools
import operator
from typing import Sequence, Annotated

from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from databricks_langchain.genie import GenieAgent

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

#this config file will be used for dev and test
#when the model is logged, the config file will be overwritten
multi_agent_config = mlflow.models.ModelConfig(development_config="config/multi_agent_config.yaml")

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

multi_agent_llm_config = multi_agent_config.get("multi_agent_llm_config")

multi_agent_llm = ChatDatabricks(
    endpoint=multi_agent_llm_config.get("llm_endpoint_name"),
    extra_params=multi_agent_llm_config.get("llm_parameters"),
)


def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, str):
        return {
            "messages": [AIMessage(content=result, name=name)]
        }
    else:
        return {
            "messages": [AIMessage(content=result["messages"][-1].content, name=name)]
        }
        

def get_final_message(resp):
    print(resp)
    return resp["messages"][-1]


###################
#Create a genie agent
genie_config = multi_agent_config.get("genie_agent_config")
genie_space_id =  genie_config.get("genie_space_id")
genie_agent = GenieAgent(genie_space_id=genie_space_id,
                         genie_agent_name="GenieAgent",
                         description="An agent to query Genie Database and answer queries related to COVID Trials")


###################
#Wire up everything

members = [
    {"name": "CovidRagAgent",
     "chain": functools.partial(agent_node, agent=covid_rag_chain, name="CovidRagAgent"), 
     "description": "An agent for answering questions about COVID-19 Research and Studies."
    },
    {"name": "GenieAgent",
     "chain": functools.partial(agent_node, agent=genie_agent, name="GenieAgent"), 
     "description": "An agent to query Genie Database and answer queries related to COVID Trials"
    },
    {"name": "GeneralHelperAgent",
     "chain": functools.partial(agent_node, agent=helper_chain, name="GeneralHelperAgent"), 
     "description": "A general purpose agent that can answer general questions and do mathematical calculations."
    }
]

member_names = [member['name'] for member in members]
member_name_desc = "\n".join([ f"{member['name']}:{member['description']}" for member in members])

system_prompt = "You are a supervisor tasked with managing a conversation between the following workers: \n \
    {member_name_desc}. \
        Given the following user request,respond with the worker to act next. \
            Each worker will perform a task and respond with their results and status. \
                If the question has been answered, respond with FINISH."

options = ["FINISH"] + member_names

class RouteResponse(BaseModel):
    next: Literal[tuple(options)]

def cleanup(in_str : str) -> str:
    return in_str.replace("'","").replace('"','')

def create_next(response):
    return RouteResponse(next=cleanup(response.content))

def supervisor_agent(state):
    supervisor_chain = prompt | multi_agent_llm | RunnableLambda(create_next)
    return supervisor_chain.invoke(state)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"
        ),
    ]
).partial(options=str(options), member_name_desc=member_name_desc)


workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_agent)
for member in members:
    workflow.add_node(member["name"], member["chain"])
    workflow.add_edge(member["name"], "supervisor")

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in member_names}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

# parse the output from the graph to get the final message, and then format into ChatCompletions
graph_with_parser = graph | RunnableLambda(get_final_message) | ChatCompletionsOutputParser()

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=graph_with_parser)
