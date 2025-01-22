import mlflow
import os
from databricks_langchain.genie import GenieAgent

genie_config = mlflow.models.ModelConfig(development_config=os.environ.get("GENIE_AGENT_CONFIG_FILE"))

#Create a genie agent
genie_space_id =  genie_config.get("genie_space_id")
genie_agent = GenieAgent(genie_space_id=genie_space_id,
                         genie_agent_name="GenieAgent",
                         description="An agent to query Genie Database and answer queries related to COVID Trials")


## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=genie_agent)