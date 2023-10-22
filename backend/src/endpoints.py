import os
import logging

from agent import MovieAgent
from database import Neo4jDatabase
from fastapi import APIRouter, HTTPException, Query
from run import get_result_and_thought_using_graph

neo4j_host = os.environ.get("NEO4J_URL")
neo4j_user = os.environ.get("NEO4J_USER")
neo4j_password = os.environ.get("NEO4J_PASS")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
# build router
router = APIRouter()
logger = logging.getLogger(__name__)
graph = Neo4jDatabase(host=neo4j_host, user=neo4j_user, password=neo4j_password)
agent_movie = MovieAgent.initialize(graph=graph, model_name=model_name)


@router.get("/predict")
def get_load(message: str = Query(...)):
    try:
        return get_result_and_thought_using_graph(agent_movie, graph, message)
    except Exception as e:
        # Log stack trace
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e
