from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool

from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.llms import OpenAI

from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from prompts import CYPHER_GENERATION_PROMPT, CYPHER_UPDATE_PROMPT

from graph import get_retriever, get_qa_chain


class EveryoneAgent(AgentExecutor):
    """Everyone agent"""

    @staticmethod
    def function_name():
        return "EveryoneAgent"

    @classmethod
    def initialize(cls, graph, model_name, *args, **kwargs):
        if model_name in ["gpt-3.5-turbo", "gpt-4"]:
            llm = ChatOpenAI(temperature=0, model_name=model_name)
        else:
            raise Exception(f"Model {model_name} is currently not supported")

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        readonlymemory = ReadOnlySharedMemory(memory=memory)

        query_tool = GraphCypherQAChain.from_llm(
            llm,
            graph=graph,
            verbose=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            #
            # top_k=10
            return_intermedate_steps=True,
        )
        update_tool = GraphCypherQAChain.from_llm(
            llm,
            graph=graph,
            verbose=True,
            cypher_prompt=CYPHER_UPDATE_PROMPT,
        )

        retriever = get_retriever()
        graph_chain = get_qa_chain(llm, retriever)

        # Load the tool configs that are needed.
        tools = [
            Tool(
                name="Exact search",
                func=query_tool.run,
                description="""
                This is the primary tool for querying when exact matches are known. Input should be full question.""",
            ),
            Tool(
                name="Update",
                func=update_tool.run,
                description="""
                This is the primary tool for updating our knowledge graph. Input should be full question.""",
            ),
            # Tool(
            #     name="Keyword search",
            #     func=fulltext_tool.run,
            #     description="""Utilize this tool when explicitly told to use keyword search.
            #     Input should be a list of relevant movies inferred from the question.
            #     Remove stop word "The" from specified movie titles.""",
            # ),
            Tool(
                name="Fuzzy search",
                func=graph_chain.run,
                # coroutine=graph_chain.acall,  # if you want to use async
                description="This is the primary tool for searching. Input should just be the movie title. Do not include agent instructions.",
            ),
        ]

        agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
        )

        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)
