from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

import os

from cypher_database_tool import LLMCypherGraphChain

# from keyword_neo4j_tool import LLMKeywordGraphChain
# from vector_neo4j_tool import LLMNeo4jVectorChain
from graph import get_retriver, get_qa_chain


class MovieAgent(AgentExecutor):
    """Movie agent"""

    @staticmethod
    def function_name():
        return "MovieAgent"

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

        cypher_tool = LLMCypherGraphChain(
            llm=llm, graph=graph, verbose=True, memory=readonlymemory
        )
        # fulltext_tool = LLMKeywordGraphChain(llm=llm, graph=graph, verbose=True)
        # vector_tool = LLMNeo4jVectorChain(llm=llm, verbose=True, graph=graph)
        retriever = get_retriver()
        graph_chain = get_qa_chain(llm, retriever)

        # Load the tool configs that are needed.
        tools = [
            Tool(
                name="Exact search",
                func=cypher_tool.run,
                description="""
                This is the primary tool for querying when exact matches are known. Input should be full question.""",
            ),
            Tool(
                name="Update",
                func=cypher_tool.run,
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
            # Tool(
            #     name="Fuzzy search",
            #     func=vector_tool.run,
            #     description="This is the primary tool for searching. Input should be full question. Do not include agent instructions.",
            # ),
            Tool(
                name="Fuzzy search",
                # func=vector_qa_chain.run,
                func=graph_chain.run,
                # coroutine=graph_chain.acall,  # if you want to use async
                description="This is the primary tool for searching. Input should be full question. Do not include agent instructions.",
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
