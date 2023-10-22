from __future__ import annotations
from database import Neo4jDatabase
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.base import Chain

from typing import Any, Dict, List

from pydantic import Field
from logger import logger

from langchain.vectorstores import Neo4jVector

import os

url = os.environ.get("NEO4J_URL")
username = os.environ.get("NEO4J_USER")
password = os.environ.get("NEO4J_PASS")


class LLMNeo4jVectorChain(Chain):
    """Chain for question-answering against a graph."""

    graph: Neo4jDatabase = Field(exclude=True)
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings()

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Embed a question and do vector search."""
        question = inputs[self.input_key]
        logger.debug(f"Vector search input: {question}")

        vector_search = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=url,
            username=username,
            password=password,
            index_name="index",
            node_label="Movie",
            text_node_properties=["title", "tagline"],
            embedding_node_property="embedding",
            search_type="hybrid",  # optional, maybe better?
        )
        # RetrievalQA.from_chain_type(
        #     llm=llm, chain_type="stuff", retriever=vector_search.as_retriever()
        # )

        retriever = vector_search.as_retriever()
        context = retriever.invoke(question)  # it will vectorise for us

        # embedding = self.embeddings.embed_query(question)

        # self.callback_manager.on_text(
        #     "Vector search embeddings:", end="\n", verbose=self.verbose
        # )
        # self.callback_manager.on_text(
        #     embedding[:5], color="green", end="\n", verbose=self.verbose
        # )
        # we do have access to the underlying graph here self.graph

        # context = self.graph.query(vector_search, {"embedding": embedding})
        return {self.output_key: context}


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0.0)
    database = Neo4jDatabase(
        host="bolt://100.27.33.83:7687",
        user="neo4j",
        password="room-loans-transmissions",
    )
    chain = LLMNeo4jVectorChain(llm=llm, verbose=True, graph=database)

    output = chain.run("What type of movie is Grumpier?")

    print(output)
