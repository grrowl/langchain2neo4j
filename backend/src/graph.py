from __future__ import annotations
from database import Neo4jDatabase
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.base import Chain
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

from langchain.chat_models import ChatOpenAI

from typing import Any, Dict, List

from pydantic import Field
from logger import logger

from langchain.vectorstores import Neo4jVector

import os

url = os.environ.get("NEO4J_URL")
username = os.environ.get("NEO4J_USER")
password = os.environ.get("NEO4J_PASS")

# https://python.langchain.com/docs/integrations/vectorstores/neo4jvector


def get_retriver():
    store = Neo4jVector.from_existing_graph(
        embedding=OpenAIEmbeddings(),
        url=url,
        username=username,
        password=password,
        node_label="Movie",
        text_node_properties=["title", "tagline"],
        embedding_node_property="embedding",
        index_name="vector",
        keyword_index_name="keyword",
        search_type="hybrid",
    )

    return store.as_retriever()


def get_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever, callbacks=[StdOutCallbackHandler]
    )


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0.0)
    database = Neo4jDatabase(
        host="bolt://localhost:7687",
        user="neo4j",
        password=password,
    )
    chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=get_retriver()
    )
    output = chain(
        {"question": "What did the president say about Justice Breyer"},
        return_only_outputs=True,
    )

    # chain = LLMNeo4jVectorChain(llm=llm, verbose=True, graph=database)

    # output = chain.run("What type of movie is Grumpier?")

    print(output)
