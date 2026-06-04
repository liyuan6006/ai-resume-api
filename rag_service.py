from llama_index.core import (
    VectorStoreIndex,
    PromptTemplate
)

from llama_index.llms.openai import OpenAI

from llama_index.vector_stores.postgres import (
    PGVectorStore
)

from langsmith import traceable

import os


class RagService:

    @traceable
    def ask(
        self,
        question: str
    ):

        llm = OpenAI(
            model="gpt-4o-mini"
        )

        vector_store = PGVectorStore.from_params(
            host=os.getenv("DB_HOST"),
            port=5432,
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            table_name="resume_chunks",
            embed_dim=1536
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

        qa_prompt = PromptTemplate(
            """
You are an AI Resume Assistant.

Use ONLY the information provided in the resume context.

Rules:

- Answer ONLY from the resume context.
- Do NOT use outside knowledge.
- Do NOT make assumptions.
- If multiple matching items exist, list them all.
- If the answer is not found, respond exactly:

I cannot find that information in the resume.

Resume Context:
---------------------
{context_str}
---------------------

Question:
{query_str}

Answer:
"""
        )

        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=10,
            text_qa_template=qa_prompt
        )

        nodes = self.retrieve_chunks(
            index,
            question
        )

        response = self.generate_answer(
            query_engine,
            question
        )

        self.log_source_nodes(
            response
        )

        return str(response)

    @traceable
    def retrieve_chunks(
        self,
        index,
        question
    ):

        retriever = index.as_retriever(
            similarity_top_k=10
        )

        nodes = retriever.retrieve(
            question
        )

        print(
            "\n===== RETRIEVED CHUNKS ====="
        )

        for i, node in enumerate(nodes):

            print(
                f"\n----- Chunk {i + 1} -----"
            )

            print(
                f"Score: {node.score}"
            )

            print(
                node.text[:1000]
            )

        return nodes

    @traceable
    def generate_answer(
        self,
        query_engine,
        question
    ):

        response = query_engine.query(
            question
        )

        print(
            "\n===== FINAL ANSWER ====="
        )

        print(
            str(response)
        )

        return response

    @traceable
    def log_source_nodes(
        self,
        response
    ):

        if not hasattr(
            response,
            "source_nodes"
        ):
            return

        print(
            "\n===== SOURCE NODES USED BY GPT ====="
        )

        for i, source in enumerate(
            response.source_nodes
        ):

            print(
                f"\nSource {i + 1}"
            )

            print(
                f"Score: {source.score}"
            )

            print(
                source.text[:1000]
            )