import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langsmith import traceable


# --- Configuration -----------------------------------------------------------

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Must match what ingestion_service.py writes to.
COLLECTION_NAME = "resume_chunks"

# How many chunks to pull from the vector store for each question.
TOP_K = 10

# The model is told to answer strictly from the retrieved resume context.
PROMPT_TEMPLATE = """
You are an AI Resume Assistant.

Use ONLY the information provided in the resume context.

Each context block is numbered like [1], [2], ... .

Rules:

- Answer ONLY from the resume context.
- Do NOT use outside knowledge.
- Do NOT make assumptions.
- If multiple matching items exist, list them all.
- Cite the block(s) you used inline with their number, e.g. [1] or [2][3].
- If the answer is not found, respond exactly:

I cannot find that information in the resume.

Resume Context:
---------------------
{context}
---------------------

Question:
{question}

Answer:
"""


class RagService:
    """Answers questions about ingested resumes using LangChain retrieval +
    an OpenAI chat model (RAG)."""

    def create_vector_store(self) -> PGVector:
        """Connect to the same Postgres vector store ingestion writes to."""
        connection = (
            f"postgresql+psycopg://{os.getenv('DB_USER')}:"
            f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/"
            f"{os.getenv('DB_NAME')}"
        )

        return PGVector(
            embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
            collection_name=COLLECTION_NAME,
            connection=connection,
            use_jsonb=True,
        )

    @traceable
    def retrieve_chunks(self, question: str) -> list:
        """Find the most relevant resume chunks for the question.

        Returns a list of (Document, similarity_score) tuples so we can log the
        scores, just like the previous implementation did.
        """
        vector_store = self.create_vector_store()
        results = vector_store.similarity_search_with_score(question, k=TOP_K)

        print("\n===== RETRIEVED CHUNKS =====")
        for i, (document, score) in enumerate(results):
            print(f"\n----- Chunk {i + 1} -----")
            print(f"Score: {score}")
            print(document.page_content[:1000])

        return results

    def build_sources(self, results: list) -> list[dict]:
        """Turn retrieved chunks into citation metadata for the caller.

        Each source is numbered to match the [n] markers we put in the context,
        so the answer's inline citations can be linked back to a real chunk.
        """
        sources = []
        for i, (document, score) in enumerate(results):
            metadata = document.metadata or {}
            sources.append(
                {
                    "citation": i + 1,
                    "file_name": metadata.get("file_name"),
                    "file_type": metadata.get("file_type"),
                    # page is present for PDFs (PyPDFLoader); 0-based -> 1-based.
                    "page": (
                        metadata["page"] + 1 if "page" in metadata else None
                    ),
                    # char offset of the chunk within its document.
                    "start_index": metadata.get("start_index"),
                    # cosine distance from PGVector: lower = more relevant.
                    "score": round(float(score), 4),
                    # short preview so the citation is human-readable.
                    "snippet": document.page_content[:200],
                }
            )
        return sources

    @traceable
    def generate_answer(self, question: str, results: list) -> dict:
        """Build the context from retrieved chunks and ask the model.

        Returns the answer plus the source metadata (citations) behind it.
        """
        # Number each chunk so the model can cite it as [n] and we can map the
        # citation back to its source in build_sources().
        context = "\n\n".join(
            f"[{i + 1}] {document.page_content}"
            for i, (document, _) in enumerate(results)
        )

        # prompt -> llm. We keep the raw AIMessage (instead of piping through
        # StrOutputParser) so we can read its token usage. temperature=0 keeps
        # answers grounded.
        chain = (
            ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            | ChatOpenAI(model=CHAT_MODEL, temperature=0)
        )

        message = chain.invoke({"context": context, "question": question})
        answer = message.content

        # usage_metadata is the provider-agnostic token count LangChain attaches
        # to the response; default to an empty dict if the provider omits it.
        usage = message.usage_metadata or {}

        print("\n===== FINAL ANSWER =====")
        print(answer)
        print(f"\nToken usage: {usage}")

        return {
            "answer": answer,
            "sources": self.build_sources(results),
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        }

    @traceable
    def ask(self, question: str) -> dict:
        """Public entry point: retrieve relevant chunks, then answer.

        Returns {"answer": str, "sources": [ {citation, file_name, ...}, ... ]}.
        """
        results = self.retrieve_chunks(question)
        return self.generate_answer(question, results)
