import os
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from urllib.parse import quote_plus

def annotate_run(**metadata) -> None:
    """Attach key/value metadata to the current LangSmith span, if we're inside
    a traced run. Shows up under the run's Metadata in the LangSmith UI. No-op
    when tracing is disabled, so it's safe to call unconditionally."""
    run = get_current_run_tree()
    if run is not None:
        run.metadata.update(metadata)


# --- Configuration -----------------------------------------------------------

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Length of each embedding vector. MUST match what ingestion_service.py uses,
# or the stored vectors and the query vector won't be comparable. 1536 is the
# native size for text-embedding-3-small.
EMBEDDING_DIMENSIONS = 1536

# Fail fast instead of hanging. The embedding HTTP call and the Postgres query
# each get a hard ceiling so a stalled request raises an error in seconds
# rather than blocking until the upstream proxy times out (Cloudflare 524).
EMBEDDING_TIMEOUT_SECONDS = 15
EMBEDDING_MAX_RETRIES = 2
# Postgres cancels any single statement that runs longer than this (in ms).
DB_STATEMENT_TIMEOUT_MS = 30_000

# Must match what ingestion_service.py writes to.
COLLECTION_NAME = "resume_chunks"

# How many chunks to pull from the vector store for each question.
TOP_K = 5

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

    # Built once and reused across requests so we don't reconnect to Postgres
    # (and rebuild the embeddings client) on every question.
    _vector_store: PGVector | None = None



    @traceable
    def build_connection_string(self) -> str:
        password = quote_plus(os.getenv("DB_PASSWORD"))

        return (
            f"postgresql+psycopg://{os.getenv('DB_USER')}:"
            f"{password}@{os.getenv('DB_HOST')}:5432/"
            f"{os.getenv('DB_NAME')}"
        )

    @traceable
    def build_embeddings_config(self) -> dict:
        return {
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "timeout": EMBEDDING_TIMEOUT_SECONDS,
            "max_retries": EMBEDDING_MAX_RETRIES,
        }

    @traceable
    def initialize_openai_embeddings(
        self,
        embeddings_config: dict,
    ) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            **embeddings_config,
        )

    @traceable
    def create_embeddings_client(self) -> OpenAIEmbeddings:
        start = time.perf_counter()
        parent_run = get_current_run_tree()
        child_trace = {"parent": parent_run} if parent_run is not None else None

        embeddings_config = self.build_embeddings_config(
            langsmith_extra=child_trace,
        )
        embeddings = self.initialize_openai_embeddings(
            embeddings_config,
            langsmith_extra=child_trace,
        )

        annotate_run(
            step="create_embeddings_client",
            elapsed_seconds=round(time.perf_counter() - start, 3),
            embedding_model=embeddings_config["model"],
            embedding_dimensions=embeddings_config["dimensions"],
            embedding_timeout_seconds=embeddings_config["timeout"],
            embedding_max_retries=embeddings_config["max_retries"],
        )
        return embeddings

    @traceable
    def initialize_pgvector(
        self,
        connection: str,
        embeddings: OpenAIEmbeddings,
    ) -> PGVector:
        return PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=connection,
            use_jsonb=True,
            engine_args={
                # Verify a pooled connection is alive before handing it out, so
                # a silently-dropped (stale) socket gets replaced instead of
                # causing a multi-minute hang on the next request.
                "pool_pre_ping": True,
                # Recycle connections older than 5 min — many proxies/DBs drop
                # idle connections, so we retire them before they go stale.
                "pool_recycle": 300,
                "connect_args": {
                    # Cap how long a single SQL statement may run server-side.
                    "options": f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}",
                },
            },
        )

    @traceable
    def create_vector_store(self) -> PGVector:
        """Connect to the Postgres vector store, reusing the connection.

        The PGVector instance owns a SQLAlchemy engine / connection pool, so we
        cache it on the class and hand back the same one on every call instead
        of paying connection-setup cost per request.
        """
        start = time.perf_counter()
        if RagService._vector_store is not None:
            annotate_run(
                step="create_vector_store",
                cache_hit=True,
                elapsed_seconds=round(time.perf_counter() - start, 3),
            )
            return RagService._vector_store

        parent_run = get_current_run_tree()
        child_trace = {"parent": parent_run} if parent_run is not None else None

        connection = self.build_connection_string(langsmith_extra=child_trace)
        embeddings = self.create_embeddings_client(langsmith_extra=child_trace)
        RagService._vector_store = self.initialize_pgvector(
            connection,
            embeddings,
            langsmith_extra=child_trace,
        )

        annotate_run(
            step="create_vector_store",
            cache_hit=False,
            elapsed_seconds=round(time.perf_counter() - start, 3),
        )
        return RagService._vector_store

    @traceable
    def embed_question(self, vector_store: PGVector, question: str) -> list[float]:
        """Embed the question via OpenAI. Split out as its own span so LangSmith
        shows the network round-trip time separately from the DB search."""
        start = time.perf_counter()
        vector = vector_store.embeddings.embed_query(question)
        annotate_run(
            step="embed_question",
            elapsed_seconds=round(time.perf_counter() - start, 3),
            embedding_model=EMBEDDING_MODEL,
            embedding_dimensions=len(vector),
        )
        return vector

    @traceable
    def search_by_vector(self, vector_store: PGVector, vector: list[float]) -> list:
        """Run the Postgres vector search. Its own span so LangSmith shows the
        DB time separately from the embedding call above."""
        start = time.perf_counter()
        results = vector_store.similarity_search_with_score_by_vector(vector, k=TOP_K)
        scores = [score for _, score in results]
        annotate_run(
            step="search_by_vector",
            elapsed_seconds=round(time.perf_counter() - start, 3),
            top_k=TOP_K,
            num_results=len(results),
            min_score=round(min(scores), 4) if scores else None,
            max_score=round(max(scores), 4) if scores else None,
        )
        return results

    @traceable
    def log_retrieved_chunks(self, results: list) -> None:
        """Print retrieved chunks. Traced separately in case local logging ever
        becomes noticeable with large chunks or slow stdout."""
        start = time.perf_counter()
        print("\n===== RETRIEVED CHUNKS =====")
        for i, (document, score) in enumerate(results):
            print(f"\n----- Chunk {i + 1} -----")
            print(f"Score: {score}")
            print(document.page_content[:1000])

        annotate_run(
            step="log_retrieved_chunks",
            elapsed_seconds=round(time.perf_counter() - start, 3),
            num_results=len(results),
        )

    @traceable
    def retrieve_chunks(self, question: str) -> list:
        """Find the most relevant resume chunks for the question.

        Returns a list of (Document, similarity_score) tuples so we can log the
        scores, just like the previous implementation did.

        The embedding call and the DB search are separate spans so LangSmith
        shows which one is slow.
        """
        start = time.perf_counter()
        parent_run = get_current_run_tree()
        child_trace = {"parent": parent_run} if parent_run is not None else None

        vector_store = self.create_vector_store(langsmith_extra=child_trace)
        vector = self.embed_question(
            vector_store,
            question,
            langsmith_extra=child_trace,
        )
        results = self.search_by_vector(
            vector_store,
            vector,
            langsmith_extra=child_trace,
        )

        annotate_run(
            step="retrieve_chunks",
            elapsed_seconds=round(time.perf_counter() - start, 3),
            top_k=TOP_K,
            num_results=len(results),
        )

        self.log_retrieved_chunks(results, langsmith_extra=child_trace)

        return results

    @traceable
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

        start = time.perf_counter()
        message = chain.invoke({"context": context, "question": question})
        answer = message.content

        # usage_metadata is the provider-agnostic token count LangChain attaches
        # to the response; default to an empty dict if the provider omits it.
        usage = message.usage_metadata or {}

        annotate_run(
            step="generate_answer",
            elapsed_seconds=round(time.perf_counter() - start, 3),
            chat_model=CHAT_MODEL,
            num_context_chunks=len(results),
            total_tokens=usage.get("total_tokens"),
        )

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
        parent_run = get_current_run_tree()
        child_trace = {"parent": parent_run} if parent_run is not None else None

        results = self.retrieve_chunks(question, langsmith_extra=child_trace)
        return self.generate_answer(question, results, langsmith_extra=child_trace)
