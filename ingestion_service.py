import os
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Configuration -----------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"

# Semantic chunking splits where the *meaning* changes rather than at a fixed
# length. "percentile" flags a breakpoint whenever the similarity gap between
# two consecutive sentences is larger than this percentile of all gaps. Higher
# => fewer, larger chunks; lower => more, smaller chunks.
SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 90

# Safety net: semantic chunks have no hard size limit, so we re-split any chunk
# that grows too large. These are characters, kept well under the embedding
# model's token limit while staying large enough to preserve context.
MAX_CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

# LangChain's PGVector stores all chunks in its own tables and groups them
# under a "collection" name (the logical equivalent of the old table name).
COLLECTION_NAME = "resume_chunks"

# A loader per supported extension. Each value is a callable that takes a file
# path and returns a LangChain document loader instance.
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
}


class IngestionService:
    """Loads a document, splits it into chunks, embeds them and stores the
    vectors in Postgres — all through LangChain."""

    def get_file_metadata(self, file_path: str) -> dict:
        """Metadata attached to every chunk so we can trace it back later."""
        path = Path(file_path)

        return {
            "file_name": path.name,
            "file_type": path.suffix.lower(),
        }

    def clean_text(self, text: str) -> str:
        """Strip NUL bytes — Postgres text columns reject them."""
        if not text:
            return ""

        return text.replace("\x00", "").strip()

    def load_documents(self, file_path: str) -> list[Document]:
        """Pick the right LangChain loader for the file type and load it."""
        extension = Path(file_path).suffix.lower()

        loader_factory = LOADERS.get(extension)
        if not loader_factory:
            raise ValueError(f"Unsupported file type: {extension}")

        return loader_factory(file_path).load()

    def split_documents(
        self,
        documents: list[Document],
        metadata: dict,
        embeddings: OpenAIEmbeddings,
    ) -> list[Document]:
        """Split documents into semantically coherent chunks.

        1. SemanticChunker groups sentences and cuts only where the meaning
           shifts, so a job entry or bullet stays intact instead of being
           sliced at an arbitrary character count.
        2. A recursive splitter then trims any chunk that exceeded
           MAX_CHUNK_SIZE, guarding against an occasional oversized block.
        3. Finally we clean each chunk, drop empties and tag file metadata.
        """
        # Step 1 — semantic, meaning-aware splitting.
        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        )
        chunks = semantic_splitter.split_documents(documents)

        # Step 2 — enforce a hard upper bound on chunk size.
        size_guard = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        chunks = size_guard.split_documents(chunks)

        # Step 3 — clean, filter and tag.
        clean_chunks = []
        for chunk in chunks:
            chunk.page_content = self.clean_text(chunk.page_content)
            if not chunk.page_content:
                continue

            chunk.metadata.update(metadata)
            clean_chunks.append(chunk)

        return clean_chunks

    def create_embeddings(self) -> OpenAIEmbeddings:
        """Single embeddings client reused for both chunking and storage."""
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def create_vector_store(self, embeddings: OpenAIEmbeddings) -> PGVector:
        """Connect to the Postgres vector store using env-based credentials."""
        connection = (
            f"postgresql+psycopg://{os.getenv('DB_USER')}:"
            f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/"
            f"{os.getenv('DB_NAME')}"
        )

        return PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=connection,
            use_jsonb=True,
        )

    def ingest_file(self, file_path: str) -> int:
        """Load -> split -> embed -> store. Returns the number of chunks stored.

        This is the public entry point and keeps the same signature as before.
        """
        print(f"Processing file: {file_path}")

        documents = self.load_documents(file_path)
        if not documents:
            raise ValueError("No text found in document")

        # Reuse one embeddings client so semantic chunking and storage share it.
        embeddings = self.create_embeddings()

        chunks = self.split_documents(
            documents=documents,
            metadata=self.get_file_metadata(file_path),
            embeddings=embeddings,
        )
        if not chunks:
            raise ValueError("No text found in document")

        print("First 500 characters:")
        print(chunks[0].page_content[:500])
        print(f"Created {len(chunks)} chunks")

        # add_documents embeds each chunk and writes the vectors to Postgres.
        self.create_vector_store(embeddings).add_documents(chunks)

        print("Embeddings saved successfully")

        return len(chunks)
