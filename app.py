"""
SPARSA-LM Application - Chat Interface with RAG and RIG

Features:
- OpenWebUI compatible API
- RAG (Retrieval Augmented Generation) with FAISS/ChromaDB
- RIG (Retrieval Interleaved Generation) for active retrieval
- Gradio standalone interface
- HuggingFace Hub integration

This code belongs to EllanorAI and is licensed under the EllanorAI Proprietary License.
"""

import os
import sys
import json
import logging
import hashlib
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, AsyncIterator, Union
from pathlib import Path
import time
import uuid

import torch
import torch.nn.functional as F
import numpy as np

# Optional imports
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from huggingface_hub import HfApi, upload_folder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import SPARSALM, SPARSAConfig
from src.inference import create_inference_engine, InferenceConfig, InferenceServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    # Vector store settings
    vector_store: str = "faiss"  # "faiss" or "chromadb"
    collection_name: str = "sparsa_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.5
    max_context_length: int = 2048

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Document sources
    document_paths: List[str] = field(default_factory=list)

    # Persistence
    persist_directory: str = "data/vector_store"


class DocumentChunker:
    """Chunk documents for vector store indexing."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk = {
                "text": chunk_text,
                "metadata": metadata or {},
                "start_idx": i,
                "end_idx": i + len(chunk_words),
            }
            chunks.append(chunk)

        return chunks

    def chunk_document(self, filepath: str) -> List[Dict]:
        """Chunk a document file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        metadata = {
            "source": filepath,
            "filename": os.path.basename(filepath),
        }

        return self.chunk_text(text, metadata)


class EmbeddingModel:
    """Wrapper for embedding model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            logger.warning("SentenceTransformers not available. Using random embeddings.")
            self.embedding_dim = 384

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        else:
            # Fallback: random embeddings (for testing)
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(self, config: RAGConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        self.config = config
        self.embedding_model = EmbeddingModel(config.embedding_model)
        self.index = None
        self.documents: List[Dict] = []

        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index."""
        # Use IVF index for larger collections
        self.index = faiss.IndexFlatIP(self.embedding_model.embedding_dim)  # Inner product for cosine sim

    def add_documents(self, documents: List[Dict]):
        """Add documents to the index."""
        if not documents:
            return

        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.encode(texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)

        logger.info(f"Added {len(documents)} documents to FAISS index")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search for similar documents."""
        top_k = top_k or self.config.top_k

        # Encode query
        query_embedding = self.embedding_model.encode(query)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            if score >= self.config.similarity_threshold:
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)

        return results

    def save(self, path: Optional[str] = None):
        """Save index to disk."""
        path = path or self.config.persist_directory
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save documents
        with open(os.path.join(path, "documents.json"), 'w') as f:
            json.dump(self.documents, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: Optional[str] = None):
        """Load index from disk."""
        path = path or self.config.persist_directory

        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "documents.json")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'r') as f:
                self.documents = json.load(f)
            logger.info(f"Loaded vector store from {path} ({len(self.documents)} documents)")
        else:
            logger.warning(f"No saved index found at {path}")


class ChromaDBVectorStore:
    """ChromaDB-based vector store."""

    def __init__(self, config: RAGConfig):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")

        self.config = config
        self.embedding_model = EmbeddingModel(config.embedding_model)

        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=config.persist_directory,
            anonymized_telemetry=False,
        ))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Dict]):
        """Add documents to ChromaDB."""
        if not documents:
            return

        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        metadatas = [doc.get("metadata", {}) for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(documents)} documents to ChromaDB")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search for similar documents."""
        top_k = top_k or self.config.top_k

        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = []
        if results["documents"] and results["distances"]:
            for doc, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0]),
            ):
                score = 1 - distance  # Convert distance to similarity
                if score >= self.config.similarity_threshold:
                    documents.append({
                        "text": doc,
                        "score": score,
                        "metadata": metadata,
                    })

        return documents


class RAGSystem:
    """Retrieval Augmented Generation system."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunker = DocumentChunker(config.chunk_size, config.chunk_overlap)

        # Initialize vector store
        if config.vector_store == "chromadb" and CHROMADB_AVAILABLE:
            self.vector_store = ChromaDBVectorStore(config)
        elif FAISS_AVAILABLE:
            self.vector_store = FAISSVectorStore(config)
        else:
            raise ImportError("No vector store available. Install faiss-cpu or chromadb.")

        # Load documents if paths provided
        if config.document_paths:
            self.load_documents(config.document_paths)

    def load_documents(self, paths: List[str]):
        """Load and index documents from paths."""
        all_chunks = []

        for path in paths:
            if os.path.isfile(path):
                chunks = self.chunker.chunk_document(path)
                all_chunks.extend(chunks)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(('.txt', '.md', '.json')):
                            filepath = os.path.join(root, file)
                            chunks = self.chunker.chunk_document(filepath)
                            all_chunks.extend(chunks)

        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            logger.info(f"Loaded {len(all_chunks)} chunks from {len(paths)} paths")

    def add_text(self, text: str, metadata: Optional[Dict] = None):
        """Add text directly to the index."""
        chunks = self.chunker.chunk_text(text, metadata)
        self.vector_store.add_documents(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents."""
        return self.vector_store.search(query, top_k)

    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents as context."""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.get("metadata", {}).get("source", "Unknown")
            context_parts.append(f"[Source {i+1}: {source}]\n{doc['text']}")

        return "\n\n".join(context_parts)


class RIGSystem:
    """
    Retrieval Interleaved Generation system.

    Allows the model to actively query the retrieval tool during generation.
    """

    def __init__(self, rag_system: RAGSystem, model, tokenizer):
        self.rag = rag_system
        self.model = model
        self.tokenizer = tokenizer

        # Special tokens for RIG
        self.retrieve_token = "<|retrieve|>"
        self.context_start = "<|context|>"
        self.context_end = "<|/context|>"

    def generate_with_retrieval(
        self,
        prompt: str,
        max_tokens: int = 512,
        max_retrieval_calls: int = 3,
    ) -> Tuple[str, List[Dict]]:
        """
        Generate response with interleaved retrieval.

        The model can output <|retrieve|>query<|/retrieve|> to trigger retrieval.
        """
        full_response = ""
        retrieval_history = []
        retrieval_count = 0

        current_prompt = prompt

        while retrieval_count < max_retrieval_calls:
            # Generate until retrieval token or end
            response = self._generate_segment(current_prompt, max_tokens)

            # Check for retrieval request
            if self.retrieve_token in response:
                # Extract query
                parts = response.split(self.retrieve_token)
                pre_retrieve = parts[0]
                full_response += pre_retrieve

                if len(parts) > 1:
                    query_part = parts[1]
                    if "<|/retrieve|>" in query_part:
                        query = query_part.split("<|/retrieve|>")[0].strip()

                        # Perform retrieval
                        docs = self.rag.retrieve(query)
                        retrieval_history.append({
                            "query": query,
                            "documents": docs,
                        })

                        # Add context to prompt
                        context = self.rag.format_context(docs)
                        current_prompt = (
                            f"{current_prompt}\n{full_response}\n"
                            f"{self.context_start}\n{context}\n{self.context_end}\n"
                        )

                        retrieval_count += 1
                    else:
                        full_response += response
                        break
            else:
                full_response += response
                break

        return full_response, retrieval_history

    def _generate_segment(self, prompt: str, max_tokens: int) -> str:
        """Generate a segment of text."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device if hasattr(self.model, 'device') else 'cuda')

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )

        return response


class SPARSAChatApp:
    """
    Main chat application with RAG, RIG, and OpenWebUI compatibility.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        rag_config: Optional[RAGConfig] = None,
        use_rig: bool = True,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Initialize inference engine
        self.engine = create_inference_engine(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )

        # Initialize RAG
        self.rag_config = rag_config or RAGConfig()
        self.rag = RAGSystem(self.rag_config)

        # Initialize RIG if enabled
        self.use_rig = use_rig
        if use_rig and hasattr(self.engine, 'model'):
            self.rig = RIGSystem(
                self.rag,
                self.engine.model if hasattr(self.engine, 'model') else None,
                self.engine.tokenizer if hasattr(self.engine, 'tokenizer') else None,
            )
        else:
            self.rig = None

        # Conversation history
        self.conversations: Dict[str, List[Dict]] = {}

    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_rag: bool = True,
        use_rig: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message.
        """
        conversation_id = conversation_id or str(uuid.uuid4())

        # Get or create conversation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        history = self.conversations[conversation_id]

        # Prepare context
        context = ""
        retrieval_results = []

        if use_rag:
            retrieval_results = self.rag.retrieve(message)
            context = self.rag.format_context(retrieval_results)

        # Build prompt
        prompt = self._build_prompt(
            message=message,
            history=history,
            context=context,
            system_prompt=system_prompt,
        )

        # Generate response
        if use_rig and self.rig is not None:
            response, rig_history = self.rig.generate_with_retrieval(prompt)
            retrieval_results.extend(rig_history)
        else:
            responses = self.engine.generate(
                [prompt],
                max_tokens=512,
                temperature=0.7,
            )
            response = responses[0]

        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "conversation_id": conversation_id,
            "retrieval_results": retrieval_results,
        }

    def _build_prompt(
        self,
        message: str,
        history: List[Dict],
        context: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build prompt from message, history, and context."""
        parts = []

        # System prompt
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}")
        else:
            parts.append("<|system|>\nYou are SPARSA-LM, a helpful AI assistant.")

        # Add context if available
        if context:
            parts.append(f"\n\nRelevant Context:\n{context}")

        # Add history
        for msg in history[-10:]:  # Last 10 messages
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"\n<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"\n<|assistant|>\n{content}")

        # Add current message
        parts.append(f"\n<|user|>\n{message}")
        parts.append("\n<|assistant|>")

        return "".join(parts)

    def clear_conversation(self, conversation_id: str):
        """Clear a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to RAG index."""
        metadatas = metadatas or [{}] * len(texts)
        for text, metadata in zip(texts, metadatas):
            self.rag.add_text(text, metadata)


# ============================================================================
# OpenWebUI Compatible API
# ============================================================================

if FASTAPI_AVAILABLE:

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "sparsa-360m"
        messages: List[ChatMessage]
        temperature: float = 0.7
        max_tokens: int = 2048
        stream: bool = False
        use_rag: bool = True

    class CompletionRequest(BaseModel):
        model: str = "sparsa-360m"
        prompt: str
        max_tokens: int = 2048
        temperature: float = 0.7
        stream: bool = False

    def create_openwebui_api(app: SPARSAChatApp) -> FastAPI:
        """Create OpenWebUI compatible FastAPI application."""

        api = FastAPI(
            title="SPARSA-LM API",
            description="OpenWebUI compatible API for SPARSA-LM",
            version="1.0.0",
        )

        @api.get("/v1/models")
        async def list_models():
            """List available models."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "sparsa-360m",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ellanor-ai",
                    }
                ],
            }

        @api.post("/v1/chat/completions")
        async def chat_completions(request: ChatRequest):
            """Chat completions endpoint."""
            # Extract messages
            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            # Get last user message
            user_message = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    user_message = msg["content"]
                    break

            # Generate response
            result = app.chat(
                message=user_message,
                use_rag=request.use_rag,
            )

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result["response"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(result["response"].split()),
                    "total_tokens": len(user_message.split()) + len(result["response"].split()),
                },
            }

            return response

        @api.post("/v1/completions")
        async def completions(request: CompletionRequest):
            """Text completions endpoint."""
            result = app.chat(message=request.prompt)

            return {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": result["response"],
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
            }

        @api.post("/v1/rag/add")
        async def add_documents(documents: List[Dict[str, str]]):
            """Add documents to RAG index."""
            texts = [d["text"] for d in documents]
            metadatas = [d.get("metadata", {}) for d in documents]
            app.add_documents(texts, metadatas)
            return {"status": "success", "count": len(texts)}

        @api.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy"}

        return api


# ============================================================================
# Gradio Interface
# ============================================================================

def create_gradio_interface(app: SPARSAChatApp):
    """Create Gradio chat interface."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not available. Install with: pip install gradio")

    def chat_fn(message: str, history: List[Tuple[str, str]], use_rag: bool):
        """Chat function for Gradio."""
        result = app.chat(message=message, use_rag=use_rag)
        return result["response"]

    def add_document_fn(text: str):
        """Add document function."""
        app.add_documents([text])
        return "Document added successfully!"

    with gr.Blocks(title="SPARSA-LM Chat") as interface:
        gr.Markdown("# SPARSA-LM Chat Interface")
        gr.Markdown("A 360M parameter LLaMA-style language model with RAG capabilities.")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Message", placeholder="Type your message here...")
            use_rag = gr.Checkbox(label="Use RAG", value=True)

            with gr.Row():
                submit = gr.Button("Send")
                clear = gr.Button("Clear")

            def respond(message, chat_history, use_rag):
                response = chat_fn(message, chat_history, use_rag)
                chat_history.append((message, response))
                return "", chat_history

            submit.click(respond, [msg, chatbot, use_rag], [msg, chatbot])
            msg.submit(respond, [msg, chatbot, use_rag], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.Tab("RAG Management"):
            gr.Markdown("### Add Documents to Knowledge Base")
            doc_text = gr.Textbox(label="Document Text", lines=10)
            add_btn = gr.Button("Add Document")
            status = gr.Textbox(label="Status")

            add_btn.click(add_document_fn, [doc_text], [status])

        with gr.Tab("Settings"):
            gr.Markdown("### Model Settings")
            gr.Markdown(f"Model Path: {app.model_path}")
            gr.Markdown(f"Tokenizer Path: {app.tokenizer_path}")

    return interface


# ============================================================================
# HuggingFace Hub Integration
# ============================================================================

def save_to_hub(
    model_path: str,
    tokenizer_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
):
    """
    Save model and tokenizer to HuggingFace Hub.
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")

    api = HfApi()

    # Create repo if not exists
    api.create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
    )

    # Upload model
    upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        token=token,
    )

    # Upload tokenizer
    if tokenizer_path != model_path:
        upload_folder(
            folder_path=tokenizer_path,
            repo_id=repo_id,
            token=token,
            path_in_repo="tokenizer",
        )

    logger.info(f"Model uploaded to https://huggingface.co/{repo_id}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SPARSA-LM Chat Application")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--mode", type=str, choices=["api", "gradio", "cli"], default="api")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--rag_documents", type=str, nargs="*", help="Paths to RAG documents")
    parser.add_argument("--push_to_hub", type=str, help="HuggingFace repo ID to push to")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token")
    args = parser.parse_args()

    # Configure RAG
    rag_config = RAGConfig(
        document_paths=args.rag_documents or [],
    )

    # Create app
    app = SPARSAChatApp(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        rag_config=rag_config,
    )

    # Push to hub if requested
    if args.push_to_hub:
        save_to_hub(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            repo_id=args.push_to_hub,
            token=args.hf_token,
        )
        return

    # Start appropriate interface
    if args.mode == "api":
        if not FASTAPI_AVAILABLE:
            print("FastAPI not available. Install with: pip install fastapi uvicorn")
            return

        api = create_openwebui_api(app)
        uvicorn.run(api, host=args.host, port=args.port)

    elif args.mode == "gradio":
        if not GRADIO_AVAILABLE:
            print("Gradio not available. Install with: pip install gradio")
            return

        interface = create_gradio_interface(app)
        interface.launch(server_name=args.host, server_port=args.port, share=True)

    elif args.mode == "cli":
        print("SPARSA-LM Chat CLI")
        print("Type 'quit' to exit, 'clear' to clear history, 'rag on/off' to toggle RAG")
        print("-" * 50)

        use_rag = True
        conversation_id = str(uuid.uuid4())

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    app.clear_conversation(conversation_id)
                    conversation_id = str(uuid.uuid4())
                    print("Conversation cleared.")
                    continue
                elif user_input.lower() == 'rag on':
                    use_rag = True
                    print("RAG enabled.")
                    continue
                elif user_input.lower() == 'rag off':
                    use_rag = False
                    print("RAG disabled.")
                    continue

                result = app.chat(
                    message=user_input,
                    conversation_id=conversation_id,
                    use_rag=use_rag,
                )

                print(f"\nAssistant: {result['response']}")

                if use_rag and result.get('retrieval_results'):
                    print(f"\n[Retrieved {len(result['retrieval_results'])} documents]")

            except KeyboardInterrupt:
                break

        print("\nGoodbye!")


if __name__ == "__main__":
    main()
