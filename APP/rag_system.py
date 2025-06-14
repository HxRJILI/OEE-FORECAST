"""
RAG (Retrieval-Augmented Generation) System for OEE Advisory
Uses Gemini API for generation and vector similarity for retrieval
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import pickle
from datetime import datetime
import sqlite3

# Vector operations
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Gemini API
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Document processing
from document_processor import DocumentProcessor, create_document_processor


class VectorStore:
    """
    Vector storage and similarity search using FAISS
    """
    
    def __init__(self, vector_db_path: str = "vector_db"):
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.index_file = self.vector_db_path / "faiss_index.bin"
        self.metadata_file = self.vector_db_path / "metadata.json"
        self.chunks_db_file = self.vector_db_path / "chunks.db"
        
        self.index = None
        self.metadata = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize SQLite database for chunks
        self._init_chunks_database()
        
        # Load existing index if available
        self._load_index()
        
        self.logger = logging.getLogger("VectorStore")
    
    def _init_chunks_database(self):
        """Initialize SQLite database for storing chunk data"""
        conn = sqlite3.connect(str(self.chunks_db_file))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                oee_relevance_score REAL,
                key_terms TEXT,
                entities TEXT,
                topics TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_hash ON chunks(document_hash)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_oee_relevance ON chunks(oee_relevance_score)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                self.logger.info(f"Loaded existing vector index with {len(self.metadata)} vectors")
            else:
                self._create_new_index()
                
        except Exception as e:
            self.logger.warning(f"Failed to load existing index: {e}. Creating new index.")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        self.metadata = []
        self.logger.info("Created new vector index")
    
    def add_document_chunks(self, document_hash: str, chunks: List[Dict]):
        """Add document chunks to the vector store"""
        vectors = []
        chunk_metadata = []
        
        conn = sqlite3.connect(str(self.chunks_db_file))
        cursor = conn.cursor()
        
        try:
            for chunk in chunks:
                if chunk.get("embedding"):
                    # Normalize vector for cosine similarity
                    embedding = np.array(chunk["embedding"], dtype=np.float32)
                    embedding = embedding / np.linalg.norm(embedding)
                    vectors.append(embedding)
                    
                    # Prepare metadata for FAISS
                    metadata_entry = {
                        "document_hash": document_hash,
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "oee_relevance_score": chunk.get("oee_relevance_score", 0.0),
                        "length": chunk.get("length", 0),
                        "word_count": chunk.get("word_count", 0),
                        "key_terms": chunk.get("key_terms", []),
                        "entities": chunk.get("entities", []),
                        "topics": chunk.get("topics", [])
                    }
                    chunk_metadata.append(metadata_entry)
                    
                    # Store in SQLite
                    cursor.execute('''
                        INSERT INTO chunks (
                            document_hash, chunk_id, text, oee_relevance_score,
                            key_terms, entities, topics, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        document_hash,
                        chunk["chunk_id"],
                        chunk["text"],
                        chunk.get("oee_relevance_score", 0.0),
                        json.dumps(chunk.get("key_terms", [])),
                        json.dumps(chunk.get("entities", [])),
                        json.dumps(chunk.get("topics", [])),
                        json.dumps(chunk.get("metadata", {}))
                    ))
            
            if vectors:
                # Add to FAISS index
                vectors_array = np.vstack(vectors)
                self.index.add(vectors_array)
                
                # Add metadata
                self.metadata.extend(chunk_metadata)
                
                # Save updated index
                self._save_index()
                
                conn.commit()
                self.logger.info(f"Added {len(vectors)} chunks from document {document_hash}")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to add document chunks: {e}")
            raise
        finally:
            conn.close()
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            faiss.write_index(self.index, str(self.index_file))
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
    
    def search_similar(self, 
                      query_embedding: np.ndarray, 
                      k: int = 5,
                      min_relevance_score: float = 0.0) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            min_relevance_score: Minimum OEE relevance score filter
            
        Returns:
            List of similar chunks with metadata
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query vector
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.index.search(query_vector, min(k * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    chunk_metadata = self.metadata[idx].copy()
                    chunk_metadata["similarity_score"] = float(score)
                    
                    # Filter by OEE relevance if specified
                    if chunk_metadata.get("oee_relevance_score", 0.0) >= min_relevance_score:
                        results.append(chunk_metadata)
            
            # Sort by similarity score and return top k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_document_chunks(self, document_hash: str) -> List[Dict]:
        """Get all chunks for a specific document"""
        conn = sqlite3.connect(str(self.chunks_db_file))
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM chunks WHERE document_hash = ?
                ORDER BY chunk_id
            ''', (document_hash,))
            
            rows = cursor.fetchall()
            chunks = []
            
            for row in rows:
                chunk = {
                    "id": row[0],
                    "document_hash": row[1],
                    "chunk_id": row[2],
                    "text": row[3],
                    "oee_relevance_score": row[4],
                    "key_terms": json.loads(row[5]) if row[5] else [],
                    "entities": json.loads(row[6]) if row[6] else [],
                    "topics": json.loads(row[7]) if row[7] else [],
                    "metadata": json.loads(row[8]) if row[8] else {},
                    "created_at": row[9]
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get document chunks: {e}")
            return []
        finally:
            conn.close()
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        conn = sqlite3.connect(str(self.chunks_db_file))
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM chunks')
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT document_hash) FROM chunks')
            total_documents = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(oee_relevance_score) FROM chunks')
            avg_relevance = cursor.fetchone()[0] or 0.0
            
            return {
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "avg_oee_relevance": avg_relevance,
                "vector_dimension": self.dimension,
                "index_total": self.index.ntotal if self.index else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}
        finally:
            conn.close()


class OEEAdvisor:
    """
    OEE Advisory system using RAG with Gemini API
    """
    
    def __init__(self, 
             gemini_api_key: str,
             vector_store: VectorStore = None,
             document_processor: DocumentProcessor = None,
             model_name: str = 'gemini-1.5-flash'):
        """
        Initialize the OEE Advisor
        
        Args:
            gemini_api_key: Google Gemini API key
            vector_store: Vector store instance
            document_processor: Document processor instance
        """
        self.api_key = gemini_api_key
        self.vector_store = vector_store or VectorStore()
        self.document_processor = document_processor or create_document_processor()
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize Gemini model
        # Initialize Gemini model - Updated to use available model
        # Initialize Gemini model with configurable model name
        self.model = genai.GenerativeModel(model_name)
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        self.logger = logging.getLogger("OEEAdvisor")
        
        # Conversation history
        self.conversation_history = []
        
        # System prompt for OEE context
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for OEE advisory context"""
        return """
You are an expert OEE (Overall Equipment Effectiveness) advisor with deep knowledge in manufacturing, 
production optimization, and industrial engineering. You specialize in helping manufacturing teams 
improve their equipment effectiveness through data-driven insights and best practices.

Your expertise includes:
- OEE calculation and analysis (Availability × Performance × Quality)
- Manufacturing process optimization
- Equipment maintenance strategies
- Quality management systems
- Lean manufacturing principles
- Root cause analysis for production issues
- Continuous improvement methodologies

When answering questions:
1. Provide practical, actionable advice
2. Reference specific data or examples when available
3. Consider the manufacturing context and constraints
4. Suggest measurable improvement strategies
5. Be clear about assumptions and limitations
6. Use data from the provided context when relevant

Always structure your responses to be helpful for manufacturing professionals at different levels,
from operators to management. Focus on solutions that can realistically be implemented in a 
manufacturing environment.
"""
    
    def add_document(self, file_path: str = None, uploaded_file = None) -> str:
        """
        Add a document to the knowledge base
        
        Args:
            file_path: Path to local file
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Document hash
        """
        try:
            if file_path:
                results = self.document_processor.process_document(file_path)
            elif uploaded_file:
                results = self.document_processor.process_uploaded_document(uploaded_file)
            else:
                raise ValueError("Either file_path or uploaded_file must be provided")
            
            # Add to vector store
            self.vector_store.add_document_chunks(
                results["document_hash"], 
                results["chunks"]
            )
            
            self.logger.info(f"Added document to knowledge base: {results['document_hash']}")
            return results["document_hash"]
            
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            raise
    
    def get_relevant_context(self, 
                           query: str, 
                           max_chunks: int = 5,
                           min_relevance_score: float = 5.0) -> Tuple[List[Dict], str]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            max_chunks: Maximum number of chunks to retrieve
            min_relevance_score: Minimum OEE relevance score
            
        Returns:
            Tuple of (relevant_chunks, formatted_context)
        """
        try:
            # Generate query embedding
            if not self.document_processor.embedding_model:
                return [], ""
            
            query_embedding = self.document_processor.embedding_model.encode(query)
            
            # Search for similar chunks
            relevant_chunks = self.vector_store.search_similar(
                query_embedding,
                k=max_chunks,
                min_relevance_score=min_relevance_score
            )
            
            # Format context
            context_parts = []
            for i, chunk in enumerate(relevant_chunks, 1):
                context_parts.append(f"""
Context {i} (Relevance: {chunk.get('similarity_score', 0):.3f}, OEE Score: {chunk.get('oee_relevance_score', 0):.1f}):
{chunk['text']}
""")
            
            formatted_context = "\n".join(context_parts)
            
            return relevant_chunks, formatted_context
            
        except Exception as e:
            self.logger.error(f"Failed to get relevant context: {e}")
            return [], ""
    
    def generate_response(self, 
                         query: str, 
                         context: str = "", 
                         include_context_search: bool = True) -> Dict:
        """
        Generate response using Gemini API with RAG
        
        Args:
            query: User query
            context: Additional context (optional)
            include_context_search: Whether to search for relevant context
            
        Returns:
            Response dictionary
        """
        try:
            # Get relevant context if enabled
            relevant_chunks = []
            retrieved_context = ""
            
            if include_context_search:
                relevant_chunks, retrieved_context = self.get_relevant_context(query)
            
            # Combine contexts
            full_context = ""
            if retrieved_context:
                full_context += f"RELEVANT KNOWLEDGE BASE CONTEXT:\n{retrieved_context}\n\n"
            if context:
                full_context += f"ADDITIONAL CONTEXT:\n{context}\n\n"
            
            # Create prompt
            prompt = self._create_prompt(query, full_context)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            # Extract response text
            response_text = response.text if response.text else "I apologize, but I couldn't generate a response for your query."
            
            # Prepare response data
            response_data = {
                "query": query,
                "response": response_text,
                "relevant_chunks": relevant_chunks,
                "context_used": bool(full_context),
                "timestamp": datetime.now().isoformat(),
                "model": "gemini-pro"
            }
            
            # Add to conversation history
            self.conversation_history.append(response_data)
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return {
                "query": query,
                "response": f"I encountered an error while processing your query: {str(e)}",
                "relevant_chunks": [],
                "context_used": False,
                "timestamp": datetime.now().isoformat(),
                "model": "gemini-pro",
                "error": str(e)
            }
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the full prompt for Gemini"""
        prompt_parts = [self.system_prompt]
        
        if context:
            prompt_parts.append(f"\nRELEVANT CONTEXT:\n{context}")
        
        # Add recent conversation history for context
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            history_text = "\n".join([
                f"Previous Q: {item['query']}\nPrevious A: {item['response'][:200]}..."
                for item in recent_history
            ])
            prompt_parts.append(f"\nRECENT CONVERSATION:\n{history_text}")
        
        prompt_parts.append(f"\nUSER QUESTION: {query}")
        prompt_parts.append("\nPlease provide a helpful, accurate response based on the context and your OEE expertise:")
        
        return "\n".join(prompt_parts)
    
    def ask_question(self, question: str, additional_context: str = "") -> Dict:
        """
        Main interface for asking questions
        
        Args:
            question: User question
            additional_context: Additional context to include
            
        Returns:
            Response dictionary
        """
        return self.generate_response(question, additional_context)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get knowledge base statistics"""
        stats = self.vector_store.get_stats()
        stats["processor_info"] = {
            "embedding_model": self.document_processor.embedding_model_name,
            "chunk_size": self.document_processor.chunk_size,
            "chunk_overlap": self.document_processor.chunk_overlap
        }
        return stats
    
    def suggest_improvements(self, oee_data: Dict) -> Dict:
        """
        Suggest improvements based on OEE data
        
        Args:
            oee_data: Dictionary containing OEE metrics
            
        Returns:
            Improvement suggestions
        """
        # Format OEE data as context
        context = f"""
Current OEE Performance Data:
- Overall OEE: {oee_data.get('oee', 0):.1%}
- Availability: {oee_data.get('availability', 0):.1%}
- Performance: {oee_data.get('performance', 0):.1%}
- Quality: {oee_data.get('quality', 0):.1%}
- Production Line: {oee_data.get('line', 'Unknown')}
- Time Period: {oee_data.get('period', 'Unknown')}
"""
        
        if 'additional_metrics' in oee_data:
            context += f"\nAdditional Metrics:\n{oee_data['additional_metrics']}"
        
        question = """
Based on the OEE performance data provided, please analyze the current performance and provide:

1. **Performance Assessment**: How does this OEE performance compare to industry standards?
2. **Key Issues Identification**: Which component (Availability, Performance, or Quality) needs the most attention?
3. **Root Cause Analysis**: What are the likely causes for the identified issues?
4. **Improvement Recommendations**: Specific, actionable steps to improve OEE
5. **Implementation Priority**: Which improvements should be tackled first?
6. **Expected Impact**: What improvement in OEE percentage can be expected from each recommendation?

Please provide practical, implementable solutions suitable for a manufacturing environment.
"""
        
        return self.generate_response(question, context)


def create_oee_advisor(gemini_api_key: str, model_name: str = 'gemini-1.5-flash') -> OEEAdvisor:
    """
    Factory function to create OEE Advisor
    
    Args:
        gemini_api_key: Google Gemini API key
        model_name: Gemini model to use (default: gemini-1.5-flash)
        
    Returns:
        Configured OEEAdvisor instance
    """
    vector_store = VectorStore()
    document_processor = create_document_processor()
    
    advisor = OEEAdvisor(
        gemini_api_key=gemini_api_key,
        vector_store=vector_store,
        document_processor=document_processor,
        model_name=model_name
    )
    
    return advisor


if __name__ == "__main__":
    # Test the RAG system
    import os
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDtIOvSP1jp7dz91ByO1U44p2HZB5Q23BI")
    
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    # Create advisor
    advisor = create_oee_advisor(api_key)
    
    # Test with a simple question
    response = advisor.ask_question("What is OEE and how is it calculated?")
    print("Response:", response["response"])
    
    # Get knowledge base stats
    stats = advisor.get_knowledge_base_stats()
    print("Knowledge base stats:", stats)