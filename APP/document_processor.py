"""
Document Processor for OEE Advisory System
Handles PDF processing, text extraction, and NLP preprocessing
"""

import os
import re
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# PDF Processing
import PyPDF2
import pdfplumber
from io import BytesIO

# NLP and Text Processing
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Embeddings and Vector Processing
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Download required NLTK data
def download_nltk_requirements():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('chunkers/maxent_ne_chunker')
        nltk.data.find('corpora/words')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('wordnet', quiet=True)

# Initialize NLP components
download_nltk_requirements()

class DocumentProcessor:
    """
    Advanced document processor for OEE-related documents
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_length: int = 100):
        """
        Initialize the document processor
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_length: Minimum length for a chunk to be valid
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize NLP components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model (try different models)
        self.nlp = self._load_spacy_model()
        
        # Initialize embedding model
        self.embedding_model = self._load_embedding_model()
        
        # OEE-specific terms and patterns
        self.oee_terms = self._load_oee_vocabulary()
        
        # Create processing directories
        self.processed_docs_dir = Path("processed_documents")
        self.embeddings_dir = Path("embeddings")
        self.metadata_dir = Path("document_metadata")
        
        for dir_path in [self.processed_docs_dir, self.embeddings_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the document processor"""
        logger = logging.getLogger("DocumentProcessor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_spacy_model(self):
        """Load spaCy model with fallback options"""
        models_to_try = ["en_core_web_sm", "en_core_web_md", "en"]
        
        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name)
                self.logger.info(f"Loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                continue
        
        self.logger.warning("No spaCy model found. Some NLP features will be limited.")
        return None
    
    def _load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            model = SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            return None
    
    def _load_oee_vocabulary(self) -> Dict[str, List[str]]:
        """Load OEE-specific vocabulary and patterns"""
        return {
            "metrics": [
                "oee", "overall equipment effectiveness", "availability", "performance", 
                "quality", "efficiency", "utilization", "throughput", "yield",
                "downtime", "uptime", "cycle time", "changeover", "setup time"
            ],
            "manufacturing": [
                "production", "manufacturing", "equipment", "machine", "line",
                "process", "operation", "maintenance", "breakdown", "repair",
                "scheduling", "planning", "capacity", "bottleneck"
            ],
            "quality": [
                "defect", "scrap", "rework", "first pass yield", "quality rate",
                "inspection", "testing", "specification", "tolerance", "variation"
            ],
            "improvement": [
                "kaizen", "lean", "six sigma", "continuous improvement", "optimization",
                "root cause", "analysis", "action plan", "corrective action",
                "preventive maintenance", "tpm", "total productive maintenance"
            ]
        }
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF file using multiple methods
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        text_content = ""
        metadata = {
            "file_path": file_path,
            "file_size": 0,
            "pages": 0,
            "extraction_method": None,
            "processing_time": None
        }
        
        start_time = datetime.now()
        
        try:
            # Get file info
            file_size = os.path.getsize(file_path)
            metadata["file_size"] = file_size
            
            # Method 1: Try pdfplumber first (better for complex layouts)
            try:
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    metadata["pages"] = len(pdf.pages)
                    
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            # Clean and add page text
                            cleaned_text = self._clean_extracted_text(page_text)
                            text_parts.append(f"[Page {page_num + 1}]\n{cleaned_text}\n")
                    
                    text_content = "\n".join(text_parts)
                    metadata["extraction_method"] = "pdfplumber"
                    
            except Exception as e:
                self.logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
                
                # Method 2: Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata["pages"] = len(pdf_reader.pages)
                    text_parts = []
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self._clean_extracted_text(page_text)
                            text_parts.append(f"[Page {page_num + 1}]\n{cleaned_text}\n")
                    
                    text_content = "\n".join(text_parts)
                    metadata["extraction_method"] = "PyPDF2"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata["processing_time"] = processing_time
            
            self.logger.info(f"Extracted {len(text_content)} characters from {metadata['pages']} pages")
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {e}")
            raise
        
        return text_content, metadata
    
    def extract_text_from_uploaded_file(self, uploaded_file) -> Tuple[str, Dict]:
        """
        Extract text from uploaded Streamlit file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        text_content = ""
        metadata = {
            "file_name": uploaded_file.name,
            "file_size": len(uploaded_file.getvalue()),
            "pages": 0,
            "extraction_method": None,
            "processing_time": None
        }
        
        start_time = datetime.now()
        
        try:
            # Read file content
            file_content = uploaded_file.getvalue()
            
            # Method 1: Try pdfplumber first
            try:
                with pdfplumber.open(BytesIO(file_content)) as pdf:
                    text_parts = []
                    metadata["pages"] = len(pdf.pages)
                    
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self._clean_extracted_text(page_text)
                            text_parts.append(f"[Page {page_num + 1}]\n{cleaned_text}\n")
                    
                    text_content = "\n".join(text_parts)
                    metadata["extraction_method"] = "pdfplumber"
                    
            except Exception as e:
                self.logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
                
                # Method 2: Fallback to PyPDF2
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                metadata["pages"] = len(pdf_reader.pages)
                text_parts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = self._clean_extracted_text(page_text)
                        text_parts.append(f"[Page {page_num + 1}]\n{cleaned_text}\n")
                
                text_content = "\n".join(text_parts)
                metadata["extraction_method"] = "PyPDF2"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata["processing_time"] = processing_time
            
            self.logger.info(f"Extracted {len(text_content)} characters from uploaded file")
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from uploaded file: {e}")
            raise
        
        return text_content, metadata
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.,;:!?\-()%/]', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Add space between number and letter
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Add space between letter and number
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with overlap and enhanced metadata
        
        Args:
            text: Input text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # First, split by paragraphs/sections
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk if it's long enough
                if len(current_chunk) >= self.min_chunk_length:
                    chunk_data = self._create_chunk_data(
                        current_chunk, chunk_id, metadata
                    )
                    chunks.append(chunk_data)
                    chunk_id += 1
                
                # Start new chunk with overlap
                if len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                current_chunk = current_chunk + " " + para if current_chunk else para
        
        # Don't forget the last chunk
        if len(current_chunk) >= self.min_chunk_length:
            chunk_data = self._create_chunk_data(current_chunk, chunk_id, metadata)
            chunks.append(chunk_data)
        
        self.logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _create_chunk_data(self, text: str, chunk_id: int, metadata: Dict = None) -> Dict:
        """Create enhanced chunk data with NLP analysis"""
        chunk_data = {
            "chunk_id": chunk_id,
            "text": text.strip(),
            "length": len(text),
            "word_count": len(text.split()),
            "embedding": None,
            "oee_relevance_score": 0.0,
            "key_terms": [],
            "entities": [],
            "sentences": [],
            "metadata": metadata or {}
        }
        
        # Perform NLP analysis
        chunk_data.update(self._analyze_chunk_content(text))
        
        # Generate embedding
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                chunk_data["embedding"] = embedding.tolist()
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding for chunk {chunk_id}: {e}")
        
        return chunk_data
    
    def _analyze_chunk_content(self, text: str) -> Dict:
        """Perform detailed NLP analysis on chunk content"""
        analysis = {
            "oee_relevance_score": 0.0,
            "key_terms": [],
            "entities": [],
            "sentences": [],
            "topics": [],
            "sentiment": "neutral"
        }
        
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            analysis["sentences"] = sentences
            
            # Calculate OEE relevance score
            analysis["oee_relevance_score"] = self._calculate_oee_relevance(text)
            
            # Extract key terms
            analysis["key_terms"] = self._extract_key_terms(text)
            
            # Extract entities using spaCy if available
            if self.nlp:
                doc = self.nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                analysis["entities"] = entities
                
                # Extract topics/themes
                analysis["topics"] = self._extract_topics(doc)
            
            # Extract entities using NLTK as fallback
            else:
                analysis["entities"] = self._extract_entities_nltk(text)
                
        except Exception as e:
            self.logger.warning(f"NLP analysis failed: {e}")
        
        return analysis
    
    def _calculate_oee_relevance(self, text: str) -> float:
        """Calculate how relevant the text is to OEE topics"""
        text_lower = text.lower()
        total_score = 0.0
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        # Score based on OEE vocabulary presence
        for category, terms in self.oee_terms.items():
            category_score = 0
            for term in terms:
                count = text_lower.count(term.lower())
                category_score += count
            
            # Weight different categories
            weights = {
                "metrics": 3.0,
                "manufacturing": 2.0,
                "quality": 2.5,
                "improvement": 2.0
            }
            
            total_score += category_score * weights.get(category, 1.0)
        
        # Normalize by text length
        relevance_score = min(total_score / total_words * 100, 100.0)
        return relevance_score
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms using various NLP techniques"""
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words 
            and len(word) > 2 
            and word.isalpha()
        ]
        
        # Get word frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top terms
        key_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [term[0] for term in key_terms]
    
    def _extract_entities_nltk(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities using NLTK"""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    entities.append((entity_text, chunk.label()))
            
            return entities
            
        except Exception as e:
            self.logger.warning(f"NLTK entity extraction failed: {e}")
            return []
    
    def _extract_topics(self, doc) -> List[str]:
        """Extract topics/themes from spaCy doc"""
        topics = []
        
        # Look for key phrases and noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:
                topics.append(chunk.text.lower())
        
        # Deduplicate and return top topics
        unique_topics = list(set(topics))[:5]
        return unique_topics
    
    def process_document(self, file_path: str) -> Dict:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Starting document processing: {file_path}")
        
        # Extract text
        text_content, file_metadata = self.extract_text_from_pdf(file_path)
        
        if not text_content:
            raise ValueError("No text content extracted from document")
        
        # Chunk text
        chunks = self.chunk_text(text_content, file_metadata)
        
        # Create document fingerprint
        doc_hash = hashlib.md5(text_content.encode()).hexdigest()
        
        # Prepare results
        results = {
            "document_hash": doc_hash,
            "file_metadata": file_metadata,
            "text_content": text_content,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "processing_timestamp": datetime.now().isoformat(),
            "processor_config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model_name
            }
        }
        
        # Save processed document
        self._save_processed_document(results, doc_hash)
        
        self.logger.info(f"Document processing completed. Created {len(chunks)} chunks.")
        return results
    
    def process_uploaded_document(self, uploaded_file) -> Dict:
        """
        Process uploaded Streamlit file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Starting uploaded document processing: {uploaded_file.name}")
        
        # Extract text
        text_content, file_metadata = self.extract_text_from_uploaded_file(uploaded_file)
        
        if not text_content:
            raise ValueError("No text content extracted from uploaded document")
        
        # Chunk text
        chunks = self.chunk_text(text_content, file_metadata)
        
        # Create document fingerprint
        doc_hash = hashlib.md5(text_content.encode()).hexdigest()
        
        # Prepare results
        results = {
            "document_hash": doc_hash,
            "file_metadata": file_metadata,
            "text_content": text_content,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "processing_timestamp": datetime.now().isoformat(),
            "processor_config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model_name
            }
        }
        
        # Save processed document
        self._save_processed_document(results, doc_hash)
        
        self.logger.info(f"Uploaded document processing completed. Created {len(chunks)} chunks.")
        return results
    
    def _save_processed_document(self, results: Dict, doc_hash: str):
        """Save processed document results"""
        try:
            # Save metadata
            metadata_file = self.metadata_dir / f"{doc_hash}_metadata.json"
            metadata = {
                "document_hash": doc_hash,
                "file_metadata": results["file_metadata"],
                "total_chunks": results["total_chunks"],
                "processing_timestamp": results["processing_timestamp"],
                "processor_config": results["processor_config"]
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save chunks with embeddings
            chunks_file = self.processed_docs_dir / f"{doc_hash}_chunks.json"
            with open(chunks_file, 'w') as f:
                json.dump(results["chunks"], f, indent=2)
            
            self.logger.info(f"Saved processed document: {doc_hash}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed document: {e}")
    
    def load_processed_document(self, doc_hash: str) -> Optional[Dict]:
        """Load previously processed document"""
        try:
            metadata_file = self.metadata_dir / f"{doc_hash}_metadata.json"
            chunks_file = self.processed_docs_dir / f"{doc_hash}_chunks.json"
            
            if not (metadata_file.exists() and chunks_file.exists()):
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            
            return {
                "document_hash": doc_hash,
                "file_metadata": metadata["file_metadata"],
                "chunks": chunks,
                "total_chunks": len(chunks),
                "processing_timestamp": metadata["processing_timestamp"],
                "processor_config": metadata["processor_config"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load processed document {doc_hash}: {e}")
            return None
    
    def get_processed_documents(self) -> List[Dict]:
        """Get list of all processed documents"""
        documents = []
        
        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                documents.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        return sorted(documents, key=lambda x: x.get("processing_timestamp", ""), reverse=True)


# Utility functions for integration
def process_default_oee_guide(processor: DocumentProcessor) -> Optional[str]:
    """Process the default OEE guide if it exists"""
    oee_guide_path = "The Complete_Guide_to_Simple_OEE.pdf"
    
    if os.path.exists(oee_guide_path):
        try:
            results = processor.process_document(oee_guide_path)
            return results["document_hash"]
        except Exception as e:
            logging.error(f"Failed to process default OEE guide: {e}")
    
    return None


def create_document_processor() -> DocumentProcessor:
    """Factory function to create document processor"""
    return DocumentProcessor(
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        min_chunk_length=100
    )


if __name__ == "__main__":
    # Test the document processor
    processor = create_document_processor()
    
    # Process default OEE guide if available
    doc_hash = process_default_oee_guide(processor)
    if doc_hash:
        print(f"Successfully processed default OEE guide: {doc_hash}")
    else:
        print("Default OEE guide not found or processing failed")
    
    # List processed documents
    documents = processor.get_processed_documents()
    print(f"Total processed documents: {len(documents)}")
    for doc in documents:
        print(f"- {doc.get('file_metadata', {}).get('file_name', 'Unknown')}")