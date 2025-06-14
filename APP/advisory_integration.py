"""
Advisory System Integration for Streamlit OEE Dashboard
Adds RAG-based advisory capabilities to the existing dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import os
import logging

# RAG System imports
try:
    from rag_system import create_oee_advisor, OEEAdvisor
    from document_processor import create_document_processor, process_default_oee_guide
    RAG_AVAILABLE = True
    RAG_ERROR = None
except ImportError as e:
    RAG_AVAILABLE = False
    RAG_ERROR = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Gemini API Key
GEMINI_API_KEY = "AIzaSyDtIOvSP1jp7dz91ByO1U44p2HZB5Q23BI"

def initialize_advisory_system():
    """Initialize the advisory system with error handling"""
    if not RAG_AVAILABLE:
        return None, f"RAG system not available: {RAG_ERROR}"
    
    try:
        if 'oee_advisor' not in st.session_state:
            with st.spinner("Initializing OEE Advisory System..."):
                advisor = create_oee_advisor(GEMINI_API_KEY, 'gemini-1.5-flash')
                
                # Try to process default OEE guide if available
                try:
                    doc_processor = create_document_processor()
                    doc_hash = process_default_oee_guide(doc_processor)
                    if doc_hash:
                        st.session_state.default_doc_processed = doc_hash
                        st.success("✅ Default OEE guide loaded successfully!")
                except Exception as e:
                    logging.warning(f"Could not process default OEE guide: {e}")
                    st.info("ℹ️ Default OEE guide not found. You can upload your own documents.")
                
                st.session_state.oee_advisor = advisor
                st.success("🤖 OEE Advisory System initialized successfully!")
        
        return st.session_state.oee_advisor, None
        
    except Exception as e:
        error_msg = f"Failed to initialize advisory system: {str(e)}"
        logging.error(error_msg)
        return None, error_msg

def show_advisory_dashboard(daily_oee_data, overall_daily_oee):
    """Show the main advisory dashboard page"""
    st.header("🤖 OEE Advisory System")
    st.markdown("*AI-powered insights and recommendations for your OEE performance*")
    
    # Initialize advisory system
    advisor, error = initialize_advisory_system()
    
    if not advisor:
        st.error(f"❌ Advisory system unavailable: {error}")
        
        # Show manual setup instructions
        with st.expander("🛠️ Setup Instructions"):
            st.markdown("""
            To enable the advisory system, you need to install the following dependencies:
            
            ```bash
            pip install google-generativeai sentence-transformers faiss-cpu PyPDF2 pdfplumber spacy nltk
            python -m spacy download en_core_web_sm
            ```
            
            Then restart the Streamlit application.
            """)
        return
    
    # Sidebar for advisory controls
    with st.sidebar:
        st.subheader("🎛️ Advisory Controls")
        
        # Knowledge base management
        st.markdown("### 📚 Knowledge Base")
        kb_stats = advisor.get_knowledge_base_stats()
        
        st.metric("Documents", kb_stats.get('total_documents', 0))
        st.metric("Knowledge Chunks", kb_stats.get('total_chunks', 0))
        
        if st.button("📊 View Knowledge Base Stats", use_container_width=True):
            st.session_state.show_kb_stats = True
        
        # Document upload
        st.markdown("### 📤 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload OEE-related PDF documents",
            type=["pdf"],
            help="Upload PDF documents to expand the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process Document", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        doc_hash = advisor.add_document(uploaded_file=uploaded_file)
                        st.success(f"✅ Document processed successfully!")
                        st.info(f"Document ID: {doc_hash[:8]}...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to process document: {str(e)}")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            advisor.clear_conversation_history()
            if 'chat_messages' in st.session_state:
                del st.session_state.chat_messages
            st.success("Conversation cleared!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Ask the OEE Expert")
        
        # Initialize chat messages
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant", 
                    "content": "Hello! I'm your OEE expert advisor. I can help you analyze your OEE data, identify improvement opportunities, and provide manufacturing best practices. How can I assist you today?"
                }
            ]
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about OEE, manufacturing efficiency, or improvement strategies..."):
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_data = advisor.ask_question(prompt)
                    response_text = response_data["response"]
                    
                    st.markdown(response_text)
                    
                    # Show context sources if available
                    if response_data.get("relevant_chunks"):
                        with st.expander("📚 Knowledge Sources Used"):
                            for i, chunk in enumerate(response_data["relevant_chunks"], 1):
                                st.markdown(f"""
                                **Source {i}** (Relevance: {chunk.get('similarity_score', 0):.3f})
                                
                                {chunk['text'][:300]}...
                                """)
            
            # Add assistant response to chat
            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
    
    with col2:
        # Quick Actions and Analysis
        st.subheader("⚡ Quick Actions")
        
        # OEE Analysis for specific line
        st.markdown("### 🎯 Line-Specific Analysis")
        selected_line = st.selectbox(
            "Select Production Line:",
            options=sorted(daily_oee_data['PRODUCTION_LINE'].unique()),
            key="advisory_line_select"
        )
        
        if st.button("📊 Analyze This Line", use_container_width=True):
            line_data = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == selected_line]
            
            if not line_data.empty:
                # Calculate metrics
                latest_data = line_data.iloc[-1]
                avg_oee = line_data['OEE'].mean()
                
                oee_data = {
                    'line': selected_line,
                    'oee': latest_data['OEE'],
                    'availability': latest_data['Availability'],
                    'performance': latest_data['Performance'],
                    'quality': latest_data['Quality'],
                    'period': f"Latest: {latest_data['Date'].strftime('%Y-%m-%d')}",
                    'additional_metrics': f"Average OEE: {avg_oee:.1%}, Total Output: {line_data['Total_Actual_Output'].sum()}"
                }
                
                with st.spinner("Analyzing line performance..."):
                    response_data = advisor.suggest_improvements(oee_data)
                    
                    # Add to chat
                    analysis_message = f"🔍 **Analysis for {selected_line}:**\n\n{response_data['response']}"
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": analysis_message
                    })
                    
                    st.success("Analysis added to chat!")
                    st.rerun()
        
        # Overall Performance Analysis
        st.markdown("### 🏭 Overall Performance")
        if st.button("📈 Analyze Overall Performance", use_container_width=True):
            # Calculate overall metrics
            overall_metrics = {
                'line': 'All Lines Combined',
                'oee': overall_daily_oee['OEE'].iloc[-1] if not overall_daily_oee.empty else 0,
                'availability': overall_daily_oee['Availability'].iloc[-1] if not overall_daily_oee.empty else 0,
                'performance': overall_daily_oee['Performance'].iloc[-1] if not overall_daily_oee.empty else 0,
                'quality': overall_daily_oee['Quality'].iloc[-1] if not overall_daily_oee.empty else 0,
                'period': f"Latest: {overall_daily_oee['Date'].iloc[-1].strftime('%Y-%m-%d')}" if not overall_daily_oee.empty else "Unknown",
                'additional_metrics': f"Average OEE: {overall_daily_oee['OEE'].mean():.1%}, Data points: {len(overall_daily_oee)}"
            }
            
            with st.spinner("Analyzing overall performance..."):
                response_data = advisor.suggest_improvements(overall_metrics)
                
                # Add to chat
                analysis_message = f"🏭 **Overall Performance Analysis:**\n\n{response_data['response']}"
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": analysis_message
                })
                
                st.success("Analysis added to chat!")
                st.rerun()
        
        # Predefined questions
        st.markdown("### 💡 Common Questions")
        common_questions = [
            "What is a good OEE benchmark?",
            "How can I reduce downtime?",
            "What causes low performance rates?",
            "How to implement TPM?",
            "Best practices for quality improvement",
            "How to calculate OEE accurately?"
        ]
        
        for question in common_questions:
            if st.button(f"❓ {question}", use_container_width=True, key=f"q_{hash(question)}"):
                # Add question to chat
                st.session_state.chat_messages.append({"role": "user", "content": question})
                
                # Generate response
                with st.spinner("Generating response..."):
                    response_data = advisor.ask_question(question)
                    
                    # Add response to chat
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": response_data["response"]
                    })
                
                st.rerun()
    
    # Knowledge Base Stats (if requested)
    if st.session_state.get('show_kb_stats', False):
        st.markdown("---")
        st.subheader("📊 Knowledge Base Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", kb_stats.get('total_documents', 0))
        with col2:
            st.metric("Total Chunks", kb_stats.get('total_chunks', 0))
        with col3:
            st.metric("Avg OEE Relevance", f"{kb_stats.get('avg_oee_relevance', 0):.1f}")
        with col4:
            st.metric("Vector Dimension", kb_stats.get('vector_dimension', 0))
        
        # Reset the flag
        st.session_state.show_kb_stats = False

def show_document_management():
    """Show document management interface"""
    st.header("📚 Document Management")
    
    # Initialize advisory system
    advisor, error = initialize_advisory_system()
    
    if not advisor:
        st.error(f"❌ Advisory system unavailable: {error}")
        return
    
    # Get knowledge base stats
    kb_stats = advisor.get_knowledge_base_stats()
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📑 Total Documents", kb_stats.get('total_documents', 0))
    with col2:
        st.metric("🧩 Knowledge Chunks", kb_stats.get('total_chunks', 0))
    with col3:
        st.metric("🎯 Avg OEE Relevance", f"{kb_stats.get('avg_oee_relevance', 0):.1f}")
    
    st.markdown("---")
    
    # Document upload section
    st.subheader("📤 Upload New Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload OEE-related PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents to expand the knowledge base. Multiple files can be selected."
        )
    
    with col2:
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) selected")
            
            if st.button("🔄 Process All Documents", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful_uploads = 0
                failed_uploads = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {uploaded_file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        doc_hash = advisor.add_document(uploaded_file=uploaded_file)
                        successful_uploads += 1
                        
                    except Exception as e:
                        failed_uploads.append((uploaded_file.name, str(e)))
                
                # Show results
                if successful_uploads > 0:
                    st.success(f"✅ Successfully processed {successful_uploads} document(s)!")
                
                if failed_uploads:
                    st.error(f"❌ Failed to process {len(failed_uploads)} document(s):")
                    for filename, error in failed_uploads:
                        st.error(f"- {filename}: {error}")
                
                progress_bar.empty()
                status_text.empty()
                
                # Refresh the page to show updated stats
                st.rerun()
    
    # Existing documents (if we can list them)
    st.markdown("---")
    st.subheader("📋 Knowledge Base Contents")
    
    # Try to get processed documents from document processor
    try:
        doc_processor = create_document_processor()
        processed_docs = doc_processor.get_processed_documents()
        
        if processed_docs:
            # Create a DataFrame for display
            docs_df = pd.DataFrame([
                {
                    "Document": doc.get('file_metadata', {}).get('file_name', 'Unknown'),
                    "Pages": doc.get('file_metadata', {}).get('pages', 0),
                    "Size (KB)": doc.get('file_metadata', {}).get('file_size', 0) // 1024,
                    "Processed": doc.get('processing_timestamp', 'Unknown'),
                    "Hash": doc.get('document_hash', '')[:8] + '...'
                }
                for doc in processed_docs
            ])
            
            st.dataframe(docs_df, use_container_width=True)
        else:
            st.info("📝 No documents in knowledge base yet. Upload some PDF documents to get started!")
    
    except Exception as e:
        st.warning(f"Could not load document list: {str(e)}")
    
    # Processing configuration
    st.markdown("---")
    st.subheader("⚙️ Processing Configuration")
    
    with st.expander("🔧 Advanced Settings"):
        processor_info = kb_stats.get('processor_info', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Settings:**")
            st.code(f"""
Embedding Model: {processor_info.get('embedding_model', 'Unknown')}
Chunk Size: {processor_info.get('chunk_size', 'Unknown')}
Chunk Overlap: {processor_info.get('chunk_overlap', 'Unknown')}
Vector Dimension: {kb_stats.get('vector_dimension', 'Unknown')}
            """)
        
        with col2:
            st.markdown("**Processing Pipeline:**")
            st.markdown("""
            1. 📄 PDF text extraction
            2. 🧹 Text cleaning and normalization
            3. ✂️ Intelligent chunking
            4. 🔍 NLP analysis and entity extraction
            5. 🧠 Embedding generation
            6. 💾 Vector storage with metadata
            """)

# Integration functions for main app
def add_advisory_system_to_sidebar():
    """Add advisory system option to sidebar navigation"""
    return ["🏠 Main Dashboard", "📈 Line-Specific Analysis", "📊 Overall Daily Analysis", 
            "🔮 OEE Forecasting", "🤖 OEE Advisory", "📚 Document Management"]

def handle_advisory_pages(page_selection, daily_oee_data, overall_daily_oee):
    """Handle advisory system page routing"""
    if page_selection == "🤖 OEE Advisory":
        show_advisory_dashboard(daily_oee_data, overall_daily_oee)
        return True
    elif page_selection == "📚 Document Management":
        show_document_management()
        return True
    return False

# Utility function to check system status
def check_advisory_system_status():
    """Check if the advisory system is properly configured"""
    status = {
        "rag_available": RAG_AVAILABLE,
        "api_key_configured": bool(GEMINI_API_KEY),
        "error_message": RAG_ERROR if not RAG_AVAILABLE else None
    }
    
    if RAG_AVAILABLE:
        try:
            # Try to create a minimal advisor instance
            advisor = create_oee_advisor(GEMINI_API_KEY)
            status["initialization_test"] = True
        except Exception as e:
            status["initialization_test"] = False
            status["initialization_error"] = str(e)
    
    return status

# Integration helper for existing app
def integrate_advisory_system():
    """
    Integration helper function to add advisory capabilities to existing app
    Call this function in your main app to add advisory features
    """
    # Add advisory system status to sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 Advisory System")
        
        status = check_advisory_system_status()
        
        if status["rag_available"] and status["api_key_configured"]:
            st.success("✅ Advisory System Ready")
        else:
            st.error("❌ Advisory System Unavailable")
            
            if not status["rag_available"]:
                st.error(f"RAG System Error: {status['error_message']}")
            
            if not status["api_key_configured"]:
                st.error("Gemini API key not configured")
    
    return status["rag_available"] and status["api_key_configured"]

if __name__ == "__main__":
    # Test the integration
    st.set_page_config(page_title="Advisory System Test", layout="wide")
    
    # Check system status
    status = check_advisory_system_status()
    st.json(status)
    
    if status.get("rag_available") and status.get("api_key_configured"):
        # Create some dummy data for testing
        dummy_oee_data = pd.DataFrame({
            'PRODUCTION_LINE': ['LINE-01', 'LINE-02'] * 10,
            'Date': pd.date_range('2024-01-01', periods=20),
            'OEE': np.random.uniform(0.6, 0.9, 20),
            'Availability': np.random.uniform(0.8, 0.95, 20),
            'Performance': np.random.uniform(0.7, 0.9, 20),
            'Quality': np.random.uniform(0.95, 1.0, 20),
            'Total_Actual_Output': np.random.randint(100, 500, 20)
        })
        
        dummy_overall_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'OEE': np.random.uniform(0.6, 0.9, 10),
            'Availability': np.random.uniform(0.8, 0.95, 10),
            'Performance': np.random.uniform(0.7, 0.9, 10),
            'Quality': np.random.uniform(0.95, 1.0, 10)
        })
        
        show_advisory_dashboard(dummy_oee_data, dummy_overall_data)
    else:
        st.error("Advisory system not available for testing")