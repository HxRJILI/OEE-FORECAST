OEE Advisory System
==================

The OEE Advisory System is an advanced AI-powered component that provides intelligent recommendations and insights for manufacturing optimization. Built using Retrieval-Augmented Generation (RAG) technology, it combines domain-specific knowledge with real-time production data to deliver actionable advice.

🤖 **System Architecture**
==========================

**Core Components:**

.. code-block::

   RAG Advisory System:
   ├── Document Processing Pipeline
   │   ├── PDF Text Extraction (PyPDF2, pdfplumber)
   │   ├── Text Chunking & Preprocessing
   │   └── Knowledge Base Construction
   ├── Embedding & Retrieval System
   │   ├── Sentence Transformers (all-MiniLM-L6-v2)
   │   ├── FAISS Vector Database
   │   └── Semantic Search Engine
   ├── Language Model Integration
   │   ├── Google Gemini API
   │   ├── Context-Aware Prompting
   │   └── Response Generation
   └── Streamlit Interface
       ├── Document Upload Interface
       ├── Chat-Based Interaction
       └── Contextual Recommendations

**Technology Stack:**

- **Embeddings**: Sentence Transformers for semantic understanding
- **Vector Database**: FAISS for efficient similarity search
- **Language Model**: Google Gemini for natural language generation
- **Document Processing**: PyPDF2 and pdfplumber for PDF handling
- **NLP Pipeline**: spaCy and NLTK for text preprocessing

🚀 **Getting Started**
======================

**Prerequisites:**

1. **API Configuration**

.. code-block:: bash

   # Set your Google Gemini API key
   export GEMINI_API_KEY="your_api_key_here"

2. **Install RAG Dependencies**

.. code-block:: bash

   pip install -r requirements_rag.txt

3. **Initialize the System**

.. code-block:: bash

   python setup_advisory.py

**Initial Setup:**

The advisory system requires a knowledge base to provide meaningful recommendations:

1. **Prepare Knowledge Documents**
   
   - Manufacturing best practices PDFs
   - OEE optimization guides
   - Equipment manuals and troubleshooting guides
   - Industry standards and benchmarks

2. **Upload Documents**
   
   Navigate to "📚 Document Management" in the application and upload your PDF documents.

3. **System Initialization**
   
   The system automatically:
   - Extracts text from uploaded PDFs
   - Creates semantic embeddings
   - Builds the searchable knowledge base
   - Validates system functionality

📚 **Document Management Interface**
===================================

**Upload Process:**

.. code-block::

   Document Management Workflow:
   
   1. File Selection
      ├── Drag & drop PDF files
      ├── File validation (size, format)
      └── Progress tracking
   
   2. Processing Pipeline
      ├── Text extraction
      ├── Content validation
      ├── Chunking & preprocessing
      └── Embedding generation
   
   3. Knowledge Base Integration
      ├── Vector indexing
      ├── Metadata storage
      └── Search optimization

**Supported Features:**

- **Multiple File Upload**: Process multiple PDFs simultaneously
- **Progress Tracking**: Real-time processing status for each document
- **Content Preview**: View extracted text before adding to knowledge base
- **Quality Validation**: Automatic detection of processing issues

**Document Requirements:**

- **Format**: PDF files only
- **Size**: Maximum 50MB per file
- **Content**: Text-based PDFs (OCR for scanned documents not included)
- **Language**: English language documents for optimal performance

🧠 **AI Chat Interface**
========================

**Query Types:**

**General OEE Questions:**

.. code-block::

   User: "What are the main factors affecting OEE?"
   
   Advisory System Response:
   Based on manufacturing best practices, the main factors affecting OEE include:
   
   1. **Availability Factors:**
      - Equipment breakdowns and unplanned downtime
      - Setup and changeover times
      - Material shortages and supply chain issues
   
   2. **Performance Factors:**
      - Reduced operating speeds
      - Idling and minor stops
      - Operator efficiency variations
   
   3. **Quality Factors:**
      - Product defects and rework
      - Process variation and capability
      - Material quality issues

**Line-Specific Recommendations:**

.. code-block::

   User: "LINE-01 has low availability (65%). What should I investigate?"
   
   Advisory System Response:
   For LINE-01's low availability of 65%, I recommend investigating:
   
   **Immediate Actions:**
   - Review maintenance logs for recurring equipment issues
   - Analyze changeover times between product runs
   - Check for material supply interruptions
   
   **Data Analysis:**
   - Identify top 5 downtime causes over last 30 days
   - Compare performance vs. similar production lines
   - Examine correlation with specific products or shifts

**Performance Optimization:**

.. code-block::

   User: "How can I improve performance efficiency on LINE-06?"
   
   Advisory System Response:
   Based on your LINE-06 data and industry best practices:
   
   **Performance Optimization Strategies:**
   
   1. **Speed Optimization:**
      - Gradually increase line speed while monitoring quality
      - Optimize material flow and feeding systems
      - Reduce micro-stops through preventive maintenance
   
   2. **Operator Training:**
      - Standardize operating procedures
      - Implement visual management systems
      - Cross-train operators for flexibility

**Context-Aware Responses:**

The system automatically considers:

- **Current Performance Data**: Uses real-time OEE metrics from your production lines
- **Historical Patterns**: Analyzes trends and recurring issues
- **Industry Benchmarks**: Compares your performance with manufacturing standards
- **Knowledge Base**: Draws from uploaded documents and best practices

🎯 **Advanced Features**
=======================

**Contextual Recommendations:**

The advisory system provides context-aware suggestions based on:

.. code-block:: python

   def generate_contextual_recommendation(line_data, query):
       """Generate recommendations based on current performance"""
       
       context = {
           'current_oee': line_data['oee'],
           'availability': line_data['availability'],
           'performance': line_data['performance'],
           'quality': line_data['quality'],
           'trend': calculate_trend(line_data),
           'industry_benchmark': get_benchmark_data(line_data['line'])
       }
       
       # Enhance query with production context
       enhanced_query = f"""
       Production Line: {line_data['line']}
       Current OEE: {context['current_oee']:.1%}
       Performance Context: {context}
       
       User Question: {query}
       
       Please provide specific, actionable recommendations.
       """
       
       return query_rag_system(enhanced_query, context)

**Smart Document Retrieval:**

The system uses intelligent retrieval strategies:

- **Semantic Search**: Understands meaning beyond keyword matching
- **Contextual Ranking**: Prioritizes relevant sections based on current performance
- **Multi-Document Synthesis**: Combines information from multiple sources
- **Confidence Scoring**: Indicates reliability of recommendations

**Performance Integration:**

Real-time data integration enhances advisory capabilities:

.. code-block::

   Advisory Context Integration:
   
   ├── Real-Time Metrics
   │   ├── Current OEE values
   │   ├── Line status information
   │   └── Performance trends
   ├── Historical Analysis
   │   ├── Performance patterns
   │   ├── Recurring issues
   │   └── Improvement trajectories
   └── Benchmark Comparison
       ├── Industry standards
       ├── Best-in-class performance
       └── Improvement potential

🔧 **System Configuration**
==========================

**API Configuration:**

.. code-block:: python

   # Configure Gemini API
   GEMINI_CONFIG = {
       'api_key': os.getenv('GEMINI_API_KEY'),
       'model': 'gemini-pro',
       'temperature': 0.3,  # Lower for more consistent responses
       'max_tokens': 1024,
       'safety_settings': {
           'harassment': 'block_medium_and_above',
           'hate_speech': 'block_medium_and_above',
           'sexually_explicit': 'block_medium_and_above',
           'dangerous_content': 'block_medium_and_above'
       }
   }

**Embedding Configuration:**

.. code-block:: python

   # Sentence Transformer settings
   EMBEDDING_CONFIG = {
       'model_name': 'all-MiniLM-L6-v2',
       'device': 'cpu',  # Use 'cuda' if GPU available
       'batch_size': 32,
       'max_seq_length': 384
   }

**FAISS Index Settings:**

.. code-block:: python

   # Vector database configuration
   FAISS_CONFIG = {
       'index_type': 'IndexFlatIP',  # Inner product for similarity
       'dimension': 384,  # Matches embedding model
       'nprobe': 10,  # For IVF indices
       'metric': 'METRIC_INNER_PRODUCT'
   }

🛠️ **Troubleshooting**
======================

**Common Issues and Solutions:**

**1. API Key Issues:**

.. code-block::

   Error: "Invalid API key"
   
   Solution:
   - Verify GEMINI_API_KEY environment variable is set
   - Check API key validity in Google AI Studio
   - Ensure proper permissions for Gemini API

**2. Document Processing Failures:**

.. code-block::

   Error: "Failed to extract text from PDF"
   
   Solutions:
   - Ensure PDF contains selectable text (not scanned images)
   - Check file size limits (max 50MB)
   - Try alternative PDF processing libraries

**3. Embedding Generation Issues:**

.. code-block::

   Error: "Sentence transformer model not found"
   
   Solution:
   - Install sentence-transformers: pip install sentence-transformers
   - Download model: sentence-transformers download all-MiniLM-L6-v2
   - Check internet connection for model download

**4. FAISS Index Problems:**

.. code-block::

   Error: "FAISS index creation failed"
   
   Solutions:
   - Install FAISS: pip install faiss-cpu
   - Check vector dimensions match
   - Verify sufficient memory for index creation

**Performance Optimization:**

- **Memory Usage**: Monitor RAM usage with large document collections
- **Response Time**: Optimize chunk size and embedding batch size
- **Quality**: Use high-quality source documents for better recommendations

📊 **Usage Analytics**
=====================

**System Metrics:**

The advisory system tracks important usage metrics:

- **Query Response Time**: Average time to generate recommendations
- **Document Retrieval Accuracy**: Relevance of retrieved knowledge
- **User Satisfaction**: Feedback on recommendation quality
- **Knowledge Base Coverage**: Topics covered by uploaded documents

**Continuous Improvement:**

- **Feedback Loop**: User ratings improve future recommendations
- **Knowledge Base Expansion**: Regular addition of new documents
- **Model Updates**: Periodic updates to embedding and language models
- **Performance Tuning**: Optimization based on usage patterns

🔗 **Integration Examples**
==========================

**Production Line Integration:**

.. code-block:: python

   def get_line_specific_advice(line_name, current_metrics):
       """Get AI advice for specific production line"""
       
       query = f"""
       Production line {line_name} performance analysis:
       - OEE: {current_metrics['oee']:.1%}
       - Availability: {current_metrics['availability']:.1%}
       - Performance: {current_metrics['performance']:.1%}
       - Quality: {current_metrics['quality']:.1%}
       
       What specific improvements should we focus on?
       """
       
       return advisory_system.query(query, context=current_metrics)

**Scheduled Reporting:**

.. code-block:: python

   def generate_weekly_advisory_report():
       """Generate automated weekly performance advisory"""
       
       for line in production_lines:
           weekly_metrics = calculate_weekly_metrics(line)
           advice = get_line_specific_advice(line, weekly_metrics)
           
           report = {
               'line': line,
               'metrics': weekly_metrics,
               'recommendations': advice,
               'priority_actions': extract_priority_actions(advice)
           }
           
           send_advisory_report(report)

**Next Steps:**

- Explore :doc:`../advanced/rag_system` for technical implementation details
- Review :doc:`../models/evaluation_metrics` for performance assessment
- Check :doc:`../troubleshooting` for additional support