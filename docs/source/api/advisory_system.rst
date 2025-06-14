Advisory System API Reference
==============================

This section provides comprehensive documentation for the AI-powered advisory system APIs. These components enable intelligent recommendations, knowledge base management, and context-aware manufacturing optimization advice.

 **Core Advisory Classes**
============================

.. py:class:: OEEAdvisor

   Main advisory system class that orchestrates RAG (Retrieval-Augmented Generation) capabilities.

   .. py:method:: __init__(api_key=None, model_name='gemini-pro', embedding_model='all-MiniLM-L6-v2')

      Initialize the OEE Advisory system.

      :param str api_key: Google Gemini API key
      :param str model_name: Language model to use for response generation
      :param str embedding_model: Sentence transformer model for embeddings
      :raises APIKeyError: If API key is invalid or missing
      :raises ModelLoadError: If models fail to load

      **Initialization Process:**

      .. code-block:: python

         def __init__(self, api_key=None, model_name='gemini-pro', embedding_model='all-MiniLM-L6-v2'):
             """
             Initialize advisory system with comprehensive setup
             
             Setup Process:
             1. Validate API credentials
             2. Load embedding model
             3. Initialize vector database
             4. Setup language model
             5. Configure safety settings
             6. Initialize knowledge base
             """
             
             # Validate API key
             self.api_key = api_key or os.getenv('GEMINI_API_KEY')
             if not self.api_key:
                 raise APIKeyError("Gemini API key is required")
             
             # Initialize components
             self.embedding_model = self._load_embedding_model(embedding_model)
             self.vector_db = self._initialize_vector_database()
             self.llm = self._initialize_language_model(model_name)
             self.knowledge_base = DocumentKnowledgeBase()
             
             # Configuration
             self.config = {
                 'max_context_length': 4000,
                 'max_response_length': 1000,
                 'similarity_threshold': 0.7,
                 'safety_level': 'high'
             }

   .. py:method:: add_documents(documents, document_type='pdf', metadata=None)

      Add documents to the knowledge base for enhanced advisory capabilities.

      :param list documents: List of document file paths or content
      :param str document_type: Type of documents ('pdf', 'text', 'url')
      :param dict metadata: Optional metadata for documents
      :returns: Processing results and success status
      :rtype: dict

      **Document Processing Pipeline:**

      .. code-block:: python

         def add_documents(self, documents, document_type='pdf', metadata=None):
             """
             Comprehensive document processing and indexing
             
             Processing Steps:
             1. Document validation and format checking
             2. Text extraction (PDF, text, web scraping)
             3. Content cleaning and preprocessing
             4. Text chunking with overlap
             5. Embedding generation
             6. Vector database indexing
             7. Metadata storage
             """
             
             processing_results = []
             
             for doc in documents:
                 try:
                     # Extract text content
                     if document_type == 'pdf':
                         content = self._extract_pdf_text(doc)
                     elif document_type == 'text':
                         content = self._load_text_file(doc)
                     elif document_type == 'url':
                         content = self._scrape_web_content(doc)
                     
                     # Process content
                     chunks = self._chunk_text(content)
                     embeddings = self._generate_embeddings(chunks)
                     
                     # Store in vector database
                     doc_id = self._store_document(
                         chunks, embeddings, metadata, doc
                     )
                     
                     processing_results.append({
                         'document': doc,
                         'doc_id': doc_id,
                         'chunks_created': len(chunks),
                         'status': 'success'
                     })
                     
                 except Exception as e:
                     processing_results.append({
                         'document': doc,
                         'error': str(e),
                         'status': 'failed'
                     })
             
             return {
                 'processed_documents': len(processing_results),
                 'successful': len([r for r in processing_results if r['status'] == 'success']),
                 'failed': len([r for r in processing_results if r['status'] == 'failed']),
                 'details': processing_results
             }

   .. py:method:: query(question, context=None, production_line=None, include_sources=True)

      Generate AI-powered advice based on question and operational context.

      :param str question: User question or request for advice
      :param dict context: Current OEE metrics and operational context
      :param str production_line: Specific production line for context
      :param bool include_sources: Include source document references
      :returns: AI-generated advice with supporting information
      :rtype: dict

      **Query Processing Workflow:**

      .. code-block:: python

         def query(self, question, context=None, production_line=None, include_sources=True):
             """
             AI-powered query processing with contextual awareness
             
             Query Pipeline:
             1. Question analysis and intent recognition
             2. Context integration (OEE metrics, line status)
             3. Relevant knowledge retrieval
             4. Context-aware prompt construction
             5. LLM response generation
             6. Response validation and formatting
             7. Source attribution
             """
             
             # Analyze question intent
             intent = self._analyze_question_intent(question)
             
             # Retrieve relevant knowledge
             relevant_docs = self._retrieve_relevant_knowledge(
                 question, context, production_line
             )
             
             # Construct enhanced prompt
             enhanced_prompt = self._construct_contextual_prompt(
                 question, context, relevant_docs, production_line
             )
             
             # Generate response
             response = self._generate_llm_response(enhanced_prompt)
             
             # Format and validate response
             formatted_response = self._format_response(
                 response, relevant_docs, include_sources
             )
             
             return {
                 'answer': formatted_response['answer'],
                 'confidence': formatted_response['confidence'],
                 'sources': formatted_response['sources'] if include_sources else [],
                 'context_used': context,
                 'production_line': production_line,
                 'intent': intent,
                 'response_metadata': {
                     'generation_time': datetime.now().isoformat(),
                     'model_used': self.llm.model_name,
                     'knowledge_sources': len(relevant_docs)
                 }
             }

 **Knowledge Base Management**
==============================

.. py:class:: DocumentKnowledgeBase

   Advanced knowledge base management for manufacturing documentation.

   .. py:method:: __init__(storage_backend='faiss', chunk_size=500, chunk_overlap=50)

      Initialize document knowledge base.

      :param str storage_backend: Vector database backend ('faiss', 'chroma')
      :param int chunk_size: Size of text chunks for processing
      :param int chunk_overlap: Overlap between consecutive chunks

   .. py:method:: index_document(content, metadata, chunk_strategy='semantic')

      Index a document with advanced chunking strategies.

      :param str content: Document text content
      :param dict metadata: Document metadata (title, source, date, etc.)
      :param str chunk_strategy: Chunking strategy ('fixed', 'semantic', 'recursive')
      :returns: Indexing results
      :rtype: dict

      **Chunking Strategies:**

      .. code-block:: python

         def chunk_text_by_strategy(self, content, strategy='semantic'):
             """
             Advanced text chunking with multiple strategies
             
             Chunking Strategies:
             
             'fixed':     Fixed-size chunks with overlap
             'semantic':  Sentence-boundary aware chunking
             'recursive': Hierarchical chunking (paragraphs -> sentences)
             'sliding':   Sliding window with custom step size
             """
             
             if strategy == 'semantic':
                 return self._semantic_chunking(content)
             elif strategy == 'recursive':
                 return self._recursive_chunking(content)
             elif strategy == 'sliding':
                 return self._sliding_window_chunking(content)
             else:  # fixed
                 return self._fixed_size_chunking(content)

   .. py:method:: search_knowledge(query, filters=None, top_k=5, similarity_threshold=0.7)

      Search knowledge base for relevant information.

      :param str query: Search query
      :param dict filters: Optional metadata filters
      :param int top_k: Number of top results to return
      :param float similarity_threshold: Minimum similarity score
      :returns: Relevant knowledge chunks with scores
      :rtype: list

      **Advanced Search Features:**

      .. code-block:: python

         def search_knowledge(self, query, filters=None, top_k=5, similarity_threshold=0.7):
             """
             Advanced knowledge search with filtering and ranking
             
             Search Features:
             - Semantic similarity search
             - Metadata-based filtering
             - Hybrid keyword + semantic search
             - Result re-ranking based on relevance
             - Contextual boosting
             """
             
             # Generate query embedding
             query_embedding = self.embedding_model.encode(query)
             
             # Perform similarity search
             raw_results = self.vector_db.similarity_search(
                 query_embedding, k=top_k * 2  # Get more for filtering
             )
             
             # Apply filters
             if filters:
                 filtered_results = self._apply_metadata_filters(raw_results, filters)
             else:
                 filtered_results = raw_results
             
             # Re-rank results
             reranked_results = self._rerank_results(
                 query, filtered_results, similarity_threshold
             )
             
             return reranked_results[:top_k]

 **Context-Aware Advisory Functions**
======================================

.. py:function:: generate_line_specific_advice(advisor, production_line, current_metrics, historical_data)

   Generate production line-specific optimization advice.

   :param OEEAdvisor advisor: Initialized advisory system
   :param str production_line: Production line identifier
   :param dict current_metrics: Current OEE metrics
   :param pd.DataFrame historical_data: Historical performance data
   :returns: Detailed line-specific recommendations
   :rtype: dict

   **Line-Specific Analysis:**

   .. code-block:: python

      def generate_line_specific_advice(advisor, production_line, current_metrics, historical_data):
          """
          Generate comprehensive line-specific optimization advice
          
          Analysis Components:
          1. Current performance assessment
          2. Historical trend analysis
          3. Benchmark comparison
          4. Root cause identification
          5. Actionable recommendations
          6. Priority ranking
          """
          
          # Analyze current performance
          performance_analysis = analyze_current_performance(
              current_metrics, production_line
          )
          
          # Historical context
          historical_context = analyze_historical_trends(
              historical_data, production_line
          )
          
          # Construct context-rich query
          context_query = f"""
          Production Line: {production_line}
          
          Current Performance:
          - OEE: {current_metrics['oee']:.1%}
          - Availability: {current_metrics['availability']:.1%}
          - Performance: {current_metrics['performance']:.1%}
          - Quality: {current_metrics['quality']:.1%}
          
          Historical Context:
          - 30-day average OEE: {historical_context['avg_oee']:.1%}
          - Trend direction: {historical_context['trend']}
          - Top issues: {historical_context['top_issues']}
          
          What specific actions should we take to improve OEE for this production line?
          Please provide prioritized recommendations with expected impact.
          """
          
          # Generate advice
          advice_response = advisor.query(
              context_query,
              context=current_metrics,
              production_line=production_line,
              include_sources=True
          )
          
          # Enhance with specific analysis
          enhanced_advice = {
              'production_line': production_line,
              'current_status': performance_analysis['status'],
              'priority_level': performance_analysis['priority'],
              'recommendations': advice_response['answer'],
              'confidence': advice_response['confidence'],
              'supporting_sources': advice_response['sources'],
              'impact_assessment': estimate_improvement_impact(
                  current_metrics, historical_context
              ),
              'implementation_timeline': suggest_implementation_timeline(
                  advice_response['answer']
              )
          }
          
          return enhanced_advice

.. py:function:: generate_comparative_analysis(advisor, production_lines_data, timeframe='30d')

   Generate comparative analysis across multiple production lines.

   :param OEEAdvisor advisor: Advisory system instance
   :param dict production_lines_data: Data for multiple production lines
   :param str timeframe: Analysis timeframe
   :returns: Comparative analysis and recommendations
   :rtype: dict

.. py:function:: generate_root_cause_analysis(advisor, issue_description, affected_metrics, context_data)

   Perform AI-powered root cause analysis for performance issues.

   :param OEEAdvisor advisor: Advisory system instance
   :param str issue_description: Description of the observed issue
   :param dict affected_metrics: Metrics showing the impact
   :param dict context_data: Additional contextual information
   :returns: Root cause analysis with recommendations
   :rtype: dict

   **Root Cause Analysis Process:**

   .. code-block:: python

      def generate_root_cause_analysis(advisor, issue_description, affected_metrics, context_data):
          """
          AI-powered root cause analysis for manufacturing issues
          
          Analysis Framework:
          1. Issue categorization and severity assessment
          2. Pattern recognition in affected metrics
          3. Historical precedent analysis
          4. Knowledge base consultation
          5. Systematic root cause identification
          6. Corrective action recommendations
          """
          
          # Categorize the issue
          issue_category = categorize_manufacturing_issue(
              issue_description, affected_metrics
          )
          
          # Analyze metric patterns
          pattern_analysis = analyze_metric_patterns(affected_metrics)
          
          # Historical precedent search
          similar_cases = search_historical_cases(
              issue_description, context_data
          )
          
          # Construct comprehensive analysis query
          analysis_query = f"""
          Manufacturing Issue Analysis:
          
          Issue Description: {issue_description}
          Issue Category: {issue_category}
          
          Affected Metrics:
          {format_metrics_for_analysis(affected_metrics)}
          
          Pattern Analysis:
          {pattern_analysis}
          
          Context:
          {format_context_for_analysis(context_data)}
          
          Based on manufacturing best practices and root cause analysis methodologies,
          what are the most likely root causes for this issue? Please provide:
          1. Primary root cause candidates with probability assessment
          2. Supporting evidence for each candidate
          3. Recommended diagnostic steps
          4. Immediate containment actions
          5. Long-term corrective measures
          """
          
          # Generate analysis
          rca_response = advisor.query(
              analysis_query,
              context=context_data,
              include_sources=True
          )
          
          # Structure the response
          structured_analysis = {
              'issue_summary': {
                  'description': issue_description,
                  'category': issue_category,
                  'severity': assess_issue_severity(affected_metrics),
                  'affected_areas': identify_affected_areas(affected_metrics)
              },
              'root_cause_analysis': rca_response['answer'],
              'confidence_level': rca_response['confidence'],
              'supporting_evidence': rca_response['sources'],
              'recommended_actions': extract_action_items(rca_response['answer']),
              'similar_historical_cases': similar_cases,
              'follow_up_monitoring': suggest_monitoring_plan(issue_category)
          }
          
          return structured_analysis

 **Advanced Advisory Features**
=================================

.. py:class:: AdvisoryAnalytics

   Advanced analytics for advisory system performance and insights.

   .. py:method:: analyze_query_patterns(query_history, timeframe='30d')

      Analyze patterns in user queries to improve advisory capabilities.

      :param list query_history: Historical queries and responses
      :param str timeframe: Analysis timeframe
      :returns: Query pattern analysis
      :rtype: dict

   .. py:method:: measure_advice_effectiveness(advice_given, outcomes_observed)

      Measure the effectiveness of provided advice based on observed outcomes.

      :param list advice_given: Previously provided advice
      :param list outcomes_observed: Observed performance outcomes
      :returns: Effectiveness metrics and insights
      :rtype: dict

.. py:function:: continuous_learning_update(advisor, feedback_data, performance_metrics)

   Update advisory system based on user feedback and performance data.

   :param OEEAdvisor advisor: Advisory system to update
   :param dict feedback_data: User feedback on advice quality
   :param dict performance_metrics: System performance metrics
   :returns: Update results and improved system
   :rtype: dict

   **Continuous Learning Process:**

   .. code-block:: python

      def continuous_learning_update(advisor, feedback_data, performance_metrics):
          """
          Continuous learning system for advisory improvement
          
          Learning Components:
          1. Feedback analysis and sentiment scoring
          2. Performance metric correlation analysis
          3. Knowledge base gap identification
          4. Response quality assessment
          5. Model fine-tuning recommendations
          6. System configuration optimization
          """
          
          # Analyze user feedback
          feedback_analysis = analyze_user_feedback(feedback_data)
          
          # Identify improvement areas
          improvement_areas = identify_improvement_opportunities(
              feedback_analysis, performance_metrics
          )
          
          # Update knowledge base
          knowledge_updates = update_knowledge_base(
              advisor, improvement_areas
          )
          
          # Optimize system parameters
          parameter_updates = optimize_system_parameters(
              advisor, performance_metrics
          )
          
          return {
              'feedback_summary': feedback_analysis,
              'improvement_areas': improvement_areas,
              'knowledge_updates': knowledge_updates,
              'parameter_updates': parameter_updates,
              'system_performance': assess_updated_performance(advisor)
          }

 **Safety and Validation**
============================

.. py:function:: validate_advice_safety(advice_text, manufacturing_context)

   Validate that generated advice is safe for manufacturing environments.

   :param str advice_text: Generated advice text
   :param dict manufacturing_context: Manufacturing context for validation
   :returns: Safety validation results
   :rtype: dict

   **Safety Validation Framework:**

   .. code-block:: python

      def validate_advice_safety(advice_text, manufacturing_context):
          """
          Comprehensive safety validation for manufacturing advice
          
          Safety Checks:
          1. Equipment safety compliance
          2. Process safety standards adherence
          3. Regulatory compliance verification
          4. Risk assessment for recommendations
          5. Feasibility validation
          6. Resource requirement assessment
          """
          
          safety_checks = {
              'equipment_safety': check_equipment_safety_compliance(advice_text),
              'process_safety': check_process_safety_standards(advice_text),
              'regulatory_compliance': check_regulatory_compliance(
                  advice_text, manufacturing_context
              ),
              'risk_assessment': assess_recommendation_risks(advice_text),
              'feasibility': assess_implementation_feasibility(
                  advice_text, manufacturing_context
              )
          }
          
          # Overall safety score
          overall_safety = calculate_overall_safety_score(safety_checks)
          
          return {
              'safety_score': overall_safety,
              'safety_checks': safety_checks,
              'approved_for_implementation': overall_safety > 0.8,
              'safety_warnings': extract_safety_warnings(safety_checks),
              'mitigation_suggestions': suggest_risk_mitigations(safety_checks)
          }

 **Performance Monitoring**
============================

.. py:class:: AdvisoryPerformanceMonitor

   Monitor and track advisory system performance and effectiveness.

   .. py:method:: track_response_quality(responses, feedback_scores)

      Track the quality of advisory responses over time.

      :param list responses: Generated responses
      :param list feedback_scores: User feedback scores
      :returns: Quality metrics and trends
      :rtype: dict

   .. py:method:: monitor_knowledge_coverage(queries, knowledge_base)

      Monitor knowledge base coverage for incoming queries.

      :param list queries: User queries
      :param DocumentKnowledgeBase knowledge_base: Knowledge base instance
      :returns: Coverage analysis and gap identification
      :rtype: dict

.. py:function:: generate_advisory_performance_report(monitor, timeframe='monthly')

   Generate comprehensive performance report for advisory system.

   :param AdvisoryPerformanceMonitor monitor: Performance monitor instance
   :param str timeframe: Report timeframe
   :returns: Detailed performance report
   :rtype: dict

 **Integration Examples**
==========================

**Basic Advisory Usage**

.. code-block:: python

   # Initialize advisory system
   advisor = OEEAdvisor(api_key="your_gemini_api_key")

   # Add manufacturing knowledge
   pdf_documents = [
       "OEE_Best_Practices.pdf",
       "Manufacturing_Optimization_Guide.pdf",
       "Equipment_Maintenance_Manual.pdf"
   ]
   
   advisor.add_documents(pdf_documents, document_type='pdf')

   # Get advice for specific situation
   current_metrics = {
       'oee': 0.72,
       'availability': 0.85,
       'performance': 0.89,
       'quality': 0.95
   }

   advice = advisor.query(
       "Our OEE is 72%. What are the main areas for improvement?",
       context=current_metrics,
       production_line='LINE-01'
   )

   print(f"Advice: {advice['answer']}")
   print(f"Confidence: {advice['confidence']}")

**Advanced Root Cause Analysis**

.. code-block:: python

   # Perform root cause analysis
   issue_description = "Sudden drop in availability from 90% to 65% over 3 days"
   
   affected_metrics = {
       'availability_before': 0.90,
       'availability_current': 0.65,
       'downtime_increase': 180,  # minutes per day
       'affected_shifts': ['morning', 'afternoon']
   }

   context_data = {
       'production_line': 'LINE-03',
       'recent_changes': ['new_operator_training', 'preventive_maintenance'],
       'environmental_factors': ['temperature_fluctuation']
   }

   rca_result = generate_root_cause_analysis(
       advisor, issue_description, affected_metrics, context_data
   )

   print("Root Cause Analysis:")
   print(rca_result['root_cause_analysis'])

**Continuous Improvement Monitoring**

.. code-block:: python

   # Monitor advisory effectiveness
   performance_monitor = AdvisoryPerformanceMonitor()

   # Track advice implementation outcomes
   advice_tracking = {
       'advice_id': 'ADV_001',
       'implementation_date': '2024-01-15',
       'pre_implementation_oee': 0.72,
       'post_implementation_oee': 0.78,
       'user_satisfaction': 4.5  # out of 5
   }

   performance_monitor.track_advice_outcome(advice_tracking)

   # Generate performance report
   monthly_report = generate_advisory_performance_report(
       performance_monitor, timeframe='monthly'
   )

**Multi-Line Comparative Analysis**

.. code-block:: python

   # Compare performance across multiple lines
   lines_data = {
       'LINE-01': {'oee': 0.68, 'trend': 'declining'},
       'LINE-03': {'oee': 0.82, 'trend': 'stable'},
       'LINE-04': {'oee': 0.75, 'trend': 'improving'},
       'LINE-06': {'oee': 0.89, 'trend': 'stable'}
   }

   comparative_analysis = generate_comparative_analysis(
       advisor, lines_data, timeframe='30d'
   )

   print("Comparative Analysis Results:")
   print(comparative_analysis['analysis_summary'])

 **Error Handling**
====================

**Advisory System Exceptions**

.. py:exception:: APIKeyError

   Raised when Gemini API key is invalid or missing.

.. py:exception:: DocumentProcessingError

   Raised when document processing fails.

.. py:exception:: KnowledgeBaseError

   Raised when knowledge base operations fail.

.. py:exception:: QueryProcessingError

   Raised when query processing encounters errors.

**Error Recovery Strategies**

.. code-block:: python

   def robust_advisory_query(advisor, question, context=None, max_retries=3):
       """
       Robust query processing with error handling and retries
       """
       
       for attempt in range(max_retries):
           try:
               return advisor.query(question, context=context)
               
           except APIKeyError:
               # API key issues require manual intervention
               raise
               
           except QueryProcessingError as e:
               if attempt < max_retries - 1:
                   # Retry with simplified query
                   simplified_question = simplify_query(question)
                   continue
               else:
                   # Return fallback response
                   return create_fallback_response(question, str(e))
                   
           except Exception as e:
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)  # Exponential backoff
                   continue
               else:
                   return create_error_response(question, str(e))

**Next Steps:**

- Explore :doc:`../advanced/rag_system` for detailed RAG implementation
- Review :doc:`../advanced/deployment` for production deployment strategies
- Check :doc:`../troubleshooting` for common advisory system issues