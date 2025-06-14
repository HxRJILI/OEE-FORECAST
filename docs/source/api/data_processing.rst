Data Processing API Reference
=============================

This section provides comprehensive documentation for the data processing components of the OEE analytics system. These APIs handle data ingestion, cleaning, transformation, and OEE calculation processes.

🔧 **Core Data Processing Functions**
====================================

**Main Data Loading Functions**

.. py:function:: load_and_process_data(line_status_file, production_file)

   Load and process raw manufacturing data files into cleaned datasets.

   :param str line_status_file: Path to line status CSV file
   :param str production_file: Path to production data CSV file
   :returns: Tuple of processed DataFrames (line_status_cleaned, production_data_cleaned)
   :rtype: tuple(pd.DataFrame, pd.DataFrame)
   :raises FileNotFoundError: If input files don't exist
   :raises ValueError: If data format is invalid

   **Example:**

   .. code-block:: python

      # Load and process manufacturing data
      line_status, production_data = load_and_process_data(
          'line_status_notcleaned.csv',
          'production_data.csv'
      )

**Data Cleaning and Validation**

.. py:function:: clean_line_status_data(raw_data)

   Clean and standardize line status data with comprehensive validation.

   :param pd.DataFrame raw_data: Raw line status data from CSV
   :returns: Cleaned and validated line status data
   :rtype: pd.DataFrame

   **Data Transformations:**

   - Convert datetime columns to proper datetime format
   - Standardize production line names
   - Validate status transitions
   - Handle missing timestamps
   - Remove invalid records

   **Implementation:**

   .. code-block:: python

      def clean_line_status_data(raw_data):
          """
          Comprehensive line status data cleaning
          
          Processing steps:
          1. Datetime conversion and validation
          2. Production line standardization
          3. Status name normalization
          4. Shift assignment validation
          5. Data quality checks
          """
          
          # Copy data to avoid modifying original
          cleaned_data = raw_data.copy()
          
          # Convert datetime columns
          datetime_columns = ['START_DATETIME', 'FINISH_DATETIME']
          for col in datetime_columns:
              if col in cleaned_data.columns:
                  cleaned_data[col] = pd.to_datetime(
                      cleaned_data[col], 
                      errors='coerce'
                  )
          
          # Standardize production line names
          if 'PRODUCTION_LINE' in cleaned_data.columns:
              cleaned_data['PRODUCTION_LINE'] = cleaned_data['PRODUCTION_LINE'].str.upper().str.strip()
          
          # Remove invalid records
          valid_mask = (
              cleaned_data['START_DATETIME'].notna() &
              cleaned_data['PRODUCTION_LINE'].notna() &
              cleaned_data['STATUS_NAME'].notna()
          )
          
          cleaned_data = cleaned_data[valid_mask].reset_index(drop=True)
          
          return cleaned_data

.. py:function:: validate_production_data(production_data)

   Validate and clean production output data.

   :param pd.DataFrame production_data: Raw production data
   :returns: Validated production data with quality flags
   :rtype: pd.DataFrame

   **Validation Checks:**

   - Datetime format consistency
   - Production line name matching
   - Duration calculations
   - Data completeness assessment

🏭 **OEE Calculation Engine**
============================

.. py:class:: OEECalculator

   Core class for calculating Overall Equipment Effectiveness metrics.

   .. py:method:: __init__(cycle_times=None, quality_assumptions=None)

      Initialize OEE calculator with configuration parameters.

      :param dict cycle_times: Theoretical cycle times per production line
      :param dict quality_assumptions: Quality rate assumptions per line

      **Default Configuration:**

      .. code-block:: python

         default_cycle_times = {
             'LINE-01': 600,  # 10 minutes per unit
             'LINE-03': 480,  # 8 minutes per unit
             'LINE-04': 720,  # 12 minutes per unit
             'LINE-06': 360   # 6 minutes per unit
         }
         
         default_quality = {
             'LINE-01': 0.98,  # 98% quality rate
             'LINE-03': 0.99,  # 99% quality rate
             'LINE-04': 0.97,  # 97% quality rate
             'LINE-06': 0.995  # 99.5% quality rate
         }

   .. py:method:: calculate_availability(line_status_data, production_line, date)

      Calculate availability metric for a specific production line and date.

      :param pd.DataFrame line_status_data: Cleaned line status data
      :param str production_line: Production line identifier
      :param datetime.date date: Target date for calculation
      :returns: Availability percentage (0-1)
      :rtype: float

      **Availability Formula:**

      .. math::

         Availability = \frac{Actual\_Run\_Time}{Planned\_Production\_Time}

      **Implementation:**

      .. code-block:: python

         def calculate_availability(self, line_status_data, production_line, date):
             """
             Calculate availability for specific line and date
             
             Steps:
             1. Filter data for target line and date
             2. Identify production vs. non-production time
             3. Calculate planned production time
             4. Calculate actual run time
             5. Compute availability ratio
             """
             
             # Filter data for specific line and date
             line_data = line_status_data[
                 (line_status_data['PRODUCTION_LINE'] == production_line) &
                 (line_status_data['START_DATETIME'].dt.date == date)
             ].copy()
             
             if line_data.empty:
                 return 0.0
             
             # Define production statuses
             production_statuses = ['Production', 'Running', 'Active']
             
             # Calculate time periods
             planned_time = self._calculate_planned_time(date)
             actual_run_time = self._calculate_run_time(line_data, production_statuses)
             
             # Availability calculation
             availability = actual_run_time / planned_time if planned_time > 0 else 0.0
             
             return min(availability, 1.0)  # Cap at 100%

   .. py:method:: calculate_performance(production_data, line_status_data, production_line, date)

      Calculate performance efficiency metric.

      :param pd.DataFrame production_data: Production output data
      :param pd.DataFrame line_status_data: Line status data
      :param str production_line: Production line identifier
      :param datetime.date date: Target date for calculation
      :returns: Performance percentage (0-1)
      :rtype: float

      **Performance Formula:**

      .. math::

         Performance = \frac{Theoretical\_Cycle\_Time \times Units\_Produced}{Actual\_Run\_Time}

   .. py:method:: calculate_quality(production_data, production_line, date, quality_data=None)

      Calculate quality metric based on production data.

      :param pd.DataFrame production_data: Production output data
      :param str production_line: Production line identifier
      :param datetime.date date: Target date
      :param pd.DataFrame quality_data: Optional quality/defect data
      :returns: Quality percentage (0-1)
      :rtype: float

      **Quality Formula:**

      .. math::

         Quality = \frac{Good\_Units\_Produced}{Total\_Units\_Produced}

   .. py:method:: calculate_oee(line_status_data, production_data, production_line, date)

      Calculate complete OEE metric combining all three factors.

      :param pd.DataFrame line_status_data: Line status data
      :param pd.DataFrame production_data: Production output data
      :param str production_line: Production line identifier
      :param datetime.date date: Target date
      :returns: OEE percentage and component breakdown
      :rtype: dict

      **Complete OEE Calculation:**

      .. code-block:: python

         def calculate_oee(self, line_status_data, production_data, production_line, date):
             """
             Calculate complete OEE with component breakdown
             
             Returns comprehensive OEE analysis including:
             - Individual component metrics
             - Combined OEE score
             - Performance diagnostics
             - Data quality indicators
             """
             
             # Calculate individual components
             availability = self.calculate_availability(
                 line_status_data, production_line, date
             )
             
             performance = self.calculate_performance(
                 production_data, line_status_data, production_line, date
             )
             
             quality = self.calculate_quality(
                 production_data, production_line, date
             )
             
             # Calculate OEE
             oee = availability * performance * quality
             
             # Prepare detailed results
             results = {
                 'date': date,
                 'production_line': production_line,
                 'availability': availability,
                 'performance': performance,
                 'quality': quality,
                 'oee': oee,
                 'data_quality': self._assess_data_quality(
                     line_status_data, production_data, production_line, date
                 ),
                 'calculation_metadata': {
                     'timestamp': datetime.now(),
                     'cycle_time_used': self.cycle_times.get(production_line),
                     'quality_assumption': self.quality_assumptions.get(production_line)
                 }
             }
             
             return results

📊 **Daily Report Generation**
=============================

.. py:function:: generate_daily_oee_reports(line_status_data, production_data, start_date=None, end_date=None)

   Generate comprehensive daily OEE reports for all production lines.

   :param pd.DataFrame line_status_data: Cleaned line status data
   :param pd.DataFrame production_data: Production output data
   :param datetime.date start_date: Start date for report period
   :param datetime.date end_date: End date for report period
   :returns: Daily OEE reports with comprehensive metrics
   :rtype: pd.DataFrame

   **Generated Report Structure:**

   .. code-block::

      Daily OEE Report Columns:
      
      ├── Date Information
      │   ├── Date
      │   ├── Day_of_Week
      │   └── Week_Number
      │
      ├── Production Line Metrics
      │   ├── Production_Line
      │   ├── Availability
      │   ├── Performance
      │   ├── Quality
      │   └── OEE
      │
      ├── Production Statistics
      │   ├── Units_Produced
      │   ├── Planned_Production_Time
      │   ├── Actual_Run_Time
      │   └── Downtime_Minutes
      │
      └── Data Quality Indicators
          ├── Data_Completeness
          ├── Calculation_Confidence
          └── Missing_Data_Flags

   **Implementation Example:**

   .. code-block:: python

      def generate_daily_oee_reports(line_status_data, production_data, 
                                   start_date=None, end_date=None):
          """
          Generate comprehensive daily OEE reports
          
          Features:
          - Multi-line processing
          - Data quality assessment
          - Performance trend calculation
          - Exception identification
          """
          
          # Initialize OEE calculator
          calculator = OEECalculator()
          
          # Determine date range
          if start_date is None:
              start_date = line_status_data['START_DATETIME'].dt.date.min()
          if end_date is None:
              end_date = line_status_data['START_DATETIME'].dt.date.max()
          
          # Get unique production lines
          production_lines = line_status_data['PRODUCTION_LINE'].unique()
          
          reports = []
          
          # Generate reports for each line and date
          for production_line in production_lines:
              for date in pd.date_range(start_date, end_date):
                  oee_result = calculator.calculate_oee(
                      line_status_data, production_data, 
                      production_line, date.date()
                  )
                  
                  # Add derived metrics
                  oee_result.update({
                      'day_of_week': date.day_name(),
                      'week_number': date.isocalendar()[1],
                      'month': date.month,
                      'quarter': f"Q{(date.month-1)//3 + 1}"
                  })
                  
                  reports.append(oee_result)
          
          # Convert to DataFrame
          daily_reports = pd.DataFrame(reports)
          
          # Add performance indicators
          daily_reports = add_performance_indicators(daily_reports)
          
          return daily_reports

🔍 **Data Quality Assessment**
=============================

.. py:function:: assess_data_quality(line_status_data, production_data)

   Comprehensive data quality assessment for manufacturing datasets.

   :param pd.DataFrame line_status_data: Line status data
   :param pd.DataFrame production_data: Production data
   :returns: Detailed data quality report
   :rtype: dict

   **Quality Assessment Framework:**

   .. code-block:: python

      def assess_data_quality(line_status_data, production_data):
          """
          Comprehensive data quality assessment
          
          Assessment Categories:
          1. Completeness - Missing data analysis
          2. Consistency - Data format and value consistency
          3. Accuracy - Data range and logical validation
          4. Timeliness - Data freshness and update frequency
          5. Integrity - Cross-dataset relationship validation
          """
          
          quality_report = {
              'completeness': assess_completeness(line_status_data, production_data),
              'consistency': assess_consistency(line_status_data, production_data),
              'accuracy': assess_accuracy(line_status_data, production_data),
              'timeliness': assess_timeliness(line_status_data, production_data),
              'integrity': assess_integrity(line_status_data, production_data),
              'overall_score': 0.0
          }
          
          # Calculate overall quality score
          scores = [quality_report[category]['score'] 
                   for category in quality_report if category != 'overall_score']
          quality_report['overall_score'] = np.mean(scores)
          
          return quality_report

**Data Quality Metrics:**

.. py:function:: assess_completeness(line_status_data, production_data)

   Assess data completeness across all required fields.

   **Completeness Checks:**

   - Missing datetime values
   - Incomplete production line information
   - Missing status transitions
   - Production data gaps

.. py:function:: assess_consistency(line_status_data, production_data)

   Evaluate data consistency and standardization.

   **Consistency Checks:**

   - Datetime format consistency
   - Production line name standardization
   - Status name variations
   - Unit measurement consistency

.. py:function:: assess_accuracy(line_status_data, production_data)

   Validate data accuracy and logical constraints.

   **Accuracy Checks:**

   - Reasonable datetime ranges
   - Logical status sequences
   - Production rate validation
   - Data range constraints

⚡ **Performance Optimization**
=============================

**Efficient Data Processing**

.. py:function:: optimize_data_processing(data, chunk_size=10000)

   Optimize large dataset processing using chunking strategies.

   :param pd.DataFrame data: Large dataset to process
   :param int chunk_size: Size of processing chunks
   :returns: Generator yielding processed chunks
   :rtype: generator

   **Implementation:**

   .. code-block:: python

      def optimize_data_processing(data, chunk_size=10000):
          """
          Memory-efficient data processing for large datasets
          
          Features:
          - Chunked processing to manage memory usage
          - Progress tracking for long operations
          - Error handling and recovery
          - Parallel processing support
          """
          
          total_chunks = len(data) // chunk_size + 1
          
          for i in range(0, len(data), chunk_size):
              chunk = data.iloc[i:i+chunk_size].copy()
              
              # Process chunk
              processed_chunk = process_data_chunk(chunk)
              
              yield {
                  'chunk_number': i // chunk_size + 1,
                  'total_chunks': total_chunks,
                  'data': processed_chunk,
                  'progress': ((i // chunk_size + 1) / total_chunks) * 100
              }

**Caching and Memoization**

.. py:function:: cache_oee_calculations(calculation_function)

   Decorator for caching OEE calculations to improve performance.

   **Usage Example:**

   .. code-block:: python

      @cache_oee_calculations
      def calculate_monthly_oee(line_data, production_data, month, year):
          """Cached monthly OEE calculation"""
          # Expensive calculation here
          return monthly_oee_results

🔗 **Integration Utilities**
===========================

**Streamlit Integration**

.. py:function:: prepare_data_for_streamlit(daily_reports, overall_reports)

   Prepare processed data for Streamlit application display.

   :param pd.DataFrame daily_reports: Daily OEE reports
   :param pd.DataFrame overall_reports: Overall aggregated reports
   :returns: Formatted data structures for Streamlit
   :rtype: dict

   **Data Preparation:**

   .. code-block:: python

      def prepare_data_for_streamlit(daily_reports, overall_reports):
          """
          Prepare data structures optimized for Streamlit display
          
          Optimizations:
          - Column formatting for display
          - Data type optimization
          - Index structure for efficient filtering
          - Pre-calculated summary statistics
          """
          
          # Format daily reports
          formatted_daily = daily_reports.copy()
          formatted_daily['Date'] = pd.to_datetime(formatted_daily['date'])
          formatted_daily['OEE_Percentage'] = formatted_daily['oee'] * 100
          
          # Create summary statistics
          summary_stats = {
              'total_lines': len(daily_reports['production_line'].unique()),
              'date_range': (daily_reports['date'].min(), daily_reports['date'].max()),
              'average_oee': daily_reports['oee'].mean(),
              'best_performing_line': daily_reports.loc[
                  daily_reports['oee'].idxmax(), 'production_line'
              ]
          }
          
          return {
              'daily_reports': formatted_daily,
              'overall_reports': overall_reports,
              'summary_stats': summary_stats,
              'last_updated': datetime.now()
          }

**API Response Formatting**

.. py:function:: format_api_response(data, request_params, status="success")

   Format data processing results for API responses.

   :param dict data: Processed data to return
   :param dict request_params: Original request parameters
   :param str status: Response status indicator
   :returns: Formatted API response
   :rtype: dict

   **Response Structure:**

   .. code-block:: python

      {
          "status": "success",
          "timestamp": "2024-01-15T10:30:00Z",
          "request_id": "req_12345",
          "data": {
              "oee_metrics": [...],
              "summary_statistics": {...},
              "data_quality": {...}
          },
          "metadata": {
              "processing_time": 1.23,
              "records_processed": 1000,
              "calculation_parameters": {...}
          },
          "warnings": [],
          "errors": []
      }

📈 **Usage Examples**
====================

**Basic OEE Calculation**

.. code-block:: python

   # Load and process data
   line_status, production_data = load_and_process_data(
       'line_status_notcleaned.csv',
       'production_data.csv'
   )

   # Initialize OEE calculator
   calculator = OEECalculator()

   # Calculate OEE for specific line and date
   oee_result = calculator.calculate_oee(
       line_status, 
       production_data, 
       'LINE-01', 
       date(2024, 1, 15)
   )

   print(f"OEE: {oee_result['oee']:.2%}")
   print(f"Availability: {oee_result['availability']:.2%}")
   print(f"Performance: {oee_result['performance']:.2%}")
   print(f"Quality: {oee_result['quality']:.2%}")

**Batch Processing**

.. code-block:: python

   # Generate comprehensive daily reports
   daily_reports = generate_daily_oee_reports(
       line_status, 
       production_data,
       start_date=date(2024, 1, 1),
       end_date=date(2024, 1, 31)
   )

   # Assess data quality
   quality_report = assess_data_quality(line_status, production_data)
   
   print(f"Data Quality Score: {quality_report['overall_score']:.1%}")

**Performance Monitoring**

.. code-block:: python

   # Monitor OEE calculations performance
   import time

   start_time = time.time()
   
   # Process large dataset
   for chunk_result in optimize_data_processing(large_dataset):
       process_chunk(chunk_result['data'])
       print(f"Progress: {chunk_result['progress']:.1f}%")
   
   processing_time = time.time() - start_time
   print(f"Total processing time: {processing_time:.2f} seconds")

🔧 **Error Handling**
====================

**Common Exceptions**

.. py:exception:: DataValidationError

   Raised when input data fails validation checks.

   **Example:**

   .. code-block:: python

      try:
          cleaned_data = clean_line_status_data(raw_data)
      except DataValidationError as e:
          print(f"Data validation failed: {e}")
          # Handle validation error

.. py:exception:: OEECalculationError

   Raised when OEE calculation encounters errors.

   **Example:**

   .. code-block:: python

      try:
          oee_result = calculator.calculate_oee(line_status, production_data, line, date)
      except OEECalculationError as e:
          print(f"OEE calculation failed: {e}")
          # Fallback to alternative calculation method

**Error Recovery Strategies**

.. code-block:: python

   def robust_oee_calculation(line_status_data, production_data, line, date):
       """
       Robust OEE calculation with comprehensive error handling
       """
       
       try:
           # Primary calculation method
           return calculator.calculate_oee(line_status_data, production_data, line, date)
           
       except DataValidationError:
           # Attempt data cleaning and retry
           cleaned_status = clean_line_status_data(line_status_data)
           return calculator.calculate_oee(cleaned_status, production_data, line, date)
           
       except OEECalculationError:
           # Fallback to simplified calculation
           return calculate_simplified_oee(line_status_data, production_data, line, date)
           
       except Exception as e:
           # Log error and return default values
           logger.error(f"Unexpected error in OEE calculation: {e}")
           return create_default_oee_result(line, date)

**Next Steps:**

- Explore :doc:`forecasting` for prediction API documentation
- Review :doc:`advisory_system` for RAG system APIs
- Check :doc:`../troubleshooting` for common data processing issues