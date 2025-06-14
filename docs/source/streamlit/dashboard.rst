Dashboard Features and Navigation
=================================

The main dashboard serves as the central command center for OEE monitoring and analysis, providing real-time insights into manufacturing performance across all production lines.

🏠 **Main Dashboard Overview**
=============================

**Purpose and Scope:**

The main dashboard provides a comprehensive overview of manufacturing performance with:

- **Real-time Performance Metrics**: Live OEE, Availability, Performance, and Quality indicators
- **Production Line Status**: Visual status indicators for each active production line
- **Comparative Analysis**: Side-by-side performance comparison across lines
- **Trend Visualization**: Historical performance trends and patterns
- **Quick Navigation**: One-click access to detailed analysis pages

**Key Design Principles:**

1. **At-a-Glance Intelligence**: Critical information visible without scrolling
2. **Visual Hierarchy**: Most important metrics prominently displayed
3. **Interactive Elements**: Clickable charts and drill-down capabilities
4. **Performance-Based Color Coding**: Immediate visual status recognition
5. **Responsive Layout**: Optimal viewing on all device sizes

📊 **Top-Level Metrics Section**
===============================

**Overview Metrics Display:**

The dashboard header presents four key performance indicators:

.. code-block:: python

   # Implementation of top-level metrics
   col1, col2, col3, col4 = st.columns(4)

   # Calculate aggregate metrics
   avg_oee = daily_oee_data['OEE'].mean()
   avg_availability = daily_oee_data['Availability'].mean()
   avg_performance = daily_oee_data['Performance'].mean()
   total_output = daily_oee_data['Total_Actual_Output'].sum()

   with col1:
       st.metric("Average OEE", f"{avg_oee:.1%}")
   with col2:
       st.metric("Average Availability", f"{avg_availability:.1%}")
   with col3:
       st.metric("Average Performance", f"{avg_performance:.1%}")
   with col4:
       st.metric("Total Output", f"{total_output:,}")

**Metric Cards Features:**

.. list-table:: Metric Card Components
   :header-rows: 1
   :widths: 25 35 40

   * - Element
     - Purpose
     - Visual Treatment
   * - Primary Value
     - Main performance indicator
     - Large, bold typography with percentage formatting
   * - Trend Indicator
     - Direction of change (↑↓)
     - Color-coded arrows (green=up, red=down)
   * - Delta Value
     - Change from previous period
     - Small text with +/- formatting
   * - Background Color
     - Performance level indication
     - Light blue background with colored left border

**Metric Calculation Logic:**

.. code-block:: python

   def calculate_dashboard_metrics(daily_oee_data):
       """Calculate top-level dashboard metrics with trend analysis"""
       
       # Current period metrics
       current_metrics = {
           'avg_oee': daily_oee_data['OEE'].mean(),
           'avg_availability': daily_oee_data['Availability'].mean(), 
           'avg_performance': daily_oee_data['Performance'].mean(),
           'total_output': daily_oee_data['Total_Actual_Output'].sum()
       }
       
       # Previous period for comparison (if available)
       if len(daily_oee_data) > 30:  # Minimum 30 days for comparison
           current_30 = daily_oee_data.tail(30)
           previous_30 = daily_oee_data.iloc[-60:-30] if len(daily_oee_data) >= 60 else None
           
           if previous_30 is not None:
               # Calculate trends
               current_metrics['oee_trend'] = (
                   current_30['OEE'].mean() - previous_30['OEE'].mean()
               )
               current_metrics['availability_trend'] = (
                   current_30['Availability'].mean() - previous_30['Availability'].mean()
               )
               # ... additional trend calculations
       
       return current_metrics

🏭 **Production Line Status Grid**
==================================

**Interactive Line Status Buttons:**

The production line status section displays real-time status for each active line:

.. code-block:: python

   def show_production_lines_status(daily_oee_data):
       """Display interactive production line status grid"""
       
       st.subheader("🏭 Production Lines Status")
       st.markdown("*Click on any line to view detailed analysis*")
       
       lines = sorted(daily_oee_data['PRODUCTION_LINE'].unique())
       cols = st.columns(len(lines))
       
       for i, line in enumerate(lines):
           with cols[i]:
               # Get current status
               status, current_oee, icon = get_line_current_status(line, daily_oee_data)
               
               # Create clickable status button
               button_clicked = st.button(
                   f"{icon} **{line}**\n\nStatus: {status}\nOEE: {current_oee:.1%}",
                   key=f"line_button_{line}",
                   help=f"Click to analyze {line}",
                   use_container_width=True
               )
               
               if button_clicked:
                   # Navigate to line-specific analysis
                   st.session_state.page = "📈 Line-Specific Analysis"
                   st.session_state.selected_line = line
                   st.rerun()

**Status Classification Logic:**

.. code-block:: python

   def get_line_current_status(line, daily_oee_data):
       """Determine current status and performance level for a production line"""
       
       line_data = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == line]
       
       if line_data.empty:
           return "No Data", 0.0, "🔴"
       
       # Get most recent OEE value
       latest_data = line_data.loc[line_data['Date'].idxmax()]
       latest_oee = latest_data['OEE']
       
       # Classify performance level
       if latest_oee >= 0.85:
           status, icon = "Excellent", "🟢"
       elif latest_oee >= 0.70:
           status, icon = "Good", "🟡" 
       elif latest_oee >= 0.50:
           status, icon = "Fair", "🟠"
       else:
           status, icon = "Poor", "🔴"
       
       return status, latest_oee, icon

**Visual Status Indicators:**

.. list-table:: Status Classification System
   :header-rows: 1
   :widths: 15 15 20 25 25

   * - Status
     - Icon
     - OEE Range
     - Color Theme
     - Business Implication
   * - Excellent
     - 🟢
     - 85%+
     - Green (#2E8B57)
     - World-class performance
   * - Good
     - 🟡
     - 70-85%
     - Yellow (#FFD700)
     - Above average, room for improvement
   * - Fair
     - 🟠
     - 50-70%
     - Orange (#FF8C00)
     - Below target, needs attention
   * - Poor
     - 🔴
     - <50%
     - Red (#DC143C)
     - Critical, immediate action required

⚖️ **Performance Comparison Section**
====================================

**Interactive Metric Selection:**

Users can compare production lines across different performance metrics:

.. code-block:: python

   def show_performance_comparison(daily_oee_data):
       """Display interactive performance comparison section"""
       
       st.subheader("⚖️ Compare Production Lines")
       
       col1, col2 = st.columns([1, 3])
       
       with col1:
           # Metric selection dropdown
           comparison_metric = st.selectbox(
               "Select Metric to Compare:", 
               options=['OEE', 'Availability', 'Performance', 'Quality'],
               key="comparison_metric",
               help="Choose which metric to compare across production lines"
           )
       
       with col2:
           # Generate comparison chart
           fig_comparison = create_comparison_chart(daily_oee_data, comparison_metric)
           st.plotly_chart(fig_comparison, use_container_width=True)

**Comparison Chart Implementation:**

.. code-block:: python

   def create_comparison_chart(daily_oee_data, metric):
       """Create interactive comparison chart for selected metric"""
       
       # Calculate average values by production line
       avg_data = daily_oee_data.groupby('PRODUCTION_LINE')[metric].mean().reset_index()
       avg_data = avg_data.sort_values(metric, ascending=False)
       
       # Assign colors based on performance levels
       colors = []
       for value in avg_data[metric]:
           if value >= 0.85:
               colors.append('#2E8B57')  # Excellent - Green
           elif value >= 0.70:
               colors.append('#FFD700')  # Good - Yellow
           elif value >= 0.50:
               colors.append('#FF8C00')  # Fair - Orange
           else:
               colors.append('#DC143C')  # Poor - Red
       
       # Create Plotly bar chart
       fig = go.Figure()
       fig.add_trace(go.Bar(
           x=avg_data['PRODUCTION_LINE'],
           y=avg_data[metric],
           marker_color=colors,
           text=[f"{val:.1%}" for val in avg_data[metric]],
           textposition='auto',
           hovertemplate='<b>%{x}</b><br>' +
                        f'{metric}: %{{y:.1%}}<br>' +
                        '<extra></extra>'
       ))
       
       # Configure layout
       fig.update_layout(
           title=f'{metric} Comparison Across Production Lines',
           xaxis_title='Production Line',
           yaxis_title=f'Average {metric}',
           yaxis=dict(
               tickformat=',.0%',
               range=[0, max(1.1, avg_data[metric].max() * 1.1)]
           ),
           height=400,
           showlegend=False
       )
       
       return fig

🏆 **Performance Ranking Section**
=================================

**Dynamic Ranking Table:**

The ranking section provides a comprehensive performance leaderboard:

.. code-block:: python

   def show_performance_ranking(daily_oee_data):
       """Display performance ranking section with interactive controls"""
       
       st.subheader("🏆 Performance Ranking")
       
       col1, col2 = st.columns([1, 3])
       
       with col1:
           # Ranking metric selection
           ranking_metric = st.selectbox(
               "Rank Lines By:", 
               options=['OEE', 'Availability', 'Performance', 'Quality'],
               key="ranking_metric",
               help="Select metric for ranking production lines"
           )
           
           # Show top performers
           st.markdown("### 🥇 Top Performers")
           ranking_df = create_ranking_table(daily_oee_data, ranking_metric)
           
           medals = ["🥇", "🥈", "🥉"]
           for i in range(min(3, len(ranking_df))):
               line_name = ranking_df.index[i]
               rank_value = ranking_df.iloc[i][f'{ranking_metric}_formatted']
               medal = medals[i] if i < 3 else f"{i+1}."
               st.markdown(f"{medal} **{line_name}**: {rank_value}")
       
       with col2:
           # Complete rankings table
           st.markdown("### 📊 Complete Rankings")
           display_ranking_table(ranking_df)

**Ranking Table Generation:**

.. code-block:: python

   def create_ranking_table(daily_oee_data, metric):
       """Create comprehensive ranking table with formatted metrics"""
       
       # Calculate aggregate statistics
       avg_data = daily_oee_data.groupby('PRODUCTION_LINE').agg({
           'OEE': 'mean',
           'Availability': 'mean', 
           'Performance': 'mean',
           'Quality': 'mean',
           'Total_Actual_Output': 'sum'
       }).round(4)
       
       # Sort by selected metric
       avg_data = avg_data.sort_values(metric, ascending=False)
       
       # Add ranking
       avg_data['Rank'] = range(1, len(avg_data) + 1)
       
       # Format percentage columns
       percentage_cols = ['OEE', 'Availability', 'Performance', 'Quality']
       for col in percentage_cols:
           avg_data[f'{col}_formatted'] = avg_data[col].apply(lambda x: f"{x:.1%}")
       
       return avg_data

**Advanced Ranking Features:**

.. list-table:: Ranking Table Features
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Description
     - Implementation
   * - Color-Coded Rows
     - Performance-based row highlighting
     - Top 3 performers get gold, silver, bronze backgrounds
   * - Sortable Columns
     - Click column headers to re-sort
     - Streamlit's native dataframe sorting
   * - Tooltip Information
     - Hover for additional details
     - Performance improvement suggestions
   * - Export Options
     - Download ranking data
     - CSV, Excel formats available

📈 **Trend Visualization Section**
=================================

**Dual Chart Layout:**

The dashboard features two complementary trend visualizations:

.. code-block:: python

   def show_trend_visualizations(daily_oee_data, overall_daily_oee):
       """Display comprehensive trend visualization section"""
       
       col1, col2 = st.columns(2)
       
       with col1:
           st.subheader("📈 Overall OEE Trend")
           fig_trend = create_oee_trend_chart(
               overall_daily_oee, 
               title_suffix="(All Lines)"
           )
           st.plotly_chart(fig_trend, use_container_width=True)
       
       with col2:
           st.subheader("📊 Average Performance by Line")
           fig_avg = create_avg_oee_chart(daily_oee_data)
           st.plotly_chart(fig_avg, use_container_width=True)

**Overall OEE Trend Chart:**

.. code-block:: python

   def create_oee_trend_chart(data, line=None, title_suffix=""):
       """Create comprehensive OEE trend chart with multiple metrics"""
       
       fig = go.Figure()
       
       # Main OEE trend line
       fig.add_trace(go.Scatter(
           x=data['Date'], 
           y=data['OEE'], 
           mode='lines+markers', 
           name='OEE',
           line=dict(color='#1f77b4', width=3), 
           marker=dict(size=6),
           hovertemplate='<b>OEE</b><br>' +
                        'Date: %{x|%Y-%m-%d}<br>' +
                        'OEE: %{y:.1%}<extra></extra>'
       ))
       
       # Component trend lines
       components = [
           ('Availability', '#ff7f0e', 'dash'),
           ('Performance', '#2ca02c', 'dash')
       ]
       
       for component, color, dash_style in components:
           if component in data.columns:
               fig.add_trace(go.Scatter(
                   x=data['Date'], 
                   y=data[component], 
                   mode='lines+markers', 
                   name=component,
                   line=dict(color=color, width=2, dash=dash_style), 
                   marker=dict(size=4),
                   hovertemplate=f'<b>{component}</b><br>' +
                                'Date: %{x|%Y-%m-%d}<br>' +
                                f'{component}: %{{y:.1%}}<extra></extra>'
               ))
       
       # Add reference lines for benchmarks
       fig.add_hline(
           y=0.85, 
           line_dash="dot", 
           line_color="green",
           annotation_text="World Class (85%)",
           annotation_position="top right"
       )
       
       fig.add_hline(
           y=0.70, 
           line_dash="dot", 
           line_color="orange", 
           annotation_text="Good Performance (70%)",
           annotation_position="bottom right"
       )
       
       # Configure layout
       fig.update_layout(
           title=f'OEE and Components Trend {title_suffix}',
           xaxis_title='Date', 
           yaxis_title='Percentage',
           yaxis=dict(tickformat=',.0%', range=[0, 1.1]),
           hovermode='x unified', 
           height=500,
           legend=dict(
               x=0, y=1, 
               bgcolor='rgba(255,255,255,0.8)',
               bordercolor='rgba(0,0,0,0.2)',
               borderwidth=1
           )
       )
       
       return fig

**Average Performance Chart:**

.. code-block:: python

   def create_avg_oee_chart(daily_oee_data):
       """Create grouped bar chart showing average performance across all metrics"""
       
       # Calculate averages by production line
       avg_oee = daily_oee_data.groupby('PRODUCTION_LINE')[
           ['OEE', 'Availability', 'Performance', 'Quality']
       ].mean().reset_index()
       
       fig = go.Figure()
       
       # Define metrics and colors
       metrics = ['OEE', 'Availability', 'Performance', 'Quality']
       colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
       
       # Add traces for each metric
       for i, metric in enumerate(metrics):
           fig.add_trace(go.Bar(
               name=metric,
               x=avg_oee['PRODUCTION_LINE'],
               y=avg_oee[metric],
               marker_color=colors[i],
               hovertemplate=f'<b>{metric}</b><br>' +
                            'Line: %{x}<br>' +
                            f'{metric}: %{{y:.1%}}<extra></extra>'
           ))
       
       # Configure layout
       fig.update_layout(
           title='Average OEE and Components per Production Line',
           xaxis_title='Production Line', 
           yaxis_title='Average Value',
           yaxis=dict(tickformat=',.0%', range=[0, 1.1]), 
           barmode='group', 
           height=500,
           legend=dict(
               orientation="h",
               yanchor="bottom",
               y=1.02,
               xanchor="right",
               x=1
           )
       )
       
       return fig

📋 **Summary Data Table**
========================

**Comprehensive Production Summary:**

The bottom section provides a detailed data table for reference:

.. code-block:: python

   def show_summary_data_table(daily_oee_data):
       """Display comprehensive production line summary table"""
       
       st.subheader("📋 Production Line Summary")
       
       # Calculate comprehensive summary statistics
       summary = daily_oee_data.groupby('PRODUCTION_LINE').agg({
           'OEE': ['mean', 'min', 'max', 'std'],
           'Availability': ['mean', 'std'],
           'Performance': ['mean', 'std'], 
           'Quality': ['mean'],
           'Total_Actual_Output': ['sum', 'mean'],
           'Date': ['count']  # Number of data points
       }).round(3)
       
       # Flatten column names
       summary.columns = [
           'Avg OEE', 'Min OEE', 'Max OEE', 'OEE Std',
           'Avg Availability', 'Availability Std',
           'Avg Performance', 'Performance Std',
           'Avg Quality',
           'Total Output', 'Avg Daily Output',
           'Data Points'
       ]
       
       # Format percentage columns
       percentage_cols = [
           'Avg OEE', 'Min OEE', 'Max OEE', 'OEE Std',
           'Avg Availability', 'Availability Std',
           'Avg Performance', 'Performance Std',
           'Avg Quality'
       ]
       
       display_summary = summary.copy()
       for col in percentage_cols:
           if col in display_summary.columns:
               display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1%}")
       
       # Format integer columns
       integer_cols = ['Total Output', 'Data Points']
       for col in integer_cols:
           if col in display_summary.columns:
               display_summary[col] = display_summary[col].apply(lambda x: f"{x:,.0f}")
       
       # Display with enhanced styling
       st.dataframe(
           display_summary, 
           use_container_width=True,
           height=300
       )

**Table Enhancement Features:**

.. list-table:: Summary Table Features
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Description
     - Business Value
   * - Statistical Measures
     - Mean, min, max, standard deviation
     - Understand performance variability
   * - Production Totals
     - Cumulative and average output
     - Capacity and efficiency analysis
   * - Data Quality Indicators
     - Number of data points per line
     - Assess data reliability
   * - Export Capability
     - Download table data
     - Offline analysis and reporting

🎯 **Quick Actions and Navigation**
==================================

**AI Advisory Integration (Optional):**

When the advisory system is available, quick action buttons provide immediate access:

.. code-block:: python

   def show_advisory_quick_actions():
       """Display quick action buttons for AI advisory features"""
       
       if ADVISORY_AVAILABLE:
           st.markdown("---")
           col1, col2, col3 = st.columns([1, 1, 1])
           
           with col1:
               if st.button("🤖 Ask AI Advisor", use_container_width=True, type="primary"):
                   st.session_state.page = "🤖 OEE Advisory"
                   st.rerun()
           
           with col2:
               if st.button("📚 Manage Documents", use_container_width=True):
                   st.session_state.page = "📚 Document Management"
                   st.rerun()
           
           with col3:
               if st.button("⚡ Quick Analysis", use_container_width=True):
                   # Trigger automatic analysis of current performance
                   st.session_state.page = "🤖 OEE Advisory"
                   st.session_state.quick_analysis_requested = True
                   st.rerun()

**Contextual Navigation:**

The dashboard provides intelligent navigation based on current performance:

.. code-block:: python

   def provide_contextual_recommendations(daily_oee_data):
       """Provide contextual navigation recommendations based on performance"""
       
       # Identify lines needing attention
       poor_performers = daily_oee_data.groupby('PRODUCTION_LINE')['OEE'].mean()
       poor_performers = poor_performers[poor_performers < 0.70].index.tolist()
       
       if poor_performers:
           st.warning(f"⚠️ Lines needing attention: {', '.join(poor_performers)}")
           
           col1, col2 = st.columns(2)
           with col1:
               if st.button(f"📈 Analyze {poor_performers[0]}", use_container_width=True):
                   st.session_state.page = "📈 Line-Specific Analysis"
                   st.session_state.selected_line = poor_performers[0]
                   st.rerun()
           
           with col2:
               if st.button("🔮 Generate Forecasts", use_container_width=True):
                   st.session_state.page = "🔮 OEE Forecasting"
                   st.rerun()

📱 **Mobile-Optimized Dashboard**
================================

**Responsive Design Features:**

The dashboard automatically adapts to different screen sizes:

**Desktop Layout (>1200px):**
   - Full 4-column metric display
   - Side-by-side trend charts
   - Complete data table with all columns

**Tablet Layout (768-1200px):**
   - 2x2 metric grid
   - Stacked trend charts
   - Simplified data table

**Mobile Layout (<768px):**
   - Single-column metric cards
   - Full-width charts
   - Essential data only

.. code-block:: python

   def create_responsive_layout():
       """Create responsive layout based on screen size"""
       
       # Use CSS media queries and Streamlit columns
       # Implementation automatically adapts based on viewport
       
       # Mobile-first approach with progressive enhancement
       if is_mobile():
           create_mobile_dashboard()
       elif is_tablet():
           create_tablet_dashboard()
       else:
           create_desktop_dashboard()

🔄 **Real-Time Updates and Refresh**
===================================

**Automatic Data Refresh:**

The dashboard supports automatic updates when new data is available:

.. code-block:: python

   def check_for_data_updates():
       """Check for new data and refresh dashboard if needed"""
       
       # Check file modification times
       current_mod_time = get_data_file_modification_time()
       last_processed_time = st.session_state.get('last_data_refresh', 0)
       
       if current_mod_time > last_processed_time:
           # New data available - trigger refresh
           st.session_state.last_data_refresh = current_mod_time
           st.cache_data.clear()  # Clear cached data
           st.rerun()

**Manual Refresh Controls:**

.. code-block:: python

   def add_refresh_controls():
       """Add manual refresh controls to the dashboard"""
       
       with st.sidebar:
           st.markdown("### 🔄 Data Controls")
           
           if st.button("🔄 Refresh Data", use_container_width=True):
               # Clear all caches and reload data
               st.cache_data.clear()
               st.cache_resource.clear()
               st.success("Data refreshed!")
               st.rerun()
           
           # Show last update time
           last_update = get_last_data_update_time()
           st.caption(f"Last updated: {last_update}")

⚙️ **Dashboard Customization Options**
======================================

**User Preferences:**

.. code-block:: python

   def dashboard_customization_sidebar():
       """Provide dashboard customization options in sidebar"""
       
       with st.sidebar:
           st.markdown("### ⚙️ Dashboard Settings")
           
           # Time range selection
           time_range = st.selectbox(
               "Time Range:",
               options=["Last 30 days", "Last 60 days", "Last 90 days", "All time"],
               index=1
           )
           
           # Metric display options
           show_targets = st.checkbox("Show Performance Targets", value=True)
           show_trends = st.checkbox("Show Trend Indicators", value=True)
           
           # Chart preferences
           chart_height = st.slider("Chart Height", 300, 800, 500)
           
           return {
               'time_range': time_range,
               'show_targets': show_targets,
               'show_trends': show_trends,
               'chart_height': chart_height
           }

**Export and Sharing:**

.. code-block:: python

   def add_export_options():
       """Add export and sharing options to the dashboard"""
       
       with st.sidebar:
           st.markdown("### 📤 Export Options")
           
           if st.button("📊 Export Dashboard PDF"):
               generate_dashboard_pdf()
               st.success("Dashboard exported!")
           
           if st.button("📈 Export Data CSV"):
               generate_csv_export()
               st.success("Data exported!")
           
           if st.button("🔗 Share Dashboard Link"):
               generate_shareable_link()
               st.success("Link copied to clipboard!")

🎯 **Performance Monitoring and Alerts**
=======================================

**Automated Alert System:**

.. code-block:: python

   def check_performance_alerts(daily_oee_data):
       """Check for performance issues and display alerts"""
       
       alerts = []
       
       # Check for lines below threshold
       current_performance = daily_oee_data.groupby('PRODUCTION_LINE')['OEE'].last()
       
       for line, oee in current_performance.items():
           if oee < 0.50:
               alerts.append({
                   'type': 'critical',
                   'line': line,
                   'oee': oee,
                   'message': f"{line} OEE critically low: {oee:.1%}"
               })
           elif oee < 0.70:
               alerts.append({
                   'type': 'warning', 
                   'line': line,
                   'oee': oee,
                   'message': f"{line} OEE below target: {oee:.1%}"
               })
       
       # Display alerts
       for alert in alerts:
           if alert['type'] == 'critical':
               st.error(f"🚨 {alert['message']}")
           else:
               st.warning(f"⚠️ {alert['message']}")

🚀 **Advanced Dashboard Features**
=================================

**Drill-Down Capabilities:**

Every chart and metric supports drill-down analysis:

.. code-block:: python

   # Example: Clickable chart with drill-down
   def create_drilldown_chart(data):
       """Create chart with drill-down capability"""
       
       fig = create_base_chart(data)
       
       # Add click event handling
       fig.update_layout(clickmode='event+select')
       
       # Display chart and handle selection
       selected_data = st.plotly_chart(fig, use_container_width=True)
       
       if selected_data:
           # Show detailed analysis for selected data point
           show_detailed_analysis(selected_data)

**Comparative Time Periods:**

.. code-block:: python

   def add_time_comparison():
       """Add time period comparison functionality"""
       
       col1, col2 = st.columns(2)
       
       with col1:
           period1 = st.date_input("Compare Period 1:")
       
       with col2:
           period2 = st.date_input("Compare Period 2:")
       
       if period1 and period2:
           comparison_chart = create_period_comparison_chart(period1, period2)
           st.plotly_chart(comparison_chart, use_container_width=True)

📈 **Business Intelligence Integration**
=======================================

**KPI Dashboard Integration:**

The dashboard can integrate with existing BI systems:

.. code-block:: python

   def integrate_with_bi_system():
       """Integration points for business intelligence systems"""
       
       # REST API endpoints for data export
       api_endpoints = {
           '/api/current_oee': 'Real-time OEE metrics',
           '/api/line_status': 'Production line status',
           '/api/trends': 'Historical trend data',
           '/api/alerts': 'Active performance alerts'
       }
       
       # Data formats supported
       export_formats = ['JSON', 'CSV', 'XML', 'Excel']
       
       return api_endpoints, export_formats

📚 **Dashboard Usage Best Practices**
====================================

**Daily Operations:**

1. **Morning Review**: Check overnight performance and alerts
2. **Status Monitoring**: Review production line status indicators
3. **Issue Investigation**: Use drill-down features for problem areas
4. **Performance Tracking**: Monitor trends and compare to targets

**Weekly Analysis:**

1. **Performance Ranking**: Review weekly rankings and improvements
2. **Trend Analysis**: Identify patterns and seasonal effects
3. **Comparative Analysis**: Compare performance across lines
4. **Planning**: Use insights for next week's production planning

**Monthly Reporting:**

1. **Export Capabilities**: Generate comprehensive reports
2. **Summary Statistics**: Review monthly performance summaries
3. **Target Setting**: Update performance targets based on trends
4. **Strategic Planning**: Use data for long-term planning decisions

**Next Steps:**

Explore additional dashboard functionality:

- :doc:`forecasting` - Advanced forecasting dashboard
- :doc:`advisory_system` - AI-powered advisory interface
- :doc:`../models/evaluation_metrics` - Understanding performance metrics
- :doc:`../advanced/deployment` - Production deployment considerations