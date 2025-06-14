# 🏭 OEE Manufacturing Analytics & AI Advisory System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive **Overall Equipment Effectiveness (OEE)** analysis platform that combines traditional manufacturing metrics with cutting-edge AI capabilities. This system provides real-time OEE monitoring, advanced forecasting, and an AI-powered advisory system for manufacturing optimization.

## 🌟 Key Features

### 📊 **Core Analytics**
- **Real-time OEE Calculation**: Automatic computation of Availability, Performance, and Quality metrics
- **Multi-line Analysis**: Compare performance across multiple production lines
- **Interactive Dashboard**: Beautiful Plotly-based visualizations with drill-down capabilities
- **Historical Trend Analysis**: Identify patterns and improvement opportunities

### 🔮 **Advanced Forecasting**
- **Deep Learning Models**: RNN, CNN, and WaveNet-style architectures for time series prediction
- **Statistical Methods**: ARIMA models with automated parameter selection
- **Walk-Forward Validation**: Realistic performance evaluation using time-aware validation
- **Multi-step Forecasting**: Predict OEE values up to 30 days ahead

### 🤖 **AI Advisory System (RAG)**
- **Intelligent Q&A**: Ask questions about OEE optimization in natural language
- **Document Processing**: Upload PDF manuals and best practices for enhanced knowledge
- **Contextual Recommendations**: AI-powered suggestions based on your specific OEE data
- **Knowledge Base**: Built-in manufacturing expertise with expandable document library

### 📈 **Professional Dashboard**
- **Multiple Views**: Main dashboard, line-specific analysis, overall trends, and forecasting
- **Real-time Metrics**: Live KPI tracking with performance alerts
- **Export Capabilities**: Generate reports and export data for further analysis
- **Mobile Responsive**: Access from any device

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│   OEE Analytics │   Forecasting   │     AI Advisory         │
│                 │                 │                         │
│ • Real-time KPIs│ • Deep Learning │ • RAG System           │
│ • Trend Analysis│ • Statistical   │ • Document Processing  │
│ • Comparisons   │ • Validation    │ • Gemini API           │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Data Processing Layer                   │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Data Ingestion │   OEE Calc     │    ML Pipeline          │
│                 │                 │                         │
│ • CSV Processing│ • Availability  │ • Feature Engineering  │
│ • Data Cleaning │ • Performance   │ • Model Training       │
│ • Validation    │ • Quality       │ • Hyperparameter Opt.  │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        Data Sources                         │
│  📄 line_status.csv  │  📄 production_data.csv  │  📚 PDFs  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager
- 4GB+ RAM (8GB recommended for deep learning)

### 1. Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/oee-manufacturing-analytics.git
cd oee-manufacturing-analytics

# Install core dependencies
pip install -r requirements.txt

# Run the basic dashboard
streamlit run app.py
```

### 2. Full Installation (with AI Advisory)

```bash
# Run the automated setup script
python setup_advisory.py

# Or install manually:
pip install -r requirements_rag.txt
python -m spacy download en_core_web_sm

# Set your Gemini API key (optional - for AI features)
export GEMINI_API_KEY="your_api_key_here"

# Run with full features
streamlit run app.py
```

### 3. Docker Installation (Coming Soon)

```bash
docker pull oee-analytics:latest
docker run -p 8501:8501 oee-analytics
```

## 📁 Project Structure

```
oee-manufacturing-analytics/
├── 📊 Core Application
│   ├── app.py                          # Main Streamlit dashboard
│   ├── requirements.txt                # Core dependencies
│   └── requirements_rag.txt            # AI system dependencies
│
├── 🤖 AI Advisory System
│   ├── advisory_integration.py         # Streamlit integration
│   ├── rag_system.py                  # RAG implementation
│   ├── document_processor.py          # PDF processing
│   └── setup_advisory.py              # Automated setup
│
├── 📈 Analysis Notebooks
│   ├── OEE_Insights_1.ipynb          # Data preprocessing & OEE calculation
│   ├── OEE_Insights_2.ipynb          # Statistical analysis & ARIMA
│   └── OEE_Insights_3.ipynb          # Deep learning forecasting
│
├── 📄 Data Files
│   ├── line_status_notcleaned.csv     # Raw production line status
│   ├── production_data.csv            # Raw production output data
│   └── The_Complete_Guide_to_Simple_OEE.pdf  # Optional knowledge base
│
└── 🔧 Generated Directories (auto-created)
    ├── processed_documents/            # Processed AI documents
    ├── vector_db/                     # AI knowledge base
    ├── models/                        # Saved ML models
    └── embeddings/                    # AI embeddings cache
```

## 📊 Data Format

### Input Data Requirements

**Line Status Data** (`line_status_notcleaned.csv`):
```csv
PRODUCTION_LINE,START_DATETIME,FINISH_DATETIME,STATUS_NAME,SHIFT
LINE-01,2024-01-01 08:00:00,2024-01-01 09:30:00,Production,1
LINE-01,2024-01-01 09:30:00,2024-01-01 09:45:00,Break Time,1
```

**Production Data** (`production_data.csv`):
```csv
LINE,START_DATETIME,FINISH_DATETIME,PRODUCT_ID
LINE-01,2024-01-01 08:00:00,2024-01-01 08:11:00,PROD_001
LINE-01,2024-01-01 08:11:00,2024-01-01 08:22:00,PROD_002
```

## 🎯 Usage Examples

### Basic OEE Analysis
```python
# Load and analyze your data
from app import load_processed_data

daily_oee_data, overall_daily_oee = load_processed_data()

# Calculate average OEE by line
avg_oee = daily_oee_data.groupby('PRODUCTION_LINE')['OEE'].mean()
print(f"Average OEE: {avg_oee}")
```

### Deep Learning Forecasting
```python
# Use the forecasting models
from app import create_forecast, RobustScaler

# Prepare your data
scaler = RobustScaler()
scaled_data = scaler.fit_transform(oee_data.reshape(-1, 1))

# Generate 7-day forecast
forecast = create_forecast(
    model_builder_func=build_stacked_simplernn_with_masking,
    data_1d=oee_data,
    scaler_obj=scaler,
    look_back=14,
    forecast_steps=7
)
```

### AI Advisory Integration
```python
# Ask the AI advisor
from rag_system import create_oee_advisor

advisor = create_oee_advisor("your_gemini_api_key")
response = advisor.ask_question("How can I improve OEE for LINE-01?")
print(response["response"])
```

## 🧪 Model Performance

Our deep learning models have been extensively validated using walk-forward time series validation:

| Model | Architecture | Best MAE | Best RMSE | Use Case |
|-------|-------------|----------|-----------|----------|
| **Stacked RNN + Masking** | 2-layer SimpleRNN with attention | 0.0591 | 0.0798 | Variable-length sequences |
| **Multi-Kernel CNN** | Parallel conv1d towers | 0.0591 | 0.0798 | Pattern recognition |
| **WaveNet-style CNN** | Dilated convolutions | 0.0605 | 0.0814 | Long-range dependencies |
| **Statistical ARIMA** | Auto-selected parameters | 0.0664 | 0.0882 | Baseline & interpretability |

*Results shown for best-performing production line (LINE-06)*

## 🔧 Configuration

### Environment Variables
```bash
# Required for AI Advisory System
export GEMINI_API_KEY="your_google_gemini_api_key"

# Optional configurations
export OEE_FORECAST_HORIZON="7"  # Days to forecast ahead
export OEE_MODEL_CACHE="true"    # Enable model caching
export OEE_LOG_LEVEL="INFO"      # Logging level
```

### Cycle Times Configuration
Edit `app.py` to set your production line cycle times:
```python
CYCLE_TIMES = {
    'LINE-01': 11.0,  # seconds per unit
    'LINE-03': 5.5,
    'LINE-04': 11.0,
    'LINE-06': 11.0
}
```

## 🤖 AI Advisory System

The RAG (Retrieval-Augmented Generation) system provides intelligent OEE optimization advice:

### Features
- **Natural Language Q&A**: Ask questions like "Why is my OEE below 60%?"
- **Document Integration**: Upload maintenance manuals, best practices, troubleshooting guides
- **Contextual Analysis**: AI considers your specific production data when giving advice
- **Multi-source Knowledge**: Combines uploaded documents with built-in manufacturing expertise

### Supported Document Types
- PDF manuals and guides
- Maintenance procedures
- Quality control documents
- Best practice guidelines
- Troubleshooting manuals

### Example Queries
- "What are the main causes of low availability on production lines?"
- "How do I reduce changeover time for LINE-01?"
- "What's the industry benchmark for OEE in automotive manufacturing?"
- "Analyze the performance data for LINE-03 and suggest improvements"

## 📚 Technical Deep Dive

### OEE Calculation
```
OEE = Availability × Performance × Quality

Where:
• Availability = Actual Run Time / Planned Production Time
• Performance = (Total Output × Ideal Cycle Time) / Actual Run Time  
• Quality = Good Output / Total Output
```

### Forecasting Models

**1. Stacked SimpleRNN with Masking**
- Handles variable-length sequences with missing data
- 2-layer architecture with dropout regularization
- Multi-step prediction with 3-day horizon
- Input: 14-day lookback, padded to 20 timesteps

**2. Multi-Kernel CNN**
- Parallel convolutional towers with kernel sizes 3, 5, 7
- Captures patterns at multiple time scales
- Global average pooling for dimensionality reduction
- Input: 30-day lookback window

**3. WaveNet-style Dilated CNN**
- Exponentially increasing dilation rates (1, 2, 4, 8, 16, 32)
- Causal padding maintains temporal order
- Efficient long-range dependency modeling
- Input: 14-day lookback window

### Validation Strategy
- **Walk-Forward Validation**: Realistic time-aware evaluation
- **70/15/15 Split**: Training/Validation/Test data split
- **Multi-step Evaluation**: Models trained for 3-step ahead prediction
- **Real-world Simulation**: Retraining at each prediction step

## 🛠️ Development

### Setting Up Development Environment
```bash
# Clone and setup
git clone https://github.com/yourusername/oee-manufacturing-analytics.git
cd oee-manufacturing-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e .
pip install -r requirements_dev.txt  # Includes testing tools

# Run tests
pytest tests/

# Code formatting
black app.py
isort .
```

### Adding New Models
1. Create model builder function in `app.py`
2. Add to model options in `show_forecasting_page()`
3. Configure hyperparameters and validation
4. Test with walk-forward validation

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Example Dashboard Screenshots

### Main Dashboard
*![Main Dashboard](docs/images/main_dashboard.png)*
- Real-time OEE metrics across all production lines
- Performance comparison charts
- Quick access to line-specific analysis

### Forecasting Interface
*![Forecasting](docs/images/forecasting.png)*
- Model selection and configuration
- Interactive forecast visualizations
- Performance metrics and validation results

### AI Advisory Chat
*![AI Advisory](docs/images/ai_advisory.png)*
- Natural language query interface
- Contextual recommendations based on your data
- Document source citations

## 🔍 Troubleshooting

### Common Issues

**TensorFlow Import Errors (Windows)**
```bash
# Install Visual C++ Redistributable
# Download from: https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

# Or use CPU-only version
pip uninstall tensorflow
pip install tensorflow-cpu
```

**Memory Issues with Large Datasets**
```python
# Reduce batch size in model training
BATCH_SIZE = 16  # Instead of 32

# Use data generators for large files
# Enable model checkpointing
```

**Advisory System Setup Issues**
```bash
# Check NLTK data
python -c "import nltk; nltk.download('punkt')"

# Verify spaCy model
python -c "import spacy; spacy.load('en_core_web_sm')"

# Test Gemini API connection
python -c "import google.generativeai as genai; genai.configure(api_key='your_key')"
```

### Performance Optimization
- **Use SSD storage** for faster data loading
- **Enable GPU** for deep learning models (if available)
- **Increase RAM** for large datasets (8GB+ recommended)
- **Use model caching** to avoid retraining

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- 🔧 Additional forecasting models
- 📊 New visualization types
- 🤖 Enhanced AI advisory capabilities
- 🌐 Multi-language support
- 📱 Mobile app development
- 🔌 ERP system integrations

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/oee-manufacturing-analytics/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/oee-manufacturing-analytics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/oee-manufacturing-analytics/discussions)
- **Email**: support@oee-analytics.com

## 🙏 Acknowledgments

- **Manufacturing Domain Knowledge**: Based on industry best practices and ISO 22400 standards
- **Deep Learning Architecture**: Inspired by WaveNet and ROCKET time series models
- **AI Advisory System**: Powered by Google Gemini API and advanced RAG techniques
- **Data Visualization**: Built with Plotly and Streamlit for interactive experiences

## 🔮 Roadmap

### Short Term (Q2 2024)
- [ ] Real-time data streaming support
- [ ] Enhanced mobile interface
- [ ] Additional statistical models
- [ ] Automated report generation

### Medium Term (Q3-Q4 2024)
- [ ] Multi-plant deployment
- [ ] Advanced anomaly detection
- [ ] Predictive maintenance integration
- [ ] REST API development

### Long Term (2025+)
- [ ] Edge computing deployment
- [ ] IoT sensor integration
- [ ] Advanced AI recommendations
- [ ] Industry 4.0 compliance

---

**⭐ Star this repository if you find it useful!**

*Built with ❤️ for the manufacturing community*