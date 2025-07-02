# Financial Fraud Detection System

A comprehensive fraud detection system that combines **traditional ML models** with **modern LLM/Agent-based AI** for robust, explainable, and production-ready fraud detection.

## 🏗️ Project Architecture

This system demonstrates a **hybrid approach** that leverages:
- **Traditional ML Models**: Fast, interpretable fraud detection using XGBoost, SHAP, LIME
- **LLM/Agent Systems**: Context-aware reasoning using LangChain, OpenAI API, RAG pipelines
- **Business Intelligence**: Real-time analytics, compliance reporting, cost analysis
- **DevOps**: Complete CI/CD pipeline, monitoring, containerization

## 📁 Project Structure

```
fraud-detection-system/
├── 📁 src/                          # Source code
│   ├── 📁 data/                     # Data generation and processing
│   │   ├── __init__.py
│   │   └── data_generator.py        # Synthetic fraud data generator
│   ├── 📁 models/                   # ML model training and inference
│   │   ├── __init__.py
│   │   ├── train_model.py           # Model training pipeline
│   │   ├── inference.py             # Real-time inference
│   │   └── explainability.py        # SHAP/LIME explanations
│   ├── 📁 api/                      # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                  # API server
│   │   ├── models.py                # Pydantic models
│   │   └── routes/                  # API endpoints
│   ├── 📁 features/                 # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py   # Feature creation pipeline
│   ├── 📁 utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── helpers.py               # Common utilities
│   └── 📁 monitoring/               # Model monitoring
│       ├── __init__.py
│       └── monitoring.py            # Performance tracking
├── 📁 data/                         # Generated datasets
├── 📁 tests/                        # Unit and integration tests
├── 📁 docs/                         # Documentation
├── 📄 main.py                       # Main entry point
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # This file
└── 📄 architecture_diagram.md       # System architecture
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Generate synthetic fraud detection data
python main.py
```

This will create:
- `data/users.csv` - User profiles with behavior patterns
- `data/transactions.csv` - Transaction data with fraud labels
- `data/sample_transactions.json` - Sample data for API testing
- `data/dataset_stats.json` - Dataset statistics

### 3. Train ML Model

```bash
# Train traditional ML model
python src/models/train_model.py
```

### 4. Start API Server

```bash
# Start FastAPI server
python src/api/main.py
```

## 📊 Dataset Overview

The synthetic dataset includes:

### **Transaction Features:**
- **Basic Info**: Transaction ID, user ID, amount, timestamp
- **Merchant**: Category, location, device type
- **User Context**: User pattern, average amounts, transaction history
- **Fraud Labels**: Binary fraud indicator, fraud reason

### **Fraud Patterns:**
- **Velocity Fraud**: Multiple transactions in short time
- **Geographic Anomaly**: Transactions from unusual locations
- **Amount Anomaly**: Unusually high transaction amounts
- **Time Anomaly**: Transactions at suspicious hours
- **Merchant Risk**: High-risk merchant categories
- **Device Mismatch**: Different device types than usual

### **Dataset Size:**
- **10,000 users** with different behavior patterns
- **100,000 transactions** with realistic fraud distribution
- **~5-8% fraud rate** (industry realistic)

## 🔧 Technology Stack

### **ML/AI:**
- **Traditional ML**: Scikit-learn, XGBoost, SHAP, LIME
- **LLM/Agents**: LangChain, OpenAI API, RAG pipelines
- **Vector DB**: Pinecone, Weaviate (for future LLM integration)

### **Backend:**
- **API**: FastAPI, Pydantic, Uvicorn
- **Database**: PostgreSQL, Redis
- **ORM**: SQLAlchemy

### **DevOps:**
- **Containerization**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana, ELK Stack
- **CI/CD**: GitHub Actions, Helm, Terraform

## 🎯 Cursor AI Impact

This project showcases how **Cursor AI** accelerates development across all domains:

### **For ML Engineers:**
- **Complete ML Pipeline**: Automated model training, feature engineering, explainability
- **Production Code**: Production-ready inference, monitoring, deployment
- **Best Practices**: Proper project structure, testing, documentation

### **For Business Analysts:**
- **Automated Dashboards**: Real-time fraud analytics, compliance reporting
- **Cost Analysis**: ROI calculations, business impact metrics
- **Data Visualization**: Interactive charts, trend analysis

### **For DevOps Engineers:**
- **Infrastructure as Code**: Complete deployment automation
- **CI/CD Pipelines**: Automated testing, deployment, monitoring
- **Security**: Secrets management, network policies, encryption

## 📈 Next Steps

1. **Feature Engineering**: Implement advanced feature creation
2. **Model Training**: Train and evaluate multiple ML models
3. **API Development**: Build FastAPI endpoints for real-time inference
4. **LLM Integration**: Add LangChain agents and RAG pipelines
5. **Monitoring**: Implement model performance tracking
6. **Deployment**: Containerize and deploy to production

## 🤝 Contributing

This project demonstrates modern ML engineering practices. Feel free to:
- Add new fraud patterns
- Implement additional ML models
- Enhance the API functionality
- Improve monitoring and observability

## 📄 License

This project is for educational and demonstration purposes. 