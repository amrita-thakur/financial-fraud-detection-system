# Financial Fraud Detection System - Architecture Diagram

## System Overview
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FINANCIAL FRAUD DETECTION SYSTEM                              │
│                        Hybrid: Traditional ML Models + LLM/Agent-Based AI + Business + DevOps    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Hybrid Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FRONTEND LAYER                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   Admin         │  │  Business       │  │  Real-time      │  │  Compliance     │            │
│  │   Dashboard     │  │  Dashboard      │  │  Monitor        │  │  Portal         │            │
│  │   (React/Vue)   │  │  (Analytics)    │  │  (Live Alerts)  │  │  (Reports)      │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    API GATEWAY LAYER                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  FastAPI Gateway                                                                             │ │
│  │  • Authentication & Authorization                                                            │ │
│  │  • Rate Limiting                                                                             │ │
│  │  • Request/Response Logging                                                                   │ │
│  │  • API Versioning                                                                            │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                HYBRID FRAUD DETECTION PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌─────────────────────────────┐      ┌─────────────────────────────┐                           │
│  │  Traditional ML Pipeline    │      │  LLM/Agent Orchestration    │                           │
│  │  • Feature Engineering      │      │  • Multi-Agent System       │                           │
│  │  • Model Inference          │      │  • LLM Integration (LangChain│                           │
│  │  • SHAP/LIME Explainability │      │    LlamaIndex, OpenAI, etc.)│                           │
│  │  • Fast, Interpretable      │      │  • RAG Pipeline (Vector DB) │                           │
│  │    Predictions              │      │  • Contextual Reasoning     │                           │
│  └───────────────┬─────────────┘      └───────────────┬─────────────┘                           │
│                  │                                    │                                         │
│                  └──────────────┬─────────────────────┘                                         │
│                                 ▼                                                             │
│                    ┌─────────────────────────────────────────────┐                             │
│                    │      Hybrid Decision & Explanation Engine   │                             │
│                    │  • Combines ML and Agent/LLM outputs       │                             │
│                    │  • Generates final fraud decision          │                             │
│                    │  • Produces human-readable explanations    │                             │
│                    │  • Can escalate to human or trigger action│                             │
│                    └─────────────────────────────────────────────┘                             │
│                                 │                                                             │
│                                 ▼                                                             │
│                    ┌─────────────────────────────────────────────┐                             │
│                    │      Business Intelligence & Monitoring     │                             │
│                    │  • Analytics, Reporting, Compliance         │                             │
│                    │  • Real-time Alerts, Dashboards             │                             │
│                    └─────────────────────────────────────────────┘                             │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA & INFRASTRUCTURE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  Feature Store  │  │  Transaction    │  │  Vector DB      │  │  Model Registry │            │
│  │                 │  │  Database       │  │  (Pinecone,     │  │  (MLflow, etc.) │            │
│  │  • Feature      │  │  (PostgreSQL)   │  │   Weaviate, etc.)│  │                 │            │
│  │    Engineering  │  │  • Real-time    │  │  • Embeddings   │  │  • Model         │            │
│  │  • Feature      │  │    Transactions │  │  • Similarity   │  │    Versioning    │            │
│  │    Serving      │  │  • Historical   │  │    Search       │  │  • A/B Testing   │            │
│  │  • Monitoring   │  │    Data         │  │                 │  │  • Performance   │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  Container      │  │  Orchestration  │  │  Monitoring &   │  │  Security &     │            │
│  │  Platform       │  │  (Kubernetes)   │  │  Observability  │  │  Compliance     │            │
│  │  (Docker)       │  │  • Auto-scaling │  │  (Prometheus,   │  │  • Secrets      │            │
│  │  • Multi-stage  │  │  • Load         │  │   Grafana, ELK) │  │    Management   │            │
│  │    Builds       │  │    Balancing    │  │                 │  │  • Network      │            │
│  │  • Health       │  │  • Service      │  │                 │  │    Policies     │            │
│  │    Checks       │  │    Discovery    │  │                 │  │  • Encryption   │            │
│  │  • Resource     │  │  • Rolling      │  │                 │  │    Control      │            │
│  │    Limits       │  │    Updates      │  │                 │  │    Control      │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Hybrid Data Flow Example

```
┌───────────────┐
│ Transaction  │
│   Input      │
└──────┬────────┘
       │
       ▼
┌──────────────────────────────┐
│ Feature Engineering (ML)     │
└──────┬───────────────┬───────┘
       │               │
       ▼               ▼
┌──────────────┐   ┌────────────────────┐
│ ML Model     │   │ LLM/Agent System   │
│ Inference    │   │ (LangChain, RAG,   │
│ (XGBoost,    │   │  OpenAI, etc.)     │
│  Sklearn)    │   └────────────────────┘
└──────┬───────┘           │
       │                   │
       └──────┬────────────┘
              ▼
   ┌───────────────────────────────┐
   │ Hybrid Decision & Explanation │
   │ Engine                        │
   └──────────────┬────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Business/Alerting  │
        │ Dashboards/API     │
        └────────────────────┘
```

---

This hybrid architecture demonstrates how both traditional ML models and modern LLM/agent-based systems can be leveraged together for robust, explainable, and context-aware fraud detection, maximizing the strengths of both approaches. 