# UltraSafe Data Processing and Analysis System

A comprehensive FastAPI-based data processing and analysis system with RAG (Retrieval-Augmented Generation) capabilities and multi-agent analysis workflows using LangGraph.

## Project Overview

This project implements a sophisticated data processing and analysis platform with two main components:

### Data Processing & RAG System
- **FastAPI Application**: Complete data processing, analysis, and visualization API
- **RAG Integration**: Vector database integration with Pinecone for similar dataset retrieval
- **JSON Database**: Lightweight file-based database for data persistence
- **Data Processing**: Multi-format support with validation, cleaning, and export capabilities

### Multi-Agent Analysis System
- **LangGraph Workflows**: Orchestrated multi-agent workflows for comprehensive data analysis
- **Specialist Agents**: Data exploration, statistical analysis, visualization, and insight generation
- **RAG Enhancement**: Knowledge base of statistical methods and best practices
- **Collaborative Analysis**: Coordinated agents working together for comprehensive insights

## ASSUMPTIONS
   **Data Source**: Used csv file or tabuler data as a data source 
   
## Features

### Core Features
- **File-based Database**: JSON-based lightweight database system
- **Multi-format Support**: CSV, Excel, JSON, Parquet file processing
- **RAG System**: Vector embeddings and similarity search with reranking
- **Agent System**: Multi-agent data analysis workflow with LangGraph
- **Visualization**: Automatic chart generation and static file serving
- **Export Capabilities**: Multiple format support (CSV, JSON, Excel, Parquet)
- **Schema Analysis**: Automatic dataset schema detection and validation
- **Similar Dataset Discovery**: Find semantically similar datasets
- **Analysis Recommendations**: AI-powered technique recommendations

### Technical Stack
- **Backend**: FastAPI with Python 3.11+
- **Vector Database**: Pinecone for embeddings and similarity search
- **Agent Framework**: LangGraph for workflow orchestration
- **Database**: JSON-based file storage system
- **Data Processing**: Pandas, NumPy
- **Embeddings**: OpenAI text-embedding-ada-002
- **LLM**: OpenAI GPT models for agent reasoning

## Installation

### Prerequisites
- Python 3.11+
- Pinecone account and API key
- OpenAI API key
- UltraSafe API access

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/rahul-55-tech/ultsafe-assignment.git
   ```

2. **Create virtual environment**
   ```bash
   make setup
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   refer env.example for the environment setup
5. **Run the Application**
   ```bash
   # Using make file
   make run
   
   # Or directly with uvicorn
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Documentation

Once the application is running, you can access:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## Project Structure

```
ULTRASAFE_ASSIGNMENT/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── json_db.py         # JSON-based database implementation
│   ├── api/                    # API routes
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── datasets.py     # Dataset management endpoints
│   │       ├── analysis.py # Analysis and visualization endpoints
│   │       └── agents.py       # Agent workflow endpoints
│   ├── core/                   # Core functionality
│   │   ├── agents/             # Agent system implementation
│   │   │   ├── __init__.py
│   │   │   ├── agents.py       # Individual agent implementations
│   │   │   └── workflow_manager.py  # LangGraph workflow orchestration
│   │   └── rag/                # RAG implementation
│   │       ├── __init__.py
│   │       ├── vector_store.py      # Pinecone vector store
│   │       ├── embedding_service.py # OpenAI embeddings
│   │       ├── retrieval_service.py # Similar dataset retrieval
│   │       ├── statistical_analysis_knowledge_base.py  # Knowledge base script
│   │       └── reranker_service.py  # Result reranking
│   ├── services/               # Business logic
│   │   └── data_processor.py   # Data processing service
│   ├── schemas/                # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset schemas
│   │   ├── analysis.py         # Analysis schemas
│   │   └── agent.py            # Agent schemas
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset models
│   │   ├── analysis.py         # Analysis models
│   │   └── agent_workflow.py   # Agent workflow models
│   ├── middleware/             # Middleware components
│   │   └── auth.py             # Authentication middleware
│   ├── data/                   # Static data
│   │   └── statistical_methods.json  # Analysis techniques knowledge base
│   └── utils/                  # Utility functions
│   │   └── common_utility.py  # the utility file that used commonly

├── configurations/             # Configuration management
│   ├── config.py               # Application settings
│   ├── logger_config.py        # Logging configuration
├── db/                         # JSON database files
│   ├── datasets.json           # Dataset storage
│   └── dataset_schemas.json    # Schema storage
├── uploads/                    # File upload directory
├── exports/                    # Export directory
├── static/                     # Static files
│   └── charts/                 # Generated charts
├── requirements.txt            # Python dependencies
├── Makefile                    # Project management commands
└── README.md                   # Project documentation
```

## Authentication

All API endpoints require Bearer token authentication. The token should be included in the Authorization header:

```python
headers = {"Authorization": "Bearer your-secret-key-here"}
```

The secret key is configured in your `.env` file as `SECRET_KEY`. For development, you can use any string as the secret key.



## Key Components

### JSON Database System
- **File-based Storage**: Lightweight JSON files for data persistence
- **Thread-safe Operations**: Lock-based concurrency control
- **Auto-incrementing IDs**: Automatic ID generation for new records
- **CRUD Operations**: Complete create, read, update, delete functionality

### RAG System
- **Vector Embeddings**: OpenAI text-embedding-ada-002 for semantic search
- **Pinecone Integration**: Scalable vector database for similarity search
- **Reranking**: Multi-stage ranking for improved result quality
- **Knowledge Base**: Statistical methods database for technique recommendations

### Multi-Agent System
- **LangGraph Orchestration**: Structured workflow management
- **Specialist Agents**: 
  - **DataExplorationAgent**: Dataset profiling and overview
  - **StatisticalAnalysisAgent**: Descriptive statistics and correlations
  - **VisualizationAgent**: Automatic chart generation
  - **InsightGenerationAgent**: Business insights and recommendations
- **State Management**: Persistent workflow state tracking
- **Error Handling**: Robust error recovery and reporting

### Data Processing
- **Multi-format Support**: CSV, Excel, JSON, Parquet
- **Schema Analysis**: Automatic column type detection and statistics
- **Data Validation**: Comprehensive data quality checks
- **Export Capabilities**: Multiple output formats
- **Background Processing**: Asynchronous file processing

## API Endpoints

### Datasets
- `POST /api/v1/datasets/upload` - Upload and process dataset
- `GET /api/v1/datasets/` - List all datasets
- `GET /api/v1/datasets/{id}` - Get dataset details
- `GET /api/v1/datasets/{id}/schema` - Get dataset schema
- `GET /api/v1/datasets/{id}/similar` - Find similar datasets
- `GET /api/v1/datasets/{id}/export` - export the dataset

### Analysis
- `POST /api/v1/analysis/` - Create analysis
- `GET /api/v1/analysis/` - List analysis
- `GET /api/v1/analysis/{id}` - Get analysis details
- `POST /api/v1/analysis/{dataset_id}/explore` - Data exploration
- `POST /api/v1/analysis/{dataset_id}/statistics` - Statistical analysis
- `POST /api/v1/analysis/{dataset_id}/visualize` - Generate visualizations
- `POST /api/v1/analysis/{dataset_id}/insights` - Generate insights
- `POST /api/v1/analysis/{dataset_id}/recommend_techniques` - Get technique recommendations

### Agent Workflows
- `POST /api/v1/agents/workflows` - Create agent workflow
- `GET /api/v1/agents/workflows` - List workflows
- `GET /api/v1/agents/workflows/{id}` - Get workflow details
- `GET /api/v1/agents/workflows/{id}/tasks` - Get workflow tasks
- `GET /api/v1/agents/workflows/{id}/status` - Get workflow status

## Security & Performance

### Security Features
- **Bearer Token Authentication**: All API endpoints require valid Bearer token
- **Input Validation**: Comprehensive Pydantic schema validation
- **File Type Validation**: Strict file type checking
- **Path Sanitization**: Prevention of directory traversal attacks

### Performance Optimizations
- **Background Processing**: Asynchronous file processing
- **Vector Search**: Optimized similarity search with reranking




## Implementation Summary


**Data Processing & RAG System:**
- FastAPI application with comprehensive API endpoints
- JSON-based database system with thread-safe operations
- Multi-format data ingestion (CSV, Excel, JSON, Parquet)
- Automatic schema analysis and validation
- RAG integration with Pinecone vector database
- Similar dataset discovery with reranking
- Analysis technique recommendations
- Background task processing

**Multi-Agent Analysis System:**
- LangGraph workflow orchestration
- Data Exploration Agent for comprehensive profiling
- Statistical Analysis Agent for descriptive statistics
- Visualization Agent for automatic chart generation
- Insight Generation Agent for business insights
- Customizable workflow configurations
- Workflow status tracking and monitoring

**Technical Deatils:**
- Modular, scalable architecture
- Clean separation of concerns
- Async/await patterns throughout
- Type hints and documentation
- Input validation with Pydantic
- Secure file handling
- Performance optimizations


## Future Scope

### **Enhanced Data Processing**
- **Real-time Data Streaming**: Support for Kafka/RabbitMQ integration for real-time data ingestion


### **Enhanced RAG System**
- **Multi-modal RAG**: Support for images, audio, and video embeddings

- **Knowledge Graph Integration**: Graph-based knowledge representation


### **Security & Compliance**
- **Role-based Access Control (RBAC)**: Granular permissions and user roles
- **Data Encryption**: End-to-end encryption for sensitive data
- **Audit Logging**: Comprehensive audit trails for compliance

### **Scalability & Performance**
- **Microservices Architecture**: Service decomposition for better scalability




