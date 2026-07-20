# ASG Transformer Framework

<div align="center">

# 🛡️ ASG Transformer Framework

### AI-Powered Cyber Attack Scenario Generation using Transformer-Based Semantic Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

Enterprise-grade framework for semantic cyber attack scenario generation based on
Transformer models, ATT&CK knowledge graphs, and AI-driven reasoning.

</div>

---

# Overview

ASG Transformer is a modern AI framework designed to generate realistic cyber attack
scenarios using semantic understanding rather than traditional rule-based matching.

Unlike conventional systems based on TF-IDF, SVM, or keyword matching, ASG Transformer
leverages Transformer encoders to understand attacker objectives, techniques,
tactics, software, threat groups, and attack paths in a unified semantic space.

The framework is designed for:

- Cyber Threat Intelligence
- Attack Simulation
- Security Training
- Purple Team Exercises
- SOC Knowledge Assistance
- Research
- Enterprise Security Automation

---

# Features

## AI & Machine Learning

- Transformer-based semantic encoding
- Context-aware attack understanding
- Multi-label prediction
- Beam Search attack path generation
- Confidence scoring
- Embedding-based similarity search
- Domain-specific fine tuning
- GPU acceleration

---

## Cyber Knowledge

Supports:

- ATT&CK Techniques
- ATT&CK Tactics
- Threat Groups
- Malware / Software
- Attack Relationships
- Transition Graphs
- Knowledge Graph Expansion

---

## API

- FastAPI
- OpenAPI / Swagger
- Async API
- REST endpoints
- JSON responses
- Automatic validation

---

## Engineering

- Clean Architecture
- Dependency Injection
- Modular Design
- Configuration Management
- Docker Support
- Logging
- Unit Testing
- Production Ready

---

# Installation

## Requirements

- Python 3.11+
- pip
- Git

Clone the repository

```bash
git clone https://github.com/your-org/ASGTransformer.git

cd ASGTransformer
```

Create virtual environment

Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

Upgrade pip

```bash
pip install --upgrade pip
```

Install dependencies

```bash
pip install -e .
```

or

```bash
pip install -e ".[train]"
```

or

```bash
pip install -e ".[dev,train]"
```

---

# Configuration

Copy the configuration file

```bash
cp .env.example .env
```

Example

```env
ASG_DATA_DIR=data/processed

ASG_MODEL_DIR=models

MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

DEVICE=auto

API_PORT=8000
```

---

# Running the API

Start the API server

```bash
uvicorn asg_transformer.api.main:app \
    --host 0.0.0.0 \
    --port 8000
```

Swagger

```
http://localhost:8000/docs
```

Redoc

```
http://localhost:8000/redoc
```

---

# Training

Train the Transformer encoder

```bash
python -m asg_transformer.training.train_encoder \
    --base-model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 10 \
    --batch-size 16
```

Advanced training

```bash
python -m asg_transformer.training.train_encoder \
    --epochs 25 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.1 \
    --batch-size 32 \
    --fp16
```

The trained model will be stored in

```
models/
    asg-transformer/
```

---

# Evaluation

Evaluate the model

```bash
python -m asg_transformer.training.evaluate
```

Example output

```
Accuracy

Technique Prediction

95.2%

Tactic Prediction

98.1%

Threat Group

94.7%

Semantic Retrieval

97.3%
```

---

# Testing

Run all tests

```bash
pytest
```

Run with coverage

```bash
pytest --cov=asg_transformer
```

Run a single module

```bash
pytest tests/test_service.py
```

---

# Docker

Build

```bash
docker build -t asg-transformer .
```

Run

```bash
docker run \
-p 8000:8000 \
asg-transformer
```

---

# API Example

Request

```json
{
    "goal":"Compromise Active Directory",
    "environment":"Windows Enterprise",
    "objective":"Privilege Escalation"
}
```

Response

```json
{
    "techniques":[
        "...",
        "...",
        "..."
    ],
    "tactics":[
        "...",
        "..."
    ],
    "software":[
        "...",
        "..."
    ],
    "attack_path":[
        "...",
        "...",
        "..."
    ],
    "confidence":0.96
}
```

---

# Project Structure

```
ASGTransformer
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── models/
│   ├── checkpoints/
│   ├── transformer/
│   └── exported/
│
├── src/
│   └── asg_transformer/
│       │
│       ├── api/
│       │   ├── routes/
│       │   ├── schemas/
│       │   └── main.py
│       │
│       ├── application/
│       │   ├── services/
│       │   └── handlers/
│       │
│       ├── domain/
│       │   ├── entities/
│       │   ├── interfaces/
│       │   └── models/
│       │
│       ├── infrastructure/
│       │   ├── repository/
│       │   ├── embeddings/
│       │   └── persistence/
│       │
│       ├── training/
│       │
│       ├── inference/
│       │
│       ├── core/
│       │
│       ├── config.py
│       │
│       └── utils/
│
├── tests/
│
├── notebooks/
│
├── scripts/
│
├── Dockerfile
│
├── docker-compose.yml
│
├── pyproject.toml
│
├── README.md
│
└── .env.example
```

---

# Architecture

```
Client

      │

      ▼

 FastAPI API

      │

      ▼

Application Layer

      │

      ▼

Transformer Service

      │

      ▼

Sentence Transformer Encoder

      │

      ▼

Semantic Retrieval Engine

      │

      ▼

Knowledge Catalog

      │

      ▼

Scenario Generator

      │

      ▼

JSON Response
```

---

# Technologies

- Python
- FastAPI
- PyTorch
- HuggingFace Transformers
- Sentence Transformers
- Pydantic
- NumPy
- Scikit-Learn
- Docker
- Uvicorn
- Pytest

---

# Future Roadmap

- Knowledge Graph Integration
- Graph Neural Networks
- Multi-Agent Planning
- Reinforcement Learning
- ATT&CK Live Synchronization
- LLM-assisted Scenario Generation
- Distributed Inference
- ONNX Runtime
- TensorRT Optimization
- Kubernetes Deployment

---

# License

MIT License

---

# Citation

```bibtex
@software{asg_transformer,
  title={ASG Transformer Framework},
  author={ASG Research Group},
  year={2026},
  url={https://github.com/your-org/asg-transformer}
}
```