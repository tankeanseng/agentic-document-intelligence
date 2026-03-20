# Agentic Document Intelligence

Production-style multi-source RAG system for complex document intelligence across vector search, GraphRAG, and SQL. This project combines retrieval quality, answer reliability, runtime safety, product UX, and AWS deployment into a full-stack AI system rather than a notebook demo.

## Why This Project Stands Out

- Multi-source retrieval across Pinecone, Kuzu GraphRAG, and SQLite
- Agentic orchestration with query decomposition, routing, latency policy, and runtime quality gating
- Citation-grounded answering with self-reflection, corrective repair, and RAGAS-style evaluation
- Conversational follow-up handling with cross-turn query resolution and conversation-meta handling
- Real product experience with frontend UI, graph panel, evidence explorer, telemetry, downloadable demo corpus, and public AWS deployment

## Frontend UI

Add a screenshot at `docs/images/frontend-ui.png` before publishing and it will fit naturally here.

Suggested markdown once you add the image:

```md
![Frontend UI](docs/images/frontend-ui.png)
```

## System Overview

This system answers complex enterprise-style questions over a fixed Microsoft FY2025 10-K demo corpus and associated analyst datasets.

### Core AI Components

1. Query Understanding
   - LLM query decomposition
   - multi-query generation
   - step-back reasoning
   - HyDE-triggered corrective retrieval
   - conversational follow-up resolution

2. Retrieval
   - Pinecone vector retrieval over precomputed child chunks
   - parent-child citation restoration
   - Kuzu-based GraphRAG over pre-extracted entities and relationships
   - read-only text-to-SQL over SQLite
   - retrieval merge, deduplication, reranking, and MMR diversification

3. Orchestration
   - multi-source routing across vector, graph, and SQL
   - latency-optimized execution policy
   - safe parallel execution where appropriate
   - final cross-source evidence fusion

4. Answer Quality
   - grounded answer generation
   - self-reflective critique
   - corrective answer repair
   - runtime quality gating and bounded retries
   - RAGAS-style LLM judge metrics

### Non-AI / Product Components

- FastAPI backend
- Next.js frontend
- citation tooltip and evidence display
- graph visualization panel
- demo corpus downloader for PDF and CSV assets
- telemetry / progress updates
- AWS Lambda + API Gateway backend deployment
- S3 runtime asset hydration
- CloudFront frontend delivery

## Architecture

```text
User Query
  -> Conversational Resolution
  -> Query Decomposition
  -> Source Routing + Latency Policy
  -> Parallel / Sequential Retrieval
       -> Pinecone vector search
       -> Kuzu graph retrieval
       -> SQLite text-to-SQL
  -> Evidence Fusion + Context Compression
  -> Grounded Answer Generation
  -> Self-Reflective Critique
  -> Corrective Repair
  -> RAGAS-style LLM Judge
  -> Final Answer + Citations + Telemetry
```

## Demo Capabilities

Example question types supported:

- pure SQL:
  - `Which Microsoft segment had the highest revenue in FY2025?`
- pure vector:
  - `What did management say about AI demand in the FY2025 10-K summary?`
- graph + SQL:
  - `Which segment includes GitHub and what was its FY2025 revenue?`
- complex multi-part:
  - `Rank Microsoft's FY2025 segments by revenue, identify which one grew the fastest, and explain what the document says about the demand drivers behind that growth.`
- conversational follow-up:
  - `Based on my previous question, what was its operating income?`

## Public Deployment

- Frontend: CloudFront
- Backend API: AWS Lambda + API Gateway
- Runtime assets: S3
- Graph store: Kuzu DB hydrated from S3 into Lambda `/tmp`
- SQL store: SQLite hydrated from S3 into Lambda `/tmp`

Live frontend:

- `https://d2mwxp9ivx7w3g.cloudfront.net`

## Repository Structure

```text
agentic_document_intelligence/
  backend/         FastAPI app and Lambda runtime integration
  frontend/        Next.js UI
  scripts/         Retrieval, orchestration, evaluation, and utility modules
  tests/           Unit and integration tests
  corpus/          Demo PDF + CSV corpus metadata and source assets
  artifacts/       Generated experiment outputs required by the runtime demo
  docs/            Component notes and deployment documentation
  deployment/      AWS build and deployment helpers
  evals/           Benchmark case definitions
  template.yaml    AWS SAM deployment template
```

## Local Development

### 1. Configure environment

Create `.env` from `.env.example` and fill in your own values.

### 2. Backend

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### 3. Frontend

```powershell
cd frontend
npm install
npm run dev -- -H 127.0.0.1 -p 3000
```

### 4. Open the UI

- `http://127.0.0.1:3000`

## Testing

Representative local test command:

```powershell
python -m unittest
```

Representative deployment validation:

```powershell
sam validate -t template.yaml --lint
sam build -t template.yaml --use-container
```

## Security Notes

- Never commit `.env`
- Never publish live API keys or AWS credentials
- Lambda runtime state is stored in S3-backed job/session records for serverless compatibility
- SQL execution is explicitly read-only

## Publish Checklist

Before pushing to GitHub:

1. Make sure `.env` is not tracked
2. Rotate any credentials that were ever exposed locally or accidentally committed
3. Add a frontend screenshot at `docs/images/frontend-ui.png`
4. Review whether you want to keep all experiment artifacts public or trim them

## License

MIT
