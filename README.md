# DARIP - Daily AI-Generated Recaps in Podcast

An automated podcast generation system that creates daily tech news episodes with AI-powered hosts.

## 🎯 Project Vision

DARIP transforms the overwhelming stream of daily tech news into digestible, engaging podcast episodes. Instead of manually consuming dozens of articles across different sources, DARIP does the heavy lifting:

- **Collects** tech news from curated RSS feeds
- **Processes** content through AI to eliminate duplicates and cluster related stories  
- **Transforms** articles into structured bullet points with proper citations
- **Generates** natural dialogue between two AI hosts with distinct personalities
- **Produces** a polished podcast episode with audio, captions, and metadata

## 🏗️ Architecture Overview

```
Tech News Sources → AI Processing Pipeline → Daily Podcast Episode
     ↓                        ↓                      ↓
  RSS Feeds            Multi-Agent System      MP3 + VTT + JSON
```

The system operates as a **multi-agent pipeline** where each agent specializes in a specific task:

1. **Collector** - Fetches and normalizes news from RSS feeds
2. **Full-Text Enricher** - Fetches complete article text for all collected items
3. **RAG Retriever** - Semantic search with fixed queries to find most relevant articles
4. **Bulletizer** - Converts articles into structured bullet points using RAG
5. **Segment Composer** - Merges bullets into cohesive story segments
6. **Scriptwriter** - Creates natural dialogue between two hosts
7. **Polisher** - Fact-checks and refines the script
8. **TTS Renderer** - Generates voice audio for both hosts
9. **Assembler** - Mixes audio with proper timing and loudness
10. **Captions** - Creates WebVTT captions synchronized with audio
11. **Publisher** - Packages everything into final episode format

## 🎭 Host Personalities

- **Host A** (Male, UK voice): Analytical, focuses on technical details and facts
- **Host B** (Female, UK voice): Contextual, explores implications and broader impact

## 🚀 Development Status

**Current Phase**: Step 4 - Summarization & Bulletization

### ✅ Completed Phases:

**Step 1: Environment & Modes (DONE)**
- Dev-server mode: `docker compose up --build` → `curl -X POST localhost:8000/run`
- Run-once mode: Switch to `entrypoint.sh` for Cloud Run parity
- Environment toggles: `USE_FAKE_TTS=1`, voice settings, file paths

**Step 2: Collection & Normalization (DONE)**
- Collector Agent implemented and tested
- Fetches recent articles from 20+ RSS feeds
- Outputs: `data/outputs/YYYY-MM-DD/raw_items.json` with count metadata (~400-500 items)
- Caches: `data/cache/YYYY-MM-DD/*.json` for article bodies

**Step 2.5: Full-Text Enrichment (DONE)** 🚀
- LangGraph workflow that fetches complete article text for all raw items using robots.txt compliance, rate limiting, and 4-library extraction fallback (trafilatura → readability → newspaper3k → goose3)
- Success rate: ~40-65%, Performance: ~15-20 min for 400-500 items
- 📖 **[Full documentation →](orchestrator/extractors/README.md)**

**Step 3: RAG Query Retrieval (DONE)** 🔍
- Replaces clustering with semantic search using 8 fixed queries and hybrid scoring (FAISS + BM25). Ensures diversity via MMR algorithm and per-query caps.
- Outputs: `queried_news.json` with 30 articles, Performance: ~30 sec, No API costs
- 📖 **[Full documentation →](orchestrator/rag/README.md)**

### 🔄 Current Phase:

**Step 4: Summarization & Bulletization**
- Convert article clusters into structured bullet points
- Use RAG (Retrieval-Augmented Generation) for accuracy
- Generate concise summaries per cluster
- Prepare content for dialogue scriptwriting

## 📁 Project Structure

```
darip/
├── data/                    # Pipeline inputs/outputs (gitignored)
│   ├── feeds/              # RSS source configurations
│   ├── cache/              # Temporary processing artifacts  
│   └── outputs/            # Generated episodes by date
├── orchestrator/           # Core AI pipeline (Python)
│   ├── agents/            # Pipeline agents (collector, enricher, retriever, etc.)
│   ├── extractors/        # Text extraction utilities (robots, HTML, text)
│   ├── rag/               # RAG utilities (chunker, embedder, indexer, scorer, MMR)
│   └── config/            # Environment settings and query configurations
├── site/                   # Static website for episodes
│   └── public/            # HTML, CSS, JS, and published content
└── scripts/               # Development and deployment helpers
```

## 🎯 Target Output

Each daily run produces:
- **Episode MP3** (~10 minutes, 128kbps)
- **WebVTT Captions** with speaker labels
- **Episode JSON** with metadata, chapters, and source citations
- **Website Updates** for today's episode and archive

## 🔧 Quick Start

1. **Setup Environment**:
   ```bash
   cp .env.example .env
   # Configure your API keys and settings
   ```

2. **Run Development Server**:
   ```bash
   docker compose up --build
   ```

3. **Trigger Episode Generation**:
   ```bash
   curl -X POST localhost:8000/run
   ```

## 🎙️ Future Roadmap

- **Phase 1**: Core pipeline implementation and testing
- **Phase 2**: Real TTS integration and audio quality optimization  
- **Phase 3**: Cloud deployment and automated daily scheduling
- **Phase 4**: Advanced features (music beds, dynamic segments, analytics)

---

*DARIP aims to make tech news consumption more efficient and engaging through AI-powered automation, creating a personalized daily briefing that adapts to your interests and time constraints.*
