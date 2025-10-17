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
2. **Dedup & Cluster** - Removes duplicates and groups related stories
3. **Bulletizer** - Converts articles into structured bullet points using RAG
4. **Segment Composer** - Merges bullets into cohesive story segments
5. **Scriptwriter** - Creates natural dialogue between two hosts
6. **Polisher** - Fact-checks and refines the script
7. **TTS Renderer** - Generates voice audio for both hosts
8. **Assembler** - Mixes audio with proper timing and loudness
9. **Captions** - Creates WebVTT captions synchronized with audio
10. **Publisher** - Packages everything into final episode format

## 🎭 Host Personalities

- **Host A** (Male, UK voice): Analytical, focuses on technical details and facts
- **Host B** (Female, UK voice): Contextual, explores implications and broader impact

## 🚀 Development Status

**Current Phase**: Step 3 - Deduplication & Clustering

### ✅ Completed Phases:

**1) Environment & Modes (DONE)**
- Dev-server mode: `docker compose up --build` → `curl -X POST localhost:8000/run`
- Run-once mode: Switch to `entrypoint.sh` for Cloud Run parity
- Environment toggles: `USE_FAKE_TTS=1`, voice settings, file paths

**2) Ingestion & Normalization (DONE)**
- Collector Agent implemented and tested
- Fetches recent articles from 15+ RSS feeds
- Outputs: `data/outputs/YYYY-MM-DD/raw_items.json` with count metadata
- Caches: `data/cache/YYYY-MM-DD/*.json` for article bodies

### 🔄 Current Phase:

**3) Deduplication & Clustering**
- Remove duplicate articles
- Group similar articles into clusters
- Prepare for RAG bulletizer stage

## 📁 Project Structure

```
darip/
├── data/                    # Pipeline inputs/outputs (gitignored)
│   ├── feeds/              # RSS source configurations
│   ├── cache/              # Temporary processing artifacts  
│   └── outputs/            # Generated episodes by date
├── orchestrator/           # Core AI pipeline (Python)
│   ├── agents/            # Individual processing agents
│   ├── rag/               # RAG and embeddings utilities
│   └── config/            # Environment and logging config
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
