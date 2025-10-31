  1. scripts/run_ingest.sh - Download articles (requires internet)
  # Download 120 articles
  source .venv/bin/activate && \
  RUN_ID=wiki_teacher SNAPSHOT=20231101.en MAX_RECORDS=120 TRACE=0 \
  bash scripts/run_ingest.sh

  2. scripts/run_process.sh - Process articles (runs offline)
  # Process with Ollama (requires Ollama running locally)
  source .venv/bin/activate && \
  RUN_ID=wiki-v6-500 PIPELINE_MAX_RECORDS=10 TRACE=0 \
  bash scripts/run_process.sh