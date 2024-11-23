# Log Analyzer with GPU Acceleration

A Python-based log analysis tool that leverages GPU acceleration for processing large log files. It uses embeddings and LLM models to provide meaningful analysis of log content.

## Features

- GPU-accelerated log processing
- Checkpoint-based progress saving and resumption
- Graceful interruption handling
- Batch processing capabilities
- Automatic log analysis using LLM
- CUDA support for NVIDIA GPUs

## Requirements

- Python 3.x
- CUDA-capable GPU
- Required packages:
  - ollama
  - chromadb
  - torch
  - psutil
  - tqdm

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python log_analyzer.py
```

The script will:
1. Detect and utilize available GPU
2. Process log files in batches
3. Save progress regularly
4. Generate analysis upon completion or interruption

## License

MIT License
