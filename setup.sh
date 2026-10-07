#!/bin/bash
# Setup script for Intelligent Resume-to-Job Matching System
# Sets up virtual environment, installs lightweight CPU ML dependencies, downloads NLTK data,
# and builds precomputed index artifacts.

set -e

echo "=== 1. Virtual Environment ==="
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "=== 2. Installing Dependencies ==="
echo "Installing CPU PyTorch and scientific packages..."
pip install --default-timeout=120 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

echo "=== 3. Verifying / Downloading NLTK Corpora ==="
python -c "
import nltk
for res, path in [('punkt_tab', 'tokenizers/punkt_tab'), ('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet')]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f'Note: NLTK could not download {res}: {e}')
print('NLTK data verified!')
"

echo "=== 4. Environment Configuration ==="
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

echo "=== 5. Building Precomputed Offline Indices ==="
if [ ! -f "data/indices/jobs_index_meta.parquet" ]; then
    echo "Building offline indices (sample size: 500)..."
    python scripts/build_index.py --sample-size 500 --output-dir data/indices
else
    echo "Offline index artifacts already present in data/indices/."
fi

echo ""
echo "=========================================="
echo "Setup complete! To run the application:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo "=========================================="
