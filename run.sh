#!/usr/bin/env bash
set -e

echo "Setting up Game Review Sentiment Dashboard..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate virtual environment (try different activation scripts for different OS)
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment (Linux/Mac)..."
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    echo "Activating virtual environment (Windows)..."
    source .venv/Scripts/activate
else
    echo "Could not find virtual environment activation script"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the dashboard
echo "Running Game Review Sentiment Dashboard..."
python src/build_dashboard.py --input data/game_reviews.csv --outdir outputs --min-reviews 5

echo ""
echo "Done! See ./outputs for charts, CSVs, and PPTX."
echo "Check out Game_Review_Sentiment_Summary.pptx for the complete dashboard!"
