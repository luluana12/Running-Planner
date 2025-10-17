# PowerShell script for Windows users
Write-Host "Setting up Game Review Sentiment Dashboard..." -ForegroundColor Green

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Run the dashboard
Write-Host "Running Game Review Sentiment Dashboard..." -ForegroundColor Yellow
python src/build_dashboard.py --input data/game_reviews.csv --outdir outputs --min-reviews 5

Write-Host ""
Write-Host "Done! See ./outputs for charts, CSVs, and PPTX." -ForegroundColor Green
Write-Host "Check out Game_Review_Sentiment_Summary.pptx for the complete dashboard!" -ForegroundColor Cyan
