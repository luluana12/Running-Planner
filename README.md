# Game Review Sentiment Dashboard 🎮📊

A production-ready Python dashboard that analyzes game review sentiment, generates visualizations, and creates a PowerPoint summary - all in one command!

## What This Project Does

This dashboard takes game review data (from a CSV file or generates synthetic data), computes sentiment scores using both numeric ratings and text analysis, creates three key visualizations, exports summary data, and generates a professional PowerPoint presentation with key insights. Perfect for demonstrating data engineering and storytelling skills.

## Quick Start (One Command!)

### For Windows Users:
```powershell
.\run.ps1
```

### For Linux/Mac Users:
```bash
chmod +x run.sh
./run.sh
```

### Manual Setup:
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python src/build_dashboard.py --input data/game_reviews.csv --outdir outputs --min-reviews 5
```

## Input Data Format

The dashboard expects a CSV file at `data/game_reviews.csv` with these columns (case-insensitive):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `title` | string | Game title | "The Elder Scrolls V: Skyrim" |
| `genre` | string | Game genre | "RPG" |
| `publisher` | string | Game publisher | "Bethesda" |
| `rating` | numeric | User rating (1-5, 0-10, or 0-100) | 4.5 |
| `review_text` | string | Written review | "Amazing game with great graphics!" |

**Note:** If no CSV file exists, the dashboard automatically generates synthetic data for demonstration.

## Outputs Generated

After running, check the `outputs/` folder for:

### 📊 Charts (PNG files)
- **`avg_score_by_genre.png`** - Bar chart showing average sentiment by game genre
- **`sentiment_pie.png`** - Pie chart showing distribution of positive/neutral/negative reviews
- **`top_publishers_avg_sentiment.png`** - Bar chart of top publishers by average sentiment

### 📋 Data Exports (CSV files)
- **`avg_score_by_genre.csv`** - Average sentiment score for each genre
- **`sentiment_distribution.csv`** - Count of reviews by sentiment category
- **`top_publishers_avg_sentiment.csv`** - Publisher performance metrics (minimum 5 reviews)
- **`game_reviews_cleaned_with_sentiment.csv`** - Original data with computed sentiment scores

### 📄 Presentation
- **`Game_Review_Sentiment_Summary.pptx`** - One-slide PowerPoint with key insights and embedded charts

## How Sentiment Analysis Works

1. **Rating Normalization**: Converts ratings to 0-100 scale:
   - 1-5 scale → rating/5 × 100
   - 1-10 scale → rating/10 × 100
   - Already 0-100 → used as-is

2. **Text Sentiment**: If no rating exists, analyzes review text using a built-in lexicon:
   - **Positive words**: amazing, awesome, fun, great, love, fantastic, etc.
   - **Negative words**: bad, boring, buggy, broken, hate, terrible, etc.
   - **Score**: 50 + (positive_count - negative_count) × 10, bounded 0-100

3. **Sentiment Categories**:
   - **Positive**: ≥67 points
   - **Neutral**: 34-66 points  
   - **Negative**: ≤33 points
   - **Unknown**: No rating or text available

## Project Structure

```
game-sentiment-dashboard/
├── data/
│   └── game_reviews.csv        # Input data (optional)
├── outputs/                    # Generated outputs
├── src/
│   └── build_dashboard.py      # Main dashboard script
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── run.sh                      # Linux/Mac run script
└── run.ps1                     # Windows run script
```

## Command Line Options

```bash
python src/build_dashboard.py [options]

Options:
  --input PATH          Input CSV file (default: data/game_reviews.csv)
  --outdir PATH         Output directory (default: outputs)
  --min-reviews N       Minimum reviews for publisher analysis (default: 5)
```

## How to Talk About This Project in Interviews

### 🎯 **Data Engineering Focus**
- "Built an end-to-end data pipeline that processes raw CSV data, applies sentiment analysis using both numeric ratings and text mining, and generates multiple output formats including visualizations and presentations."

### 📊 **Data Storytelling Focus**  
- "Created an automated dashboard that transforms game review data into actionable insights, with automatic key takeaway generation and professional PowerPoint output for stakeholder presentations."

### 🔧 **Technical Skills Demonstration**
- "Implemented a production-ready Python solution with proper error handling, synthetic data generation for testing, configurable parameters, and comprehensive documentation suitable for beginner developers."

## Requirements

- Python 3.11+
- pandas, numpy, matplotlib, python-pptx
- No external APIs or downloads required
- Works on Windows, Linux, and Mac

## Troubleshooting

**Virtual environment issues?** Make sure you're using Python 3.11+ and try recreating the environment:
```bash
rm -rf .venv
python -m venv .venv
```

**Permission errors on Windows?** Run PowerShell as Administrator or use:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Missing dependencies?** Ensure all packages are installed:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

**Ready to analyze some game reviews? Run the dashboard and see the magic happen! 🎉**
