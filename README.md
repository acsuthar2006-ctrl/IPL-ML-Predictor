# IPL Match Prediction & Win Probability 🏏

This repository contains a complete Data Science and Machine Learning pipeline for the Indian Premier League (IPL). It features data cleaning scripts, exploratory data analysis, and two distinct Machine Learning models built using `scikit-learn`.

## 📁 Project Structure

```
IPL-ML/
│
├── data/
│   ├── raw/                 # Original, untouched Kaggle datasets
│   ├── interim/             # Cleaned datasets ready for modeling
│
├── ipl-ds-cleaning.ipynb    # Data cleaning and standardization pipeline
├── ipl-model-training.ipynb # Pre-Match Winner Prediction Model
├── ipl-inplay-model.ipynb   # In-Play (Ball-by-Ball) Win Probability Model
└── README.md                # Project documentation
```

---

## 🧹 1. Data Cleaning Pipeline (`ipl-ds-cleaning.ipynb`)
Before training any models, the raw data underwent several preprocessing steps:
* **Handling Missing Data:** Cleaned up missing `city` values by inferring them from the `venue`. Handled rain-washed matches by labeling the winner as `"No Result"`.
* **Standardization:** Mapped long team names (e.g., "Royal Challengers Bangalore") to standard abbreviations (e.g., "RCB") to ensure consistency across the entire dataset.
* **Date Parsing:** Converted string dates into Pandas datetime objects for future time-series analysis.

---

## 🤖 2. The Machine Learning Models

This project features two different models that answer the question *"Who will win?"* at two completely different stages of a match.

### Model A: Pre-Match Predictor (`ipl-model-training.ipynb`)
**Goal:** Predict the winner of the match *before the first ball is bowled.*

* **Algorithm:** Logistic Regression
* **Features Used:** 
  * `team1` & `team2`
  * `toss_winner`
  * `toss_decision` (bat or field)
  * `venue`
* **How it works:** It uses One-Hot Encoding to convert categorical data into numerical features. It learns historical patterns (e.g., "Does CSK usually win when they choose to bat first at Chepauk?").
* **Accuracy:** **~54%**
* *Note: Since there are 13 possible winners, a random guess is ~7%. Achieving 54% using only basic pre-match coin-toss data is a very strong baseline!*

### Model B: In-Play Win Probability (`ipl-inplay-model.ipynb`)
**Goal:** Predict the probability of the chasing team winning the match, updated in real-time *after every single ball is bowled* (just like the predictors shown on TV broadcasts).

* **Algorithm:** Logistic Regression (chosen for its well-calibrated `predict_proba` outputs).
* **Feature Engineering:** This model requires complex Pandas manipulation. We extract the 1st innings total score to set a `target`, and then calculate the following features for every ball in the 2nd innings:
  * `runs_left` (Target - current score)
  * `balls_left` (120 - balls bowled)
  * `wickets_left` (10 - wickets fallen)
  * `crr` (Current Run Rate)
  * `rrr` (Required Run Rate)
* **How it works:** By feeding the model ~72,000 individual ball deliveries, it learns the exact mathematical scenarios that lead to a win or loss. 
* **Accuracy:** **~82%** (Significantly higher because the model has access to the *current state* of the match).

---

## 🚀 How to Run

1. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas scikit-learn jupyter
   ```
2. Open the Jupyter notebooks sequentially:
   * Run `ipl-ds-cleaning.ipynb` to generate the clean data in `data/interim/`.
   * Run `ipl-model-training.ipynb` to train the pre-match model.
   * Run `ipl-inplay-model.ipynb` to train the live predictor and test out your own custom match scenarios at the bottom of the file!
