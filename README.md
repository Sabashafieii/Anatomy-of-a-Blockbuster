# ⭐ Anatomy of a Blockbuster  
### *What Makes a Movie a Hit?*

This project explores the key factors behind movie success—budget, revenue, profit, ROI, genres, and audience signals—using the Movies Dataset from Kaggle.  
It includes a full Streamlit dashboard, a polished data-cleaning pipeline, and a suite of visualizations to analyze blockbuster patterns across 100+ years of cinema.

---

## 📦 Dataset

The Movies Dataset — Kaggle  
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

---

## 🧹 Data Cleaning Pipeline

- Removed duplicates and dropped irrelevant ID/URL fields  
- Converted numeric fields; treated zero budget/revenue as missing  
- Parsed release dates into year/month  
- Extracted main_genre from the JSON-like genres field  
- Filled missing budget/revenue/runtime via genre-level medians, then global medians  
- Filled remaining numeric fields (votes, ratings, popularity) with medians  
- Computed profit and ROI; winsorized ROI to reduce extreme outliers  
- Generated season & budget-band features  
- Built a hybrid blockbuster score:  
  - 40% revenue  
  - 30% ROI  
  - 20% popularity  
  - 10% vote_count  
- Assigned is_blockbuster using quantile thresholds  
  (85th revenue, 75th ROI, 75th popularity, or top 10% score)

---

## 📊 Visual Analytics

- Blockbuster vs non-blockbuster split  
- Budget vs revenue (log-log) with profit encoding  
- ROI distribution (linear & log)  
- Median profit over time  
- Release volume over time  
- Genre distribution & blockbuster rates  
- Budget quintiles vs blockbuster rate  
- Runtime vs audience rating  
- Profit vs revenue (log-log)  
- Budget & revenue distributions  
- Combined genre dashboard (hit rate + median profit + volume)

---

## 🌐 Streamlit Dashboard

Title: *What Makes a Movie a Hit?*

The dashboard allows users to:

- Explore interactive analytics  
- Filter by year, genre, season, and budget band  
- Study ROI, profit, and blockbuster likelihood  
- Compare patterns across thousands of films

- 

---

## 📁 Project Structure

/project
│── app.py
│── analysis.py
│── data_cleaner.py
│── data_loader.py
│── visualizer.py
│── data/
│── plots/
│── requirements.txt
│── README.md

## 🛠️ Tech Stack

- Python 3.10+  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit  

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
Run the Streamlit app

streamlit run app.py
 Add the dataset
Place movies_metadata.csv into:

data/movies_metadata.csv
## 🧪 Version Control

- [ ] .gitignore configured (cache files, large data, pycache excluded)
- [ ] Frequent, atomic commits with clear & descriptive messages
