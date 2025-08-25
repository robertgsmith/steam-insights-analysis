import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import json
import re
import sys
import csv
from transformers import pipeline
from tqdm import tqdm
import time as t

# ────── Shared variables ────────────────────────────────────────────────────
start = None
# Clean languages: remove HTML, normalize list, detect full-audio ones ---
TAG_RE = re.compile(r"<.*?>")

# ────── Statistical functions ────────────────────────────────────────────────────
def stat_histogram(df,column):
    df[column].hist(bins='auto', edgecolor="black")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title("Distribution of X")
    plt.show()

def stat_simple_plot(df, col):
    # Sort values descending
    sorted_vals = df[col].sort_values(ascending=False).reset_index(drop=True)
    # Bar plot
    sorted_vals.plot(kind="bar", figsize=(10, 4))
    plt.ylabel(col)
    plt.title("Column X sorted (descending)")
    plt.show()

# ────── Processing Functions ────────────────────────────────────────────────────
def parse_price(j):
    if pd.isna(j):
        return {}
    try:
        obj = json.loads(j)
        return {"price_final_cents": obj.get("final"),"price_initial_cents": obj.get("initial"),
            "currency": obj.get("currency"),"discount_percent": obj.get("discount_percent"),
            "price_final": (obj.get("final") or 0) / 100.0,"price_initial": (obj.get("initial") or 0) / 100.0,
            "price_final_formatted": obj.get("final_formatted"),}
    except Exception:
        # Fallback: return empty fields if a row is malformed
        return {"price_final_cents": None,"price_initial_cents": None,"currency": None,"discount_percent": None,
            "price_final": None,"price_initial": None,"price_final_formatted": None,}

def clean_languages(raw):
    if pd.isna(raw):
        return [], []
    # Remove <br> and other tags, keep the "*" that indicates full audio support
    txt = TAG_RE.sub("", str(raw))
    # Now split on commas and normalize
    langs = [p.strip() for p in txt.split(",") if p.strip()]
    # Items may look like "English*" where * = full audio support
    full_audio = [l.replace("*", "").strip() for l in langs if l.endswith("*")]
    normal = [l.replace("*", "").strip() for l in langs]
    return normal, full_audio
def time_helper(text:str):
    print(f'── {text}   Time used: {int((t.time()-start)//60)}m  {round((t.time()-start)%60,2)}s')


# ────── Normalisation Functions ────────────────────────────────────────────────────
def add_minmax(df, col, new_col=None):
    """
    Adds a min–max normalized version of a column to the DataFrame.
    Parameters:
    df      : pandas DataFrame
    col     : str, column name to normalize
    new_col : str, optional, new column name (default: col + "_norm")
    """
    if new_col is None:
        new_col = col + "_linear_norm"

    vals = df[col]
    norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
    df[new_col] = norm_vals
    return df

def add_log_minmax(df, col, new_col=None):
    """
    Adds a log + min-max normalized version of a column to the DataFrame.
    Parameters:
    df      : pandas DataFrame
    col     : str, column name to normalize
    new_col : str, optional, new column name (default: col + "_norm")
    """
    if new_col is None:
        new_col = col + "_log_norm"
    # Apply log transform (add 1 to avoid log(0))
    log_vals = np.log1p(df[col])
    # Min-max normalization
    norm_vals = (log_vals - log_vals.min()) / (log_vals.max() - log_vals.min())
    # Assign new column
    df[new_col] = norm_vals
    return df


# ────── KPI-related Functions ────────────────────────────────────────────────────
def kpi_owner_acquisition_rate(df):
    # 1) Keep only needed columns
    df = df[['type', 'publisher_technical', 'date of release', 'average_owner']].copy()
    # 2) Filter only games
    df = df[df['type'].astype(str).str.strip().str.lower() == 'game']
    # 3) Convert columns
    df['average_owner'] = pd.to_numeric(df['average_owner'], errors='coerce')
    df['date of release'] = pd.to_datetime(df['date of release'], errors='coerce')
    # Remove bad rows
    df = df[df['average_owner'].notna() & df['date of release'].notna()].copy()
    # 4) Calculate days on market until 2024-10-31
    end_date = pd.to_datetime("2024-10-31")
    df['days_on_market'] = (end_date - df['date of release']).dt.days
    # Remove future releases or invalid (negative days)
    df = df[df['days_on_market'] > 0]
    # 5) Compute growth potential = owners / days on market
    df['growth_potential'] = df['average_owner'] / df['days_on_market']

    # 6) Group by publisher
    publisher_df = (df.groupby('publisher_technical').agg(growth_potential_mean=('growth_potential', 'mean'),
              n_titles=('growth_potential', 'size')).reset_index())

    # ✅ 7) Keep only publishers with at least 20 games
    publisher_df = publisher_df[publisher_df['n_titles'] >= 20]

    # rename the n_titles for cobsistency for future merger
    #publisher_df = publisher_df.rename(columns = {'n_titles': 'app_id'})

    # Sort after filtering
    publisher_df = publisher_df.sort_values('growth_potential_mean', ascending=False)
    # normalising the values
    publisher_df = add_log_minmax(publisher_df,'growth_potential_mean')
    publisher_df = add_minmax(publisher_df,'growth_potential_mean')

    return publisher_df

def kpi_positive_review_share(final_data):
    print(len(final_data['publisher_technical'].unique()))
    df = final_data.loc[final_data['type'] == 'game', ['app_id', 'publisher', '% positive reviews','publisher_technical']]
    # group by and take only the one publisher per publisher_technical that has the highest count of app_ids
    df = df.groupby('publisher_technical', as_index=False).agg({
        '% positive reviews': 'mean',
        'app_id': 'count',
        'publisher': lambda x: x.value_counts().idxmax()})
    # df.to_excel('df.xlsx')
    # take only the publishers with 20+ games
    df = df[df['app_id'] >= 20]
    return df

def kpi_monitisation_efficiency(df):
    df = df[["type", "publisher_technical", "average_owner","app_id","Price(eur)"]]
    # 3) Filter games and clean data
    df = df[df["type"].astype(str).str.lower().eq("game")]
    df = df.dropna(subset=["publisher_technical", "average_owner", "Price(eur)"])
    df = df[(df["average_owner"] > 0) & (df["Price(eur)"] > 0)]

    # 4) Compute KPI per game
    df["monetizing_efficiency"] = df["average_owner"] * df["Price(eur)"]

    # 5) Aggregate KPI per publisher (SUM) and sort
    top = df.groupby('publisher_technical', as_index=False).agg({'monetizing_efficiency': 'mean', 'app_id': 'count'})
    top = top[top['app_id'] >= 20]

    # Normalisation
    top = add_minmax(top, 'monetizing_efficiency')
    top = add_log_minmax(top, 'monetizing_efficiency')

    # stat_simple_plot(top, 'monetizing_efficiency_linear_norm')
    # stat_simple_plot(top, 'monetizing_efficiency_log_norm')

    return top

def kpi_engagement_ratio(df):
    eng_col = '% Engagement Score'

    # Clean Data
    df_clean = df[df["type"].astype(str).str.lower() != "demo"].copy()
    #df_clean = df_clean[df_clean[eng_col] != 0].reset_index(drop=True)
    publisher_stats = df_clean.groupby('publisher_technical', as_index=False).agg(
        {eng_col: 'mean', 'app_id': 'count'}).reset_index()

    # Filter to only publishers with >= 20 unique games (adjust as needed)
    publisher_stats = publisher_stats[publisher_stats["app_id"] >= 20]
    publisher_stats = add_minmax(publisher_stats,'% Engagement Score')
    return publisher_stats

def kpi_three_other_quality_kpis(df):
    """
    Review Score KPI + Semantic Sentiment Score KPI (of text reviews) + Recommendation Rate KPI
    Code generated by ChatGPT
    """
    # Keep only rows where type == 'game' and store in variable 'df_games'
    df_games = df[df['type'] == 'game'].copy()
    # print(df_games.head())

    # Fix column data types
    numeric_cols = ["review_score","semantic review score","num of recommendations","average_owner"]

    for col in numeric_cols:
        df_games[col] = pd.to_numeric(df_games[col], errors="coerce")

    # Creating KPI: review score index (average review score per publisher converted ranging from 0 to 10)
    review_score_index = df_games.groupby("publisher_technical")["review_score"].mean().reset_index()
    review_score_index.rename(columns={"review_score": "review_score_index"}, inplace=True)

    # Creating KPI: semantic score index (average semantic score based on comments per publisher ranging from -1 to 1)
    semantic_index = df_games.groupby("publisher_technical")["semantic review score"].mean().reset_index()
    semantic_index.rename(columns={"semantic review score": "semantic_sentiment_index"}, inplace=True)

    # Creating KPI: Recommendation Rate (natural virality) (recommendations / max ownership bucket for each game, and then averaged per publisher)
    # Avoid division by zero
    df_games["recommendation_rate"] = df_games["num of recommendations"] / df_games["average_owner"].replace(0, pd.NA)

    recommendation_index = df_games.groupby("publisher_technical")["recommendation_rate"].mean().reset_index()
    recommendation_index.rename(columns={"recommendation_rate": "recommendation_index"}, inplace=True)

    # Create new table of data with KPIs grouped by publisher
    publisher_kpis = review_score_index.merge(semantic_index, on="publisher_technical", how="outer")
    publisher_kpis = publisher_kpis.merge(recommendation_index, on="publisher_technical", how="outer")

    # Copy to avoid changing original
    publisher_kpis_norm = publisher_kpis.copy()

    # add the game count
    publisher_game_counts = df_games.groupby("publisher_technical")["app_id"].count().reset_index()

    # Drop duplicates before merge, just to be safe
    publisher_game_counts = publisher_game_counts.drop_duplicates(subset="publisher_technical")

    # Merge cleanly into table of KPIs of publishers
    publisher_kpis_norm = publisher_kpis_norm.drop(columns=[col for col in publisher_kpis_norm.columns if "num_games" in col])
    publisher_kpis_norm = publisher_kpis_norm.merge(publisher_game_counts, on="publisher_technical", how="left")

    # Filter by publishers by number of games (only with more than 20 games)
    publisher_kpis_norm = publisher_kpis_norm[publisher_kpis_norm["app_id"] > 20].copy()

    #normalisation
    add_minmax(publisher_kpis_norm, 'review_score_index')
    add_log_minmax(publisher_kpis_norm, 'semantic_sentiment_index')
    add_log_minmax(publisher_kpis_norm, 'recommendation_index')

    # stat_simple_plot(publisher_kpis_norm,'review_score_index_linear_norm')
    # stat_simple_plot(publisher_kpis_norm, 'semantic_sentiment_index_log_norm')
    # stat_simple_plot(publisher_kpis_norm, 'recommendation_index_log_norm')
    return publisher_kpis_norm