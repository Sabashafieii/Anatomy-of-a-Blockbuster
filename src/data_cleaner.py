import ast
from typing import Any, Optional

import numpy as np
import pandas as pd


class DataCleaner:
    """
    Cleans and enriches the movies dataset, including robust blockbuster tagging.
    """

    @staticmethod
    @staticmethod
    def _scale_0_1(series: pd.Series) -> pd.Series:
        s = series.copy()
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            return pd.Series([0] * len(series), index=series.index)
        min_v, max_v = s.min(), s.max()
        if max_v == min_v:
            return pd.Series([0.0] * len(series), index=series.index)
        return (s - min_v) / (max_v - min_v)

    @staticmethod
    def _parse_genres(raw: Any) -> str:
        """
        Parse the genres column (stringified JSON, list of dicts, or dict)
        and return the primary genre name.
        """
        if pd.isna(raw):
            return "Unknown"

        parsed: Optional[Any] = raw
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = None

        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict) and "name" in first:
                return first["name"]

        if isinstance(parsed, dict) and "name" in parsed:
            return parsed["name"]

        return "Unknown"

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates()

        # Drop columns that are identifiers/urls and not useful for analysis
        drop_cols = [col for col in ["imdb_id", "homepage", "belongs_to_collection", "collection_name", "is_franchise"] if col in df.columns]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

        # Convert numeric fields; keep coercion safe for messy raw data.
        numeric_cols = ["budget", "revenue", "runtime", "vote_average", "vote_count", "popularity"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Treat zeros in budget/revenue as missing (illogical).
        for col in ("budget", "revenue"):
            if col in df.columns:
                df[col].replace(0, np.nan, inplace=True)
        if "runtime" in df.columns:
            df["runtime"] = df["runtime"].where(df["runtime"] > 0)

        # Dates
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
        df["release_month"] = df["release_date"].dt.month
        # Backward-compat: keep 'year' column for consumers expecting it
        df["year"] = df["release_year"]

        # Genre extraction
        df["main_genre"] = df["genres"].apply(DataCleaner._parse_genres)
        df["main_genre"].fillna("Unknown", inplace=True)

        # Genre-level median fill then global median for selected cols
        for col in ["budget", "revenue", "runtime"]:
            if col in df.columns:
                genre_median = df.groupby("main_genre")[col].transform("median")
                df[col] = df[col].fillna(genre_median)
                df[col] = df[col].fillna(df[col].median())

        # Impute remaining non-critical numeric values (runtime, votes, ratings, popularity)
        for col in numeric_cols:
            if col in df.columns and col not in ("budget", "revenue"):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # Financials computed only where both budget & revenue are positive
        mask_valid = (df["budget"] > 0) & (df["revenue"] > 0)
        df["profit"] = np.where(mask_valid, df["revenue"] - df["budget"], np.nan)
        df["roi"] = np.where(mask_valid, df["profit"] / df["budget"], np.nan)
        df["roi"].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Winsorize ROI to limit extreme leverage (only on valid ROI)
        valid_roi = df.loc[df["roi"].notna(), "roi"]
        if not valid_roi.empty:
            upper_roi = valid_roi.quantile(0.995)
            lower_roi = valid_roi.quantile(0.005)
            df.loc[df["roi"].notna(), "roi"] = df.loc[df["roi"].notna(), "roi"].clip(lower=lower_roi, upper=upper_roi)

        # Fill any remaining numeric gaps post-calculation except ROI (keep NaN for invalid rows)
        numeric_after = df.select_dtypes(include=[np.number]).columns
        for col in numeric_after:
            if col == "roi":
                continue
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

        # Fill remaining categorical gaps to keep column lengths aligned
        categorical_cols = df.select_dtypes(exclude=["datetime64[ns]", np.number]).columns
        for col in categorical_cols:
            df[col].fillna("Unknown", inplace=True)

        # ======================
        #    BLOCKBUSTER LOGIC (stricter, industry-like)
        # ======================
        valid_mask = (df["budget"] > 0) & (df["revenue"] > 0)
        valid_df = df[valid_mask]

        # Scaled signals for hybrid scoring
        revenue_scaled = DataCleaner._scale_0_1(valid_df["revenue"]) if not valid_df.empty else pd.Series([], dtype=float)
        roi_scaled = DataCleaner._scale_0_1(valid_df["roi"]) if not valid_df.empty else pd.Series([], dtype=float)
        popularity_scaled = DataCleaner._scale_0_1(valid_df["popularity"]) if "popularity" in df.columns else pd.Series([], dtype=float)
        votes_scaled = DataCleaner._scale_0_1(np.log1p(valid_df["vote_count"])) if "vote_count" in df.columns else pd.Series([], dtype=float)

        df["blockbuster_chance"] = (
            0.40 * revenue_scaled.reindex(df.index).fillna(0)
            + 0.30 * roi_scaled.reindex(df.index).fillna(0)
            + 0.20 * popularity_scaled.reindex(df.index).fillna(0)
            + 0.10 * votes_scaled.reindex(df.index).fillna(0)
        )

        # Thresholds (quantile-based + ROI floor)
        rev_cut = valid_df["revenue"].quantile(0.85) if not valid_df.empty else np.inf
        roi_cut = valid_df["roi"].quantile(0.75) if not valid_df.empty else np.inf
        pop_cut = valid_df["popularity"].quantile(0.75) if ("popularity" in df.columns and not valid_df.empty) else np.inf
        score_cut = df["blockbuster_chance"].quantile(0.90) if not df.empty else np.inf

        df["is_blockbuster"] = (
            valid_mask
            & (
                ((df["revenue"] >= rev_cut) & (df["roi"] >= roi_cut) & (df["popularity"] >= pop_cut))
                | (df["blockbuster_chance"] >= score_cut)
            )
        ).astype(int)

        # Season and budget bands
        season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
        df["season"] = df["release_month"].map(season_map).fillna("Unknown") if "release_month" in df.columns else "Unknown"
        if "budget" in df.columns:
            df["budget_band"] = pd.cut(
                df["budget"],
                bins=[-1, 10_000_000, 40_000_000, 100_000_000, 250_000_000, np.inf],
                labels=["<10M", "10–40M", "40–100M", "100–250M", "250M+"],
            )

        print("Dataset cleaned and blockbuster flags created.")
        return df
