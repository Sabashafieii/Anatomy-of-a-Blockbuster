import pandas as pd

class MovieAnalysis:

    @staticmethod
    def correlation(df: pd.DataFrame) -> pd.DataFrame:
        corr = df[["budget", "revenue", "profit", "roi"]].corr()
        return corr

    @staticmethod
    def yearly_profit(df):
        return df.groupby("year")["profit"].mean()

    @staticmethod
    def blockbuster_stats(df):
        """
        Returns summary of blockbuster vs non-blockbuster.
        """
        summary = df["is_blockbuster"].value_counts(normalize=True) * 100
        return summary

    @staticmethod
    def blockbuster_by_genre(df):
        """
        % of blockbuster films in each genre.
        """
        genre_stats = df.groupby("main_genre")["is_blockbuster"].mean() * 100
        return genre_stats.sort_values(ascending=False)
