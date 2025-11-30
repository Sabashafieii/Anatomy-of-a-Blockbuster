from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.analysis import MovieAnalysis
from src.visualizer import Visualizer

def main():
    df = DataLoader.load_movies("data/movies_metadata.csv")
    df = DataCleaner.clean(df)

    print("Correlation matrix (budget/revenue/profit/roi):")
    print(MovieAnalysis.correlation(df))

    # Core visuals
    Visualizer.plot_blockbuster_pie(df)
    Visualizer.plot_budget_vs_revenue(df)
    Visualizer.plot_budget_distribution(df)
    Visualizer.plot_revenue_distribution(df)
    Visualizer.plot_roi_distribution(df)
    Visualizer.plot_roi_distribution_log(df)
    Visualizer.plot_correlation_heatmap(df)
    Visualizer.plot_yearly_profit(df)
    Visualizer.plot_genre_blockbuster_rate(df)
    Visualizer.plot_budget_buckets(df)
    Visualizer.plot_runtime_vs_rating(df)
    Visualizer.plot_movie_count_by_year(df)

if __name__ == "__main__":
    main()
