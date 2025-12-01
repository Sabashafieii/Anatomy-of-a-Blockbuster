import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


class Visualizer:
    """
    A collection of polished visuals to explain what makes a movie a blockbuster.
    """

    palette = {
        "primary": "#1f78b4",
        "accent": "#e31a1c",
        "muted": "#6b7280",
        "bg": "#f5f7fa",
        "heatmap": "rocket",
    }

    @staticmethod
    def _abbr_money(x: float) -> str:
        abs_x = abs(x)
        if abs_x >= 1_000_000_000:
            return f"${x/1_000_000_000:.1f}B"
        if abs_x >= 1_000_000:
            return f"${x/1_000_000:.1f}M"
        if abs_x >= 1_000:
            return f"${x/1_000:.1f}K"
        return f"${x:.0f}"

    @staticmethod
    def _apply_money_axis(ax, axis: str = "x"):
        fmt = FuncFormatter(lambda x, _: Visualizer._abbr_money(x))
        if axis == "x":
            ax.xaxis.set_major_formatter(fmt)
        else:
            ax.yaxis.set_major_formatter(fmt)

    @staticmethod
    def _maybe_show(save_path: str | None = None, show: bool = True, return_fig: bool = False):
        fig = plt.gcf()
        # Apply dark theme facecolors for Streamlit dark mode
        fig.patch.set_facecolor("#050608")
        for ax in fig.axes:
            ax.set_facecolor("#0f1118")
            ax.tick_params(colors="#e6e9f0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#e6e9f0")
            if ax.title:
                ax.title.set_color("#f5f7fb")
            ax.xaxis.label.set_color("#e6e9f0")
            ax.yaxis.label.set_color("#e6e9f0")
            # Adjust legend text color if present
            leg = ax.get_legend()
            if leg:
                for text in leg.get_texts():
                    text.set_color("#e6e9f0")
                leg.get_title().set_color("#e6e9f0")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        if return_fig:
            return fig

    # ------------------------------------------------------------------ #
    # Core visuals
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_blockbuster_pie(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        counts = df["is_blockbuster"].value_counts().reindex([False, True]).fillna(0)
        labels = ["Not Blockbuster", "Blockbuster"]
        colors = [Visualizer.palette["muted"], Visualizer.palette["accent"]]

        plt.figure(figsize=(7, 7))
        _, texts, autotexts = plt.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            explode=[0, 0.08],
            shadow=False,
            startangle=120,
            textprops={"fontsize": 12, "weight": "bold"},
        )
        for t in autotexts:
            t.set_color("white")
            t.set_weight("bold")
        plt.title("Blockbuster vs Non-Blockbuster", fontsize=18, weight="bold", pad=18)
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_budget_vs_revenue(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        df = df.copy()
        df = df[(df["budget"] > 0) & (df["revenue"] > 0) & df["profit"].notna()]
        if df.empty:
            print("No valid rows for budget vs revenue plot.")
            return

        df["log_budget"] = np.log10(df["budget"])
        df["log_revenue"] = np.log10(df["revenue"])

        # Profit bins and sizes (smaller dots for less clutter)
        q1, q2 = df["profit"].quantile([0.33, 0.66])
        def profit_bucket(p):
            if p <= q1:
                return "Low profit"
            elif p <= q2:
                return "Mid profit"
            return "High profit"
        df["profit_bucket"] = df["profit"].apply(profit_bucket)
        size_map = {"Low profit": 12, "Mid profit": 26, "High profit": 44}

        plt.figure(figsize=(13, 7))
        plt.subplots_adjust(right=0.72, top=0.8)
        scatter = sns.scatterplot(
            data=df,
            x="log_budget",
            y="log_revenue",
            hue="is_blockbuster",
            size="profit_bucket",
            sizes=size_map,
            palette={True: Visualizer.palette["accent"], False: Visualizer.palette["primary"]},
            alpha=0.45,
            edgecolor="none",
        )

        plt.xlabel("log10(Budget)")
        plt.ylabel("log10(Revenue)")
        plt.title("Budget vs Revenue (size = profit, color = blockbuster)", fontsize=16, weight="bold")
        plt.grid(alpha=0.25)

        # Custom size legend
        from matplotlib.lines import Line2D
        size_handles = [
            Line2D([0], [0], marker="o", color="w", label="Low profit", markerfacecolor="#9ca3af", markersize=np.sqrt(size_map["Low profit"]/np.pi)),
            Line2D([0], [0], marker="o", color="w", label="Mid profit", markerfacecolor="#9ca3af", markersize=np.sqrt(size_map["Mid profit"]/np.pi)),
            Line2D([0], [0], marker="o", color="w", label="High profit", markerfacecolor="#9ca3af", markersize=np.sqrt(size_map["High profit"]/np.pi)),
        ]

        handles, labels = scatter.get_legend_handles_labels()
        label_map = {"False": "Non-blockbuster", "True": "Blockbuster"}
        color_handles = [(h, label_map.get(l, l)) for h, l in zip(handles, labels) if l in ["False", "True"]]

        ax = plt.gca()
        # Clear any auto legend
        leg = ax.get_legend()
        if leg:
            leg.remove()

        fig = plt.gcf()
        from matplotlib.patches import Patch
        color_handles = [
            Patch(facecolor=Visualizer.palette["primary"], edgecolor="none", label="Non-blockbuster", alpha=0.7),
            Patch(facecolor=Visualizer.palette["accent"], edgecolor="none", label="Blockbuster", alpha=0.7),
        ]
        color_legend = fig.legend(
            color_handles,
            [h.get_label() for h in color_handles],
            title="Blockbuster",
            loc="upper left",
            bbox_to_anchor=(0.72, 0.78),
            frameon=True,
            fancybox=True,
            framealpha=0.8,
            facecolor="#111827",
            edgecolor="#4b5563",
        )
        for text in color_legend.get_texts():
            text.set_color("#ffffff")
        if color_legend.get_title():
            color_legend.get_title().set_color("#ffffff")
        fig.add_artist(color_legend)

        size_legend = fig.legend(
            size_handles,
            [h.get_label() for h in size_handles],
            title="Profit bucket",
            loc="upper left",
            bbox_to_anchor=(0.72, 0.60),
            frameon=True,
            fancybox=True,
            framealpha=0.85,
            facecolor="#111827",
            edgecolor="#4b5563",
        )
        for text in size_legend.get_texts():
            text.set_color("#ffffff")
        if size_legend.get_title():
            size_legend.get_title().set_color("#ffffff")
        fig.add_artist(size_legend)
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_budget_revenue_distribution(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        def _plot_series(series, ax, title, color):
            series = series.dropna()
            series = series[series > 0]
            if series.empty:
                ax.set_visible(False)
                return
            quantiles = series.quantile([0.01, 0.995])
            low, high = quantiles.iloc[0], quantiles.iloc[1]
            if pd.isna(low) or pd.isna(high) or low <= 0 or high <= 0 or low >= high:
                low, high = series.min(), series.max()
            series = series[(series >= low) & (series <= high)]
            if series.empty:
                ax.set_visible(False)
                return
            if low > 0 and high > 0 and low < high:
                bins = np.logspace(np.log10(low), np.log10(high), 50)
                log_scale = True
            else:
                bins = "auto"
                log_scale = False
            try:
                sns.histplot(series, bins=bins, color=color, log_scale=log_scale, ax=ax)
            except Exception:
                sns.histplot(series, bins="auto", color=color, log_scale=False, ax=ax)
            Visualizer._apply_money_axis(ax, "x")
            ax.set_xlabel(f"{title} (USD)")
            ax.set_ylabel("Count")
            ax.set_title(title, fontsize=15, weight="bold")

        _plot_series(df.get("budget", pd.Series(dtype=float)), axes[0], "Budget distribution", Visualizer.palette["primary"])
        _plot_series(df.get("revenue", pd.Series(dtype=float)), axes[1], "Revenue distribution", Visualizer.palette["accent"])
        plt.tight_layout()
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_roi_distribution(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        df = df.copy()
        df = df[np.isfinite(df["roi"])]
        if df.empty:
            print("No ROI values to plot.")
            return

        # Focus on central mass; show tails count in the title.
        q_low, q_high = df["roi"].quantile([0.01, 0.99])
        if pd.isna(q_low) or pd.isna(q_high) or q_low >= q_high:
            q_low, q_high = df["roi"].min(), df["roi"].max()
        df_clip = df[(df["roi"] >= q_low) & (df["roi"] <= q_high)]
        tail_low = (df["roi"] < q_low).sum()
        tail_high = (df["roi"] > q_high).sum()

        plt.figure(figsize=(9, 5))
        sns.histplot(df_clip["roi"], bins=50, color=Visualizer.palette["primary"], kde=True)
        plt.axvline(0, linestyle="--", color=Visualizer.palette["muted"], alpha=0.7)
        plt.xlim(q_low, q_high)
        plt.xlabel("ROI (profit / budget)")
        plt.ylabel("Count")
        plt.title(
            f"Return on Investment Distribution (tails: low={tail_low}, high={tail_high})",
            fontsize=16,
            weight="bold",
        )
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_roi_distribution_log(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        df = df.copy()
        df = df[df["roi"] > 0]
        df["roi_log10"] = np.log10(df["roi"])

        plt.figure(figsize=(9, 5))
        sns.histplot(df["roi_log10"], bins=50, color=Visualizer.palette["accent"], kde=True)
        plt.xlabel("ROI (log10 scale)")
        plt.ylabel("Count")
        plt.title("ROI (Log Scale) Highlights Outliers", fontsize=16, weight="bold")
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_correlation_heatmap(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            print("Not enough numeric columns for correlation heatmap.")
            return
        # Drop date-related columns to focus on meaningful signals
        drop_cols = [c for c in ["year", "release_year", "release_month"] if c in numeric_df.columns]
        numeric_df = numeric_df.drop(columns=drop_cols, errors="ignore")
        if numeric_df.shape[1] < 2:
            print("Not enough numeric columns for correlation heatmap after dropping date fields.")
            return

        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=Visualizer.palette["heatmap"],
            linewidths=0.4,
            cbar_kws={"shrink": 0.85, "label": "Correlation"},
            annot_kws={"size": 9},
            square=True,
        )
        plt.title("Correlation Heatmap of Numeric Features", fontsize=18, weight="bold")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_yearly_profit(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        # Accept either full dataframe or a precomputed Series.
        if isinstance(df, pd.DataFrame):
            yearly = (
                df[["year", "profit"]]
                .dropna(subset=["year", "profit"])
                .groupby("year")["profit"]
                .median()
            )
        else:
            yearly = pd.Series(df).dropna()

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=yearly.index, y=yearly.values, color=Visualizer.palette["primary"])
        plt.fill_between(yearly.index, yearly.values, color=Visualizer.palette["primary"], alpha=0.15)
        plt.xlabel("Release Year")
        plt.ylabel("Median Profit (USD)")
        plt.title("Median Profit Over Time", fontsize=16, weight="bold")
        plt.grid(alpha=0.25)
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_genre_blockbuster_rate(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        genre_rate = (
            df.groupby("main_genre")["is_blockbuster"]
            .mean()
            .mul(100)
            .sort_values()
        )
        plt.figure(figsize=(10, 8))
        bars = plt.barh(
            genre_rate.index,
            genre_rate.values,
            color=Visualizer.palette["primary"],
            alpha=0.9,
        )
        for bar, pct in zip(bars, genre_rate.values):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center")
        plt.xlabel("% of Blockbusters")
        plt.title("Which Genres Over-index for Blockbusters?", fontsize=16, weight="bold")
        plt.grid(axis="x", alpha=0.25)
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_budget_buckets(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        df = df.copy()
        df["budget_bucket"] = pd.qcut(df["budget"], q=5, duplicates="drop")
        bucket_stats = df.groupby("budget_bucket")["is_blockbuster"].mean().mul(100)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=bucket_stats.index.astype(str),
            y=bucket_stats.values,
            palette=sns.color_palette("Blues", len(bucket_stats)),
        )
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Blockbuster Rate (%)")
        plt.xlabel("Budget Quintile")
        plt.title("Blockbuster Rate by Budget Quintile", fontsize=16, weight="bold")
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_runtime_vs_rating(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=df,
            x="runtime",
            y="vote_average",
            hue="is_blockbuster",
            palette={True: Visualizer.palette["accent"], False: Visualizer.palette["primary"]},
            alpha=0.35,
        )
        sns.regplot(
            data=df,
            x="runtime",
            y="vote_average",
            scatter=False,
            color=Visualizer.palette["muted"],
        )
        plt.xlabel("Runtime (minutes)")
        plt.ylabel("Average Rating")
        plt.title("Runtime vs Audience Rating", fontsize=16, weight="bold")
        plt.legend(title="Blockbuster")
        plt.grid(alpha=0.25)
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_movie_count_by_year(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        yearly_counts = df.groupby("year").size().dropna()
        years = sorted(yearly_counts.index.tolist())
        counts = yearly_counts.loc[years].values

        plt.figure(figsize=(12, 5))
        ax = sns.barplot(
            x=list(range(len(years))),
            y=counts,
            color=Visualizer.palette["primary"],
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Movies Released")
        ax.set_title("Release Volume Over Time", fontsize=16, weight="bold")

        # Show a subset of ticks to increase spacing between labels.
        if len(years) > 20:
            step = max(1, len(years) // 20)  # slightly denser than before
        else:
            step = 1
        tick_positions = list(range(0, len(years), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([years[i] for i in tick_positions], rotation=80, ha="right")

        plt.grid(axis="y", alpha=0.25)
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_genre_barplot(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        genre_counts = df["main_genre"].value_counts()
        percentages = (genre_counts / genre_counts.sum()) * 100

        genres = genre_counts.index
        counts = genre_counts.values

        plt.figure(figsize=(12, 9))
        bars = plt.barh(
            genres,
            counts,
            color=Visualizer.palette["primary"],
            edgecolor="black",
            alpha=0.9,
        )

        for bar, pct in zip(bars, percentages):
            plt.text(
                bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())} films | {pct:.1f}%",
                va="center",
                fontsize=11,
                color="#333",
            )

        plt.title("Film Count & Share by Genre", fontsize=18, weight="bold", pad=15)
        plt.xlabel("Count")
        plt.ylabel("Genre")
        plt.grid(axis="x", linestyle="--", alpha=0.35)
        plt.tight_layout()
        return Visualizer._maybe_show(save_path, show, return_fig)

    # Backward-compatibility alias for older notebook calls.
    plot_correlation_heatmapp = plot_correlation_heatmap

    @staticmethod
    def plot_genre_blockbuster_dashboard(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        """
        Compact dashboard: blockbuster rate, average profit, and volume by genre.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

        genre_rate = df.groupby("main_genre")["is_blockbuster"].mean().mul(100).sort_values(ascending=False)
        sns.barplot(
            x=genre_rate.values,
            y=genre_rate.index,
            ax=axes[0],
            palette=sns.color_palette("Blues_r", len(genre_rate)),
        )
        axes[0].set_title("Blockbuster Rate (%) by Genre", fontsize=14, weight="bold")
        axes[0].set_xlabel("% Blockbusters")
        axes[0].set_ylabel("Genre")

        profit_by_genre = df.groupby("main_genre")["profit"].median().loc[genre_rate.index]
        sns.barplot(
            x=profit_by_genre.values,
            y=profit_by_genre.index,
            ax=axes[1],
            palette=sns.color_palette("Greens_r", len(profit_by_genre)),
        )
        axes[1].set_title("Median Profit by Genre", fontsize=14, weight="bold")
        axes[1].set_xlabel("Median Profit")
        axes[1].set_ylabel("")

        count_by_genre = df["main_genre"].value_counts().loc[genre_rate.index]
        sns.barplot(
            x=count_by_genre.values,
            y=count_by_genre.index,
            ax=axes[2],
            palette=sns.color_palette("Oranges_r", len(count_by_genre)),
        )
        axes[2].set_title("Films Produced per Genre", fontsize=14, weight="bold")
        axes[2].set_xlabel("Count")
        axes[2].set_ylabel("")

        plt.tight_layout()
        return Visualizer._maybe_show(save_path, show, return_fig)

    @staticmethod
    def plot_profit_vs_revenue(
        df: pd.DataFrame,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
    ):
        df = df[(df["profit"] > 0) & (df["revenue"] > 0)]
        plt.figure(figsize=(9, 7))
        sns.scatterplot(
            data=df,
            x="revenue",
            y="profit",
            hue="is_blockbuster",
            palette={True: Visualizer.palette["accent"], False: Visualizer.palette["primary"]},
            alpha=0.35,
            edgecolor="none",
        )
        # Reference line profit = revenue (100% margin)
        max_rev = df["revenue"].max()
        ref = np.linspace(df["revenue"].min(), max_rev, 50)
        plt.plot(ref, ref, linestyle="--", color=Visualizer.palette["muted"], label="profit = revenue")

        plt.xscale("log")
        plt.yscale("log")
        ax = plt.gca()
        Visualizer._apply_money_axis(ax, "x")
        Visualizer._apply_money_axis(ax, "y")
        plt.xlabel("Revenue (USD, log scale)")
        plt.ylabel("Profit (USD, log scale)")
        plt.title("Profit vs Revenue (USD, log-log)", fontsize=16, weight="bold")
        plt.legend(title="Blockbuster")
        plt.grid(alpha=0.25)
        return Visualizer._maybe_show(save_path, show, return_fig)

    # ------------------------------------------------------------------ #
    # Markdown helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def markdown_descriptions() -> dict[str, str]:
        """
        Markdown snippets that explain each visualization for reporting.
        """
        return {
            "blockbuster_pie": (
                "### Blockbuster split\n"
                "Shows the share of movies flagged as blockbusters. A heavy skew toward the red slice "
                "indicates strong revenue/profit outperformance is rare."
            ),
            "budget_vs_revenue": (
                "### Budget vs. Revenue (log10)\n"
                "Each point is a film. X-axis: log10(budget), Y-axis: log10(revenue). Color shows blockbuster status; "
                "point size encodes profit bucket (low/mid/high). This highlights whether big budgets reliably yield big box office "
                "and which films achieve outsized profits."
            ),
            "roi_distribution": (
                "### ROI distribution\n"
                "Histogram of return on investment. A long right tail implies a few films massively outperform. "
                "Use alongside the log-scale view to spot outliers."
            ),
            "roi_distribution_log": (
                "### ROI (Log Scale)\n"
                "Most movies make only small returns, but a tiny number earn **massive** ROI. The log scale makes these rare breakout hits "
                "visible—showing how the industry relies on just a few exceptional performers."
            ),
            "correlation_heatmap": (
                "### Correlation heatmap\n"
                "Pairwise correlations among numeric features. Look for strong positive ties (e.g., budget–revenue) "
                "and weak relationships (e.g., runtime–profit)."
            ),
            "yearly_profit": (
                "### Median profit over time\n"
                "Tracks typical profitability by release year. Dips or peaks often align with market shocks or "
                "industry shifts."
            ),
            "genre_blockbuster_rate": (
                "### Blockbuster rate by genre\n"
                "Percentage of films in each genre that become blockbusters. Highlights genres that over-index for hits."
            ),
            "budget_buckets": (
                "### Blockbuster Rate by Budget Quintile\n"
                "Higher-budget films are much more likely to become blockbusters. Low-budget movies have a very small hit rate, "
                "while the top budget tier shows a clear jump in blockbuster frequency."
            ),
            "runtime_vs_rating": (
                "### Runtime vs audience rating\n"
                "Checks whether longer movies earn higher ratings, and whether blockbusters differ in reception."
            ),
            "movie_count_by_year": (
                "### Release volume over time\n"
                "This chart shows how many films were released each year in the dataset, with a slow build-up in the early decades "
                "and a sharp surge from the 1990s onward, peaking in the early 2010s. The recent drop is likely due to incomplete "
                "data rather than a real decline in production. This trend highlights how competition has intensified over time: "
                "modern films have to stand out in a much more crowded market to become hits."
            ),
            "genre_barplot": (
                "### Genre mix\n"
                "Shows the production volume share of each genre, giving context for where studios invest."
            ),
            "genre_blockbuster_dashboard": (
                "### Genre dashboard\n"
                "Blockbuster hit rate, median profit, and production volume by genre in one view."
            ),
            "budget_distribution": (
                "### Budget & revenue distribution\n"
                "Side-by-side histograms with log-aware binning to show production spend and box office spread.\n"
                "Both distributions are extremely skewed: most films cluster in the low-to-mid budget and revenue ranges, while a small group "
                "of titles occupy the very high end. This imbalance shows that the industry is “hits-driven”: a few big productions absorb "
                "most of the spending, and a few top-earning films generate a disproportionate share of total box-office revenue. It also "
                "highlights how risky the market is—many films operate on limited budgets and earn modest amounts, while only a tiny fraction "
                "break into the high-revenue tier."
            ),
            "profit_vs_revenue": (
                "### Profit vs revenue (log-log)\n"
                "Shows how profits scale with revenue; the dashed line is 100% margin. Points above it indicate exceptional profitability."
            ),
        }
