import streamlit as st

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.visualizer import Visualizer


@st.cache_data(show_spinner=False)
def load_and_clean(path: str):
    df = DataLoader.load_movies(path)
    df = DataCleaner.clean(df)
    return df


def render_fig(fig):
    st.pyplot(fig, clear_figure=False, use_container_width=True)


def sidebar_filters(df):
    st.sidebar.markdown('<p class="sidebar-title">Controls</p>', unsafe_allow_html=True)

    # Year selection: presets or custom range
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    year_mode = st.sidebar.radio("Year filter", ["All years", "Custom range"], horizontal=False)
    if year_mode == "Custom range" and years:
        year_min, year_max = min(years), max(years)
        year_range = st.sidebar.slider("Release window", min_value=year_min, max_value=year_max, value=(year_min, year_max))
    else:
        year_range = (None, None)

    # Genre selection: mode switch
    genres = sorted(df["main_genre"].dropna().unique().tolist())
    genre_mode = st.sidebar.radio("Genre filter", ["All genres", "Select genres"], horizontal=False)
    if genre_mode == "Select genres" and genres:
        selected_genres = st.sidebar.multiselect("Choose genres", options=genres, default=genres[:5])
    else:
        selected_genres = genres

    return year_range, selected_genres


def apply_filters(df, year_range, selected_genres):
    filtered = df.copy()
    if year_range != (None, None):
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
    if selected_genres:
        filtered = filtered[filtered["main_genre"].isin(selected_genres)]
    return filtered


def _abbr(num: float) -> str:
    """Abbreviate large numbers for display (e.g., 14.8M)."""
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    if abs_num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if abs_num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:.0f}"


def main():
    st.set_page_config(
        page_title="Blockbuster Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark theme & layout polish
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        :root, html, body, [class*="css"]  {
            background: #050608 !important;
            color: #e6e9f0 !important;
            font-family: 'Space Grotesk', sans-serif;
            margin: 0;
            padding: 0;
        }
        [data-testid="stAppViewContainer"], .appview-container, .main, .block-container {
            background: #050608 !important;
            color: #e6e9f0 !important;
        }
        [data-testid="stHeader"] {
            background: #050608 !important;
            color: #e6e9f0 !important;
        }
        .block-container {padding-top: 1.2rem; max-width: 1300px; background: #050608;}
        h1, h2, h3, h4, h5 {color: #f5f7fb;}
        p, li, span, label {color: #e6e9f0;}
        .markdown-text-container {color: #e6e9f0;}
        .kpi-title {font-size: 0.9rem; color: #9ba4b5;}
        .stMetric > label {color: #9ba4b5;}
        [data-testid="stMetricValue"] {color: #f5f7fb !important;}
        .stTabs [data-baseweb="tab-list"] {gap: 1rem;}
        .stTabs [data-baseweb="tab"] {background: #0f1118; padding: 0.65rem 1.2rem; border-radius: 999px; color: #9ba4b5;}
        .stTabs [aria-selected="true"] {background: #1a1f2a; color: #f5f7fb;}
        /* Sidebar */
        section[data-testid="stSidebar"] {background: #080a0f !important;}
        .sidebar-title {font-weight: 700; color: #f5f7fb; font-size: 1.05rem;}
        .sidebar-sub {color: #9ba4b5;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("What Makes a Movie a Hit?")
    st.caption("Insights from 45,000+ films on budgets, profits, and audience patterns.")

    # Sidebar controls (no path input)
    st.sidebar.markdown('<p class="sidebar-title">Filters</p>', unsafe_allow_html=True)
    df = load_and_clean("data/movies_metadata.csv")

    # Interactive filters
    year_range, selected_genres = sidebar_filters(df)
    df = apply_filters(df, year_range, selected_genres)

    if df.empty:
        st.warning("No data after filters. Relax filters to see insights.")
        return

    # KPI strip
    st.markdown("### Performance Snapshot")
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Movies", f"{len(df):,}")
    kpi_cols[1].metric("Blockbusters", f"{int(df['is_blockbuster'].sum()):,}")
    kpi_cols[2].metric("Median ROI", f"{df['roi'].median():.2f}x")
    kpi_cols[3].metric("Median Profit", f"${_abbr(df['profit'].median())}")
    kpi_cols[4].metric("Median Budget", f"${_abbr(df['budget'].median())}")

    md = Visualizer.markdown_descriptions()

    tab_overview, tab_finance, tab_genre, tab_roi, tab_corr, tab_audience = st.tabs(
        ["Overview", "Financials", "Genres", "ROI & Risk", "Correlation", "Audience"]
    )

    with tab_overview:
        st.subheader("Hit Mix")
        st.markdown(md["blockbuster_pie"])
        render_fig(Visualizer.plot_blockbuster_pie(df, show=False, return_fig=True))

        # Release Momentum
        st.markdown(md["movie_count_by_year"])
        render_fig(Visualizer.plot_movie_count_by_year(df, show=False, return_fig=True))

        st.subheader("Budget → Box Office")
        st.markdown(md["budget_vs_revenue"])
        render_fig(Visualizer.plot_budget_vs_revenue(df, show=False, return_fig=True))

    with tab_finance:
        st.subheader("Profit Trajectory")
        st.markdown(md["yearly_profit"])
        render_fig(Visualizer.plot_yearly_profit(df, show=False, return_fig=True))

        # Budget & Revenue Footprint
        st.markdown(md["budget_distribution"])
        render_fig(Visualizer.plot_budget_revenue_distribution(df, show=False, return_fig=True))

    with tab_genre:
        st.subheader("Genre P&L Dashboard")
        st.markdown(md["genre_blockbuster_dashboard"])
        render_fig(Visualizer.plot_genre_blockbuster_dashboard(df, show=False, return_fig=True))

        # Genre Mix
        st.markdown(md["genre_barplot"])
        render_fig(Visualizer.plot_genre_barplot(df, show=False, return_fig=True))

        # Hit Rate by Genre
        st.markdown(md["genre_blockbuster_rate"])
        render_fig(Visualizer.plot_genre_blockbuster_rate(df, show=False, return_fig=True))

    with tab_roi:
        roi_min, roi_max = float(df["roi"].min()), float(df["roi"].max())
        st.markdown("#### ROI filter")
        roi_low, roi_high = st.slider(
            "ROI range",
            min_value=roi_min,
            max_value=roi_max,
            value=(roi_min, roi_max),
            step=max((roi_max - roi_min) / 100, 0.01),
        )
        df_roi = df[(df["roi"] >= roi_low) & (df["roi"] <= roi_high)]

        st.subheader("ROI Spread")
        st.markdown(md["roi_distribution_log"])
        render_fig(Visualizer.plot_roi_distribution_log(df_roi, show=False, return_fig=True))

    with tab_corr:
        st.subheader("Feature Correlations")
        st.markdown(md["correlation_heatmap"])
        fig = Visualizer.plot_correlation_heatmap(df, show=False, return_fig=True)
        if fig:
            render_fig(fig)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

    with tab_audience:
        # Runtime vs Audience Rating (title removed per request)
        st.markdown(md["runtime_vs_rating"])
        render_fig(Visualizer.plot_runtime_vs_rating(df, show=False, return_fig=True))


if __name__ == "__main__":
    main()
