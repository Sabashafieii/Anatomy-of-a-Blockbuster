import pandas as pd

# Rough ISO country code mapping; adjust based on dataset if codes differ.
COUNTRY_NAME_MAP = {
    "US": "United States",
    "GB": "United Kingdom",
    "FR": "France",
    "DE": "Germany",
    "CA": "Canada",
    "AU": "Australia",
    "IN": "India",
    "CN": "China",
    "JP": "Japan",
    "KR": "South Korea",
    "ES": "Spain",
    "IT": "Italy",
    "BR": "Brazil",
    "MX": "Mexico",
}

def _bbox_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    return [
        [min_lon, min_lat],
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
    ]


def blockbuster_rate_by_country(df: pd.DataFrame, country_col: str = "production_countries") -> pd.DataFrame:
    if country_col not in df.columns:
        return pd.DataFrame(columns=["iso", "name", "blockbuster_rate", "lon", "lat", "top_movies", "top_movies_data", "polygon"])

    def extract_first_code(val):
        if pd.isna(val):
            return None
        try:
            if isinstance(val, str):
                import ast

                parsed = ast.literal_eval(val)
            else:
                parsed = val
            if isinstance(parsed, list) and parsed:
                # TMDB style: [{"iso_3166_1": "US", "name": "United States of America"}]
                code = parsed[0].get("iso_3166_1") if isinstance(parsed[0], dict) else parsed[0]
                return code
            return None
        except Exception:
            return None

    df_local = df.copy()
    df_local["iso"] = df_local[country_col].apply(extract_first_code)
    df_local["title_for_top"] = (
        df_local["original_title"]
        if "original_title" in df_local.columns
        else df_local.get("title", "Unknown")
    )
    df_local["revenue_for_top"] = df_local.get("revenue", 0)

    if "is_blockbuster" not in df_local.columns:
        df_local["is_blockbuster"] = 0

    rates = df_local.dropna(subset=["iso"]).groupby("iso")["is_blockbuster"].mean().reset_index()
    rates["blockbuster_rate"] = (rates["is_blockbuster"] * 100).round(2)
    rates["name"] = rates["iso"].map(COUNTRY_NAME_MAP).fillna(rates["iso"])
    # Top 5 blockbusters per country by revenue
    top_titles = (
        df_local.dropna(subset=["iso"])
        .sort_values("revenue_for_top", ascending=False)
        .groupby("iso")
        .apply(
            lambda g: g.loc[g["is_blockbuster"] == 1, ["title_for_top", "revenue_for_top", "is_blockbuster"]]
            .head(5)
            .to_dict(orient="records")
        )
    )
    mapped = rates["iso"].map(top_titles)
    rates["top_movies_data"] = mapped.apply(lambda v: v if isinstance(v, list) else [])
    rates["top_movies"] = rates["top_movies_data"].apply(
        lambda lst: ", ".join([d.get("title_for_top", "N/A") for d in lst]) if lst else "N/A"
    )
    rates["top_movies_tooltip"] = rates["top_movies_data"].apply(
        lambda lst: "\n".join(
            [
                f"- {d.get('title_for_top','N/A')} (rev=${d.get('revenue_for_top',0):,.0f})"
                for d in lst
            ]
        )
        if lst
        else "N/A"
    )

    # Approximate centroid coords for common markets (lon, lat)
    # Rough bounding boxes per country for polygon coloring (min_lon, min_lat, max_lon, max_lat)
    bbox = {
        "US": (-125, 24, -66, 49),
        "GB": (-8, 49, 2, 59),
        "FR": (-5.5, 41, 8.5, 51.5),
        "DE": (5.5, 47, 15.5, 55),
        "CA": (-141, 42, -52, 83),
        "AU": (112, -44, 154, -10),
        "IN": (68, 6, 97, 35),
        "CN": (73, 18, 135, 54),
        "JP": (128, 30, 146, 46),
        "KR": (124, 33, 131, 39),
        "ES": (-10, 36, 4, 44),
        "IT": (6, 36, 19, 47),
        "BR": (-75, -34, -34, 5),
        "MX": (-118, 14, -86, 33),
    }
    rates["bbox"] = rates["iso"].map(bbox)
    rates = rates.dropna(subset=["bbox"])
    rates[["min_lon", "min_lat", "max_lon", "max_lat"]] = pd.DataFrame(rates["bbox"].tolist(), index=rates.index)
    rates["lon"] = (rates["min_lon"] + rates["max_lon"]) / 2
    rates["lat"] = (rates["min_lat"] + rates["max_lat"]) / 2
    rates["polygon"] = rates.apply(lambda r: _bbox_polygon(r["min_lon"], r["min_lat"], r["max_lon"], r["max_lat"]), axis=1)
    return rates[["iso", "name", "blockbuster_rate", "lon", "lat", "polygon", "top_movies", "top_movies_data", "top_movies_tooltip"]]
