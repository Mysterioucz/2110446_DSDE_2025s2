import json
from typing import Union

import pandas as pd

"""
Optimized helper versions of the student assignment functions.

Functions:
- Q1(): count rows after removing duplicates
- Q2(vdo_df): number of unique titles where dislikes > likes
- Q3(vdo_df): number of trending videos on 22 Jan 2018 with >10000 comments
- Q4(vdo_df): trending date with minimum average comments per video
- Q5(vdo_df): count of days where total daily views in Sports > Comedy

Note: paths are the same as the original assignment: '/data/videos.csv' and
 '/data/category_id.json'. Functions accepting `vdo_df` assume a pandas
DataFrame shaped like videos.csv; they defensively drop duplicate rows.
"""


def Q1(path: str = "/data/videos.csv") -> int:
    # Read the CSV from the provided path into a DataFrame. We avoid any
    # unnecessary transformations here because counting rows after deduplication
    # is an I/O-bound followed by a simple in-memory operation.
    df = pd.read_csv(path)

    # Drop duplicate rows across all columns to match the assignment's
    # requirement of counting unique rows. `drop_duplicates()` keeps the first
    # occurrence and removes subsequent identical rows.
    unique_count = df.drop_duplicates().shape[0]

    # Return as plain Python int for callers that may expect a native type.
    return int(unique_count)


def Q2(vdo_df: pd.DataFrame) -> int:
    # Defensively remove duplicate rows first so we count each video title only
    # once per the assignment instructions. This also prevents double-counting
    # when multiple identical rows for the same video appear.
    df = vdo_df.drop_duplicates()

    # Select rows where dislikes are strictly greater than likes. Then use
    # `.nunique()` on the `title` column to count distinct titles that meet
    # the criterion (ensures we count unique videos by title, not occurrences).
    mask = df["dislikes"] > df["likes"]
    unique_titles = df.loc[mask, "title"].nunique()

    return int(unique_titles)


def Q3(vdo_df: pd.DataFrame) -> int:
    # Remove duplicates to ensure each trending entry is counted once.
    df = vdo_df.drop_duplicates()

    # The dataset encodes dates as 'YY.DD.MM'. The assignment asks specifically
    # for entries trending on January 22, 2018 which is represented as
    # '18.22.01'. Build a boolean mask for that date and comment_count > 10000.
    mask = (df["trending_date"] == "18.22.01") & (df["comment_count"] > 10000)

    # `mask.sum()` counts True values; convert to int for consistency.
    return int(mask.sum())


def Q4(vdo_df: pd.DataFrame) -> Union[str, None]:
    # Drop duplicates so each video/trending entry is only counted once.
    df = vdo_df.drop_duplicates()

    # If the DataFrame is empty after dropping duplicates, there is no valid
    # trending date to return — return `None` to signal absence of data.
    if df.empty:
        return None

    # Group by `trending_date` and compute the mean number of comments for the
    # videos on each date. Using `.mean()` yields a float representing the
    # average comment_count per video on that date.
    means = df.groupby("trending_date")["comment_count"].mean()

    # `idxmin()` returns the index (trending_date) corresponding to the
    # smallest mean value. This directly answers the question: which trending
    # date has the minimum average number of comments per video.
    return means.idxmin()


def Q5(vdo_df: pd.DataFrame, cat_path: str = "/data/category_id.json") -> int:
    # Keep only the columns necessary for the comparison to reduce memory
    # footprint: trending_date, category_id (to map to names), and views.
    df = vdo_df.drop_duplicates()[["trending_date", "category_id", "views"]].copy()

    # Load the category JSON and construct a mapping from numeric category id
    # to its human-readable title. The JSON layout is expected to have an
    # `items` list where each item has `id` and `snippet.title`.
    with open(cat_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    items = payload.get("items", [])
    id_map = {int(it["id"]): it.get("snippet", {}).get("title", "") for it in items}

    # Map numeric `category_id` values to the category title; any unmapped
    # ids become 'Unknown' so they don't raise exceptions during grouping.
    df["cat"] = df["category_id"].map(id_map).fillna("Unknown")

    # Aggregate total views per (trending_date, category). This reduces the
    # data to the daily totals per category which we need to compare.
    grouped = df.groupby(["trending_date", "cat"])["views"].sum().reset_index()

    # Pivot so each row is a trending_date and each column is a category's
    # total views for that date. Missing categories for a date become 0.
    pivot = grouped.pivot(index="trending_date", columns="cat", values="views").fillna(
        0
    )

    # Extract the two series we want to compare. If either category is absent
    # in the entire dataset, treat its daily views as zero for all dates.
    if "Sports" in pivot.columns:
        sports = pivot["Sports"]
    else:
        sports = pd.Series(0, index=pivot.index)

    if "Comedy" in pivot.columns:
        comedy = pivot["Comedy"]
    else:
        comedy = pd.Series(0, index=pivot.index)

    # Compare the two series elementwise; this yields a boolean series that
    # indicates on which dates Sports had strictly more total daily views than
    # Comedy. Sum the True values to get the count of such days.
    return int((sports > comedy).sum())


__all__ = ["Q1", "Q2", "Q3", "Q4", "Q5"]
