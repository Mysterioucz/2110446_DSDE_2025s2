import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from (videos.csv and category_id.json) and answer the questions.
"""


def Q1():
    """
    1. How many rows are there in the videos.csv after removing duplications?
    - To access 'videos.csv', use the path '/data/videos.csv'.
    """
    # TODO: Paste your code here
    df = pd.read_csv("/data/videos.csv")
    df = df.drop_duplicates()
    return len(df)


def Q2(vdo_df):
    """
    2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    # TODO: Paste your code here
    return vdo_df.loc[vdo_df["dislikes"] > vdo_df["likes"], "title"].nunique()


def Q3(vdo_df):
    """
    3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    """
    # TODO: Paste your code here
    return sum(
        (vdo_df["trending_date"] == "18.22.01") & (vdo_df["comment_count"] > 10000)
    )


def Q4(vdo_df):
    """
    4. Which trending date that has the minimum average number of comments per VDO?
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
    """
    # TODO:  Paste your code here
    df = vdo_df[["trending_date", "comment_count"]].copy()
    df["vdo_count"] = 1
    df = df.groupby(["trending_date"]).sum().reset_index()
    df["ratio"] = df["comment_count"] / df["vdo_count"]
    date = df.loc[df["ratio"] == df["ratio"].min(), "trending_date"].iloc[0]
    return date


def Q5(vdo_df):
    """
    5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
        - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
        - The duplicate rows of vdo_df have been removed.
        - You must load the additional data from 'category_id.json' into memory before executing any operations.
        - To access 'category_id.json', use the path '/data/category_id.json'.
    """
    # TODO:  Paste your code here
    cat = pd.read_json("/data/category_id.json")
    cat = cat.drop(labels=["kind", "etag"], axis=1)
    id_mapper = {}
    for e in cat["items"].values:
        id_mapper[int(e["id"])] = e["snippet"]["title"]
    df = vdo_df[["trending_date", "category_id", "views"]].copy()
    df["cat"] = df["category_id"].map(id_mapper)
    df = (
        df.drop(labels=["category_id"], axis=1)
        .groupby(["trending_date", "cat"])
        .sum()
        .reset_index()
    )
    sport = df.loc[df["cat"] == "Sports"].reset_index()
    comedy = df.loc[df["cat"] == "Comedy"].reset_index()
    count = sum(sport["views"] > comedy["views"])
    return count
