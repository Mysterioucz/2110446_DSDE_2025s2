import pandas as pd
from sklearn.model_selection import train_test_split

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic_to_student.csv) and answer the questions.
    (Note that the following functions already take the Titanic dataset as a DataFrame, so you don’t need to use read_csv.)

"""


def Q1(df):
    """
    Problem 1:
        How many rows are there in the "titanic_to_student.csv"?
    """
    # TODO: Code here

    return df.shape[0]


def Q2(df):
    """
    Problem 2:
        2.1 Drop variables with missing > 50%
        2.2 Check all columns except 'Age' and 'Fare' for flat values, drop the columns where flat value > 70%
        From 2.1 and 2.2, how many columns do we have left?
        Note:
        -Ensure missing values are considered in your calculation. If you use normalize in .value_counts(), please include dropna=False.
    """
    # 2.1 Drop columns with missing > 50%
    drop_threshold = len(df) * 0.5
    df = df.dropna(thresh=drop_threshold, axis=1)

    # 2.2 Drop columns with flat values > 70% (except Age and Fare)
    rem_col = []
    for col in df.columns:
        if col not in ["Age", "Fare"]:
            top_count = df[col].value_counts(dropna=False).iloc[0]
            top_pct = top_count / df.shape[0]
            if top_pct > 0.7:
                rem_col.append(col)

    df = df.drop(rem_col, axis=1)
    return df.shape[1]


def Q3(df):
    """
    Problem 3:
         Remove all rows with missing targets (the variable "Survived")
         How many rows do we have left?
    """
    df = df.dropna(subset=["Survived"], axis=0)
    return df.shape[0]


def Q4(df):
    """
    Problem 4:
         Handle outliers
         For the variable “Fare”, replace outlier values with the boundary values
         If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
         If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
         What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
         Hint: Use function round(_, 2)
    """
    quartiles = df["Fare"].quantile([0.25, 0.5, 0.75])
    q1, q2, q3 = quartiles.values
    iqr = q3 - q1

    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    df["Fare"] = df["Fare"].clip(lower=lower_bound, upper=upper_bound)

    return round(df["Fare"].mean(), 2)


def Q5(df):
    """
    Problem 5:
         Impute missing value
         For number type column, impute missing values with mean
         What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
         Hint: Use function round(_, 2)
    """
    number_cols = ["Age", "SibSp", "Fare"]
    for col in number_cols:
        if col in df.columns:
            df.loc[df[col].isna(), col] = df[col].mean()

    return round(df["Age"].mean(), 2)


def Q6(df):
    """
    Problem 6:
        Convert categorical to numeric values
        For the variable “Embarked”, perform the dummy coding.
        What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
        Hint: Use function round(_, 2)
    """
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    return round(df["Embarked_Q"].mean(), 2)


def Q7(df):
    """
    Problem 7:
        Split train/test split with stratification using 70%:30% and random seed with 123
        Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
        What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
        Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection,
        Don't forget to impute missing values with mean.
    """
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    train_props = sum(y_train == 1) / y_train.shape[0]
    return round(train_props, 2)
