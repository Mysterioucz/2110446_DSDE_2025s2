import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings  # DO NOT modify this line
from sklearn.exceptions import ConvergenceWarning  # DO NOT modify this line

warnings.filterwarnings(
    "ignore", category=ConvergenceWarning
)  # DO NOT modify this line


class BankLogistic:
    def __init__(self, data_path):  # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=",")
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Q1(self):  # DO NOT modify this line
        """
        Problem 1:
            Load ‘bank-st.csv’ data from the “Attachment”
            How many rows of data are there in total?

        """
        # TODO: Paste your code here
        self.df = pd.read_csv(self.data_path)
        return self.df.shape[0]

    def Q2(self):  # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        # TODO: Paste your code here
        self.df = pd.read_csv(self.data_path)
        cat_col = self.df.select_dtypes(include="object")
        num_col = self.df.select_dtypes(include=[np.number])
        return (num_col.shape[1], cat_col.shape[1])

    def Q3(self):  # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        # TODO: Paste your code here
        self.df = pd.read_csv(self.data_path)
        yes_c = sum(self.df["y"] == "yes") / self.df.shape[0]
        no_c = sum(self.df["y"] == "no") / self.df.shape[0]
        return (round(no_c, 3), round(yes_c, 3))

    def Q4(self):  # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        # TODO: Paste your code here
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop_duplicates()
        return self.df.shape

    def Q5(self):  # DO NOT modify this line
        """
        Problem 5:
            5. Replace unknown value with null
            6. Remove features with more than 99% flat values.
                Hint: There is only one feature should be drop
            7. Split Data
            -	Split the dataset into training and testing sets with a 70:30 ratio.
            -	random_state=0
            -	stratify option
            return the tuple of shapes of X_train and X_test.

        """
        # TODO: Paste your code here

        # Step 1-4
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop_duplicates()

        # Step 5
        self.df = self.df.replace("unknown", None)

        # Step 6
        rem_cols = []
        for col in self.df.columns:
            top_count = self.df[col].value_counts().iloc[0]
            total = self.df.shape[0] - self.df[col].isnull().sum()
            if top_count / total > 0.99:
                rem_cols.append(col)
        self.df = self.df.drop(columns=rem_cols, axis=1)

        # Step 7
        X = self.df.drop(columns=["y"])
        y = self.df["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.3, stratify=y
        )
        return (X_train.shape, X_test.shape)

    def Q6(self):
        """
        Problem 6:
            8. Impute missing
                -	For numeric variables: Impute missing values using the mean.
                -	For categorical variables: Impute missing values using the mode.
                Hint: Use statistics calculated from the training dataset to avoid data leakage.
            9. Categorical Encoder:
                Map the nominal data for the education variable using the following order:
                education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7}
                Hint: Use One hot encoder or pd.dummy to encode nominal category
            return the shape of X_train.

        """
        # Step 1-4
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop_duplicates()

        # Step 5
        self.df = self.df.replace("unknown", None)

        # Step 6
        rem_cols = []
        for col in self.df.columns:
            top_count = self.df[col].value_counts().iloc[0]
            total = self.df.shape[0] - self.df[col].isnull().sum()
            if top_count / total > 0.99:
                rem_cols.append(col)
        self.df = self.df.drop(columns=rem_cols, axis=1)

        # Step 7
        X = self.df.drop(columns=["y"])
        y = self.df["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.3, stratify=y
        )
        # Step 8
        X_train = X_train.fillna(X_train.mean(numeric_only=True))
        X_train = X_train.fillna(X_train.select_dtypes(include="object").mode().iloc[0])

        X_test = X_test.fillna(X_train.mean(numeric_only=True))
        X_test = X_test.fillna(X_train.select_dtypes(include="object").mode().iloc[0])

        # Step 9
        education_order = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "professional.course": 6,
            "university.degree": 7,
        }
        X_train["education"] = X_train["education"].map(education_order)
        X_test["education"] = X_test["education"].map(education_order)

        cat_col = X_train.select_dtypes(include="object").columns
        X_train = pd.get_dummies(X_train, columns=cat_col)
        X_test = pd.get_dummies(X_test, columns=cat_col)

        return X_train.shape

    def Q7(self):
        """Problem7: Use Logistic Regression as the model with
        random_state=2025,
        class_weight='balanced' and
        max_iter=500.
        Train the model using all the remaining available variables.
        What is the macro F1 score of the model on the test data? in 3 digits
        """
        # TODO: Paste your code here
        # Step 1-4
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop_duplicates()

        # Step 5
        self.df = self.df.replace("unknown", None)

        # Step 6
        rem_cols = []
        for col in self.df.columns:
            top_count = self.df[col].value_counts().iloc[0]
            total = self.df.shape[0] - self.df[col].isnull().sum()
            if top_count / total > 0.99:
                rem_cols.append(col)
        self.df = self.df.drop(columns=rem_cols, axis=1)

        # Step 7
        X = self.df.drop(columns=["y"])
        y = self.df["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.3, stratify=y
        )
        # Step 8
        X_train = X_train.fillna(X_train.mean(numeric_only=True))
        X_train = X_train.fillna(X_train.select_dtypes(include="object").mode().iloc[0])

        X_test = X_test.fillna(X_train.mean(numeric_only=True))
        X_test = X_test.fillna(X_train.select_dtypes(include="object").mode().iloc[0])

        # Step 9
        education_order = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "professional.course": 6,
            "university.degree": 7,
        }
        X_train["education"] = X_train["education"].map(education_order)
        X_test["education"] = X_test["education"].map(education_order)

        cat_col = X_train.select_dtypes(include="object").columns
        X_train = pd.get_dummies(X_train, columns=cat_col)
        X_test = pd.get_dummies(X_test, columns=cat_col)

        model = LogisticRegression(
            random_state=2025, class_weight="balanced", max_iter=500
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)
        return round(report["macro avg"]["f1-score"], 2)
        # return float(f"{report["macro avg"]["f1-score"]:.2f}")
