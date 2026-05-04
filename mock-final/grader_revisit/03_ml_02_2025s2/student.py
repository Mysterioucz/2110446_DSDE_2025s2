import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
        return self.df.shape[0]

    def Q2(self):  # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        # TODO: Paste your code here
        num_col = self.df.select_dtypes(include=["number"]).columns.tolist()
        cat_col = [col for col in self.df.columns.tolist() if col not in num_col]
        return (len(num_col), len(cat_col))

    def Q3(self):  # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        # TODO: Paste your code here
        tmp = self.df["y"].value_counts().to_dict()
        return (tmp["no"], tmp["yes"])

    def Q4(self):  # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        # TODO: Paste your code here

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
        self.df = self.df.replace("unknown", np.nan)
        flagged = []

        for col in self.df.columns.tolist():
            val_count = self.df[col].value_counts()
            # print(f"val count for {col} is {val_count.to_dict()}", end='')
            max_count = val_count.max()
            # print(f"flat_val for {col} is {max_count/val_count.sum()} : {max_count / self.df.shape[0]}")
            if max_count / val_count.sum() > 0.99:
                flagged.append(col)
        # print("Removing:", flagged)
        self.df = self.df.drop(labels=flagged, axis=1)

        y = self.df["y"]
        X = self.df.drop(labels=["y"], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        return (self.X_train.shape, self.X_test.shape)

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
        # TODO: Paste your code here

        num_col = self.X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_col = [col for col in self.X_train.columns.tolist() if col not in num_col]
        numeric_transformer = Pipeline(
            steps=[
                ("num imputer", SimpleImputer(missing_values=np.nan, strategy="mean"))
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                (
                    "cat imputer",
                    SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
                )
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num imp", numeric_transformer, num_col),
                ("cat imp", categorical_transformer, cat_col),
            ]
        )
        X_train_trans = preprocessor.fit_transform(self.X_train)
        X_test_trans = preprocessor.transform(self.X_test)
        # Reconstruct
        X_test_trans = pd.DataFrame(X_test_trans, columns=num_col + cat_col)
        self.X_test = X_test_trans[self.X_test.columns.tolist()]
        X_train_trans = pd.DataFrame(X_train_trans, columns=num_col + cat_col)
        self.X_train = X_train_trans[self.X_train.columns.tolist()]

        education_order = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "professional.course": 6,
            "university.degree": 7,
        }
        self.X_train["education"] = self.X_train["education"].map(education_order)
        self.X_test["education"] = self.X_test["education"].map(education_order)

        self.X_train = pd.get_dummies(
            self.X_train,
            columns=[col for col in cat_col if col != "education"],
        )
        self.X_test = pd.get_dummies(
            self.X_test,
            columns=[col for col in cat_col if col != "education"],
        )

        return self.X_train.shape

    def Q7(self):
        """Problem7: Use Logistic Regression as the model with
        random_state=2025,
        class_weight='balanced' and
        max_iter=500.
        Train the model using all the remaining available variables.
        What is the macro F1 score of the model on the test data? in 3 digits
        """
        # TODO: Paste your code here
        model = LogisticRegression(random_state=2025, class_weight='balanced', max_iter=500)
        model.fit(self.X_train, self.y_train)
        
        pred = model.predict(self.X_test)
        res = classification_report(pred, self.y_test, output_dict=True)
        
        return round(res['macro avg']['f1-score'], 2)
