def Q1():
    """
    Output [Grader: Yes]: Return the dictionary containing the three values:
    {
        "actual_0": <actual value of y_test at index 0 rounded to 3 decimals >,
        "predict_0": <predicted value y_pred[0] rounded to 3 decimals>,
        "AE": <absolute error rounded to 3 decimals>
    }
    Hint: Compute the absolute error using the rounded values (actual_0, predict_0),
    and round the absolute error to 3 decimal places.
    [first_ae = round(abs(actual_0 – predict_0), 3)]
    """

    return {"actual_0": 15.246, "predict_0": 17.844, "AE": 2.598}
