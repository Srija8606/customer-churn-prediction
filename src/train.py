from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def train_model(preprocessor, X_train, y_train):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
