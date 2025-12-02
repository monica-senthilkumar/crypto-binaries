from xgboost import XGBClassifier

class XGBoostWrapper:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=16,
            eval_metric="mlogloss"
        )

    def load(self, path="xgb_model.json"):
        self.model.load_model(path)

    def predict(self, X):
        return self.model.predict(X)
