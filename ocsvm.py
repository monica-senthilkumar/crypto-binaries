class OCSVMWrapper:
    def __init__(self):
        from sklearn.svm import OneClassSVM
        self.model = OneClassSVM(
            kernel="rbf",
            nu=0.05,
            gamma="scale"
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
