class Crafter:
    def __init__(self, steps):
        """
        Initialize the Crafter with a list of modeling steps.
        Each step is a tuple containing a model.
        """
        self.steps = steps
        self.models = []
        self.predictions = {}  # To store predictions from each model

    def fit(self, X, y):
        """
        Fit the models in the pipeline independently on the same input data and store their predictions.
        """
        for (
            model,
            _,
            _,
        ) in self.steps:  # Only the model is used, other flags are ignored
            model.fit(X, y)
            self.models.append(model)

        return self

    def predict(self, X):
        """
        Make predictions using each fitted model and store them.
        """
        self.predictions = {}
        for i, model in enumerate(self.models):
            model_name = f"Step {i + 1}: {model.__class__.__name__}"
            self.predictions[model_name] = model.predict(X)

        return self.predictions

    def visualize_pipeline(self):
        """
        Prints a textual representation of the pipeline.
        """
        print("Crafter Pipeline Visualization:")
        for i, (model, _, _) in enumerate(self.steps):
            print(f"Step {i + 1}: {model.__class__.__name__}")
