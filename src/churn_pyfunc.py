"""mlflow pyfunc wrapper: bundles the xgb model + isotonic calibrator + feature order into ONE
artifact that emits CALIBRATED P(churn). kept standalone (no src-package imports) so mlflow can
re-import it anywhere via code_paths. the whole point of bundling: the calibrator can never be
forgotten at scoring time, and the feature order is pinned to the model it was trained with.
"""
import json
import joblib
import mlflow.pyfunc


class CalibratedChurn(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["xgb"])
        self.calibrator = joblib.load(context.artifacts["calibrator"])
        with open(context.artifacts["features"]) as f:
            self.features = json.load(f)

    def predict(self, context, model_input):
        # model_input: a DataFrame holding (at least) the base12 columns. select in pinned order,
        # take the raw churn score, then map it through the val-fit isotonic calibrator.
        raw = self.model.predict_proba(model_input[self.features])[:, 1]
        return self.calibrator.predict(raw)
