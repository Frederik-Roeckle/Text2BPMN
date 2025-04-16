_model = None

def set_model(model):
    global _model
    _model = model

def get_model():
    if _model is None:
        raise ValueError("Model not set. Call `set_model()` first.")
    return _model