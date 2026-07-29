"""
Phase 6b - Neural network (TensorFlow / Keras).

A feedforward network trained on the same data as the classical models, so the
comparison in Phase 7 is fair.

Architecture: 25 inputs -> 64 -> 32 -> 16 -> 1

Each hidden layer uses ReLU (which lets the network learn non-linear
combinations of risk factors), followed by batch normalisation and dropout.
Dropout randomly switches off a fraction of neurons during training, which
stops the network memorising individual patients. The single output neuron
uses a sigmoid so its value can be read directly as a probability of disease.

Worth stating plainly: on structured tabular data like this, neural networks
frequently do *not* beat gradient boosting. That result is reported honestly in
Phase 7 rather than hidden - knowing when the more complex model is not the
right answer matters more than always picking the fanciest one.
"""

import os

# Keep TensorFlow's startup logging out of the pipeline output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import config
import feature_engineering

NEURAL_NETWORK = "Neural Network"


def _keras():
    """Import Keras lazily so the rest of the pipeline runs without TensorFlow."""
    import tensorflow as tf

    tf.random.set_seed(config.RANDOM_STATE)
    return tf.keras


def build_network(n_features: int):
    """Assemble the feedforward architecture."""
    keras = _keras()
    layers = keras.layers

    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),

            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(16, activation="relu"),
            layers.Dropout(0.1),

            # Sigmoid squashes the output to 0-1 so it reads as a probability.
            layers.Dense(1, activation="sigmoid"),
        ],
        name="risk_network",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def model_path(disease: str):
    return config.MODELS_DIR / f"{disease}_neural_network.keras"


def train(dataset: feature_engineering.Dataset, epochs: int = 40,
          batch_size: int = 512, save: bool = True):
    """Train the network for one disease."""
    keras = _keras()
    config.ensure_directories()

    print(f"\n{dataset.label}  ({dataset.X_train.shape[0]:,} training rows, "
          f"{dataset.X_train.shape[1]} features)")

    model = build_network(dataset.X_train.shape[1])

    callbacks = [
        # Stop as soon as validation performance stops improving, and rewind to
        # the best epoch. This is what prevents the network from overfitting.
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6,
            restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0,
        ),
    ]

    history = model.fit(
        dataset.X_train, dataset.y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    trained_epochs = len(history.history["loss"])
    best_val_auc = max(history.history["val_auc"])
    print(f"  trained {trained_epochs} epochs "
          f"(early stopping from a {epochs}-epoch budget)")
    print(f"  best validation AUC: {best_val_auc:.4f}")

    if save:
        model.save(model_path(dataset.disease))
        print(f"  saved -> models/{model_path(dataset.disease).name}")

    return model, history


def load(disease: str):
    keras = _keras()
    path = model_path(disease)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found - run src/train_neural_network.py first."
        )
    return keras.models.load_model(path)


class KerasProbabilityWrapper:
    """
    Give the Keras model the same interface as the scikit-learn models.

    Evaluation, SHAP and LIME all expect `predict_proba` returning one column
    per class. Keras returns a single sigmoid column, so it is expanded here.
    This keeps every downstream phase model-agnostic.
    """

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X) -> np.ndarray:
        positive = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_wrapped(disease: str) -> KerasProbabilityWrapper:
    return KerasProbabilityWrapper(load(disease))


def main() -> None:
    print("=" * 70)
    print("PHASE 6b: NEURAL NETWORK")
    print("=" * 70)

    datasets = feature_engineering.prepare_all(verbose=False)
    for disease in config.DISEASES:
        train(datasets[disease])

    print("\nNeural networks trained.")


if __name__ == "__main__":
    main()
