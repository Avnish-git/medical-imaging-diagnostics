
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_pipeline import build_label_dataframe, split_data, make_dataset, compute_class_weights
from model import build_model, weighted_binary_crossentropy
from config import LEARNING_RATE, EPOCHS


def train(csv_path, image_dir, checkpoint_path="best_model.keras"):
    df = build_label_dataframe(csv_path, image_dir)
    train_df, val_df, _ = split_data(df)

    train_ds = make_dataset(train_df, augment=True, shuffle=True)
    val_ds = make_dataset(val_df, augment=False, shuffle=False)

    pos_weights = compute_class_weights(train_df)
    model, _ = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=weighted_binary_crossentropy(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=14),
            tf.keras.metrics.BinaryAccuracy(name="bin_acc")
        ]
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_auc", mode="max",
                        save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return model, history


if __name__ == "__main__":
    CSV_PATH = "data/sample_labels.csv"
    IMAGE_DIR = "data/sample/images"
    model, history = train(CSV_PATH, IMAGE_DIR)
    print("Training complete.")
