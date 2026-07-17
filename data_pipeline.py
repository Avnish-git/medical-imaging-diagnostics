
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CLASS_NAMES, IMG_SIZE, BATCH_SIZE

AUTOTUNE = tf.data.AUTOTUNE
random_rotation = tf.keras.layers.RandomRotation(0.05)
random_zoom = tf.keras.layers.RandomZoom(0.1)


def build_label_dataframe(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    df["path"] = df["Image Index"].apply(lambda x: f"{image_dir}/{x}")
    for label in CLASS_NAMES:
        df[label] = df["Finding Labels"].apply(
            lambda x: 1.0 if label in x.split("|") else 0.0
        )
    return df


def split_data(df, test_size=0.15, val_size=0.15, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _load_and_preprocess(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        img = random_rotation(tf.expand_dims(img, 0))[0]
        img = random_zoom(tf.expand_dims(img, 0))[0]

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def make_dataset(df, batch_size=BATCH_SIZE, augment=False, shuffle=False):
    paths = df["path"].values
    labels = df[CLASS_NAMES].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 5000), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _load_and_preprocess(p, l, augment=augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def compute_class_weights(df):
    pos_counts = df[CLASS_NAMES].sum().values
    total = len(df)
    pos_weights = (total - pos_counts) / (pos_counts + 1e-7)
    return dict(zip(CLASS_NAMES, pos_weights))
