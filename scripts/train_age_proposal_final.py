import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf


def is_colab() -> bool:
    return os.path.exists("/content") and "COLAB_GPU" in os.environ


def default_data_root() -> str:
    if is_colab():
        return "/content/data"
    return "data"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def autodetect_column(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols:
            return df.columns[cols.index(cand.lower())]
    return None


def load_split_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def build_image_path(data_root: str, rel_or_abs_path: str) -> str:
    if os.path.isabs(rel_or_abs_path):
        return rel_or_abs_path

    rel_or_abs_path = str(rel_or_abs_path).replace("/", os.sep).replace("\\", os.sep)

    if rel_or_abs_path.startswith("data" + os.sep):
        rel_or_abs_path = rel_or_abs_path.replace("data" + os.sep, "", 1)

    if rel_or_abs_path.startswith("utk_face" + os.sep):
        return os.path.normpath(os.path.join(data_root, rel_or_abs_path))

    return os.path.normpath(os.path.join(data_root, "utk_face", rel_or_abs_path))


def decode_and_resize(img_bytes: tf.Tensor, image_size: int) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def load_image_tensor(path: tf.Tensor, image_size: int) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    return decode_and_resize(img_bytes, image_size)


def maybe_set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@dataclass
class BackboneInfo:
    model: tf.keras.Model
    preprocess_fn: callable
    name: str


def build_backbone(image_size: int, force_backbone: str = "resnet50") -> BackboneInfo:
    allowed = ["resnet50"]
    if force_backbone.lower() not in allowed:
        raise ValueError(f"force_backbone must be one of: {allowed}")

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )

    return BackboneInfo(
        model=base,
        preprocess_fn=tf.keras.applications.resnet50.preprocess_input,
        name="resnet50_imagenet",
    )


def build_age_model(backbone: tf.keras.Model, image_size: int, dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="predicted_age")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_age_regressor")


def df_to_dataset(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    img_path_col: str,
    label_col: str,
    preprocess_fn,
) -> tf.data.Dataset:
    paths = df[img_path_col].astype(str).values
    labels = df[label_col].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 4096), reshuffle_each_iteration=True)

    def _map(path, y):
        img = load_image_tensor(path, image_size)
        img = preprocess_fn(img)
        y = tf.expand_dims(y, axis=-1)
        return img, y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_run_config(out_dir: str, cfg: dict) -> None:
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=default_data_root())


    parser.add_argument("--img_col", type=str, default="auto")
    parser.add_argument("--label_col", type=str, default="auto")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=1e-5)

    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--run_dir", type=str, default="runs/age_run_1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    maybe_set_seed(args.seed)

    ensure_dir(args.run_dir)
    ensure_dir("runs")

    # Load male + female splits and combine them
    train_male = load_split_csv("data/splits/train_age_male.csv")
    train_female = load_split_csv("data/splits/train_age_female.csv")

    val_male = load_split_csv("data/splits/val_age_male.csv")
    val_female = load_split_csv("data/splits/val_age_female.csv")

    train_df = pd.concat([train_male, train_female], ignore_index=True)
    val_df = pd.concat([val_male, val_female], ignore_index=True)

    print(f"Combined training samples: {len(train_df)}")
    print(f"Combined validation samples: {len(val_df)}")

    if args.limit > 0:
        train_df = train_df.iloc[: args.limit].reset_index(drop=True)
        val_df = val_df.iloc[: min(args.limit, len(val_df))].reset_index(drop=True)

    if args.img_col == "auto":
        img_col = autodetect_column(train_df, ["path", "filepath", "file", "filename", "image", "img"])
        if img_col is None:
            raise ValueError(f"Could not auto detect image column in {args.train_csv}. Use --img_col explicitly.")
    else:
        img_col = args.img_col

    if args.label_col == "auto":
        label_col = autodetect_column(train_df, ["age", "label", "target"])
        if label_col is None:
            raise ValueError(f"Could not auto detect label column in {args.train_csv}. Use --label_col explicitly.")
    else:
        label_col = args.label_col

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["resolved_image_path"] = train_df[img_col].apply(lambda p: build_image_path(args.data_root, p))
    val_df["resolved_image_path"] = val_df[img_col].apply(lambda p: build_image_path(args.data_root, p))

    train_df = train_df[train_df["resolved_image_path"].apply(os.path.exists)].reset_index(drop=True)
    val_df = val_df[val_df["resolved_image_path"].apply(os.path.exists)].reset_index(drop=True)

    if len(train_df) == 0:
        raise RuntimeError("No valid training images found after resolving paths.")
    if len(val_df) == 0:
        raise RuntimeError("No valid validation images found after resolving paths.")

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    backbone_info = build_backbone(args.image_size, force_backbone=args.backbone)
    backbone = backbone_info.model
    preprocess_fn = backbone_info.preprocess_fn

    backbone.trainable = False
    model = build_age_model(backbone, args.image_size, args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_stage1),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    train_ds = df_to_dataset(
        train_df,
        args.image_size,
        args.batch_size,
        True,
        "resolved_image_path",
        label_col,
        preprocess_fn,
    )
    val_ds = df_to_dataset(
        val_df,
        args.image_size,
        args.batch_size,
        False,
        "resolved_image_path",
        label_col,
        preprocess_fn,
    )

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    stage1_best = os.path.join(ckpt_dir, "age_stage1_best.keras")
    stage2_best = os.path.join(ckpt_dir, "age_stage2_best.keras")
    last_model = os.path.join(ckpt_dir, "age_last.keras")
    history_csv = os.path.join(args.run_dir, "history.csv")

    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(stage1_best, monitor="val_mae", save_best_only=True, mode="min"),
        tf.keras.callbacks.ModelCheckpoint(last_model, save_best_only=False),
        tf.keras.callbacks.CSVLogger(history_csv, append=os.path.exists(history_csv)),
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=2, min_lr=1e-7),
    ]

    if os.path.exists(last_model):
        try:
            model = tf.keras.models.load_model(last_model)
            print(f"Resumed from: {last_model}")
        except Exception:
            pass

    cfg = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "img_col": img_col,
        "label_col": label_col,
        "data_root": args.data_root,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_stage1": args.epochs_stage1,
        "epochs_stage2": args.epochs_stage2,
        "backbone_used": backbone_info.name,
        "backbone_requested": args.backbone,
        "run_dir": args.run_dir,
        "seed": args.seed,
    }
    save_run_config(args.run_dir, cfg)

    print(f"Backbone used: {backbone_info.name}")
    print("Stage 1 Training (Frozen Backbone)")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1,
        callbacks=callbacks_stage1,
    )

    print("Stage 2 Fine tuning")

    backbone = model.layers[1] if isinstance(model.layers[1], tf.keras.Model) else None
    if backbone is not None:
        backbone.trainable = True

        # Freeze most layers, unfreeze only the top portion for fine tuning.
        # This is a safer generic setting for ResNet50.
        fine_tune_from = max(0, len(backbone.layers) - 30)
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False

        for layer in backbone.layers[fine_tune_from:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_stage2),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(stage2_best, monitor="val_mae", save_best_only=True, mode="min"),
        tf.keras.callbacks.ModelCheckpoint(last_model, save_best_only=False),
        tf.keras.callbacks.CSVLogger(history_csv, append=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=2, min_lr=1e-7),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage2,
        callbacks=callbacks_stage2,
    )

    print("Training complete")
    print(f"Best stage 1: {stage1_best}")
    print(f"Best stage 2: {stage2_best}")
    print(f"Last model: {last_model}")
    print(f"History: {history_csv}")


if __name__ == "__main__":
    main()