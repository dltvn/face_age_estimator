import os
import json
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

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


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


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


    if rel_or_abs_path.startswith("data" + os.sep) or rel_or_abs_path.startswith("utk_face" + os.sep):
        return os.path.normpath(os.path.join(data_root, rel_or_abs_path.replace("data" + os.sep, "")))

 
    return os.path.normpath(os.path.join(data_root, "utk_face", rel_or_abs_path))


def decode_and_resize(img_bytes: tf.Tensor, image_size: int) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def load_image_tensor(path: tf.Tensor, image_size: int) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    return decode_and_resize(img_bytes, image_size)


def maybe_set_seed(seed: int) -> None:
    if seed is None:
        return
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def try_import_mtcnn():
    try:
        import mtcnn 
        return mtcnn.MTCNN
    except Exception:
        return None


def try_import_vggface():
    try:
        from keras_vggface.vggface import VGGFace  
        return VGGFace
    except Exception:
        return None


def align_faces_with_mtcnn(
    df: pd.DataFrame,
    data_root: str,
    img_col: str,
    out_dir: str,
    image_size: int,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Proposal alignment step: MTCNN detects face and 5 landmarks.
    For a baseline, we crop using the detected bounding box.
    This keeps the pipeline aligned with the proposal and is reliable.

    Output: new column 'aligned_path' pointing to saved aligned face images.
    """
    MTCNN = try_import_mtcnn()
    if MTCNN is None:
        raise RuntimeError(
            "MTCNN is not installed. Install it with: python -m pip install mtcnn opencv-python"
        )

    import cv2  # type: ignore

    ensure_dir(out_dir)
    detector = MTCNN()

    rows = []
    n = len(df) if limit is None else min(len(df), limit)

    for i in range(n):
        row = df.iloc[i].to_dict()
        src = str(row[img_col])
        src_path = build_image_path(data_root, src)

        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dets = detector.detect_faces(img_rgb)

        if not dets:
            continue

        det = max(dets, key=lambda d: d.get("confidence", 0.0))
        x, y, w, h = det["box"]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        face = img_rgb[y : y + h, x : x + w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        out_path = os.path.join(out_dir, f"aligned_{i:07d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        row["aligned_path"] = out_path
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        raise RuntimeError("No faces were aligned. Check your CSV paths and images.")
    return out_df


@dataclass
class BackboneInfo:
    model: tf.keras.Model
    preprocess_fn: callable
    name: str


def build_backbone(image_size: int, force_backbone: str = "auto") -> BackboneInfo:
    """
    Proposal wants VGGFace backbone.
    We try keras_vggface first.
    If unavailable, we use VGG16 as a compatible baseline backbone.
    """
    if force_backbone.lower() not in ["auto", "vggface", "vgg16"]:
        raise ValueError("force_backbone must be one of: auto, vggface, vgg16")

    VGGFace = try_import_vggface()
    if force_backbone.lower() in ["auto", "vggface"] and VGGFace is not None:
        # keras_vggface returns a Keras model
        base = VGGFace(model="vgg16", include_top=False, input_shape=(image_size, image_size, 3))
        def preprocess(x):
            # keras_vggface expects BGR with mean subtraction typically.
            # We keep it simple and use tf.keras VGG16 preprocess which is close.
            return tf.keras.applications.vgg16.preprocess_input(x)
        return BackboneInfo(model=base, preprocess_fn=preprocess, name="vggface_vgg16")

    # Fallback baseline
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    return BackboneInfo(
        model=base,
        preprocess_fn=tf.keras.applications.vgg16.preprocess_input,
        name="vgg16_imagenet_fallback",
    )


def build_age_model(backbone: tf.keras.Model, image_size: int, dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


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

 
    parser.add_argument("--train_csv", type=str, default="data/splits/train_age_male.csv")
    parser.add_argument("--val_csv", type=str, default="data/splits/val_age_male.csv")

    # Auto detect if your CSV uses different column names
    parser.add_argument("--img_col", type=str, default="auto")
    parser.add_argument("--label_col", type=str, default="auto")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=1e-5)

    # Proposal style backbone
    parser.add_argument("--backbone", type=str, default="auto")  # auto, vggface, vgg16

    parser.add_argument("--run_dir", type=str, default="runs/age_run_1")
    parser.add_argument("--limit", type=int, default=0) 
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    maybe_set_seed(args.seed)

    ensure_dir(args.run_dir)
    ensure_dir("runs")

    train_df = load_split_csv(args.train_csv)
    val_df = load_split_csv(args.val_csv)

    # Detect columns
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

    # Proposal: MTCNN detection and alignment
    aligned_dir_train = os.path.join(args.run_dir, "aligned_train")
    aligned_dir_val = os.path.join(args.run_dir, "aligned_val")

    limit = None if args.limit == 0 else args.limit

    train_aligned = align_faces_with_mtcnn(
        df=train_df,
        data_root=args.data_root,
        img_col=img_col,
        out_dir=aligned_dir_train,
        image_size=args.image_size,
        limit=limit,
    )

    val_aligned = align_faces_with_mtcnn(
        df=val_df,
        data_root=args.data_root,
        img_col=img_col,
        out_dir=aligned_dir_val,
        image_size=args.image_size,
        limit=limit,
    )

    # Save aligned CSVs for reproducibility
    train_aligned_csv = os.path.join(args.run_dir, "train_aligned.csv")
    val_aligned_csv = os.path.join(args.run_dir, "val_aligned.csv")
    train_aligned.to_csv(train_aligned_csv, index=False)
    val_aligned.to_csv(val_aligned_csv, index=False)

    backbone_info = build_backbone(args.image_size, force_backbone=args.backbone)
    backbone = backbone_info.model
    preprocess_fn = backbone_info.preprocess_fn

    # Stage 1: freeze backbone
    backbone.trainable = False
    model = build_age_model(backbone, args.image_size, args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_stage1),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    train_ds = df_to_dataset(
        train_aligned, args.image_size, args.batch_size, True, "aligned_path", label_col, preprocess_fn
    )
    val_ds = df_to_dataset(
        val_aligned, args.image_size, args.batch_size, False, "aligned_path", label_col, preprocess_fn
    )

    # Checkpointing
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    stage1_best = os.path.join(ckpt_dir, "age_stage1_best.keras")
    stage2_best = os.path.join(ckpt_dir, "age_stage2_best.keras")
    last_model = os.path.join(ckpt_dir, "age_last.keras")
    history_csv = os.path.join(args.run_dir, "history.csv")

    callbacks_common = [
        tf.keras.callbacks.ModelCheckpoint(stage1_best, monitor="val_mae", save_best_only=True, mode="min"),
        tf.keras.callbacks.ModelCheckpoint(last_model, save_best_only=False),
        tf.keras.callbacks.CSVLogger(history_csv, append=os.path.exists(history_csv)),
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=2, min_lr=1e-7),
    ]

    # Resume if last exists
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
        callbacks=callbacks_common,
    )

    # Stage 2: fine tune top of backbone
    print("Stage 2 Fine tuning")
    backbone = model.layers[1] if isinstance(model.layers[1], tf.keras.Model) else None

    # Unfreeze last blocks gradually
    # This works for VGG style models
    if backbone is not None:
        backbone.trainable = True
        for layer in backbone.layers[:-4]:
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
    print(f"Aligned train csv: {train_aligned_csv}")
    print(f"Aligned val csv: {val_aligned_csv}")


if __name__ == "__main__":
    main()