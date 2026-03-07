import os
import argparse
import numpy as np
import tensorflow as tf


def decode_and_resize(img_bytes: tf.Tensor, image_size: int) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


def load_image_for_inference(image_path: str, image_size: int) -> np.ndarray:
    img_bytes = tf.io.read_file(image_path)
    img = decode_and_resize(img_bytes, image_size)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runs/age_run_1/checkpoints/age_stage2_best.keras",
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image for prediction",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size used during training",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)

    x = load_image_for_inference(args.image, args.image_size)

    pred = model.predict(x, verbose=0)

    if pred.ndim == 2 and pred.shape[1] == 1:
        predicted_age = float(pred[0][0])
    else:
        predicted_age = float(np.squeeze(pred))

    print(f"Predicted age: {predicted_age:.2f}")


if __name__ == "__main__":
    main()