"""CNNを使用した7セグメント数字認識機能"""

from cores.detector import Detector
import os
import logging
from typing import Optional, Union, List, Tuple, Any
import cv2
import numpy as np

# TensorFlow と h5py のログを無効にする
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)


class CNNCore(Detector):
    """CNNを使用した7セグメント数字認識のための基底クラス"""

    def __init__(self, num_digits: int) -> None:
        self.num_digits = num_digits
        self.model: Optional[Any]

        self.logger = logging.getLogger("__main__").getChild(__name__)

        # 画像の各種設定
        # 空白の表示に対応させるため、blankのところを「' '」で空白に設定
        self.folder = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ""])
        self.image_width = 100  # 使用する学習済みモデルと同じwidth（横幅）を指定
        self.image_height = 100  # 使用する学習済みモデルと同じheight（縦の高さ）を指定
        self.color_setting = 1  # 学習済みモデルと同じ画像のカラー設定にする。モノクロ・グレースケールの場合は「1」。カラーの場合は「3」
        self.cv2_color_setting = 0  # 同上。cv2.imreadではモノクロ・グレースケールの場合は「0」。カラーの場合は「1」
        self.crop_size = 100  # 画像をトリミングするサイズ

    def inference_7seg_classifier(self, image_bin: np.ndarray) -> List[int]:
        """画像から7セグメント数字を推論する

        Args:
            image_bin (np.ndarray): 推論対象の画像

        Raises:
            NotImplementedError: サブクラスで実装されていない場合

        Returns:
            List[int]: 各桁の推論結果
        """
        raise NotImplementedError("This method must be implemented in the subclass")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """画像を準備する

        各桁を一度に処理できるように画像をトリミングし、リサイズする

        Args:
            image (np.ndarray): 入力画像

        Returns:
            np.ndarray: 画像の準備結果
        """
        images = []
        for index in range(self.num_digits):
            # 画像を一桁にトリミング
            img = image[
                0 : self.crop_size,
                index * self.crop_size : (index + 1) * self.crop_size,
            ]
            img = cv2.resize(img, (self.image_width, self.image_height))
            img_ = (
                img.reshape(
                    self.image_width, self.image_height, self.color_setting
                ).astype("float32")
                / 255
            )
            images.append(img_)
        return np.array(images, dtype=np.float32)

    def predict(
        self,
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        binarize_th: Optional[int] = None,
    ) -> tuple[int, float]:
        """画像のリストから7セグメント数字を推論する

        Args:
            images (Union[str, np.ndarray, List[np.ndarray], List[str]]): 推論対象の画像またはパスのリスト
            binarize_th (Optional[int], optional): 二値化の閾値。Noneの場合は自動設定。デフォルトはNone。

        Returns:
            tuple[int, float]: 推論結果とエラー率
        """

        images_ = images if isinstance(images, list) else [images]

        results = np.zeros((0, self.num_digits))

        for image in images_:

            image_gs = self.load_image(image)

            if image_gs is None:
                self.logger.error("Error: Could not read image file.")
                continue

            image_gs = cv2.resize(
                image_gs, (self.crop_size * self.num_digits, self.crop_size)
            )

            image_bin = self.preprocess_binarization(
                image_gs, binarize_th, output_grayscale=True
            )

            argmax_indices = self.inference_7seg_classifier(image_bin)

            results = np.vstack([results, argmax_indices])

        # 最頻値を取得
        result, errors_per_digit = self.find_mode_per_column_np(results)

        # 最後にラベルに対応させる
        result_digits = self.folder[result.astype(int)]
        result_str = "".join(result_digits)
        result_int = int(result_str) if result_str != "" else 0

        failed_rate = np.mean(errors_per_digit)

        self.logger.debug("Detected label: %s", result)
        self.logger.debug("Error: %s", failed_rate)
        return result_int, failed_rate

    def find_mode_per_column_np(
        self, predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """各列の最頻値を取得する

        Args:
            predictions (np.ndarray): 推論結果の配列

        Returns:
            Tuple[np.ndarray, np.ndarray]: 最頻値の配列と各桁のエラー率の配列
        """
        result = []
        errors_per_digit = []
        for i in range(predictions.shape[1]):  # 各桁に対して
            digit_predictions = predictions[:, i]  # 各列を抽出

            values, counts = np.unique(digit_predictions, return_counts=True)
            mode_value = values[np.argmax(counts)]  # 最も頻出する値を取得

            incorrect_predictions = np.sum(digit_predictions != mode_value)
            error_rate = incorrect_predictions / len(digit_predictions)

            result.append(mode_value)
            errors_per_digit.append(error_rate)
        return np.array(result), np.array(errors_per_digit)


def cnn_init(num_digits: int, model_filename: Optional[str] = None) -> CNNCore:
    """インストールされているライブラリに応じてCNNモデルを選択する

    Args:
        num_digits (int): 推論する桁数
        model_filename (Optional[str], optional): モデルファイル名。Noneの場合はデフォルトのモデルを使用。デフォルトはNone。
    """
    logger = logging.getLogger("__main__").getChild(__name__)

    try:
        from cores.cnn_tflite import CNNLite

        logger.info("TensorFlow Lite Runtime detected. Using TFLite model.")

        model_filename = (
            "model_100x100.tflite" if model_filename is None else model_filename
        )
        return CNNLite(num_digits=num_digits, model_filename=model_filename)
    except ImportError:
        logger.debug(
            "TensorFlow Lite runtime not found. Attempting to import TensorFlow."
        )

    try:
        from cores.cnn_tf import CNNTf

        logger.info("TensorFlow detected. Using Keras model.")
        model_filename = (
            "model_100x100.keras" if model_filename is None else model_filename
        )
        return CNNTf(num_digits=num_digits, model_filename=model_filename)
    except ImportError:
        logger.debug("TensorFlow not found. Attempting to import ONNX Runtime.")

    try:
        from cores.cnn_onnx import CNNOnnx

        logger.info("ONNX Runtime detected. Using ONNX model.")
        model_filename = (
            "model_100x100.onnx" if model_filename is None else model_filename
        )
        return CNNOnnx(num_digits=num_digits, model_filename=model_filename)
    except ImportError:
        logger.error(
            "No compatible machine learning library found. Cannot select a model."
        )
        raise ImportError(
            "No compatible model library found. Please install TensorFlow, TensorFlow Lite, or ONNX Runtime."
        )
