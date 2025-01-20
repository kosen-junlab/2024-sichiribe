"""リアルタイム解析の実行画面"""

from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QSlider,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from gui.widgets.custom_qwidget import CustomQWidget
from gui.widgets.mpl_canvas_widget import MplCanvas
from gui.utils.screen_manager import ScreenManager
from gui.utils.common import convert_cv_to_qimage
from gui.utils.exporter import export_result, export_settings
from gui.workers.live_detect_worker import DetectWorker
from cores.settings_manager import SettingsManager
from cores.frame_editor import FrameEditor
import logging
from typing import List, Optional
import numpy as np
from datetime import timedelta


class LiveExeWindow(CustomQWidget):
    """リアルタイム解析を行うViewクラス"""

    def __init__(self, screen_manager: ScreenManager) -> None:
        self.logger = logging.getLogger("__main__").getChild(__name__)
        self.settings_manager = SettingsManager("live")
        self.screen_manager = screen_manager
        self.results: List[int]
        self.failed_rates: List[float]
        self.timestamps: List[str]
        self.graph_results: List[int]
        self.graph_failed_rates: List[float]
        self.graph_timestamps: List[str]
        self.worker: Optional[DetectWorker] = None

        super().__init__()
        screen_manager.add_screen("live_exe", self, "ライブ解析中")

    def initUI(self):
        """ウィジェットのUIを初期化する"""
        main_layout = QVBoxLayout()
        graph_layout = QVBoxLayout()
        extracted_image_layout = QHBoxLayout()
        form_layout = QFormLayout()
        footer_layout = QHBoxLayout()
        self.setLayout(main_layout)

        graph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        extracted_image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.graph_label = MplCanvas()
        graph_layout.addWidget(self.graph_label)

        self.extracted_label = QLabel()
        self.extracted_label.setMinimumHeight(100)
        extracted_image_layout.addWidget(self.extracted_label)

        slider_layout = QHBoxLayout()
        self.binarize_th = QSlider()
        self.binarize_th.setFixedWidth(200)
        self.binarize_th.setRange(0, 255)
        self.binarize_th.setOrientation(Qt.Orientation.Horizontal)
        self.binarize_th.valueChanged.connect(self.update_binarize_th)
        self.binarize_th_label = QLabel()
        slider_layout.addWidget(self.binarize_th)
        slider_layout.addWidget(self.binarize_th_label)
        form_layout.addRow("画像二値化しきい値：", slider_layout)

        self.graph_clear_button = QPushButton("グラフクリア")
        self.graph_clear_button.setFixedWidth(100)
        self.graph_clear_button.clicked.connect(self.graph_clear)

        self.term_label = QLabel()
        self.term_label.setStyleSheet("color: red")
        self.remaining_time_label = QLabel("ここに残り時間を表示")
        self.term_button = QPushButton("途中終了")
        self.term_button.setFixedWidth(100)
        self.term_button.clicked.connect(self.cancel)

        footer_layout.addWidget(self.graph_clear_button)
        footer_layout.addStretch()
        footer_layout.addWidget(self.term_label)
        footer_layout.addWidget(self.remaining_time_label)
        footer_layout.addWidget(self.term_button)

        main_layout.addLayout(graph_layout)
        main_layout.addLayout(extracted_image_layout)
        main_layout.addLayout(form_layout)
        main_layout.addStretch()
        main_layout.addLayout(footer_layout)

    def trigger(self, action, *args):
        """
        ウィジェットのアクションをトリガーする

        Args:
            action (str): 実行するアクションの名前

        Raises:
            ValueError: actionが不正な場合
        """
        if action == "startup":
            self.startup(*args)
        else:
            raise ValueError(f"Invalid action: {action}")

    def cancel(self) -> None:
        """解析を中止する"""
        if self.worker is not None:
            self.term_label.setText("中止中...")
            self.worker.cancel()

    def update_binarize_th(self, value: Optional[int]) -> None:
        """
        画像二値化しきい値を更新する
        しきい値が0の場合は自動設定にする

        Args:
            value (Optional[int]): 画像二値化しきい値
        """
        value = None if value == 0 else value
        binarize_th_str = "自動設定" if value is None else str(value)
        self.binarize_th_label.setText(binarize_th_str)
        if self.worker is not None:
            self.worker.update_binarize_th(value)

    def graph_clear(self) -> None:
        """グラフをクリアする"""
        self.graph_results = []
        self.graph_failed_rates = []
        self.graph_timestamps = []
        self.update_graph(self.results[-1], self.failed_rates[-1], self.timestamps[-1])

    def startup(self) -> None:
        """各種初期化処理を行う

        初期化後、推論処理のワーカーを起動する
        """

        self.logger.info("Starting LiveExeWindow.")
        self.screen_manager.show_screen("log")
        settings = self.settings_manager.remove_non_require_keys(
            self.data_store.get_all()
        )
        self.settings_manager.save(settings)
        self.fe = FrameEditor(self.data_store.get("num_digits"))
        p_, s_ = self.screen_manager.save_screen_size()

        self.binarize_th.setValue(0)
        self.binarize_th_label.setText("自動設定")
        self.term_label.setText("")
        self.results = []
        self.failed_rates = []
        self.timestamps = []
        self.graph_results = []
        self.graph_failed_rates = []
        self.graph_timestamps = []

        self.graph_label.gen_graph(
            title="Results",
            xlabel="Timestamp",
            ylabel1="Failed Rate",
            ylabel2="Detected results",
            dark_theme=self.screen_manager.check_if_dark_mode(),
        )

        self.worker = DetectWorker()
        self.worker.ready.connect(lambda: self.screen_manager.show_screen("live_exe"))
        self.worker.progress.connect(self.detect_progress)
        self.worker.send_image.connect(self.display_extract_image)
        self.worker.remaining_time.connect(self.update_remaining_time)
        self.worker.finished.connect(self.detect_finished)
        self.worker.error.connect(lambda msg: self.screen_manager.popup(msg))

        self.worker.start()
        self.logger.info("Detect started.")

    def detect_progress(self, result: int, failed_rate: float, timestamp: str) -> None:
        """解析進捗を表示する

        Args:
            result (int): 推論結果
            failed_rate (float): 失敗率
            timestamp (str): 開始からの経過時間
        """
        self.results.append(result)
        self.failed_rates.append(failed_rate)
        self.timestamps.append(timestamp)
        self.update_graph(result, failed_rate, timestamp)

    def update_graph(self, result: int, failed_rate: float, timestamp: str) -> None:
        """グラフを更新する

        Args:
            result (int): 推論結果
            failed_rate (float): 失敗率
            timestamp (str): 開始からの経過時間
        """
        self.graph_results.append(result)
        self.graph_failed_rates.append(failed_rate)
        self.graph_timestamps.append(timestamp)
        self.graph_label.update_existing_plot(
            self.graph_timestamps, self.graph_failed_rates, self.graph_results
        )

    def display_extract_image(self, image: np.ndarray) -> None:
        """推論した7セグディスプレイを表示する

        Args:
            image (np.ndarray): 抽出画像
        """
        image = self.fe.draw_separation_lines(image)
        q_image = convert_cv_to_qimage(image)
        self.extracted_label.setPixmap(QPixmap.fromImage(q_image))

    def update_remaining_time(self, remaining_time: float) -> None:
        """残り時間を表示する

        Args:
            remaining_time (float): 残り時間
        """
        self.remaining_time_label.setText(str(timedelta(seconds=int(remaining_time))))

    def detect_finished(self) -> None:
        """解析結果の保存し、環境をクリアしたあと、メニュー画面に戻る"""
        self.logger.info("Detect finished.")
        self.data_store.set("results", self.results)
        self.data_store.set("failed_rates", self.failed_rates)
        self.data_store.set("timestamps", self.timestamps)
        self.export_process()
        self.screen_manager.show_screen("menu")
        self.clear_env()

    def export_process(self) -> None:
        """解析結果をエクスポートする"""
        self.logger.info("Data exporting...")
        export_result(self.data_store.get_all())
        export_settings(self.data_store.get_all())
        self.screen_manager.popup(f"保存場所：{self.data_store.get('out_dir')}")

    def clear_env(self) -> None:
        """環境をクリアする"""
        self.graph_label.clear()
        self.extracted_label.clear()
        self.term_label.setText("")
        self.logger.info("Environment cleared.")
        self.screen_manager.restore_screen_size()
