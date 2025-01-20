"""システムからのログ出力を画面に表示する機能"""

from PySide6.QtWidgets import QVBoxLayout, QTextEdit
from PySide6.QtCore import Signal, QObject
from gui.widgets.custom_qwidget import CustomQWidget
from gui.utils.screen_manager import ScreenManager
import logging


class LogEmitter(QObject):
    """ログをメインスレッドに送信するためのクラス"""

    new_log = Signal(str)


class QTextEditLogger(logging.Handler):
    """ログをQTextEditに表示するためのハンドラ

    スレッド間通信のためにQObjectを継承している
    """

    def __init__(self, emitter: LogEmitter) -> None:
        super().__init__()
        self.emitter = emitter

    def emit(self, record) -> None:
        log_entry = self.format(record)
        self.emitter.new_log.emit(log_entry)


class LogWindow(CustomQWidget):
    """
    ログウィンドウのViewクラス

    1. 黒い背景に白い文字でログを表示するやつ
    2. ログはログハンドラを使って受け取ることで、別スレッドで実行中のワーカーからもログを受け取れる
    3. 処理中で、表示するコンテンツがない場合に使う
    """

    def __init__(self, screen_manager: ScreenManager) -> None:
        self.screen_manager = screen_manager

        super().__init__()
        screen_manager.add_screen("log", self, "ログ")

    def initUI(self):
        """UIの初期化"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # ログエミッタを作成し、ログハンドラを設定
        self.emitter = LogEmitter()
        self.logger = logging.getLogger("__main__")
        log_handler = QTextEditLogger(self.emitter)
        self.logger.addHandler(log_handler)

        # シグナルとスロットの接続
        self.emitter.new_log.connect(self.append_log)

    def display(self):
        """画面表示時の処理

        画面表示時にログをクリアする
        """
        self.log_display.clear()

    def append_log(self, message: str) -> None:
        """ログを画面に追加する"""
        self.log_display.append(message)
