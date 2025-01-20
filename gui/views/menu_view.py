"""アプリケーション起動後に表示されるメニュー画面"""

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt
from gui.widgets.custom_qwidget import CustomQWidget
from gui.utils.screen_manager import ScreenManager
import logging


class MenuWindow(CustomQWidget):
    """
    アプリケーション起動後に表示されるメニュー画面のViewクラス

    1. リアルタイム処理とファイル読み込み処理の選択ができる
    2. 処理が終わった場合はこの画面に戻って来る
    """

    def __init__(self, screen_manager: ScreenManager) -> None:
        self.logger = logging.getLogger("__main__").getChild(__name__)
        self.screen_manager = screen_manager

        super().__init__()
        screen_manager.add_screen("menu", self, "")

    def initUI(self):
        """UIの初期化"""
        main_layout = QVBoxLayout()
        button_layout = QVBoxLayout()
        footer_layout = QHBoxLayout()
        self.setLayout(main_layout)

        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.live_button = QPushButton("ライブカム")
        self.live_button.setFixedHeight(50)
        self.live_button.setFixedWidth(200)
        self.live_button.clicked.connect(
            lambda: self.screen_manager.show_screen("live_setting")
        )
        button_layout.addWidget(self.live_button)

        self.replay_button = QPushButton("ファイル読み込み")
        self.replay_button.setFixedHeight(50)
        self.replay_button.setFixedWidth(200)
        self.replay_button.clicked.connect(
            lambda: self.screen_manager.show_screen("replay_setting")
        )
        button_layout.addWidget(self.replay_button)

        self.quit_button = QPushButton("終了")
        self.quit_button.setFixedWidth(100)
        self.quit_button.clicked.connect(lambda: self.screen_manager.quit())
        footer_layout.addWidget(self.quit_button)
        footer_layout.addStretch()

        main_layout.addStretch()
        main_layout.addLayout(button_layout)
        main_layout.addStretch()
        main_layout.addLayout(footer_layout)

    def display(self):
        """画面表示時の処理

        画面間共有データストアをクリアする
        """
        self.data_store.clear()
