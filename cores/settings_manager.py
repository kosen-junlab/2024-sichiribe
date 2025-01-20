"""設定ファイルの管理機能"""

from pathlib import Path
from cores.export_utils import get_supported_formats
from cores.common import filter_dict, is_directory_writable
from typing import Dict, Any, Callable, Union
from platformdirs import user_data_dir
import json
import logging

logger = logging.getLogger("__main__").getChild(__name__)


class SettingsManager:
    """設定ファイルの読み込み、保存、検証を行うクラス"""

    def __init__(self, pattern: str) -> None:
        self.required_keys = self._get_required_keys(pattern)
        self.default_path = self._get_default_setting_path(pattern)

    def _get_required_keys(self, pattern: str) -> Dict[str, Callable[[Any], bool]]:
        """設定ファイルに必要なキーとその検証関数を取得する

        Args:
            pattern (str): "live" or "replay"

        Raises:
            ValueError: patternが"live"または"replay"でない場合

        Returns:
            Dict[str, Callable[[Any], bool]]: 必要なキーとその検証関数
        """
        base_settings = {
            "num_digits": lambda x: isinstance(x, int) and x >= 1,
            "sampling_sec": lambda x: isinstance(x, int) and x >= 1,
            "batch_frames": lambda x: isinstance(x, int) and x >= 1,
            "format": lambda x: x in get_supported_formats(),
            "save_frame": lambda x: isinstance(x, bool),
            "out_dir": lambda x: is_directory_writable(Path(x).parent),
            "click_points": lambda x: isinstance(x, list),
        }
        if pattern == "live":
            additional_settings = {
                "device_num": lambda x: isinstance(x, int) and x >= 0,
                "total_sampling_sec": lambda x: isinstance(x, int) and x >= 1,
                "cap_size": lambda x: isinstance(x, list) or isinstance(x, tuple),
            }
        elif pattern == "replay":
            additional_settings = {
                "video_path": lambda x: isinstance(x, str) and Path(x).exists(),
                "video_skip_sec": lambda x: isinstance(x, int) and x >= 0,
            }
        else:
            raise ValueError(f"Invalid pattern: {pattern}")
        return {**base_settings, **additional_settings}

    def _get_default_setting_path(self, pattern: str) -> Path:
        """設定ファイルのデフォルトパスを取得する"""
        appname = "sichiribe"
        appauthor = "EbinaKai"
        user_dir = user_data_dir(appname, appauthor)

        if pattern == "live":
            return Path(user_dir) / "live_settings.json"
        elif pattern == "replay":
            return Path(user_dir) / "replay_settings.json"
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

    def load_default(self) -> Dict[str, Any]:
        """デフォルトの設定ファイルを読み込む

        Returns:
            Dict[str, Any]: 設定ファイルの内容
        """
        try:
            return self.load(self.default_path)
        except FileNotFoundError:
            logger.warning(
                f"Default settings file not found. Creating a new one at {self.default_path}."
            )
            initial_settings = {key: None for key in self.required_keys}
            self.save(initial_settings)  # 初期設定を保存
            return initial_settings

    def load(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """設定ファイルを読み込む

        Args:
            filepath (Union[str, Path]): 設定ファイルのパス

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            TypeError: ファイルの内容が辞書でない場合
            KeyError: 必要なキーが存在しない場合

        Returns:
            Dict[str, Any]: 設定ファイルの内容
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as file:
            settings = json.load(file)
            if not isinstance(settings, dict):
                raise TypeError(f"Data in {filepath} is not a dictionary")

            for key in self.required_keys:
                if key not in settings:
                    raise KeyError(f"Key '{key}' not found in {filepath}")

        return self.remove_non_require_keys(settings)

    def validate(self, settings: Dict[str, Any]) -> bool:
        """設定ファイルの内容を検証する

        Args:
            settings (Dict[str, Any]): 設定ファイルの内容

        Returns:
            bool: 検証結果
        """
        for key, rules in self.required_keys.items():
            value = settings.get(key)
            if value is None:
                logger.error(f"Missing key: {key}")
                return False
            if not rules(value):
                logger.error(f"Invalid value: {key}={value}")
                return False
        return True

    def remove_non_require_keys(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """設定ファイルの不要なキーを削除する

        Args:
            settings (Dict[str, Any]): 設定ファイルの内容

        Returns:
            Dict[str, Any]: 不要なキーを削除した設定ファイルの内容
        """
        return filter_dict(settings, lambda k, _: k in self.required_keys)

    def save(self, settings: Dict[str, Any]) -> None:
        """設定ファイルを保存する

        Args:
            settings (Dict[str, Any]): 保存する設定ファイルの内容

        Raises:
            ValueError: 検証に失敗した場合
        """
        if not self.validate(settings):
            raise ValueError("Invalid settings")
        settings = self.remove_non_require_keys(settings)
        with open(self.default_path, "w") as file:
            json.dump(settings, file, indent=4)
