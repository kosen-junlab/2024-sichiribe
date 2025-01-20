"""エクスポート機能"""

import logging
import json
import csv
from itertools import zip_longest
from typing import List, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger("__main__").getChild(__name__)


def get_supported_formats() -> List[str]:
    """サポートされているエクスポートフォーマットを取得する

    Returns:
        List[str]: サポートされているエクスポートフォーマット
    """
    return ["csv", "json"]


def export(
    data: Union[List, Dict], format: str, out_dir: Union[str, Path], prefix: str
) -> None:
    """データをエクスポートする

    Args:
        data (Union[List, Dict]): エクスポートするデータ
        format (str): エクスポートフォーマット
        out_dir (Union[str, Path]): 出力ディレクトリ
        prefix (str): 出力ファイル名のプレフィックス
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if format == "csv":
        if isinstance(data, dict):
            data = [data]
        to_csv(data, out_dir, prefix)
    elif format == "json":
        to_json(data, out_dir, prefix)
    elif format == "dummy":
        return
    else:
        raise ValueError("Invalid export method.")


def to_json(data: Union[Dict, List], out_dir: Union[str, Path], prefix: str) -> None:
    """JSON形式でデータをエクスポートする

    Args:
        data (Union[Dict, List]): エクスポートするデータ
        out_dir (Union[str, Path]): 出力ディレクトリ
        prefix (str): 出力ファイル名のプレフィックス
    """
    out_path = Path(out_dir) / f"{prefix}.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    logger.debug("Exported data to json.")


def to_csv(data: List[Dict], out_dir: Union[str, Path], prefix: str) -> None:
    """CSV形式でデータをエクスポートする

    Args:
        data (List[Dict]): エクポートするデータ
        out_dir (Union[str, Path]): 出力ディレクトリ
        prefix (str): 出力ファイル名のプレフィックス
    """
    out_path = Path(out_dir) / f"{prefix}.csv"
    keys = data[0].keys()
    with open(out_path, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    logger.debug("Exported data to csv.")


def build_data_records(data_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """データをレコード形式に変換する

    Args:
        data_dict (Dict[str, List[Any]]): データ

    Returns:
        List[Dict[str, Any]]: レコード形式のデータ

    Example:
        ```python
        data_dict = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [20, 25, 30],
        }

        build_data_records(data_dict)
        # => [
        #      {"name": "Alice", "age": 20},
        #      {"name": "Bob", "age": 25},
        #      {"name": "Charlie", "age": 30},
        #    ]
        ```
    """
    records = []
    field_names = list(data_dict.keys())
    data_lists = list(data_dict.values())

    for values in zip_longest(*data_lists):
        record = {field: value for field, value in zip(field_names, values)}
        records.append(record)

    return records
