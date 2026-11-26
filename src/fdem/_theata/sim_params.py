# Adapted from THeatA for the bundled workbench.
"""Simulation parameter loading — 時間 / フォノン / cooling パラメータの一元管理。

正本 (single source of truth) は estimator.TimeParameters の dataclass デフォルト。
このモジュールはそれを
  - パラメータファイル (JSON / YAML) で部分上書きして読み込む (load_time_parameters)
  - QCCDState.phonon_params 形式の dict へ変換する (phonon_params_from)
  - 実験条件の記録用にファイルへダンプする (dump_time_parameters)
する機能を提供する。

Why this module exists:
  従来デフォルト値が per_liz_compiler.DEFAULT_PHONON_PARAMS /
  qccd_compiler.MovementPlanner 内のインライン dict / TimeParameters の
  3 箇所に重複していた。外部ファイルからの上書きを可能にするにあたり、
  正本を TimeParameters に一本化し、他はここを経由して導出する。

パラメータファイルの形式: TimeParameters のフィールド名をキーとする flat dict。
部分指定可 (未指定キーはデフォルト値)。未知キーはエラー (typo 検出)。

    {
      "ms_gate": 100.0,
      "cool": 300.0,
      "phonon_cool_residual_axial": 0.1
    }
"""

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional, Union

from .estimator import TimeParameters

PathLike = Union[str, Path]


def load_time_parameters(path: Optional[PathLike] = None,
                         overrides: Optional[dict] = None) -> TimeParameters:
    """パラメータファイル + dict override から TimeParameters を構築する。

    優先順位: overrides > ファイル > dataclass デフォルト。

    Args:
        path: JSON (.json) / YAML (.yaml/.yml, 要 pyyaml) ファイル。None ならデフォルト。
        overrides: 個別上書き dict (CLI 引数からの注入用)。

    Raises:
        ValueError: 未知のパラメータキーが含まれる場合。
    """
    values: dict = {}
    if path is not None:
        p = Path(path)
        text = p.read_text(encoding='utf-8')
        if p.suffix in ('.yaml', '.yml'):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "YAML パラメータファイルには pyyaml が必要です "
                    "(pip install pyyaml)。JSON 形式なら追加依存なしで使えます。")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"パラメータファイルの中身は dict である必要があります: {path}")
        values.update(data)
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    valid_keys = {f.name for f in fields(TimeParameters)}
    unknown = set(values) - valid_keys
    if unknown:
        raise ValueError(
            f"未知のパラメータキー: {sorted(unknown)}\n"
            f"有効なキー: {sorted(valid_keys)}")
    return TimeParameters(**values)


def phonon_params_from(params: Optional[TimeParameters] = None) -> dict:
    """TimeParameters から QCCDState.phonon_params 形式の dict を作る。

    QCCDState 側は歴史的に短いキー名 ('split_axial' 等) を使うため、
    ここで写像する。None を渡すとデフォルト値の dict を返す。
    """
    p = params or TimeParameters()

    def _tr2(value2, value1):
        # *_transversal2 が None のとき第 1 モードと同値 (TimeParameters の規約)
        return value1 if value2 is None else value2

    return {
        'split_axial': p.phonon_split_axial,
        'split_transversal': p.phonon_split_transversal,
        'split_transversal2': _tr2(p.phonon_split_transversal2,
                                   p.phonon_split_transversal),
        'merge_axial': p.phonon_merge_axial,
        'merge_transversal': p.phonon_merge_transversal,
        'merge_transversal2': _tr2(p.phonon_merge_transversal2,
                                   p.phonon_merge_transversal),
        'swap_axial': p.phonon_swap_axial,
        'swap_transversal': p.phonon_swap_transversal,
        'swap_transversal2': _tr2(p.phonon_swap_transversal2,
                                  p.phonon_swap_transversal),
        'shuttle_axial_per_180um': p.phonon_shuttle_axial_per_180um,
        'shuttle_transversal': p.phonon_shuttle_transversal,
        'shuttle_transversal2': _tr2(p.phonon_shuttle_transversal2,
                                     p.phonon_shuttle_transversal),
        'shuttle_base_distance': p.shuttle_base_distance,
        'shuttle_ion_factor': p.shuttle_ion_factor,
        'cool_residual_axial': p.phonon_cool_residual_axial,
        'cool_residual_transversal': p.phonon_cool_residual_transversal,
        'cool_residual_transversal2': _tr2(p.phonon_cool_residual_transversal2,
                                           p.phonon_cool_residual_transversal),
        'heating_rate_axial_per_s': p.heating_rate_axial_per_s,
        'heating_rate_transversal_per_s': p.heating_rate_transversal_per_s,
        'heating_rate_transversal2_per_s': _tr2(
            p.heating_rate_transversal2_per_s,
            p.heating_rate_transversal_per_s),
    }


def dump_time_parameters(params: TimeParameters, path: PathLike) -> None:
    """全パラメータを JSON へ保存する (実験条件の記録・再現用)。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(params), indent=2) + "\n", encoding='utf-8')
