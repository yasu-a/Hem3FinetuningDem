# Adapted from THeatA for the bundled workbench.
"""Pauli-twirled error probability lookup for MS gates.

The bundled CSV tabulates the master-equation and Pauli transfer matrix calculation.

  n_bar  ∈ {0.01, 0.25, 0.5, 0.75, ..., 20.0}   (81 values)
  pauli  ∈ {II, IX, IY, IZ, XI, XX, ..., ZZ}    (16 labels, II = no-error)

各 (n_bar, pauli) で 1 行、計 81*16=1296 行。

API:
  - `PauliErrorLookup(csv_path)`: load table
  - `lookup.lookup(n_bar, mode)`: return 15 probabilities in Stim PAULI_CHANNEL_2 order

mode:
  - "strict"     : ceil  (n_bar_table_low ≤ n_bar ≤ n_bar_table_high のとき high 側を採用 = よりエラー高め)
  - "optimistic" : floor (low 側を採用 = エラー低め)
  - "round_off"  : nearest (最近接)

n_bar が table の範囲外の場合:
  - n_bar < min: 最小キー (0.01) を採用
  - n_bar > max: 最大キー (20.0) を採用 + warning（strict として safe な動作）
"""

import csv
import warnings
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, List, Optional

# Stim PAULI_CHANNEL_2 の引数順 (II 除外、15 個)
STIM_PAULI_ORDER: List[str] = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]

DEFAULT_CSV_PATH = Path(__file__).parent / "pauli_twirled_error_probabilities.csv"


class PauliErrorLookup:
    """Pauli-twirled error probability lookup table for 2-qubit MS gates."""

    SUPPORTED_MODES = ("strict", "optimistic", "round_off")

    def __init__(self, csv_path: Optional[str | Path] = None):
        path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
        if not path.exists():
            raise FileNotFoundError(f"Lookup table CSV not found: {path}")
        self.csv_path = path
        # _table[n_bar][pauli] = probability
        self._table: Dict[float, Dict[str, float]] = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n = float(row["n_bar"])
                pauli = row["pauli"]
                p = float(row["probability"])
                self._table.setdefault(n, {})[pauli] = p

        self._n_keys: List[float] = sorted(self._table.keys())
        if not self._n_keys:
            raise ValueError(f"Lookup table is empty: {path}")
        self._n_min = self._n_keys[0]
        self._n_max = self._n_keys[-1]

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    @property
    def n_bar_range(self) -> tuple:
        return (self._n_min, self._n_max)

    def select_n_bar(self, n_bar: float, mode: str = "strict",
                     warn_on_clip: bool = True) -> float:
        """与えられた n_bar に対し、 mode に従って table のキー値を 1 つ選ぶ。

        warn_on_clip=False で table 上限超過時の RuntimeWarning を抑制する
        (クリップ動作自体は変わらない)。
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}; supported: {self.SUPPORTED_MODES}"
            )

        # 範囲外クリップ
        if n_bar <= self._n_min:
            return self._n_min
        if n_bar >= self._n_max:
            if n_bar > self._n_max and warn_on_clip:
                warnings.warn(
                    f"n_bar={n_bar:.4f} exceeds lookup table max {self._n_max}; "
                    f"clipping to {self._n_max}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self._n_max

        # table 内: 上下のキーを取得
        keys = self._n_keys
        idx_right = bisect_left(keys, n_bar)  # n_bar 以上の最初の index
        n_high = keys[idx_right]
        if n_high == n_bar:
            return n_high
        n_low = keys[idx_right - 1]

        if mode == "strict":
            return n_high
        if mode == "optimistic":
            return n_low
        # round_off
        if abs(n_bar - n_low) <= abs(n_high - n_bar):
            return n_low
        return n_high

    def lookup(
        self,
        n_bar: float,
        mode: str = "strict",
        warn_on_clip: bool = True,
    ) -> List[float]:
        """n_bar に対応する 15 Pauli 確率を Stim PAULI_CHANNEL_2 順で返す。

        Returns:
            [p_IX, p_IY, p_IZ, p_XI, p_XX, p_XY, p_XZ,
             p_YI, p_YX, p_YY, p_YZ, p_ZI, p_ZX, p_ZY, p_ZZ]
        """
        selected_n = self.select_n_bar(n_bar, mode=mode, warn_on_clip=warn_on_clip)
        row = self._table[selected_n]
        return [row[p] for p in STIM_PAULI_ORDER]

    def lookup_dict(
        self,
        n_bar: float,
        mode: str = "strict",
        warn_on_clip: bool = True,
    ) -> Dict[str, float]:
        """同じ lookup を dict 形式で返す (II も含む)。デバッグ・確認用。"""
        selected_n = self.select_n_bar(n_bar, mode=mode, warn_on_clip=warn_on_clip)
        return dict(self._table[selected_n])

    def __repr__(self) -> str:
        return (
            f"PauliErrorLookup(path={self.csv_path!s}, "
            f"n_bar range=[{self._n_min}, {self._n_max}], "
            f"n_keys={len(self._n_keys)}, paulis={len(STIM_PAULI_ORDER)+1})"
        )
