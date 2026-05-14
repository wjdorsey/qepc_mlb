from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import os


def detect_project_root(start: Optional[Path] = None) -> Path:
    """Best-effort project root detection.

    Walks upward from the provided path (or current working directory) looking for
    a likely project marker. Falls back to cwd.
    """
    start_path = Path(start or Path.cwd()).resolve()
    markers = {
        ".git",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "main.py",
    }
    for candidate in [start_path, *start_path.parents]:
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return start_path


PROJECT_ROOT = detect_project_root()


@dataclass(frozen=True)
class MLBPaths:
    project_root: Path = PROJECT_ROOT

    @property
    def package_root(self) -> Path:
        return self.project_root / "qepc_mlb"

    @property
    def cache_root(self) -> Path:
        return self.project_root / "cache" / "mlb"

    @property
    def raw_root(self) -> Path:
        return self.cache_root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.cache_root / "processed"

    @property
    def ids_root(self) -> Path:
        return self.cache_root / "ids"

    @property
    def artifacts_root(self) -> Path:
        return self.project_root / "artifacts" / "mlb"

    @property
    def models_root(self) -> Path:
        return self.artifacts_root / "models"

    @property
    def predictions_root(self) -> Path:
        return self.artifacts_root / "predictions"

    @property
    def reports_root(self) -> Path:
        return self.artifacts_root / "reports"

    def ensure(self) -> None:
        for path in [
            self.package_root,
            self.cache_root,
            self.raw_root,
            self.processed_root,
            self.ids_root,
            self.artifacts_root,
            self.models_root,
            self.predictions_root,
            self.reports_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)


PATHS = MLBPaths()


TEAM_ALIASES: Dict[str, str] = {
    "diamondbacks": "ARI",
    "arizona diamondbacks": "ARI",
    "braves": "ATL",
    "atlanta braves": "ATL",
    "orioles": "BAL",
    "baltimore orioles": "BAL",
    "red sox": "BOS",
    "boston red sox": "BOS",
    "cubs": "CHC",
    "chicago cubs": "CHC",
    "white sox": "CWS",
    "chicago white sox": "CWS",
    "reds": "CIN",
    "cincinnati reds": "CIN",
    "guardians": "CLE",
    "cleveland guardians": "CLE",
    "rockies": "COL",
    "colorado rockies": "COL",
    "tigers": "DET",
    "detroit tigers": "DET",
    "astros": "HOU",
    "houston astros": "HOU",
    "royals": "KC",
    "kansas city royals": "KC",
    "angels": "LAA",
    "los angeles angels": "LAA",
    "dodgers": "LAD",
    "los angeles dodgers": "LAD",
    "marlins": "MIA",
    "miami marlins": "MIA",
    "brewers": "MIL",
    "milwaukee brewers": "MIL",
    "twins": "MIN",
    "minnesota twins": "MIN",
    "mets": "NYM",
    "new york mets": "NYM",
    "yankees": "NYY",
    "new york yankees": "NYY",
    "athletics": "ATH",
    "a's": "ATH",
    "oakland athletics": "ATH",
    "phillies": "PHI",
    "philadelphia phillies": "PHI",
    "pirates": "PIT",
    "pittsburgh pirates": "PIT",
    "padres": "SD",
    "san diego padres": "SD",
    "giants": "SF",
    "san francisco giants": "SF",
    "mariners": "SEA",
    "seattle mariners": "SEA",
    "cardinals": "STL",
    "st. louis cardinals": "STL",
    "rays": "TB",
    "tampa bay rays": "TB",
    "rangers": "TEX",
    "texas rangers": "TEX",
    "blue jays": "TOR",
    "toronto blue jays": "TOR",
    "nationals": "WSH",
    "washington nationals": "WSH",
}


def normalize_team_name(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) in (2, 3, 4) and text.upper() in set(TEAM_ALIASES.values()):
        return text.upper()
    key = text.lower().replace(".", "")
    return TEAM_ALIASES.get(key, text.upper())


DEFAULT_SEASON = int(os.environ.get("QEPC_MLB_DEFAULT_SEASON", "2025"))
RANDOM_SEED = int(os.environ.get("QEPC_MLB_RANDOM_SEED", "42"))
MIN_GAMES_FOR_MODEL = int(os.environ.get("QEPC_MLB_MIN_GAMES_FOR_MODEL", "8"))


if __name__ == "__main__":
    PATHS.ensure()
    print(f"PROJECT_ROOT={PATHS.project_root}")
    print(f"CACHE_ROOT={PATHS.cache_root}")
    print(f"ARTIFACTS_ROOT={PATHS.artifacts_root}")
