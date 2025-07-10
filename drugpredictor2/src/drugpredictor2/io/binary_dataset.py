from kedro.io import AbstractDataSet
from typing import Any
import fsspec


class BinaryDataSet(AbstractDataSet[bytes, bytes]):
    def __init__(self, filepath: str):
        self._filepath = filepath
        self._fs, self._path = fsspec.core.url_to_fs(self._filepath)

    def _load(self) -> bytes:
        with self._fs.open(self._path, mode="rb") as f:
            return f.read()

    def _save(self, data: bytes) -> None:
        with self._fs.open(self._path, mode="wb") as f:
            f.write(data)

    def _describe(self) -> dict[str, Any]:
        return dict(filepath=self._filepath)
