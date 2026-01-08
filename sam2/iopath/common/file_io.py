from __future__ import annotations

import os
import shutil


class _PathManager:
    def open(self, path: str, mode: str = "r", buffering: int = -1):
        return open(path, mode=mode, buffering=buffering)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def isdir(self, path: str) -> bool:
        return os.path.isdir(path)

    def isfile(self, path: str) -> bool:
        return os.path.isfile(path)

    def mkdirs(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def copy(self, src: str, dst: str) -> str:
        return shutil.copyfile(src, dst)

    def rm(self, path: str) -> None:
        os.remove(path)

    def mv(self, src: str, dst: str) -> bool:
        shutil.move(src, dst)
        return True


g_pathmgr = _PathManager()
