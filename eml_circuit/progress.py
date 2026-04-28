from __future__ import annotations

from collections.abc import Iterable, Iterator


try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


class _TqdmFallback:
    def __init__(
        self,
        iterable: Iterable | None = None,
        *,
        total: int | None = None,
        disable: bool = False,
        desc: str | None = None,
        leave: bool = False,
    ) -> None:
        del total, disable, desc, leave
        self._iterable = iterable

    def __iter__(self) -> Iterator:
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def set_postfix(self, *args, **kwargs) -> None:
        del args, kwargs

    def write(self, message: str) -> None:
        print(message)

    def close(self) -> None:
        return None


def maybe_tqdm(
    iterable: Iterable | None = None,
    *,
    total: int | None = None,
    disable: bool = False,
    desc: str | None = None,
    leave: bool = False,
):
    if _tqdm is None:
        return _TqdmFallback(
            iterable,
            total=total,
            disable=disable,
            desc=desc,
            leave=leave,
        )
    return _tqdm(
        iterable,
        total=total,
        disable=disable,
        desc=desc,
        leave=leave,
    )
