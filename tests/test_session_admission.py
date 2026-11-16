"""Admission control without a GPU: a one-off prefix cannot evict one that is reused.

    python tests/test_session_admission.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flide.runtime.session as session
from flide.runtime.session import Entry, SessionCache


def entry(tokens, size=64):
    return Entry(tokens, {pos: {"S": torch.zeros(size)} for pos in (len(tokens) - 1,)}, {})


def main():
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'' if ok else f' ({detail})'}")

    old_room, old_seen = session.ADMIT_ROOM, session.SEEN_MAX
    session.ADMIT_ROOM, session.SEEN_MAX = 0.3, 1 << 16   # room for one 256-byte entry of 1200, not two
    try:
        cache = SessionCache(budget_bytes=1200, min_tokens=1, stride=8)
        first = [1] * 20
        second = [2] * 20
        cache.insert(entry(first))
        check("a prefix is kept while there is room", len(cache.entries) == 1, len(cache.entries))

        one_off = entry(second)
        cache.insert(one_off)
        check("a one-off prefix is not kept once the cache is past its room", len(cache.entries) == 1,
              [(len(e.tokens), e.tokens[:1]) for e in cache.entries.values()])

        cache.insert(entry(second))
        check("the same prefix is kept when it is seen a second time", len(cache.entries) == 2, len(cache.entries))

        third = [3] * 24
        cache.insert(entry(third), parent=next(iter(cache.entries.values())))
        check("an extension of a cached entry is kept", len(cache.entries) == 3, len(cache.entries))

        session.SEEN_MAX = 2
        before = len(cache.seen)
        for i in range(5):
            cache.insert(entry([10 + i] * 20))
        check("the seen set is bounded", len(cache.seen) <= 4, len(cache.seen))
    finally:
        session.ADMIT_ROOM, session.SEEN_MAX = old_room, old_seen

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()