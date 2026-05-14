from __future__ import annotations


def can_pair(a: str, b: str, allow_wobble: bool = False) -> bool:
    canonical = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
    wobble = {("G", "U"), ("U", "G")}
    pair = (a.upper(), b.upper())
    return pair in canonical or (allow_wobble and pair in wobble)


def nussinov(
    sequence: str,
    min_loop_length: int = 0,
    allow_wobble: bool = False,
) -> tuple[str, set[tuple[int, int]]]:
    sequence = sequence.upper()
    n = len(sequence)
    if n == 0:
        return "", set()

    dp = [[0] * n for _ in range(n)]
    backtrack: list[list[tuple | None]] = [[None] * n for _ in range(n)]

    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            best = dp[i + 1][j]
            choice: tuple | None = ("skip_i", i + 1, j)

            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
                choice = ("skip_j", i, j - 1)

            if j - i > min_loop_length and can_pair(sequence[i], sequence[j], allow_wobble):
                value = 1 + (dp[i + 1][j - 1] if i + 1 <= j - 1 else 0)
                if value > best:
                    best = value
                    choice = ("pair", i + 1, j - 1)

            for k in range(i + 1, j):
                value = dp[i][k] + dp[k + 1][j]
                if value > best:
                    best = value
                    choice = ("split", i, k, k + 1, j)

            dp[i][j] = best
            backtrack[i][j] = choice

    pairs: set[tuple[int, int]] = set()

    def trace(i: int, j: int) -> None:
        if i >= j:
            return
        choice = backtrack[i][j]
        if choice is None:
            return
        if choice[0] in {"skip_i", "skip_j"}:
            trace(choice[1], choice[2])
        elif choice[0] == "pair":
            pairs.add((i, j))
            trace(choice[1], choice[2])
        elif choice[0] == "split":
            _, i1, k1, i2, j2 = choice
            trace(i1, k1)
            trace(i2, j2)

    trace(0, n - 1)

    dot_bracket = ["." for _ in range(n)]
    for i, j in pairs:
        dot_bracket[i] = "("
        dot_bracket[j] = ")"

    return "".join(dot_bracket), pairs
