import numpy as np


UNMATCHED_COST = 1_000_000.0


def hungarian(cost_matrix):
    if cost_matrix.size == 0:
        return []

    transposed = False
    cost = cost_matrix.copy()

    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True

    rows, cols = cost.shape
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(cols + 1, dtype=np.float64)
    p = np.zeros(cols + 1, dtype=np.int32)
    way = np.zeros(cols + 1, dtype=np.int32)

    for row in range(1, rows + 1):
        p[0] = row
        minv = np.full(cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(cols + 1, dtype=bool)
        col0 = 0

        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0

            for col in range(1, cols + 1):
                if used[col]:
                    continue

                cur = cost[row0 - 1, col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col

            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta

            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = []
    for col in range(1, cols + 1):
        if p[col] == 0:
            continue

        row = p[col] - 1
        col_index = col - 1
        if transposed:
            assignment.append((col_index, row))
        else:
            assignment.append((row, col_index))

    return assignment

