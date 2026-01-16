import numpy as np
import json
from typing import Optional, Tuple


class SimplexSolver:
    def __init__(self, c, A, b, constraint_types, maximize=True, eps=1e-9):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.types = constraint_types
        self.maximize = maximize
        self.eps = eps

        self.m, self.n = self.A.shape

    # ======================= PUBLIC =======================

    def solve(self) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        tableau, basis, art_cols = self._build_phase1_tableau()
        z = self._simplex(tableau, basis, phase=1)

        if abs(z) > self.eps:
            return None, None, "infeasible"

        tableau, basis = self._build_phase2_tableau(tableau, basis, art_cols)
        z = self._simplex(tableau, basis, phase=2)

        if z is None:
            return None, None, "unbounded"

        solution = np.zeros(self.n)
        for i, bi in enumerate(basis):
            if bi < self.n:
                solution[bi] = tableau[i, -1]

        return solution, z, "optimal"

    # ======================= TABLEAU =======================

    def _build_phase1_tableau(self):
        A = self.A.copy()
        b = self.b.copy()

        slack = []
        artificial = []

        for i, t in enumerate(self.types):
            if t == "<=":
                col = np.zeros(self.m)
                col[i] = 1
                A = np.column_stack((A, col))
                slack.append(A.shape[1] - 1)
            elif t == ">=":
                col1 = np.zeros(self.m)
                col1[i] = -1
                A = np.column_stack((A, col1))

                col2 = np.zeros(self.m)
                col2[i] = 1
                A = np.column_stack((A, col2))
                artificial.append(A.shape[1] - 1)
            else:
                col = np.zeros(self.m)
                col[i] = 1
                A = np.column_stack((A, col))
                artificial.append(A.shape[1] - 1)

        tableau = np.zeros((self.m + 1, A.shape[1] + 1))
        tableau[:-1, :-1] = A
        tableau[:-1, -1] = b

        for j in artificial:
            tableau[-1, j] = -1

        basis = []
        for i in range(self.m):
            for j in artificial + slack:
                if abs(tableau[i, j] - 1) < self.eps:
                    basis.append(j)
                    break

        for i, bi in enumerate(basis):
            tableau[-1] += tableau[i]

        return tableau, basis, artificial

    def _build_phase2_tableau(self, tableau, basis, art_cols):
        tableau = np.delete(tableau, art_cols, axis=1)

        basis = [b for b in basis if b < tableau.shape[1] - 1]

        c = np.zeros(tableau.shape[1] - 1)
        c[:self.n] = self.c if self.maximize else -self.c

        tableau[-1, :-1] = -c
        tableau[-1, -1] = 0

        for i, bi in enumerate(basis):
            tableau[-1] += tableau[-1, bi] * tableau[i]

        return tableau, basis

    # ======================= SIMPLEX =======================

    def _simplex(self, tableau, basis, phase):
        while True:
            col = np.argmax(tableau[-1, :-1])
            if tableau[-1, col] <= self.eps:
                return tableau[-1, -1]

            ratios = []
            for i in range(len(basis)):
                if tableau[i, col] > self.eps:
                    ratios.append(tableau[i, -1] / tableau[i, col])
                else:
                    ratios.append(np.inf)

            if min(ratios) == np.inf:
                return None

            row = np.argmin(ratios)
            basis[row] = col

            self._pivot(tableau, row, col)

    def _pivot(self, T, r, c):
        T[r] /= T[r, c]
        for i in range(T.shape[0]):
            if i != r:
                T[i] -= T[i, c] * T[r]


# ======================= JSON =======================

def solve_from_file(filename):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    solver = SimplexSolver(
        data["objective"],
        data["constraints_matrix"],
        data["constraints_rhs"],
        data["constraint_types"],
        data["maximize"]
    )

    return solver.solve()
