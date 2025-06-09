"""
I used ChatGPT to assist me while writing this code.

find_decomposition.py — feasibility checker for equal‑tax decompositions
########################################################################
This helper answers a yes/no question that pops up in many participatory‑
budgeting exercises:

    *Given a public budget and citizens’ preferences, can the money be split so
    that* (a) *each topic keeps its allocated amount,* (b) *each citizen pays
    the same share of the total,* **and** (c) *nobody pays for a topic they
    dislike?*

If the answer is *yes* the script produces one concrete split (not
necessarily the fairest or “nicest” one, just a valid witness).  Otherwise
it returns ``None``.

This is a feasibility problem, so we can solve it with a linear program.
"""

from typing import List, Set, Dict, Tuple
import numpy as np
from scipy.optimize import linprog

###############################################################################
# Main routine
###############################################################################

def find_decomposition(budget: List[float],
                       preferences: List[Set[int]]):
    """Attempt to *fraction* the budget under an equal‑tax rule.

    Parameters
    ----------
    budget : list[float]
        ``budget[j]`` is the amount already approved for topic *j*.
    preferences : list[set[int]]
        ``preferences[i]`` is the set of topics citizen *i* agrees to fund.

    Returns
    -------
    list[dict[int, float]] | None
        * A list of length ``n`` (number of citizens).  Entry *i* is a mapping
          *topic → cash* that citizen *i* contributes.  Satisfies all three
          constraints explained in the module docstring.
        * ``None`` if **no** such mapping exists.
    """
    # ---------------------------
    # Quick sanity & edge cases
    # ---------------------------
    m = len(budget)                 # topics
    n = len(preferences)            # citizens

    total_budget = float(sum(budget))
    if total_budget == 0.0:
        # Nothing needs funding → everyone pays 0 and we’re done.
        return [dict() for _ in range(n)]

    # ------------------------------------------------------------------
    # 1) Create LP variables x_(i,j)  only where j in preferences[i].
    # ------------------------------------------------------------------
    var_list: List[Tuple[int, int]] = []           # keeps (i,j) tuples
    var_index: Dict[Tuple[int, int], int] = {}     # (i,j) → column in LP

    for i, support in enumerate(preferences):
        for j in support:
            var_index[(i, j)] = len(var_list)
            var_list.append((i, j))

    if not var_list:               # nobody supports *any* topic
        return None                # but total_budget > 0 ⇒ impossible

    num_vars = len(var_list)

    c = np.zeros(num_vars)

    # ------------------------------------------------------------------
    # 2) Build equality matrix  A_eq · x = b_eq.
    #    • First m rows  – ensure each topic j receives exactly budget[j].
    #    • Next  n rows  – each citizen pays total_budget / n.
    # ------------------------------------------------------------------
    A_eq = np.zeros((m + n, num_vars))
    b_eq = np.zeros(m + n)

    # Topic‑coverage block ---------------------------------------------
    for j in range(m):
        for i in range(n):
            col_id = var_index.get((i, j))
            if col_id is not None:
                A_eq[j, col_id] = 1.0
        b_eq[j] = budget[j]

    # Equal‑tax block ---------------------------------------------------
    equal_share = total_budget / n
    for i in range(n):
        row = m + i                          # first m rows already used
        for j in preferences[i]:             # only iterate over allowed topics
            col_id = var_index[(i, j)]       # always exists by construction
            A_eq[row, col_id] = 1.0
        b_eq[row] = equal_share

    # Non‑negativity for every decision variable
    bounds = [(0.0, None)] * num_vars

    # ------------------------------------------------------------------
    # 3) Fire up HiGHS.  If it returns `success=False`, the LP is infeasible
    #    → no valid split exists.
    # ------------------------------------------------------------------
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        return None

    # ------------------------------------------------------------------
    # 4) Translate the raw solution vector back into per‑citizen dicts.
    # ------------------------------------------------------------------
    split: List[Dict[int, float]] = [dict() for _ in range(n)]

    for k, (i, j) in enumerate(var_list):
        contribution = float(result.x[k])
        if contribution > 1e-9:                # ignore numerical noise < 1 milli‑cent
            # Better readability: round *after* the noise filter.
            split[i][j] = round(contribution, 10)

    return split

if __name__ == "__main__":
    # Example usage
    budget = [400, 50, 50, 0]
    preferences = [
        {0, 1},   # citizen 0 funds topics 0,1
        {0, 2},   # citizen 1 funds topics 0,2
        {0, 3},   # citizen 2 funds topics 0,3
        {1, 2},   # citizen 3 funds topics 1,2
        {0},      # citizen 4 funds topic 0 only
    ]

    allocation = find_decomposition(budget, preferences)

    if allocation is None:
        print("No decomposable split exists — somebody must pay for a disliked topic.")
    else:
        print("One valid way to split the bill (each pays", sum(budget) / len(preferences), "):")
        for i, contrib in enumerate(allocation):
            print(f"  citizen {i}: {contrib}")
