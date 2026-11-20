from itertools import combinations
import pandas as pd

transactions = [
    {"Milk", "Bread", "Butter"},
    {"Bread", "Butter", "Jam"},
    {"Milk", "Bread", "Eggs"},
    {"Milk", "Bread", "Butter", "Eggs"},
    {"Bread", "Butter"},
    {"Milk", "Eggs", "Coffee"},
    {"Milk", "Bread", "Butter", "Coffee"},
    {"Bread", "Jam", "Coffee"},
    {"Milk", "Bread", "Butter"},
    {"Bread", "Eggs", "Coffee"}
]


def get_support(itemset, data):
    count = sum(1 for t in data if itemset.issubset(t))
    return count / len(data)


def apriori(data, min_support=0.3):
    all_items = sorted({item for t in data for item in t})
    frequent_itemsets = {}
    current_candidates = [frozenset([item]) for item in all_items]
    k = 1

    while current_candidates:
        valid = {}
        for candidate in current_candidates:
            sup = get_support(candidate, data)
            if sup >= min_support:
                valid[candidate] = sup

        if not valid:
            break

        frequent_itemsets.update(valid)
        prev_sets = list(valid.keys())
        k += 1

        next_candidates = set()
        for i in range(len(prev_sets)):
            for j in range(i + 1, len(prev_sets)):
                union_set = prev_sets[i] | prev_sets[j]
                if len(union_set) == k:
                    subsets = [frozenset(s) for s in combinations(union_set, k - 1)]
                    if all(s in valid for s in subsets):
                        next_candidates.add(union_set)
        current_candidates = list(next_candidates)

    return frequent_itemsets


def generate_rules(frequent_itemsets, min_confidence=0.6):
    rules = []
    for itemset, sup_both in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent_tuple in combinations(itemset, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                sup_a = frequent_itemsets[antecedent]
                sup_b = frequent_itemsets[consequent]
                confidence = sup_both / sup_a
                lift = confidence / sup_b
                if confidence >= min_confidence:
                    rules.append({
                        "Antecedent": ", ".join(sorted(antecedent)),
                        "Consequent": ", ".join(sorted(consequent)),
                        "Support": sup_both,
                        "Confidence": confidence,
                        "Lift": lift
                    })
    return pd.DataFrame(rules).sort_values(by="Lift", ascending=False)


freq_sets = apriori(transactions, min_support=0.3)
rules_df = generate_rules(freq_sets, min_confidence=0.65)

print("Discovered Association Rules (Sorted by Lift):\n")
print(rules_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
