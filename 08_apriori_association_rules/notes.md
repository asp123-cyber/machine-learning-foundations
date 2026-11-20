# 08. Unsupervised Learning: Apriori Algorithm (Part 1 & 2)

## Part 1: Market Basket Analysis & The Three Metrics
Ever wondered why supermarkets place pasta right next to sauce, or why online stores suggest a screen protector the second you add a phone? That is **Association Rule Mining**, and **Apriori** is its classic algorithm.

We search transaction logs for rules of the form:
$$\{\text{Bread}, \text{Butter}\} \rightarrow \{\text{Milk}\}$$
("If a shopper puts Bread and Butter in their basket, they tend to add Milk as well.")

### The Three Numbers That Decide Whether a Rule Is Useful
1. **Support (Popularity):** What fraction of *all* receipts contain this itemset?
   $$\text{Support}(A \cup B) = \frac{\text{Transactions containing both } A \text{ and } B}{\text{Total Transactions}}$$
2. **Confidence (Reliability):** Given someone already put $A$ in their basket, what is the conditional probability they also buy $B$?
   $$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
3. **Lift (Real Signal vs. Coincidence):** Almost everyone buys Milk anyway! So a high Confidence for $\{\text{Toothpaste}\} \rightarrow \{\text{Milk}\}$ might just reflect how popular Milk is. **Lift** checks whether buying $A$ genuinely increases the odds of buying $B$:
   $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
   - $\text{Lift} > 1$: True positive association (put them together!).
   - $\text{Lift} = 1$: Independent coincidence.
   - $\text{Lift} < 1$: Substitutes or negative correlation (buying one makes the other *less* likely).

## Part 2: Why Is It Called "Apriori"? (The Pruning Shortcut)
If a store sells 1,000 products, testing every possible subset ($2^{1000} - 1$) is computationally impossible. The **Apriori Principle** (downward closure property) solves this cleanly:

> **If an itemset is infrequent (below `min_support`), every larger combination containing it MUST also be infrequent.**

So we build itemsets level by level:
1. Find all frequent 1-itemsets (`k = 1`).
2. Combine only frequent $(k-1)$-itemsets to propose $k$-item candidates.
3. Prune any candidate whose subsets are not all frequent.
4. Repeat until no frequent itemsets remain, then extract high-confidence, high-lift rules.
