# A/B Testing: Epsilon-Greedy vs. Thompson Sampling

Four ads with true mean rewards `[1, 2, 3, 4]`, 20000 trials, Normal reward noise (known precision = 1).

## Run
```bash
pip install -r requirements.txt
python Bandit.py
```
Outputs are written to `output/`: `rewards.csv` (Bandit, Reward, Algorithm), per-algorithm CSVs, `plot1_*.png` (learning process, linear + log), `plot2_comparison.png` (cumulative reward and regret).

## Results (seed 42)
| Algorithm | Cumulative reward | Cumulative regret |
|---|---|---|
| Epsilon-Greedy (eps = 1/t) | 73907.23 | 6181.00 |
| Thompson Sampling (known precision) | 79937.44 | 37.00 |

## Bonus: better implementation plan
1. Epsilon = 1/t decays too fast: the greedy arm can lock in on a suboptimal ad after a few noisy early pulls. Use a slower schedule (e.g. `eps = c / sqrt(t)`) or optimistic initial values.
2. Replace the known-precision assumption with a Normal-Gamma posterior so the noise level is learned too.
3. Run many seeds and report mean +/- std of regret instead of a single run.
4. Add a stopping rule (e.g. stop when P(best arm) > 95%) so fewer users see a worse ad in a real A/B test.
5. Vectorize/batch updates for large-scale experiments.
