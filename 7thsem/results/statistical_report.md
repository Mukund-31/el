# Statistical Validation Report

Generated: 2025-12-26T14:55:40.395250

Sample size: 300 episodes per agent

## Results Summary

| Metric | RL Agent | Baseline | Improvement | p-value | Cohen's d | Effect Size |
|--------|----------|----------|-------------|---------|-----------|-------------|
| avg_wait | 22.99 ± 235.31 | 28.90 ± 89.84 | 20.5% | 0.6854 | 0.03 | negligible |
| renege_rate | 2.11 ± 5.81 | 10.19 ± 10.77 | 79.3% | 0.0000 | 0.93 | large |
| staffing_cost | 398.05 ± 71.56 | 372.07 ± 76.72 | -7.0% | 0.0000 | -0.35 | small |
| total_cost | 512.99 ± 1155.65 | 516.58 ± 411.19 | 0.7% | 0.9597 | 0.00 | negligible |

## Interpretation

- **renege_rate**: Statistically significant difference (p < 0.05) with large effect size. RL agent shows 79.3% improvement.
- **staffing_cost**: Statistically significant difference (p < 0.05) with small effect size. RL agent shows -7.0% improvement.

## Detailed Statistics

```json
{
  "avg_wait": {
    "rl_mean": 22.986872023809525,
    "rl_std": 235.31114675139054,
    "rl_ci": [
      -3.793492210499938,
      49.767236258118984
    ],
    "baseline_mean": 28.901976388888887,
    "baseline_std": 89.83818912519114,
    "baseline_ci": [
      18.677643797176756,
      39.12630898060102
    ],
    "improvement_pct": 20.46608953480902,
    "t_statistic": -0.40549946885297977,
    "p_value": 0.6854004091083082,
    "u_statistic": 9696.5,
    "u_p_value": 1.1989981588947962e-63,
    "cohens_d": 0.03321147536733869,
    "effect_size_interpretation": "negligible"
  },
  "renege_rate": {
    "rl_mean": 2.1124338624338628,
    "rl_std": 5.810656963618005,
    "rl_ci": [
      1.4511328060469226,
      2.7737349188208027
    ],
    "baseline_mean": 10.194067460317461,
    "baseline_std": 10.768463308926604,
    "baseline_ci": [
      8.968526836177233,
      11.419608084457689
    ],
    "improvement_pct": 79.27781162272123,
    "t_statistic": -12.669727634222607,
    "p_value": 9.7169400326393e-30,
    "u_statistic": 8997.0,
    "u_p_value": 4.0216662262297196e-66,
    "cohens_d": 0.9340478310980612,
    "effect_size_interpretation": "large"
  },
  "staffing_cost": {
    "rl_mean": 398.05333333333334,
    "rl_std": 71.56376286610113,
    "rl_ci": [
      389.9087823335596,
      406.1978843331071
    ],
    "baseline_mean": 372.06666666666666,
    "baseline_std": 76.72469543040813,
    "baseline_ci": [
      363.3347586471812,
      380.7985746861521
    ],
    "improvement_pct": -6.984411395807206,
    "t_statistic": 4.763605011976008,
    "p_value": 2.9701228550188483e-06,
    "u_statistic": 52284.0,
    "u_p_value": 0.0006009953257146932,
    "cohens_d": -0.35027597950304185,
    "effect_size_interpretation": "small"
  },
  "total_cost": {
    "rl_mean": 512.987693452381,
    "rl_std": 1155.651073371961,
    "rl_ci": [
      381.4649963768804,
      644.5103905278816
    ],
    "baseline_mean": 516.5765486111112,
    "baseline_std": 411.19245017101724,
    "baseline_ci": [
      469.77943025100143,
      563.373666971221
    ],
    "improvement_pct": 0.6947383051707156,
    "t_statistic": -0.05060155903709678,
    "p_value": 0.9596768071030691,
    "u_statistic": 41045.0,
    "u_p_value": 0.06251578173451397,
    "cohens_d": 0.00413770147430249,
    "effect_size_interpretation": "negligible"
  }
}
```
