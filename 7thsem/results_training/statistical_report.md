# Statistical Validation Report

Generated: 2025-12-26T17:18:08.424972

Sample size: 500 episodes per agent

## Results Summary

| Metric | RL Agent | Baseline | Improvement | p-value | Cohen's d | Effect Size |
|--------|----------|----------|-------------|---------|-----------|-------------|
| avg_wait | 12.09 ± 126.99 | 35.01 ± 112.28 | 65.5% | 0.0024 | 0.19 | negligible |
| renege_rate | 1.93 ± 5.57 | 10.71 ± 11.20 | 82.0% | 0.0000 | 0.99 | large |
| staffing_cost | 443.44 ± 60.77 | 367.14 ± 80.54 | -20.8% | 0.0000 | -1.07 | large |
| total_cost | 503.91 ± 621.62 | 542.17 ± 521.38 | 7.1% | 0.2873 | 0.07 | negligible |

## Interpretation

- **avg_wait**: Statistically significant difference (p < 0.05) with negligible effect size. RL agent shows 65.5% improvement.
- **renege_rate**: Statistically significant difference (p < 0.05) with large effect size. RL agent shows 82.0% improvement.
- **staffing_cost**: Statistically significant difference (p < 0.05) with large effect size. RL agent shows -20.8% improvement.

## Detailed Statistics

```json
{
  "avg_wait": {
    "rl_mean": 12.094156428571429,
    "rl_std": 126.98719546821351,
    "rl_ci": [
      0.9252050339003919,
      23.263107823242464
    ],
    "baseline_mean": 35.006834166666664,
    "baseline_std": 112.27909597626667,
    "baseline_ci": [
      25.131509643018695,
      44.88215869031463
    ],
    "improvement_pct": 65.45201325263675,
    "t_statistic": -3.057456710226443,
    "p_value": 0.0023520012096577246,
    "u_statistic": 25300.5,
    "u_p_value": 3.4588611215179346e-110,
    "cohens_d": 0.19116365345409117,
    "effect_size_interpretation": "negligible"
  },
  "renege_rate": {
    "rl_mean": 1.9291817460317462,
    "rl_std": 5.565509124546134,
    "rl_ci": [
      1.4396764800129092,
      2.4186870120505835
    ],
    "baseline_mean": 10.709800793650793,
    "baseline_std": 11.201888997533546,
    "baseline_ci": [
      9.724556940672954,
      11.69504464662863
    ],
    "improvement_pct": 81.98676349633465,
    "t_statistic": -16.227752398962757,
    "p_value": 7.275313188831576e-48,
    "u_statistic": 23507.0,
    "u_p_value": 4.065945696317738e-114,
    "cohens_d": 0.9927555965119594,
    "effect_size_interpretation": "large"
  },
  "staffing_cost": {
    "rl_mean": 443.435,
    "rl_std": 60.766580247698656,
    "rl_ci": [
      438.0903746721904,
      448.7796253278096
    ],
    "baseline_mean": 367.136,
    "baseline_std": 80.53695117149643,
    "baseline_ci": [
      360.05250398457224,
      374.2194960154278
    ],
    "improvement_pct": -20.78221694412969,
    "t_statistic": 18.462115271699304,
    "p_value": 2.1628459634081823e-58,
    "u_statistic": 207368.5,
    "u_p_value": 9.194719999761113e-73,
    "cohens_d": -1.0695129002124222,
    "effect_size_interpretation": "large"
  },
  "total_cost": {
    "rl_mean": 503.90578214285716,
    "rl_std": 621.6187333275399,
    "rl_ci": [
      449.2323216183262,
      558.5792426673881
    ],
    "baseline_mean": 542.1701708333333,
    "baseline_std": 521.3804745562151,
    "baseline_ci": [
      496.3130026599758,
      588.0273390066907
    ],
    "improvement_pct": 7.057634438217522,
    "t_statistic": -1.065251874776004,
    "p_value": 0.28727712755272833,
    "u_statistic": 176707.0,
    "u_p_value": 1.0103762135539964e-29,
    "cohens_d": 0.06669836453115709,
    "effect_size_interpretation": "negligible"
  }
}
```
