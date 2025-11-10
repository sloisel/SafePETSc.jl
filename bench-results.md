# SafePETSc Benchmarks

- Date: 2025-11-09 18:11:40
- Ranks: 4
- Size: N = 100
- Repetitions: 10 when default_check < 10; 100 when default_check ≥ 10

## create_mat — dense

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.048248 | 0.049212 |
| 3 | 0.014695 | 0.015177 |
| 10 | 0.005018 | 0.005235 |
| 100 | 0.000638 | 0.000826 |

## create_mat — sparse_diag

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.048487 | 0.049346 |
| 3 | 0.014875 | 0.014974 |
| 10 | 0.005159 | 0.005203 |
| 100 | 0.000779 | 0.000794 |

## create_vec — vec

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.048224 | 0.049475 |
| 3 | 0.014620 | 0.014985 |
| 10 | 0.004944 | 0.004998 |
| 100 | 0.000569 | 0.000602 |

## matmat — dense

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.049122 | 0.049421 |
| 3 | 0.015109 | 0.015157 |
| 10 | 0.005574 | 0.005481 |
| 100 | 0.001040 | 0.001086 |

## matmat — sparse_diag

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.048874 | 0.049854 |
| 3 | 0.015035 | 0.015088 |
| 10 | 0.005319 | 0.005295 |
| 100 | 0.000856 | 0.000889 |

## matvec — dense

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.048098 | 0.048917 |
| 3 | 0.014682 | 0.014747 |
| 10 | 0.004958 | 0.005139 |
| 100 | 0.000578 | 0.000591 |

## matvec — sparse_diag

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.048454 | 0.048960 |
| 3 | 0.014670 | 0.014741 |
| 10 | 0.004956 | 0.004981 |
| 100 | 0.000561 | 0.000573 |

## solve — dense_spd

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.097262 | 0.097781 |
| 3 | 0.029435 | 0.029556 |
| 10 | 0.009982 | 0.010189 |
| 100 | 0.001175 | 0.001210 |

## solve — sparse_diag

| default_check | assert=false | assert=true |
|---:|---:|---:|
| 1 | 0.096979 | 0.097918 |
| 3 | 0.029432 | 0.029604 |
| 10 | 0.010045 | 0.010036 |
| 100 | 0.001165 | 0.001202 |

