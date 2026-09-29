# COL333-A2 checker

The checker generates three deterministic adversarial suites with 500 cases each, stores reference model scores in `model_scores/`, and compares an external project in `evaluate` mode. The score files are named `suite_001.json`, `suite_002.json`, and `suite_003.json`.

## Checker beside the starter code

```bash
cd /path/to/COL333-A2
git clone git@github.com:AbhinavPJ/COL333-A2-CHECKER.git checker
python3 checker/benchmark.py overwrite --part a --project-dir .
python3 checker/benchmark.py evaluate --part a --project-dir .
```

If you are already inside `A2-starter-code` and the checker is beside it, use:

```bash
cd /path/to/COL333-A2/A2-starter-code
python3 ../checker/benchmark.py overwrite --part a --project-dir ..
python3 ../checker/benchmark.py evaluate --part a --project-dir ..
```

Use `--part b` for Part B.

## Contributing test cases

Testcase contributions are welcome. Submit deterministic, valid cases that target an edge condition, numerical corner case, or difficult decision pattern. Part A contributions should follow the documented layout and probability-file contract. Part B contributions should use the standard environment with a reproducible seed and discount factor. Do not include assignment implementation files. Include a short explanation of what the case is intended to catch; accepted contributions will be incorporated into the numbered suites and reference scores.

## Checker inside the starter-code folder

```bash
cd /path/to/COL333-A2/A2-starter-code
git clone git@github.com:AbhinavPJ/COL333-A2-CHECKER.git checker
python3 checker/benchmark.py overwrite --part a --project-dir ..
python3 checker/benchmark.py evaluate --part a --project-dir ..
```

Use `--part b` for Part B.
