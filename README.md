# COL333-A2 checker

The checker generates three deterministic adversarial suites with 500 cases each, stores reference model scores in `model_scores/`, and compares an external project in `evaluate` mode.

## Checker beside the starter code

```bash
cd /path/to/COL333-A2
git clone git@github.com:AbhinavPJ/COL333-A2-CHECKER.git checker
python3 checker/benchmark.py overwrite --part a --project-dir .
python3 checker/benchmark.py evaluate --part a --project-dir .
```

Use `--part b` for Part B.

## Checker inside the starter-code folder

```bash
cd /path/to/COL333-A2/A2-starter-code
git clone git@github.com:AbhinavPJ/COL333-A2-CHECKER.git checker
python3 checker/benchmark.py overwrite --part a --project-dir ..
python3 checker/benchmark.py evaluate --part a --project-dir ..
```
