import argparse
import importlib.util
import json
import math
import random
import signal
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np


CHECKER_ROOT = Path(__file__).resolve().parent
SUITES = ("suite_001", "suite_002", "suite_003")
SUITE_PROFILES = {
    "suite_001": "small",
    "suite_002": "structured",
    "suite_003": "large",
}
DEFAULT_COUNT = 500
DEFAULT_SEED = 20260929
DEFAULT_TRAIN_SECONDS = {"a": 0.2, "b": 0.2}
DEFAULT_EVAL_RUNS = 5
DEFAULT_TOLERANCE = 1e-5


class BenchmarkTimeout(Exception):
    pass


class _TreasureHunt:
    def __init__(self, layout_path, prob_path):
        rows = [line.strip() for line in Path(layout_path).read_text(encoding="utf-8").splitlines()]
        self.N = len(rows)
        self.locations = {
            "ship": [],
            "land": [],
            "pirate_area": [],
            "pirate": [None, None],
            "treasure": [],
            "fort": [],
        }
        for row, line in enumerate(rows):
            for col, symbol in enumerate(line):
                location = (row, col)
                if symbol == "S":
                    self.locations["ship"].append(location)
                elif symbol == "L":
                    self.locations["land"].append(location)
                elif symbol in "!12":
                    self.locations["pirate_area"].append(location)
                    if symbol == "1":
                        self.locations["pirate"][0] = location
                    elif symbol == "2":
                        self.locations["pirate"][1] = location
                elif symbol == "T":
                    self.locations["treasure"].append(location)
                elif symbol == "F":
                    self.locations["fort"].append(location)
        lines = [line.strip() for line in Path(prob_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.ship_prob = [float(lines[0]), 1.0 - float(lines[0])]
        self.pirate_prob = [
            [float(value) for value in lines[1].split()],
            [float(value) for value in lines[2].split()],
        ]
        step, treasure, fort, pirate = [float(value) for value in lines[3].split()]
        self.rewards = {"step": step, "treasure": treasure, "fort": fort, "pirate": pirate}
        self.df = float(lines[4])
        self._action_delta = ((1, 0), (-1, 0), (0, -1), (0, 1))
        self.done = False

    def get_state(self):
        return self.locations["ship"][0], self.locations["pirate"], self.locations["treasure"]

    def _move(self, location, action):
        delta_row, delta_col = self._action_delta[action]
        return location[0] + delta_row, location[1] + delta_col

    def _sample_action(self, probabilities):
        return int(np.random.multinomial(1, probabilities).nonzero()[0][0])

    def _random_action(self, excluded):
        valid_actions = list(set(range(4)) - excluded)
        return int(np.random.choice(valid_actions))

    def _valid_ship_location(self, location):
        return 0 <= location[0] < self.N and 0 <= location[1] < self.N and location not in self.locations["land"]

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0, True
        for pirate in range(2):
            next_location = self._move(self.locations["pirate"][pirate], self._sample_action(self.pirate_prob[pirate]))
            if next_location not in self.locations["pirate_area"]:
                next_location = self.locations["pirate"][pirate]
            self.locations["pirate"][pirate] = next_location
        if random.random() > self.ship_prob[0]:
            action = self._random_action({action})
        next_ship = self._move(self.locations["ship"][0], action)
        if self._valid_ship_location(next_ship):
            self.locations["ship"][0] = next_ship
        ship_location = self.locations["ship"][0]
        reward = self.rewards["step"]
        if ship_location in self.locations["pirate"]:
            self.done = True
            reward += self.rewards["pirate"]
        elif ship_location in self.locations["fort"]:
            self.done = True
            reward += self.rewards["fort"]
        elif ship_location in self.locations["treasure"]:
            reward += self.rewards["treasure"]
            self.locations["treasure"].remove(ship_location)
        return self.get_state(), reward, self.done


def _timeout_handler(signum, frame):
    raise BenchmarkTimeout


def _run_with_timeout(function, seconds):
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, seconds))
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _normalise(values):
    total = sum(values)
    return [value / total for value in values]


def _random_probability_vector(rng):
    shape = rng.randrange(6)
    if shape == 0:
        return [1.0, 0.0, 0.0, 0.0]
    if shape == 1:
        return [0.0, 1.0, 0.0, 0.0]
    if shape == 2:
        return [0.25, 0.25, 0.25, 0.25]
    if shape == 3:
        return [0.97, 0.01, 0.01, 0.01]
    if shape == 4:
        return [0.005, 0.005, 0.49, 0.5]
    return _normalise([rng.random() ** 3 + 1e-9 for _ in range(4)])


def _neighbours(size, location):
    row, col = location
    for delta_row, delta_col in ((1, 0), (-1, 0), (0, -1), (0, 1)):
        next_row = row + delta_row
        next_col = col + delta_col
        if 0 <= next_row < size and 0 <= next_col < size:
            yield next_row, next_col


def _reachable(size, start, target, blocked):
    if start in blocked or target in blocked:
        return False
    queue = deque([start])
    visited = {start}
    while queue:
        location = queue.popleft()
        if location == target:
            return True
        for neighbour in _neighbours(size, location):
            if neighbour not in blocked and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return False


def _connected_region(size, root, row_range, col_range, target_size, rng):
    region = {root}
    while len(region) < target_size:
        candidates = []
        for location in region:
            for neighbour in _neighbours(size, location):
                if (
                    neighbour not in region
                    and row_range[0] <= neighbour[0] < row_range[1]
                    and col_range[0] <= neighbour[1] < col_range[1]
                ):
                    candidates.append(neighbour)
        if not candidates:
            break
        region.add(rng.choice(candidates))
    return region


def _part_a_case(suite, index, seed):
    rng = random.Random(seed)
    profile = SUITE_PROFILES[suite]
    if profile == "small":
        size = rng.randint(5, 9)
        region_limit = 5
        density = rng.uniform(0.02, 0.18)
    elif profile == "structured":
        size = rng.randint(10, 18)
        region_limit = 14
        density = rng.uniform(0.12, 0.38)
    else:
        size = rng.randint(20, 30)
        region_limit = 36
        density = rng.uniform(0.18, 0.48)

    first_row_range = (0, max(2, size // 2))
    first_col_range = (0, max(2, size // 2))
    second_row_range = (min(size - 1, size // 2 + 1), size)
    second_col_range = (min(size - 1, size // 2 + 1), size)
    first_root = (rng.randrange(first_row_range[0], first_row_range[1]), rng.randrange(first_col_range[0], first_col_range[1]))
    second_root = (rng.randrange(second_row_range[0], second_row_range[1]), rng.randrange(second_col_range[0], second_col_range[1]))
    first_size = rng.randint(1, max(1, min(region_limit, 1 + size // 2)))
    second_size = rng.randint(1, max(1, min(region_limit, 1 + size // 2)))
    first_region = _connected_region(size, first_root, first_row_range, first_col_range, first_size, rng)
    second_region = _connected_region(size, second_root, second_row_range, second_col_range, second_size, rng)
    pirate_cells = first_region | second_region

    candidates = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) not in pirate_cells
    ]
    if index % 4 == 0:
        ship = (0, size - 1)
        fort = (size - 1, 0)
    else:
        ship = rng.choice(candidates)
        far_candidates = [location for location in candidates if abs(location[0] - ship[0]) + abs(location[1] - ship[1]) >= max(2, size // 2)]
        fort = rng.choice(far_candidates or candidates)
    remaining = [location for location in candidates if location not in {ship, fort}]
    rng.shuffle(remaining)
    treasures = remaining[:2]

    land = set()
    pattern = index % 6
    for row, col in candidates:
        location = (row, col)
        if location in {ship, fort, *treasures}:
            continue
        structured = False
        if pattern == 1:
            structured = col % 4 == 1 and (row + index) % 7 != 0
        elif pattern == 2:
            structured = row % 4 == 2 and (col + index) % 7 != 0
        elif pattern == 3:
            structured = (row + col + index) % 5 == 0
        elif pattern == 4:
            middle = size // 2
            structured = abs(row - middle) + abs(col - middle) in {2, 3}
        elif pattern == 5:
            structured = (row * 13 + col * 7 + index) % 11 < 4
        if (structured or rng.random() < density) and _reachable(size, ship, fort, land | {location}):
            land.add(location)

    grid = [["W" for _ in range(size)] for _ in range(size)]
    for row, col in land:
        grid[row][col] = "L"
    for row, col in first_region:
        grid[row][col] = "!"
    for row, col in second_region:
        grid[row][col] = "!"
    grid[first_root[0]][first_root[1]] = "1"
    grid[second_root[0]][second_root[1]] = "2"
    grid[ship[0]][ship[1]] = "S"
    grid[fort[0]][fort[1]] = "F"
    for row, col in treasures:
        grid[row][col] = "T"

    ship_success = rng.choice([0.0, 0.05, 0.5, 0.8, 0.99, 1.0])
    step_reward = -rng.choice([0.001, 0.01, 0.05, 0.2, 1.0])
    treasure_reward = rng.choice([0.001, 0.1, 1.0, 3.0, 25.0, 100.0])
    fort_reward = rng.choice([0.1, 1.0, 5.0, 25.0, 100.0])
    pirate_reward = -rng.choice([0.01, 0.1, 1.0, 5.0, 50.0])
    discount = rng.choice([0.0, 0.5, 0.9, 0.99, 0.999, 0.9999])
    return {
        "id": f"{suite}-{index:04d}",
        "suite": suite,
        "part": "a",
        "seed": seed,
        "layout": ["".join(row) for row in grid],
        "prob": {
            "ship_success": ship_success,
            "pirate_prob": [_random_probability_vector(rng), _random_probability_vector(rng)],
            "rewards": [step_reward, treasure_reward, fort_reward, pirate_reward],
            "discount": discount,
        },
    }


def _part_b_case(suite, index, seed):
    rng = random.Random(seed)
    return {
        "id": f"{suite}-{index:04d}",
        "suite": suite,
        "part": "b",
        "seed": seed,
        "discount_factor": rng.choice([0.5, 0.8, 0.95, 0.99, 0.999]),
    }


def _make_cases(part, suite, count, seed):
    if part == "a":
        return [_part_a_case(suite, index, seed + index * 1009) for index in range(count)]
    return [_part_b_case(suite, index, seed + index * 1009) for index in range(count)]


def _format_number(value):
    return f"{value:.12g}"


def _write_part_a_files(case, directory):
    layout_path = directory / "layout.txt"
    prob_path = directory / "prob.txt"
    layout_path.write_text("\n".join(case["layout"]) + "\n", encoding="utf-8")
    prob = case["prob"]
    lines = [
        _format_number(prob["ship_success"]),
        " ".join(_format_number(value) for value in prob["pirate_prob"][0]),
        " ".join(_format_number(value) for value in prob["pirate_prob"][1]),
        " ".join(_format_number(value) for value in prob["rewards"]),
        _format_number(prob["discount"]),
    ]
    prob_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return layout_path, prob_path


def _load_modules(part, agent_path, project_root):
    part_dir = project_root / "A2-starter-code" / f"part_{part}"
    old_path = list(sys.path)
    old_env = sys.modules.get("env")
    env_module = None
    if part == "b":
        env_spec = importlib.util.spec_from_file_location("benchmark_env", part_dir / "env.py")
        env_module = importlib.util.module_from_spec(env_spec)
        sys.modules["env"] = env_module
        env_spec.loader.exec_module(env_module)
    sys.path.insert(0, str(part_dir))
    agent_spec = importlib.util.spec_from_file_location("benchmark_agent", agent_path)
    if agent_spec is None or agent_spec.loader is None:
        raise ValueError(f"could not load agent from {agent_path}")
    agent_module = importlib.util.module_from_spec(agent_spec)
    agent_spec.loader.exec_module(agent_module)
    return agent_module, env_module, old_path, old_env


def _unload_modules(old_path, old_env):
    sys.path[:] = old_path
    if old_env is None:
        sys.modules.pop("env", None)
    else:
        sys.modules["env"] = old_env


def _part_a_score(agent_module, case, train_seconds, eval_runs, temp_directory):
    layout_path, prob_path = _write_part_a_files(case, temp_directory)
    agent = agent_module.Agent(str(layout_path), str(prob_path))
    random.seed(case["seed"])
    np.random.seed(case["seed"] % (2**32 - 1))
    _run_with_timeout(lambda: agent.learn_policy(train_seconds), train_seconds + 1.0)
    scores = []
    for run_index in range(eval_runs):
        random.seed(case["seed"] + run_index * 104729)
        np.random.seed((case["seed"] + run_index * 104729) % (2**32 - 1))
        env = _TreasureHunt(str(layout_path), str(prob_path))
        rewards = []
        steps = 0
        state = env.get_state()
        while not env.done and steps < 2 * (env.N**2):
            action = agent.get_action(*state)
            state, reward, _ = env.step(action)
            rewards.append(float(reward))
            steps += 1
        discounted = 0.0
        for reward in reversed(rewards):
            discounted = discounted * env.df + reward
        scores.append(discounted)
    return float(sum(scores) / len(scores))


def _part_b_score(agent_module, env_module, case, train_seconds, eval_runs):
    original_reset = env_module.HighwayEnv.reset

    def deterministic_reset(self, seed=None):
        return original_reset(self, case["seed"] if seed is None else seed)

    env_module.HighwayEnv.reset = deterministic_reset
    try:
        np.random.seed(case["seed"] % (2**32 - 1))
        train_env = env_module.HighwayEnv()
        agent = agent_module.Agent(train_env, discount_factor=case["discount_factor"])
        _run_with_timeout(lambda: agent.learn_policy(train_seconds), train_seconds + 1.0)
        scores = []
        for run_index in range(eval_runs):
            eval_seed = case["seed"] + run_index * 104729
            np.random.seed(eval_seed % (2**32 - 1))
            env = env_module.HighwayEnv()
            env.reset(seed=eval_seed)
            rewards = []
            state = env.get_state()
            while not env.done:
                action = agent.get_action(*state)
                state, reward, _ = env.step(action)
                rewards.append(float(reward))
            discounted = 0.0
            for reward in reversed(rewards):
                discounted = discounted * case["discount_factor"] + reward
            scores.append(discounted)
        return float(sum(scores) / len(scores))
    finally:
        env_module.HighwayEnv.reset = original_reset


def _score_case(part, agent_module, env_module, case, train_seconds, eval_runs, temp_directory):
    if part == "a":
        return _part_a_score(agent_module, case, train_seconds, eval_runs, temp_directory)
    return _part_b_score(agent_module, env_module, case, train_seconds, eval_runs)


def _score_file(scores_dir, part, suite):
    return scores_dir / f"part_{part}" / f"{suite}.json"


def _serialise_score(value):
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _run_overwrite(args, suites):
    scores_dir = Path(args.scores_dir).resolve()
    project_root = Path(args.project_dir).resolve()
    agent_path = Path(args.agent).resolve() if args.agent else project_root / "A2-starter-code" / f"part_{args.part}" / "agent.py"
    train_seconds = args.train_seconds if args.train_seconds is not None else DEFAULT_TRAIN_SECONDS[args.part]
    eval_runs = args.eval_runs if args.eval_runs is not None else DEFAULT_EVAL_RUNS
    started = time.perf_counter()
    for suite in suites:
        cases = _make_cases(args.part, suite, args.count, args.seed + SUITES.index(suite) * 1000003)
        agent_module, env_module, old_path, old_env = _load_modules(args.part, agent_path, project_root)
        scores = []
        failures = 0
        with tempfile.TemporaryDirectory(prefix=f"benchmark_{args.part}_{suite}_") as temporary:
            temporary_path = Path(temporary)
            for number, case in enumerate(cases, 1):
                try:
                    score = _score_case(args.part, agent_module, env_module, case, train_seconds, eval_runs, temporary_path)
                    score = _serialise_score(score)
                except Exception as error:
                    score = None
                    failures += 1
                    print(f"{case['id']} error={type(error).__name__}: {error}")
                scores.append(score)
                print(f"overwrite {number}/{len(cases)} {case['id']} score={score}")
        _unload_modules(old_path, old_env)
        payload = {
            "version": 1,
            "part": args.part,
            "suite": suite,
            "seed": args.seed + SUITES.index(suite) * 1000003,
            "count": len(cases),
            "settings": {
                "train_seconds": train_seconds,
                "eval_runs": eval_runs,
                "tolerance": args.tolerance,
            },
            "cases": cases,
            "scores": scores,
        }
        output_path = _score_file(scores_dir, args.part, suite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved {output_path} failures={failures}")
    print(f"total time taken: {time.perf_counter() - started:.3f}s")


def _comparison(score, reference, tolerance):
    if score is None:
        return "error"
    margin = max(tolerance, tolerance * max(1.0, abs(reference)))
    if abs(score - reference) <= margin:
        return "matched"
    if score > reference:
        return "more optimal"
    return "suboptimal"


def _run_evaluate(args, suites):
    scores_dir = Path(args.scores_dir).resolve()
    project_root = Path(args.project_dir).resolve()
    agent_path = Path(args.agent).resolve() if args.agent else project_root / "A2-starter-code" / f"part_{args.part}" / "agent.py"
    started = time.perf_counter()
    totals = Counter()
    for suite in suites:
        score_path = _score_file(scores_dir, args.part, suite)
        if not score_path.exists():
            raise FileNotFoundError(f"missing model scores: {score_path}; run overwrite first")
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        cases = payload["cases"]
        references = payload["scores"]
        limit = args.limit if args.limit else len(cases)
        limit = min(limit, len(cases))
        train_seconds = args.train_seconds if args.train_seconds is not None else payload["settings"]["train_seconds"]
        eval_runs = args.eval_runs if args.eval_runs is not None else payload["settings"]["eval_runs"]
        tolerance = args.tolerance if args.tolerance is not None else payload["settings"].get("tolerance", DEFAULT_TOLERANCE)
        agent_module, env_module, old_path, old_env = _load_modules(args.part, agent_path, project_root)
        with tempfile.TemporaryDirectory(prefix=f"benchmark_eval_{args.part}_{suite}_") as temporary:
            temporary_path = Path(temporary)
            for number in range(limit):
                case = cases[number]
                reference = references[number]
                try:
                    score = _serialise_score(_score_case(args.part, agent_module, env_module, case, train_seconds, eval_runs, temporary_path))
                    status = _comparison(score, reference, tolerance) if reference is not None else "error"
                except Exception as error:
                    score = None
                    status = "error"
                    print(f"{case['id']} error={type(error).__name__}: {error}")
                totals[status] += 1
                print(f"{case['id']} {status} score={score} reference={reference} delta={None if score is None or reference is None else score - reference}")
        _unload_modules(old_path, old_env)
        print(f"suite {suite}: {limit} cases")
    print(f"matched: {totals['matched']}")
    print(f"more optimal: {totals['more optimal']}")
    print(f"suboptimal: {totals['suboptimal']}")
    print(f"error: {totals['error']}")
    print(f"total time taken: {time.perf_counter() - started:.3f}s")


def _parser():
    parser = argparse.ArgumentParser(description="Generate and evaluate adversarial Part A and Part B benchmark suites.")
    parser.add_argument("mode", choices=("overwrite", "evaluate"))
    parser.add_argument("--part", choices=("a", "b"), required=True)
    parser.add_argument("--suite", choices=SUITES + ("all",), default="all")
    parser.add_argument("--agent", type=str)
    parser.add_argument("--project-dir", type=str, default=str(CHECKER_ROOT.parent))
    parser.add_argument("--scores-dir", type=str, default=str(CHECKER_ROOT / "model_scores"))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-seconds", type=float)
    parser.add_argument("--eval-runs", type=int)
    parser.add_argument("--tolerance", type=float, default=None)
    return parser


def main():
    args = _parser().parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if args.train_seconds is not None and args.train_seconds <= 0:
        raise ValueError("--train-seconds must be positive")
    if args.eval_runs is not None and args.eval_runs < 1:
        raise ValueError("--eval-runs must be positive")
    if args.tolerance is not None and args.tolerance < 0:
        raise ValueError("--tolerance cannot be negative")
    suites = SUITES if args.suite == "all" else (args.suite,)
    if args.mode == "overwrite":
        if args.tolerance is None:
            args.tolerance = DEFAULT_TOLERANCE
        _run_overwrite(args, suites)
    else:
        _run_evaluate(args, suites)


if __name__ == "__main__":
    main()
