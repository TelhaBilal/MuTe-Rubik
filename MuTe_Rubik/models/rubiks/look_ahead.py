"""branch out a tree of possible actions"""

import random
from typing import Callable, Iterable, List, Tuple

from tqdm.auto import tqdm


def score_actions(
    state_0,
    actions,
    get_next_state: Callable,
    get_state_score: Callable,
    depth_decay: float,
    n_branches: int,
    max_depth: int,
    greedy_fraction: float = 0.95,
):
    rewards = []
    for action in actions:
        next_state = get_next_state(state_0, action)
        _, possible_rewards = search_best_paths(
            state_0=next_state,
            actions=actions,
            get_next_state=get_next_state,
            get_state_score=get_state_score,
            depth_decay=depth_decay,
            n_branches=n_branches,
            max_depth=max_depth,
            greedy_fraction=greedy_fraction,
        )
        if random.random() < greedy_fraction:
            rewards.append(max(possible_rewards))
        else:
            rewards.append(
                random.choices(possible_rewards, weights=possible_rewards, k=1)[0]
            )

    return actions, rewards


def _prune(data, scores: Iterable[float], n_top, n_random) -> Tuple[List, List[float]]:
    indexed_sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    indexed_scores = indexed_sorted_scores[:n_top] + sorted(
        set(
            random.choices(
                indexed_sorted_scores[n_top:],
                weights=[score+1e-5 for _, score in indexed_sorted_scores[n_top:]],
                k=n_random,
            )
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    return (
        [data[i] for i, _ in indexed_scores],
        [score for _, score in indexed_scores],
    )


def search_best_paths(
    state_0,
    actions,
    get_next_state: Callable,
    get_state_score: Callable,
    remove_action_cycles: Callable,
    depth_decay: float,
    n_branches: int,
    max_depth: int,
    greedy_fraction: float = 0.75,
):
    next_states = [get_next_state(state_0, action) for action in actions]
    rewards = list(map(get_state_score, next_states))

    action_paths = list(zip([(action,) for action in actions], next_states))

    for depth in tqdm(range(max_depth), leave=False):
        # look ahead
        num_branches = len(action_paths)
        for branch_index in range(num_branches):
            action_path, state = action_paths[branch_index]
            next_states = [get_next_state(state, action) for action in actions]
            next_rewards = map(get_state_score, next_states)
            action_paths.extend(
                zip(
                    [action_path + (action,) for action in actions],
                    next_states,
                )
            )
            # rewards[branch_index] * 0.05 + 0.95 * reward * depth_decay**depth
            rewards.extend([reward * depth_decay**depth for reward in next_rewards])

        # drop_duplicates
        unique_by_path = {
            remove_action_cycles(paths[0]): {"state": paths[1], "score": score}
            for paths, score in sorted(
                zip(action_paths, rewards), key=lambda item: item[1]
            )
        }
        action_paths = [(key, val["state"]) for key, val in unique_by_path.items()]
        rewards = [val["score"] for val in unique_by_path.values()]

        # filter best options
        action_paths, rewards = _prune(
            action_paths,
            rewards,
            n_top=round(n_branches * greedy_fraction),
            n_random=round(n_branches * (1 - greedy_fraction)),
        )

    return action_paths, rewards
