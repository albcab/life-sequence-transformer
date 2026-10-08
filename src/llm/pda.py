import re
import json
import string
from tqdm import tqdm
from collections import deque
from functools import lru_cache
from typing import List, Tuple, Optional, Set, Dict

CHARSET = (
    "." +
    string.ascii_lowercase +
    string.ascii_uppercase +
    string.digits +
    string.punctuation +
    " "
)

class PDA:
    def __init__(self, states: Set[str], input_alphabet: Set[str], stack_alphabet: Set[str],
                 final_states: Set[str], initial_state: str, transitions: Dict,
                 stack_bottom: Optional[str] = None, name: Optional[str] = None,  nb_max_iterations: int = 50, create_ma: bool = True, tokenizer = None, input_alphabet_dfas=None):
        self.states = states
        self.input_alphabet = input_alphabet
        self.stack_alphabet = stack_alphabet
        self.final_states = final_states
        self.initial_state = initial_state
        self.transitions = transitions
        self.stack_bottom = stack_bottom
        self.name = name if name is not None else "PDA"
        self.tokenizer = tokenizer
        self.new_configurartion_found = False

        if stack_bottom is not None and stack_bottom not in stack_alphabet:
            raise ValueError(f"Bottom symbol '{stack_bottom}' must be in stack_alphabet")

        self.configuration_moves = {}
        for q in states:
            for top in stack_alphabet:
                self.configuration_moves[(q, top)] = self.get_configuration_moves(q=q, gamma=[top])

        self.input_alphabet_dfas = input_alphabet_dfas or {}

        self.precomputed_configurations = {}
        if create_ma:
            pre_star = PRESTAR(self,self.tokenizer)
            
            ma, sigma_gamma, dist_gamma = pre_star.pre_star(nb_max_iterations)
            self.ma = ma
            self.sigma_gamma = sigma_gamma
            self.dist_gamma = dist_gamma
        if not hasattr(self, '_accepting_states'):
            self._accepting_states = set(self.final_states) | {"s_acc"}

    def stats(self) -> Dict[str, int]:
        ret_symbols = {
            s for s in self.stack_alphabet
            if isinstance(s, str) and s.startswith("RET::")
        }
        epsilon_keys = [k for k in self.transitions if k[1] == ""]
        return {
            "num_states": len(self.states),
            "num_input_symbols": len(self.input_alphabet),
            "num_stack_symbols": len(self.stack_alphabet),
            "num_ret_symbols": len(ret_symbols),
            "num_transition_keys": len(self.transitions),
            "num_transitions": sum(len(v) for v in self.transitions.values()),
            "num_epsilon_transition_keys": len(epsilon_keys),
            "num_epsilon_transitions": sum(len(self.transitions[k]) for k in epsilon_keys),
        }
    
    @lru_cache(maxsize=1000000)
    def a_starts_with_t(self, t: str, a: str) -> str | None:

        dfa, deadlock_state = self.input_alphabet_dfas.get(a)
        state = dfa.initial

        for i, char in enumerate(t):
            for charclass in dfa.map[state].keys():
                if charclass.accepts(char):
                    next_state = dfa.map[state][charclass]
                    if next_state == deadlock_state:
                        return False
                    state = next_state
                    break

        return True

    @lru_cache(maxsize=1000000)
    def a_starts_with_t_completion(self, t: str, a: str, num_beams: int = len(CHARSET)) -> str | None:

        if a == "":
            return ""
        
        dfa, deadlock_state = self.input_alphabet_dfas.get(a)
        state = dfa.initial

        for char in t:
            for charclass, next_state in dfa.map[state].items():
                if charclass.accepts(char):
                    if next_state == deadlock_state:
                        return None
                    state = next_state
                    break
            else:
                return None

        if state in dfa.finals:
            return ""

        # Beam search: keep at most num_beams states per depth
        frontier = [(state, [])]   # (state, path) pairs

        while frontier:
            next_frontier = []
            for cur_state, path in frontier:
                for charclass, next_state in dfa.map[cur_state].items():
                    if next_state == deadlock_state:
                        continue
                    for ch in CHARSET:
                        if charclass.accepts(ch):
                            new_path = path + [ch]
                            if next_state in dfa.finals:
                                return ''.join(new_path)
                            next_frontier.append((next_state, new_path))
            if not next_frontier:
                break
            frontier = next_frontier[:num_beams]

        return None
    
    @lru_cache(maxsize=1000000)
    def t_starts_with_a(self, t: str, a: str) -> str | None:
        
        dfa, deadlock_state = self.input_alphabet_dfas.get(a)
        state = dfa.initial
        longest = ""
        last_return = (False, "")

        for i, char in enumerate(t):
            for charclass in dfa.map[state].keys():
                if charclass.accepts(char):
                    longest += char
                    next_state = dfa.map[state][charclass]
                    if next_state == deadlock_state:
                        return last_return
                    if next_state in dfa.finals:
                        last_return = (True, longest)
                    state = next_state
                    break
        
        return last_return
    
    def create_ma(self, nb_max_iterations: int = 50, debug=False):
        if hasattr(self, 'ma'):
            print("MA already created.")
            
        pre_star = PRESTAR(self)
        ma, sigma_gamma, dist_gamma = pre_star.pre_star(nb_max_iterations)
        self.ma = ma
        self.sigma_gamma = sigma_gamma
        self.dist_gamma = dist_gamma
    
    def visualize(self) -> None:
        def format_set(s):
            return "{" + ", ".join(f'"{x}"' for x in sorted(s)) + "}"

        def format_tuple(t):
            if not t:
                return "()"
            return "(" + ", ".join(f'"{x}"' for x in t) + ("," if len(t) == 1 else "") + ")"

        print("{")
        print(f'    "states": {format_set(self.states)},')
        print(f'    "input_alphabet": {format_set(self.input_alphabet)},')
        print(f'    "stack_alphabet": {format_set(self.stack_alphabet)},')
        print(f'    "start_state": "{self.initial_state}",')
        print(f'    "start_stack": "{[self.stack_bottom]}",')
        print(f'    "transitions": {{')

        for (state, symbol, stack_top), outcomes in self.transitions.items():
            key = f'("{state}", "{symbol}", "{stack_top}")'
            values = []

            for next_state, push_symbols in outcomes:
                values.append(f'("{next_state}", {format_tuple(push_symbols)})')

            print(f'        {key}: [{", ".join(values)}],')

        print(f'    }},')
        print()
        print(f'    "final_states": {format_set(self.final_states)},')
        print(f'    "stack_bottom": "{[self.stack_bottom]}"')
        print("}")

    def get_configuration_moves(self, q: str, gamma: List[str]) -> Set[str]:
        if not gamma:
            return set()
        moves = set()
        for (state, symbol, stack_top), outcomes in self.transitions.items():
            if q == state and (stack_top == "" or stack_top == gamma[0]):
                for next_state, push_symbols in outcomes:
                    moves.add((next_state, symbol, stack_top, push_symbols))
        return moves
    
    @lru_cache(maxsize=1000000)
    def get_configuration_moves_with_prefix(self, q: str, gamma: Tuple[str], prefix: str) -> Set[str]:
        if not gamma:
            return set()
        moves = set()
        for next_state, symbol, stack_top, push_symbols in self.configuration_moves[(q, gamma[0])]:
            completion = self.a_starts_with_t_completion(prefix, symbol)
            if completion == None: continue 
            moves.add((next_state, tuple(self.apply_stack_operation(gamma, stack_top, push_symbols)), completion))
        return moves
    
    @lru_cache(maxsize=1000000)
    def apply_stack_operation(self, gamma: tuple[str], pop: str, push: Tuple[str]) -> Optional[List[str]]:
        if type(gamma) == tuple:
            if pop == "":
                return push + gamma
            return push + gamma[1:]
        
        if pop == "":
            return list(push) + gamma
        return list(push) + gamma[1:]
    
    def to_MA(self):
        """Build a Multi-Automaton recognizing all final configurations (any stack)."""
        initial_states = self.states.copy()
        final_state = 's_acc'
        states = self.states.copy()
        states.add(final_state)
        transitions = set()

        for fs in self.final_states:
            transitions.add((fs, "", final_state))

        for state in states:
            transitions.add((state, "", state))

        return MA(states, self.stack_alphabet.copy(), transitions, initial_states, final_state)
    
    def get_reachable_configs(self, initial_state: str, initial_stack: List[str], max_height: int) -> Set[Tuple[str, Tuple[str, ...]]]:
        """
        Compute all configurations (state, stack) reachable from the initial configuration
        with stack height <= max_height, using a forward BFS.
        Much more efficient than calling pre* for every candidate.
        """
        start_config = (initial_state, tuple(initial_stack))
        visited = {start_config}
        queue = deque([start_config])

        while queue:
            state, stack = queue.popleft()
            stack_list = list(stack)
            if not stack_list:
                continue  # empty stack: no transitions possible
            top = stack_list[0]

            # Iterate over all PDA transitions matching (state, top)
            for (q, inp, top_sym), destinations in self.transitions.items():
                if q == state and (top_sym == top or top_sym == ""):
                    for next_state, pushed in destinations:
                        # New stack: pop the top, then push 'pushed' in order

                        if top_sym != "":
                            new_stack_list = list(pushed) + stack_list[1:]
                        else:
                            new_stack_list = list(pushed) + stack_list

                        if len(new_stack_list) <= max_height:
                            new_config = (next_state, tuple(new_stack_list))
                            if new_config not in visited:
                                visited.add(new_config)
                                queue.append(new_config)

        return visited
    
    def get_min_transitions(self) -> Dict[Tuple[str, str, str], Set[Tuple[str, Tuple[str, ...]]]]:
        min_transitions = {}
        for (q, sigma, A), destinations in self.transitions.items():
            try:
                regex_min_ = self.a_starts_with_t_completion("", sigma)
            except Exception as e:
                print(f"Error computing completion for {sigma} in transition {(q, sigma, A)}: {e}")
                regex_min_ = None
            
            if regex_min_ is None:
                print(f"oups we have a None completion for regex {sigma} in transition {(q, sigma, A)}. This transition will be ignored in min_transitions.")
            min_transitions[(q, regex_min_, A)] = destinations
        return min_transitions
    
    def precompute_configurations(self, tokenizer, max_height: int = 10, load_path: Optional[str] = None, eos: str = "<|eot_id|>") -> Dict[Tuple[str, Tuple[str, ...]], Tuple[int, str]]:
        # If load path is provided, try to load from file
        if load_path is not None:
            self.path = load_path
            try:
                with open(load_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Convert keys back to tuples and values back to tuples
                loaded_configs = {}
                for key_str, value_list in tqdm(data.items()):
                    key_tuple = json.loads(key_str)  # should be [state, stack]
                    state = key_tuple[0]
                    stack = tuple(key_tuple[1])      # stack was stored as list
                    loaded_configs[(state, stack)] = (value_list[0], value_list[1].replace("<|eot_id|>", eos))
                self.precomputed_configurations = loaded_configs
                return "Done"
            except Exception as e:
                print(e)
 
        # Original computation (unchanged, but ensure self.precomputed_configurations is set)
        reachable_configs_max_height = self.get_reachable_configs(
            self.initial_state, [self.stack_bottom], max_height=max_height
        )
        if reachable_configs_max_height:
            num_at_cap = sum(
                1 for _state, stack in reachable_configs_max_height if len(stack) == max_height
            )
            if num_at_cap:
                ratio_at_cap = num_at_cap / len(reachable_configs_max_height)
                if ratio_at_cap >= 0.20:
                    print(
                        f"Warning: {num_at_cap}/{len(reachable_configs_max_height)} reachable "
                        f"configurations are already at max_height={max_height}. "
                        "Precomputation is likely being truncated; consider increasing max_height."
                    )
        bos = bool(tokenizer.bos_token_id)
 
        final_confs = []
        for state, stack in tqdm(reachable_configs_max_height):
            results = self.consume_stack(state, list(stack))
            min_dst = float("inf")
            lead_to_accepting = False
            min_str = ""
            for reached_state, reached_stack, accumulated_str in results:
                if reached_state not in self._accepting_states:
                    continue
                lead_to_accepting = True
                temp_str = tokenizer.encode(accumulated_str)
                temp_dst = len(temp_str)
                if bos:
                    temp_dst -= 1
                    temp_str = temp_str[1:]
                if min_dst > temp_dst:
                    min_dst = temp_dst
                    min_str = tokenizer.decode(temp_str)
            if lead_to_accepting:
                final_confs.append((state, stack, (min_dst, min_str)))
 
        self.precomputed_configurations = {}
        for state, stack, min_len in final_confs:
            self.precomputed_configurations[(state, stack)] = min_len

    def update_configurations(self, tokenizer, load_path: str, max_height: int = 10) -> Dict[Tuple[str, Tuple[str, ...]], Tuple[int, str]]:
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            loaded_configs = {}
            for key_str, value_list in data.items():
                key_tuple = json.loads(key_str)  # should be [state, stack]
                state = key_tuple[0]
                stack = tuple(key_tuple[1])      # stack was stored as list
                loaded_configs[(state, stack)] = (value_list[0], value_list[1])
            self.precomputed_configurations = loaded_configs
        except Exception as e:
            print(e)
 
        # Original computation (unchanged, but ensure self.precomputed_configurations is set)
        reachable_configs_max_height = self.get_reachable_configs(
            self.initial_state, [self.stack_bottom], max_height=max_height
        )

        bos = bool(tokenizer.bos_token_id)

        final_confs = []
        previusly_computed = 0
        new_confs = 0
        new_good_confs = 0
        for state, stack in tqdm(reachable_configs_max_height):
            if (state, stack) in self.precomputed_configurations.keys():
                previusly_computed += 1
                continue
            new_confs += 1
            results = self.consume_stack(state, list(stack))
            min_dst = float("inf")
            lead_to_accepting = False
            min_str = ""
            for reached_state, reached_stack, accumulated_str in results:
                if reached_state not in self._accepting_states:
                    continue
                lead_to_accepting = True
                temp_str = tokenizer.encode(accumulated_str)
                temp_dst = len(temp_str)
                if bos:
                    temp_dst -= 1
                    temp_str = temp_str[1:]
                if min_dst > temp_dst:
                    min_dst = temp_dst
                    min_str = tokenizer.decode(temp_str)
            if lead_to_accepting:
                new_good_confs += 1
                final_confs.append((state, stack, (min_dst, min_str)))

        print("All reachable confs", len(final_confs))
        print("Same as before", previusly_computed)
        print("New", new_confs)
        print("New good", new_good_confs)
        
        for state, stack, min_len in final_confs:
            self.precomputed_configurations[(state, stack)] = min_len
    
    def consume_stack(self, current_q: str, current_stack: list, max_steps=100):
        """
        Returns a list of tuples (final_state, residual_stack, accumulated_string)
        for all possible consumption paths.
        Returns [] if max_steps is exceeded.
        """

        results = []
        stack = [(current_q, current_stack, "")]

        steps = 0

        while stack:
            steps += 1
            if steps > max_steps:
                return []

            q, s_stack, acc = stack.pop()

            if hasattr(self, "precomputed_configurations"):
                key = (q, tuple(s_stack))
                if key in self.precomputed_configurations:
                    new_acc = acc + self.precomputed_configurations[key][1]
                    stack.append((list(self.final_states)[0], [self.stack_bottom], new_acc))
                    continue

            if not s_stack:
                results.append((q, s_stack, acc))
                continue

            top = s_stack[0]
            found = False

            for (q_src, top_sym, q_dst) in self.ma.transitions:
                if q_src == q and top_sym == top:
                    found = True
                    new_stack = s_stack[1:]
                    new_acc = acc + self.sigma_gamma.get((q, top, q_dst), "")
                    stack.append((q_dst, new_stack, new_acc))

            if not found:
                results.append((q, s_stack, acc))

        return results

    def suggest_continuations(self, q: str, stack: list, t_rem: str, eos: str):

        if not stack:
            return []
        
        if not hasattr(self, '_eos_pattern'):
            self._eos_pattern = re.compile(eos)
        
        continuations = set()
        visited = set()
        queue = deque([(q, stack)])

        while queue:

            current_state, current_stack = queue.popleft()

            for next_q, a, curr_stack_top, curr_push in self.get_configuration_moves(current_state, current_stack):

                if a == "":
                    if (curr_stack_top, curr_push) in visited:
                        continue
                    visited.add((curr_stack_top, curr_push))
                    queue.append((next_q, self.apply_stack_operation(gamma=current_stack, pop=curr_stack_top, push=curr_push)))
                    continue

                if self._eos_pattern.fullmatch(a):
                    continuations.add(eos)
                    continue
                    
                completion = self.a_starts_with_t_completion(t_rem, a)
                if completion is None:
                    continue

                continuations.add(completion)
                
                new_stack = tuple(self.apply_stack_operation(current_stack, curr_stack_top, curr_push))
                config_key = (next_q, new_stack)

                if config_key in self.precomputed_configurations:
                    continuations.add(completion + self.precomputed_configurations[config_key][1])

                else:
                    results = self.consume_stack(next_q, new_stack)
                    min_dst = float("inf")
                    lead_to_accepting = False
                    min_str = ""
                    for reached_state, reached_stack, accumulated_str in results:
                        if reached_state not in self._accepting_states:
                            continue
                        lead_to_accepting = True
                        temp_str = self.tokenizer.encode(accumulated_str)
                        temp_dst = len(temp_str)
                        if min_dst > temp_dst:
                            min_dst = temp_dst
                            min_str = self.tokenizer.decode(temp_str)
                    if lead_to_accepting:
                        if self.tokenizer.bos_token_id:
                            min_str = min_str[self.tokenizer.bos_token_id:]
                        continuations.add(completion + min_str)
        
        return list(continuations)

    def suggest_alphabet(self):
        return [self.a_starts_with_t_completion("", a) for a in self.input_alphabet]

    @lru_cache(maxsize=1000000)
    def suggest_continuations_top(self, q: str, stack: tuple, t_rem: str, eos: str):

        if not stack:
            return []

        visited = set()
        queue = deque([(q, stack)])

        suggestions = set()

        while queue:

            current_state, current_stack = queue.popleft()
            moves = self.configuration_moves[(current_state, current_stack[0])]
            for next_q, a, curr_stack_top, curr_push in list(moves):

                if a == "":
                    if (curr_stack_top, curr_push) in visited:
                        continue
                    visited.add((curr_stack_top, curr_push))
                    queue.append((next_q, self.apply_stack_operation(gamma=current_stack, pop=curr_stack_top, push=curr_push)))
                    continue

                else:

                    suggestions.add(self.a_starts_with_t_completion(t_rem, a))
                
        return suggestions

    def save_precomputation(self, path: str) -> None:
        """
        Save the current precomputed configurations to a JSON file.

        Args:
            path: File path where to save the data.
        """
        if not hasattr(self, 'precomputed_configurations') or self.precomputed_configurations is None:
            raise RuntimeError("No precomputed configurations to save. Run precompute_configurations first.")

        # Convert keys to JSON strings and values to lists
        serializable = {}
        for (state, stack), (min_dst, min_str) in self.precomputed_configurations.items():
            # state is a string, stack is a tuple of strings
            key = json.dumps((state, stack), separators=(',', ':'))
            serializable[key] = [min_dst, min_str]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    
class MA:
    """Multi-Automaton (MA) structure."""

    def __init__(self, states: set, stack_alphabet: set, transitions: set,
                 initial_states: set, final_state: str, tokenizer=None):
        self.states = states
        self.stack_alphabet = stack_alphabet
        self.transitions = transitions
        self.initial_states = initial_states
        self.final_state = final_state
        self.tokenizer = tokenizer

    def visualize(self):
        print("\n=== MULTI-AUTOMATON ===")
        print("States:", self.states)
        print("Stack alphabet:", self.stack_alphabet)
        print("Initial states:", self.initial_states)
        print("Final state:", self.final_state)
        print("Transitions:")
        for t in self.transitions:
            print("  ", t)
        print(f"Total: {len(self.states)} states, {len(self.transitions)} transitions")
        print("=======================\n")

    def increased_MA(self, pda: PDA, sigma_gamma: dict, dist_gamma: dict, min_transitions: dict, tokenizer=None):
        # Copy current MA data
        new_transitions = self.transitions.copy()
        new_sigma_gamma = sigma_gamma.copy()
        new_dist_gamma = dist_gamma.copy()

        distance_func = (lambda x: len(tokenizer.encode(x)) if tokenizer is not None else (len(x) if x != "" else float("inf")))

        def epsilon_closure(start_map):
            result = start_map.copy()
            queue = deque(start_map.keys())
            while queue:
                s = queue.popleft()
                d, l = result[s]
                for (src, sym, t) in self.transitions:
                    if src == s and sym == "":
                        if t not in result or d < result[t][0]:
                            result[t] = (d, l)
                            queue.append(t)
            return result

        # Process each minimized PDA transition
        for (q, sigma, A), destinations in min_transitions.items():
            for (q_prime, B) in destinations:
                # Start from q_prime, apply epsilon closure
                dp = epsilon_closure({q_prime: (0, "")})

                # Read each symbol of B in order
                for b in B:
                    new_dp = {}
                    for s, (d, l) in dp.items():
                        for (src, sym, t) in self.transitions:
                            if src == s and sym == b:
                                key = (s, b, t)
                                if key not in sigma_gamma:   # safety check
                                    continue
                                new_d = d + dist_gamma[key]
                                new_l = l + sigma_gamma[key]
                                if t not in new_dp or new_d < new_dp[t][0]:
                                    new_dp[t] = (new_d, new_l)
                    dp = epsilon_closure(new_dp)

                # For every state reachable after reading B (and epsilon),
                # add or update the MA transition (q, A, r)
                for r, (d, l) in dp.items():
                    total_label = sigma + l
                    total_dist = distance_func(total_label)
                    key = (q, A, r)
                    if key in new_sigma_gamma:
                        if total_dist < new_dist_gamma[key]:
                            new_sigma_gamma[key] = total_label
                            new_dist_gamma[key] = total_dist
                    else:
                        new_transitions.add((q, A, r))
                        new_sigma_gamma[key] = total_label
                        new_dist_gamma[key] = total_dist

        # Return the updated MA together with its label and distance maps
        return MA(self.states.copy(),
                self.stack_alphabet.copy(),
                new_transitions,
                self.initial_states.copy(),
                self.final_state), new_sigma_gamma, new_dist_gamma
    
class PRESTAR:
    """Compute pre* for a given PDA, starting from a target configuration."""

    def __init__(self, pda: PDA, tokenizer = None):
        self.pda = pda
        self.tokenizer = tokenizer

    def pre_star(self, nb_max_iterations: int = 50, debug=False ):
        """ Compute the pre* from a PDA by first converting it to a PDS, then to an MA, and then iteratively increasing the MA until a fixpoint is reached. """
        
        ma = self.pda.to_MA()
        nb_iterations = 0
        min_transitions = self.pda.get_min_transitions()
        iteration_stats = []

        while nb_iterations < nb_max_iterations:
            transitions_before = len(ma.transitions)
            if debug:
                print(f"PRE* iteration {nb_iterations}")
                print(f"Number of MA transitions: {transitions_before}")
            
            if nb_iterations == 0:
                sigma_gamma = {}
                dist_gamma = {}
                for f in self.pda.final_states:
                    sigma_gamma[(f, "", ma.final_state)] = ""
                    dist_gamma[(f, "", ma.final_state)] = 0
                for q in self.pda.states:
                    sigma_gamma[(q, "", q)] = ""
                    dist_gamma[(q, "", q)] = 0
            
            new_ma, new_sigma_gamma, new_dist_gamma = ma.increased_MA(self.pda, sigma_gamma, dist_gamma, min_transitions, self.tokenizer)
            transitions_after = len(new_ma.transitions)
            iteration_stats.append(
                {
                    "iteration": nb_iterations,
                    "ma_transitions_before": transitions_before,
                    "ma_transitions_after": transitions_after,
                    "ma_transitions_added": transitions_after - transitions_before,
                }
            )
            if debug:
                print(f"MA transitions added this iteration: {transitions_after - transitions_before}")
            
            if new_ma.transitions == ma.transitions:
                break
                
            ma = new_ma
            sigma_gamma = new_sigma_gamma
            dist_gamma = new_dist_gamma
            
            nb_iterations += 1
        self.last_pre_star_stats = {
            "pda": self.pda.stats(),
            "iterations": iteration_stats,
            "final_ma_transitions": len(ma.transitions),
        }
        return ma, sigma_gamma, dist_gamma