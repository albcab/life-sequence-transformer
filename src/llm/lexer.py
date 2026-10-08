import heapq
from functools import lru_cache
from typing import Tuple, Set, List
        
class LEXER:
    
    def __init__(self, automaton, tokenizer):
        self.automaton = automaton
        self.tokenizer = tokenizer

        if automaton.__class__.__name__ == "PDA":
            self.automaton_type = "PDA"
        else:
            raise ValueError("Type not supported")
    
    def remove_prefix(self, t: str, a: str) -> str:
        """
        Remove the matched prefix from string s.
        """
        return t[len(a):]
    
    def get_configurations(
        self,
        t: str,
        q: str,
        gamma: List[str],
        s_pref: str,
        max_steps: int = 100000,
        max_stack_height: int = 12,
        max_results: int = 1000000,
        return_first: bool = False,
        remaining_tokens: int = 1000,
        bos: bool = False,
        search_strategy: str = "dfs",
    ) -> Set[Tuple[str, Tuple[str, ...], str]]:
        
        if search_strategy == "bfs":
            return self._get_configurations_cached_bfs(
                t, q, tuple(gamma), s_pref, max_steps, max_stack_height, max_results, return_first, remaining_tokens, bos
            )
        elif search_strategy == "dfs":
            return self._get_configurations_cached_dfs(
                t, q, tuple(gamma), s_pref, max_steps, max_stack_height, max_results, return_first, remaining_tokens, bos
            )
        else:
            raise ValueError("Invalid search strategy. Use 'bfs' or 'dfs'.")

    @lru_cache(maxsize=1000000)
    def _get_configurations_cached_bfs(
        self,
        t: str,
        q: str,
        gamma_tuple: Tuple[str, ...],
        s_pref: str,
        max_steps: int,
        max_stack_height: int,
        max_results: int,
        return_first: bool = False,
        remaining_tokens: int = 1000,
        bos: bool = False,
    ) -> Set[Tuple[str, Tuple[str, ...], str]]:
        
        C = set()
        seen = set()
        heap = []
        item = (len(t), len(gamma_tuple), t, q, gamma_tuple, s_pref)
        heapq.heappush(heap, item)
        seen.add(item)
        steps = 0

        while heap:
            steps += 1
            if steps > max_steps:
                break

            # =========================
            # POP HEAP
            # =========================
            _, _, t_rem, q_curr, gamma_curr, s_pref_curr = heapq.heappop(heap)

            # =========================
            # CASE 1
            # =========================
            if s_pref_curr != "" and t_rem != "":

                combined = s_pref_curr + t_rem
                moves = self.automaton.configuration_moves[q_curr, gamma_curr[0]]

                for q_next, a, pop, push in moves:
                    
                    if a == "":
                        match, longest = True, ""
                    else:
                        match, longest = self.automaton.t_starts_with_a(combined, a)

                    if match:
                        new_gamma = self.automaton.apply_stack_operation(
                            gamma_curr, pop, push
                        )

                        if len(new_gamma) <= max_stack_height:

                            remaining = self.remove_prefix(combined, longest)

                            item = (len(remaining), len(new_gamma), remaining, q_next, new_gamma, "")
                            if item not in seen:
                                seen.add(item)
                                heapq.heappush(heap, item)

                    else:
                        cond = self.automaton.a_starts_with_t(combined, a)

                        if cond:
                            C.add((q_curr, tuple(gamma_curr), combined))

                            if return_first:
                                dst = self._compute_dst(
                                    automata=self.automaton,
                                    tokenizer=self.tokenizer,
                                    state=q_curr,
                                    stack=tuple(gamma_curr),
                                    remaining=combined,
                                    bos=bos
                                )

                                if dst < remaining_tokens - 1:
                                    return (q_curr, tuple(gamma_curr), combined), dst

            # =========================
            # CASE t_rem == ""
            # =========================
            if t_rem == "":
                C.add((q_curr, tuple(gamma_curr), ""))

                if return_first:
                    dst = self._compute_dst(
                        state=q_curr,
                        stack=tuple(gamma_curr),
                        remaining="",
                        bos=bos
                    )

                    if dst < remaining_tokens - 1:
                        return (q_curr, tuple(gamma_curr), ""), dst

            # =========================
            # CASE 2
            # =========================
            if s_pref_curr == "" and t_rem != "":

                moves = self.automaton.configuration_moves[q_curr, gamma_curr[0]]

                for q_next, a, pop, push in moves:
                    
                    if a == "":
                        match, longest = True, ""
                    else:
                        match, longest = self.automaton.t_starts_with_a(t_rem, a)

                    if match:
                        new_gamma = self.automaton.apply_stack_operation(
                            gamma_curr, pop, push
                        )

                        if len(new_gamma) <= max_stack_height:
                            remaining = self.remove_prefix(t_rem, longest)
                            item = (len(remaining), len(new_gamma), remaining, q_next, new_gamma, "")
                            if item not in seen:
                                seen.add(item)
                                heapq.heappush(heap, item)

                    else:
                        cond = self.automaton.a_starts_with_t(t_rem, a)

                        if cond:
                            C.add((q_curr, tuple(gamma_curr), t_rem))

                            if return_first:
                                dst = self._compute_dst(
                                    state=q_curr,
                                    stack=tuple(gamma_curr),
                                    remaining=t_rem,
                                    bos=bos
                                )

                                if dst < remaining_tokens - 1:
                                    return (q_curr, tuple(gamma_curr), t_rem), dst

        if not return_first:
            return C
        else:
            return (), float("inf")
    
    @lru_cache(maxsize=1000000)
    def _get_configurations_cached_dfs(
        self,
        t: str,
        q: str,
        gamma_tuple: Tuple[str, ...],
        s_pref: str,
        max_steps: int,
        max_stack_height: int,
        max_results: int,
        return_first: bool = False,
        remaining_tokens: int = 1000,
        bos: bool = False,
    ) -> Set[Tuple[str, Tuple[str, ...], str]]:

        C = set()
        seen = set()
        stack = []
        start_item = (t, q, gamma_tuple, s_pref)
        stack.append(start_item)
        seen.add(start_item)
        steps = 0

        while stack and steps < max_steps:
            steps += 1
            t_rem, q_curr, gamma_curr, s_pref_curr = stack.pop()

            # ---- CASE 1 ----
            if s_pref_curr != "" and t_rem != "":
                combined = s_pref_curr + t_rem
                moves = self.automaton.configuration_moves[q_curr, gamma_curr[0]]
                for q_next, a, pop, push in moves:
                    if a == "":
                        match, longest = True, ""
                    else:
                        match, longest = self.automaton.t_starts_with_a(combined, a)
                    if match:
                        new_gamma = self.automaton.apply_stack_operation(gamma_curr, pop, push)
                        if len(new_gamma) <= max_stack_height:
                            remaining = self.remove_prefix(combined, longest)
                            new_item = (remaining, q_next, new_gamma, "")
                            if new_item not in seen:
                                seen.add(new_item)
                                stack.append(new_item)
                    else:
                        if self.automaton.a_starts_with_t(combined, a):
                            C.add((q_curr, tuple(gamma_curr), combined))
                            if return_first:
                                dst = self._compute_dst(
                                    automata=self.automaton,
                                    tokenizer=self.tokenizer,
                                    state=q_curr,
                                    stack=tuple(gamma_curr),
                                    remaining=combined,
                                    bos=bos
                                )
                                if dst < remaining_tokens - 1:
                                    return (q_curr, tuple(gamma_curr), combined), dst

            # ---- CASE : t_rem == "" ----
            if t_rem == "":
                C.add((q_curr, tuple(gamma_curr), ""))
                if return_first:
                    dst = self._compute_dst(
                        state=q_curr,
                        stack=tuple(gamma_curr),
                        remaining="",
                        bos=bos
                    )
                    if dst < remaining_tokens - 1:
                        return (q_curr, tuple(gamma_curr), ""), dst

            # ---- CASE 2 ----
            if s_pref_curr == "" and t_rem != "":
                moves = self.automaton.configuration_moves[q_curr, gamma_curr[0]]
                for q_next, a, pop, push in moves:
                    if a == "":
                        match, longest = True, ""
                    else:
                        match, longest = self.automaton.t_starts_with_a(t_rem, a)
                    if match:
                        new_gamma = self.automaton.apply_stack_operation(gamma_curr, pop, push)
                        if len(new_gamma) <= max_stack_height:
                            remaining = self.remove_prefix(t_rem, longest)
                            new_item = (remaining, q_next, new_gamma, "")
                            if new_item not in seen:
                                seen.add(new_item)
                                stack.append(new_item)
                    else:
                        if self.automaton.a_starts_with_t(t_rem, a):
                            C.add((q_curr, tuple(gamma_curr), t_rem))
                            if return_first:
                                dst = self._compute_dst(
                                    state=q_curr,
                                    stack=tuple(gamma_curr),
                                    remaining=t_rem,
                                    bos=bos
                                )
                                if dst < remaining_tokens - 1:
                                    return (q_curr, tuple(gamma_curr), t_rem), dst

        if not return_first:
            return C
        else:
            return (), float("inf")
    
    @lru_cache(maxsize=1000000)
    def _compute_dst(self, bos: bool, state: int, stack: tuple, remaining: str) -> int:
        if remaining == "":
            if (state, stack) in self.automaton.precomputed_configurations:
                return self.automaton.precomputed_configurations[(state, stack)][0]
            temp = self.automaton.consume_stack(state, list(stack))
            #print(f"Precomputing configuration ({state}, {stack}) with {len(temp)} reachable configurations")
            #print(f"After consuming the stack, we have the following reachable configurations:")
            #for reached_state, reached_stack, accumulated_str in temp:
                #print(f"  - State: {reached_state}, Stack: {reached_stack}, Accumulated: {accumulated_str}")
            best = float("inf")
            for reached_state, reached_stack, accumulated_str in temp:
                if reached_state not in self.automaton._accepting_states:
                    continue
                if bos:
                    tokens = len(self.tokenizer.encode(accumulated_str)[1:])
                else:
                    tokens = len(self.tokenizer.encode(accumulated_str))
                best = min(best, tokens)
            if best != float("inf"):
                self.automaton.precomputed_configurations[(state, stack)] = (best, accumulated_str)
                self.automaton.new_configuration_found = True
            return best
        else:
            possible_moves = self.automaton.get_configuration_moves_with_prefix(state, stack, remaining)
            best = float("inf")
            for q_next, new_stack, completion in possible_moves:
                if completion is None:
                    continue
                if (q_next, new_stack) in self.automaton.precomputed_configurations:
                    cached = self.automaton.precomputed_configurations[(q_next, new_stack)]
                    if bos:
                        tokens = len(self.tokenizer.encode(completion + cached[1])[1:])
                    else:
                        tokens = len(self.tokenizer.encode(completion + cached[1]))
                    best = min(best, tokens)
                else:
                    temp = self.automaton.consume_stack(q_next, list(new_stack))
                    #print(f"Precomputing configuration ({state}, {stack}) with {len(temp)} reachable configurations")
                    #print(f"After consuming the stack, we have the following reachable configurations:")
                    #for reached_state, reached_stack, accumulated_str in temp:
                    #    print(f"  - State: {reached_state}, Stack: {reached_stack}, Accumulated: {accumulated_str}")
                    for reached_state, reached_stack, accumulated_str in temp:
                        if reached_state not in self.automaton._accepting_states:
                            continue
                        full_str = completion + accumulated_str
                        if bos:
                            tokens = len(self.tokenizer.encode(full_str)[1:])
                        else:
                            tokens = len(self.tokenizer.encode(full_str))
                        if tokens < best:
                            best = tokens
                            best_str = full_str
                    if best != float("inf"):
                        self.automaton.precomputed_configurations[(q_next, new_stack)] = (best, best_str)
                        self.automaton.new_configuration_found = True
        return best