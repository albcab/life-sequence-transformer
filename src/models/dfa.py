import numpy as np
from typing import Callable, List, Tuple

def build_init_matcher(gender: str):
    gender_id = {"F": 11, "M": 12}[gender]

    def matches(tokens):
        t = np.asarray(tokens)

        if t.shape != (6,):
            return False
        if t[0] != 5: # PLCH0
            return False
        if t[-1] != 1: # BOL
            return False

        middle = t[1:-1]

        gid = np.full(middle.shape, -1, dtype=np.int64)
        gid[((13 <= middle) & (middle < 19)) | (middle == 10)] = 0 # A1..A6 + rarely [UNK]
        gid[(19 <= middle) & (middle < 31)] = 1 # MONTH_1..MONTH_12
        gid[(31 <= middle) & (middle < 117)] = 2 # YEAR_1913..YEAR_1998
        gid[middle == gender_id] = 3

        if np.any(gid < 0):
            return False

        return np.unique(gid).size == 4

    return matches


def _strictly_increasing_groups(tokens, group_masks):
    t = np.asarray(tokens)

    if t.size > len(group_masks):
        return False

    if t.size == 0:
        return True

    gid = np.full(t.shape, -1, dtype=np.int64)

    for i, mask in enumerate(group_masks):
        gid[mask(t)] = i

    return np.all(gid >= 0) and np.all(gid[1:] > gid[:-1])


def _one_value_set(set):
    return lambda tokens: all(token in set for token in tokens) #short-circuits


TOKEN_INDICES_DICT = {
    # INIT
    "initf": build_init_matcher("F"),
    "initm": build_init_matcher("M"),

    # MONTHS
    "monthone": _one_value_set({19}),
    "monthtwo": _one_value_set({20}),
    "monththree": _one_value_set({21}),
    "monthfour": _one_value_set({22}),
    "monthfive": _one_value_set({23}),
    "monthsix": _one_value_set({24}),
    "monthseven": _one_value_set({25}),
    "montheight": _one_value_set({26}),
    "monthnine": _one_value_set({27}),
    "monthten": _one_value_set({28}),
    "montheleven": _one_value_set({29}),
    "monthtwelve": _one_value_set({30}),

    # TIPOS
    "tipooneeleven": _one_value_set(set(range(129, 140))),  # 129..139
    
    # DURATIONS
    "durone": _one_value_set({117}),
    "duronetwo": _one_value_set(set(range(117, 119))),       # 117..118
    "duronethree": _one_value_set(set(range(117, 120))),     # 117..119
    "duronefour": _one_value_set(set(range(117, 121))),      # 117..120
    "duronefive": _one_value_set(set(range(117, 122))),      # 117..121
    "duronesix": _one_value_set(set(range(117, 123))),       # 117..122
    "duroneseven": _one_value_set(set(range(117, 124))),     # 117..123
    "duroneeight": _one_value_set(set(range(117, 125))),     # 117..124
    "duronenine": _one_value_set(set(range(117, 126))),      # 117..125
    "duroneten": _one_value_set(set(range(117, 127))),       # 117..126
    "duroneeleven": _one_value_set(set(range(117, 128))),    # 117..127
    "duronetwelve": _one_value_set(set(range(117, 129))),    # 117..128
    "durtwelve": _one_value_set({128}),

    # EOY / EOL
    "eoy": _one_value_set({3}),
    "eol": _one_value_set({2}),

    # AFTER TIPOS
    "aftertipooneelevenf": lambda tokens: _strictly_increasing_groups(
        tokens,
        [
            lambda t: (140 <= t) & (t < 238),   # INCOME
            lambda t: (238 <= t) & (t < 247),   # WRKT
            lambda t: (247 <= t) & (t < 357),   # WRKP
            lambda t: (357 <= t) & (t < 636),   # ATE
            lambda t: (636 <= t) & (t < 641),   # FSIZE
            lambda t: (t == 641) | (t == 642),  # FULL_TIME / PART_TIME
            lambda t: (643 <= t) & (t < 649),   # WRKINT
            lambda t: (649 <= t) & (t < 655),   # SIKINT
            lambda t: (655 <= t) & (t < 661),   # MATINT
        ],
    ),

    "aftertipooneelevenm": lambda tokens: _strictly_increasing_groups(
        tokens,
        [
            lambda t: (140 <= t) & (t < 238),   # INCOME_1..INCOME_98
            lambda t: (238 <= t) & (t < 247),   # WRKT_1..8
            lambda t: (247 <= t) & (t < 357),   # WRKP
            lambda t: (357 <= t) & (t < 636),   # ATE
            lambda t: (636 <= t) & (t < 641),   # FSIZE
            lambda t: (t == 641) | (t == 642),  # FULL/PART_TIME
            lambda t: (643 <= t) & (t < 649),   # WRKINT
            lambda t: (649 <= t) & (t < 655),   # SIKINT
        ]
    ),
}

class LIFESEQUENCEDFA:
    def __init__(self):
        self.initial_state = "q0"
        self.accepting_states = {"qfinal", "qm19", "qf19"}
        self.transitions = {

            # initilization states
            ("q0", TOKEN_INDICES_DICT["initf"]): "qf1",
            ("q0", TOKEN_INDICES_DICT["initm"]): "qm1",

            # F BRANCH
            # transitions for month tokens F
            ("qf1", TOKEN_INDICES_DICT["monthone"]): "qf2",
            ("qf1", TOKEN_INDICES_DICT["monthtwo"]): "qf3",
            ("qf1", TOKEN_INDICES_DICT["monththree"]): "qf4",
            ("qf1", TOKEN_INDICES_DICT["monthfour"]): "qf5",
            ("qf1", TOKEN_INDICES_DICT["monthfive"]): "qf6",
            ("qf1", TOKEN_INDICES_DICT["monthsix"]): "qf7",
            ("qf1", TOKEN_INDICES_DICT["monthseven"]): "qf8",
            ("qf1", TOKEN_INDICES_DICT["montheight"]): "qf9",
            ("qf1", TOKEN_INDICES_DICT["monthnine"]): "qf10",
            ("qf1", TOKEN_INDICES_DICT["monthten"]): "qf11",
            ("qf1", TOKEN_INDICES_DICT["montheleven"]): "qf12",
            ("qf1", TOKEN_INDICES_DICT["monthtwelve"]): "qf13",

            # MONTH_1 F
            ("qf2", TOKEN_INDICES_DICT["tipooneeleven"]): "qf15",
            ("qf15", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf17",
            ("qf15", TOKEN_INDICES_DICT["duronetwelve"]): "qf18",
            ("qf17", TOKEN_INDICES_DICT["duronetwelve"]): "qf18",

            ("qf2", TOKEN_INDICES_DICT["durtwelve"]): "qf16",

            # MONTH_2 F
            ("qf3", TOKEN_INDICES_DICT["tipooneeleven"]): "qf20",
            ("qf20", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf21",
            ("qf20", TOKEN_INDICES_DICT["duroneeleven"]): "qf18",
            ("qf21", TOKEN_INDICES_DICT["duroneeleven"]): "qf18",

            # MONTH_3 F
            ("qf4", TOKEN_INDICES_DICT["tipooneeleven"]): "qf22",
            ("qf22", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf23",
            ("qf22", TOKEN_INDICES_DICT["duroneten"]): "qf18",
            ("qf23", TOKEN_INDICES_DICT["duroneten"]): "qf18",

            # MONTH_4 F
            ("qf5", TOKEN_INDICES_DICT["tipooneeleven"]): "qf24",
            ("qf24", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf25",
            ("qf24", TOKEN_INDICES_DICT["duronenine"]): "qf18",
            ("qf25", TOKEN_INDICES_DICT["duronenine"]): "qf18",

            # MONTH_5 F
            ("qf6", TOKEN_INDICES_DICT["tipooneeleven"]): "qf26",
            ("qf26", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf27",
            ("qf26", TOKEN_INDICES_DICT["duroneeight"]): "qf18",
            ("qf27", TOKEN_INDICES_DICT["duroneeight"]): "qf18",

            # MONTH_6 F
            ("qf7", TOKEN_INDICES_DICT["tipooneeleven"]): "qf28",
            ("qf28", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf29",
            ("qf28", TOKEN_INDICES_DICT["duroneseven"]): "qf18",
            ("qf29", TOKEN_INDICES_DICT["duroneseven"]): "qf18",

            # MONTH_7 F
            ("qf8", TOKEN_INDICES_DICT["tipooneeleven"]): "qf30",
            ("qf30", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf31",
            ("qf30", TOKEN_INDICES_DICT["duronesix"]): "qf18",
            ("qf31", TOKEN_INDICES_DICT["duronesix"]): "qf18",

            # MONTH_8 F
            ("qf9", TOKEN_INDICES_DICT["tipooneeleven"]): "qf32",
            ("qf32", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf33",
            ("qf32", TOKEN_INDICES_DICT["duronefive"]): "qf18",
            ("qf33", TOKEN_INDICES_DICT["duronefive"]): "qf18",

            # MONTH_9 F
            ("qf10", TOKEN_INDICES_DICT["tipooneeleven"]): "qf34",
            ("qf34", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf35",
            ("qf34", TOKEN_INDICES_DICT["duronefour"]): "qf18",
            ("qf35", TOKEN_INDICES_DICT["duronefour"]): "qf18",

            # MONTH_10 F
            ("qf11", TOKEN_INDICES_DICT["tipooneeleven"]): "qf36",
            ("qf36", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf37",
            ("qf36", TOKEN_INDICES_DICT["duronethree"]): "qf18",
            ("qf37", TOKEN_INDICES_DICT["duronethree"]): "qf18",

            # MONTH_11 F
            ("qf12", TOKEN_INDICES_DICT["tipooneeleven"]): "qf38",
            ("qf38", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf39",
            ("qf38", TOKEN_INDICES_DICT["duronetwo"]): "qf18",
            ("qf39", TOKEN_INDICES_DICT["duronetwo"]): "qf18",

            # MONTH_12 F
            ("qf13", TOKEN_INDICES_DICT["tipooneeleven"]): "qf40",
            ("qf40", TOKEN_INDICES_DICT["aftertipooneelevenf"]): "qf41",
            ("qf40", TOKEN_INDICES_DICT["durone"]): "qf18",
            ("qf41", TOKEN_INDICES_DICT["durone"]): "qf18",

            # RESTART F
            ("qf18", TOKEN_INDICES_DICT["monthone"]): "qf2",
            ("qf18", TOKEN_INDICES_DICT["monthtwo"]): "qf3",
            ("qf18", TOKEN_INDICES_DICT["monththree"]): "qf4",
            ("qf18", TOKEN_INDICES_DICT["monthfour"]): "qf5",
            ("qf18", TOKEN_INDICES_DICT["monthfive"]): "qf6",
            ("qf18", TOKEN_INDICES_DICT["monthsix"]): "qf7",
            ("qf18", TOKEN_INDICES_DICT["monthseven"]): "qf8",
            ("qf18", TOKEN_INDICES_DICT["montheight"]): "qf9",
            ("qf18", TOKEN_INDICES_DICT["monthnine"]): "qf10",
            ("qf18", TOKEN_INDICES_DICT["monthten"]): "qf11",
            ("qf18", TOKEN_INDICES_DICT["montheleven"]): "qf12",
            ("qf18", TOKEN_INDICES_DICT["monthtwelve"]): "qf13",

            ("qf19", TOKEN_INDICES_DICT["monthone"]): "qf2",
            ("qf19", TOKEN_INDICES_DICT["monthtwo"]): "qf3",
            ("qf19", TOKEN_INDICES_DICT["monththree"]): "qf4",
            ("qf19", TOKEN_INDICES_DICT["monthfour"]): "qf5",
            ("qf19", TOKEN_INDICES_DICT["monthfive"]): "qf6",
            ("qf19", TOKEN_INDICES_DICT["monthsix"]): "qf7",
            ("qf19", TOKEN_INDICES_DICT["monthseven"]): "qf8",
            ("qf19", TOKEN_INDICES_DICT["montheight"]): "qf9",
            ("qf19", TOKEN_INDICES_DICT["monthnine"]): "qf10",
            ("qf19", TOKEN_INDICES_DICT["monthten"]): "qf11",
            ("qf19", TOKEN_INDICES_DICT["montheleven"]): "qf12",
            ("qf19", TOKEN_INDICES_DICT["monthtwelve"]): "qf13",

            # EOY or EOL f
            ("qf18", TOKEN_INDICES_DICT["eoy"]): "qf19",
            ("qf18", TOKEN_INDICES_DICT["eol"]): "qfinal",
            ("qf16", TOKEN_INDICES_DICT["eoy"]): "qf19",
            ("qf16", TOKEN_INDICES_DICT["eol"]): "qfinal",

            # M BRANCH
            # transitions for month tokens M
            ("qm1", TOKEN_INDICES_DICT["monthone"]): "qm2",
            ("qm1", TOKEN_INDICES_DICT["monthtwo"]): "qm3",
            ("qm1", TOKEN_INDICES_DICT["monththree"]): "qm4",
            ("qm1", TOKEN_INDICES_DICT["monthfour"]): "qm5",
            ("qm1", TOKEN_INDICES_DICT["monthfive"]): "qm6",
            ("qm1", TOKEN_INDICES_DICT["monthsix"]): "qm7",
            ("qm1", TOKEN_INDICES_DICT["monthseven"]): "qm8",
            ("qm1", TOKEN_INDICES_DICT["montheight"]): "qm9",
            ("qm1", TOKEN_INDICES_DICT["monthnine"]): "qm10",
            ("qm1", TOKEN_INDICES_DICT["monthten"]): "qm11",
            ("qm1", TOKEN_INDICES_DICT["montheleven"]): "qm12",
            ("qm1", TOKEN_INDICES_DICT["monthtwelve"]): "qm13",

            # MONTH_1 M
            ("qm2", TOKEN_INDICES_DICT["tipooneeleven"]): "qm15",
            ("qm15", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm17",
            ("qm15", TOKEN_INDICES_DICT["duronetwelve"]): "qm18",
            ("qm17", TOKEN_INDICES_DICT["duronetwelve"]): "qm18",

            ("qm2", TOKEN_INDICES_DICT["durtwelve"]): "qm16",

            # MONTH_2 M
            ("qm3", TOKEN_INDICES_DICT["tipooneeleven"]): "qm20",
            ("qm20", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm21",
            ("qm20", TOKEN_INDICES_DICT["duroneeleven"]): "qm18",
            ("qm21", TOKEN_INDICES_DICT["duroneeleven"]): "qm18",

            # MONTH_3 M
            ("qm4", TOKEN_INDICES_DICT["tipooneeleven"]): "qm22",
            ("qm22", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm23",
            ("qm22", TOKEN_INDICES_DICT["duroneten"]): "qm18",
            ("qm23", TOKEN_INDICES_DICT["duroneten"]): "qm18",

            # MONTH_4 M
            ("qm5", TOKEN_INDICES_DICT["tipooneeleven"]): "qm24",
            ("qm24", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm25",
            ("qm24", TOKEN_INDICES_DICT["duronenine"]): "qm18",
            ("qm25", TOKEN_INDICES_DICT["duronenine"]): "qm18",

            # MONTH_5 M
            ("qm6", TOKEN_INDICES_DICT["tipooneeleven"]): "qm26",
            ("qm26", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm27",
            ("qm26", TOKEN_INDICES_DICT["duroneeight"]): "qm18",
            ("qm27", TOKEN_INDICES_DICT["duroneeight"]): "qm18",

            # MONTH_6 M
            ("qm7", TOKEN_INDICES_DICT["tipooneeleven"]): "qm28",
            ("qm28", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm29",
            ("qm28", TOKEN_INDICES_DICT["duroneseven"]): "qm18",
            ("qm29", TOKEN_INDICES_DICT["duroneseven"]): "qm18",

            # MONTH_7 M
            ("qm8", TOKEN_INDICES_DICT["tipooneeleven"]): "qm30",
            ("qm30", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm31",
            ("qm30", TOKEN_INDICES_DICT["duronesix"]): "qm18",
            ("qm31", TOKEN_INDICES_DICT["duronesix"]): "qm18",

            # MONTH_8 M 
            ("qm9", TOKEN_INDICES_DICT["tipooneeleven"]): "qm32",
            ("qm32", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm33",
            ("qm32", TOKEN_INDICES_DICT["duronefive"]): "qm18",
            ("qm33", TOKEN_INDICES_DICT["duronefive"]): "qm18",

            # MONTH_9 M
            ("qm10", TOKEN_INDICES_DICT["tipooneeleven"]): "qm34",
            ("qm34", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm35",
            ("qm34", TOKEN_INDICES_DICT["duronefour"]): "qm18",
            ("qm35", TOKEN_INDICES_DICT["duronefour"]): "qm18",

            # MONTH_10 M
            ("qm11", TOKEN_INDICES_DICT["tipooneeleven"]): "qm36",
            ("qm36", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm37",
            ("qm36", TOKEN_INDICES_DICT["duronethree"]): "qm18",
            ("qm37", TOKEN_INDICES_DICT["duronethree"]): "qm18",

            # MONTH_11 M
            ("qm12", TOKEN_INDICES_DICT["tipooneeleven"]): "qm38",
            ("qm38", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm39",
            ("qm38", TOKEN_INDICES_DICT["duronetwo"]): "qm18",
            ("qm39", TOKEN_INDICES_DICT["duronetwo"]): "qm18",

            # MONTH_12 M
            ("qm13", TOKEN_INDICES_DICT["tipooneeleven"]): "qm40",
            ("qm40", TOKEN_INDICES_DICT["aftertipooneelevenm"]): "qm41",
            ("qm40", TOKEN_INDICES_DICT["durone"]): "qm18",
            ("qm41", TOKEN_INDICES_DICT["durone"]): "qm18",

            # RESTART M
            ("qm18", TOKEN_INDICES_DICT["monthone"]): "qm2",
            ("qm18", TOKEN_INDICES_DICT["monthtwo"]): "qm3",
            ("qm18", TOKEN_INDICES_DICT["monththree"]): "qm4",
            ("qm18", TOKEN_INDICES_DICT["monthfour"]): "qm5",
            ("qm18", TOKEN_INDICES_DICT["monthfive"]): "qm6",
            ("qm18", TOKEN_INDICES_DICT["monthsix"]): "qm7",
            ("qm18", TOKEN_INDICES_DICT["monthseven"]): "qm8",
            ("qm18", TOKEN_INDICES_DICT["montheight"]): "qm9",
            ("qm18", TOKEN_INDICES_DICT["monthnine"]): "qm10",
            ("qm18", TOKEN_INDICES_DICT["monthten"]): "qm11",
            ("qm18", TOKEN_INDICES_DICT["montheleven"]): "qm12",
            ("qm18", TOKEN_INDICES_DICT["monthtwelve"]): "qm13",

            ("qm19", TOKEN_INDICES_DICT["monthone"]): "qm2",
            ("qm19", TOKEN_INDICES_DICT["monthtwo"]): "qm3",
            ("qm19", TOKEN_INDICES_DICT["monththree"]): "qm4",
            ("qm19", TOKEN_INDICES_DICT["monthfour"]): "qm5",
            ("qm19", TOKEN_INDICES_DICT["monthfive"]): "qm6",
            ("qm19", TOKEN_INDICES_DICT["monthsix"]): "qm7",
            ("qm19", TOKEN_INDICES_DICT["monthseven"]): "qm8",
            ("qm19", TOKEN_INDICES_DICT["montheight"]): "qm9",
            ("qm19", TOKEN_INDICES_DICT["monthnine"]): "qm10",
            ("qm19", TOKEN_INDICES_DICT["monthten"]): "qm11",
            ("qm19", TOKEN_INDICES_DICT["montheleven"]): "qm12",
            ("qm19", TOKEN_INDICES_DICT["monthtwelve"]): "qm13",

            # EOY or EOL M
            ("qm18", TOKEN_INDICES_DICT["eoy"]): "qm19",
            ("qm18", TOKEN_INDICES_DICT["eol"]): "qfinal",
            ("qm16", TOKEN_INDICES_DICT["eoy"]): "qm19",
            ("qm16", TOKEN_INDICES_DICT["eol"]): "qfinal",
        }

        self.states = set()
        for (s, _), t in self.transitions.items():
            self.states.add(s)
            self.states.add(t)

        self.min_token_lengths = {
            'aftertipooneelevenf': 0,
            'aftertipooneelevenm': 0,
            'durone': 1,
            'duroneeight': 1,
            'duroneeleven': 1,
            'duronefive': 1,
            'duronefour': 1,
            'duronenine': 1,
            'duroneseven': 1,
            'duronesix': 1,
            'duroneten': 1,
            'duronethree': 1,
            'duronetwelve': 1,
            'duronetwo': 1,
            'durtwelve': 1,
            'eol': 1,
            'eoy': 1,
            'initf': 6,
            'initm': 6,
            'montheight': 1,
            'montheleven': 1,
            'monthfive': 1,
            'monthfour': 1,
            'monthnine': 1,
            'monthone': 1,
            'monthseven': 1,
            'monthsix': 1,
            'monthten': 1,
            'monththree': 1,
            'monthtwelve': 1,
            'monthtwo': 1,
            'tipooneeleven': 1,
        }


        self.state_distances = {
            'q0': 9,
            'qf1': 3,
            'qf10': 3,
            'qf11': 3,
            'qf12': 3,
            'qf13': 3,
            'qf15': 2,
            'qf16': 1,
            'qf17': 2,
            'qf18': 1,
            'qf19': 3,
            'qf2': 2,
            'qf20': 2,
            'qf21': 2,
            'qf22': 2,
            'qf23': 2,
            'qf24': 2,
            'qf25': 2,
            'qf26': 2,
            'qf27': 2,
            'qf28': 2,
            'qf29': 2,
            'qf3': 3,
            'qf30': 2,
            'qf31': 2,
            'qf32': 2,
            'qf33': 2,
            'qf34': 2,
            'qf35': 2,
            'qf36': 2,
            'qf37': 2,
            'qf38': 2,
            'qf39': 2,
            'qf4': 3,
            'qf40': 2,
            'qf41': 2,
            'qf5': 3,
            'qf6': 3,
            'qf7': 3,
            'qf8': 3,
            'qf9': 3,
            'qfinal': 0,
            'qm1': 3,
            'qm10': 3,
            'qm11': 3,
            'qm12': 3,
            'qm13': 3,
            'qm15': 2,
            'qm16': 1,
            'qm17': 2,
            'qm18': 1,
            'qm19': 3,
            'qm2': 2,
            'qm20': 2,
            'qm21': 2,
            'qm22': 2,
            'qm23': 2,
            'qm24': 2,
            'qm25': 2,
            'qm26': 2,
            'qm27': 2,
            'qm28': 2,
            'qm29': 2,
            'qm3': 3,
            'qm30': 2,
            'qm31': 2,
            'qm32': 2,
            'qm33': 2,
            'qm34': 2,
            'qm35': 2,
            'qm36': 2,
            'qm37': 2,
            'qm38': 2,
            'qm39': 2,
            'qm4': 3,
            'qm40': 2,
            'qm41': 2,
            'qm5': 3,
            'qm6': 3,
            'qm7': 3,
            'qm8': 3,
            'qm9': 3,
        }

    def get_initial_conf(self, sequence, verbose=False):
        configurations = {(self.initial_state, ())}
        temp_configurations = set()

        for element in sequence:

            if verbose:
                print("Element:", element)
        
                print("configs", configurations)
            while configurations:
                state, prefix = configurations.pop()

                if verbose:
                    print("Conf", state, prefix)
                for (current_state, matcher), next_state in self.transitions.items():
                    if state == current_state:
                        satisfied, extendable, fixable = evaluate_matcher(matcher=matcher, tokens=list(prefix) + [element])

                        if verbose:
                            print(f"satisfied={satisfied}, extendable={extendable}, partial={fixable}")
                        if satisfied and not extendable and not fixable:
                            if verbose:
                                print("add", (next_state, ()))
                            temp_configurations.add((next_state, ()))
                        elif satisfied and extendable and not fixable:
                            if verbose:
                                print("add", (next_state, ()))
                                print("add", ((current_state, prefix + (element,))))

                            temp_configurations.add((next_state, ()))
                            temp_configurations.add((current_state, prefix + (element,)))
                        elif not satisfied and not extendable and fixable:
                            if verbose:
                                print("add", ((current_state, prefix + (element,))))
                            temp_configurations.add((current_state, prefix + (element,)))
                        elif not satisfied and not extendable and not fixable:
                            pass

            #import time
            #time.sleep(5)

            configurations = set([x for x in temp_configurations])
            temp_configurations = set()
        
        return configurations

# ------------------------------------------------------------
# Metadata extraction (assuming the exact functions from the dict)
# ------------------------------------------------------------

def _get_one_value_set_param(f: Callable) -> frozenset:
    # The lambda is of the form: lambda tokens: all(token in SET for token in tokens)
    # We can inspect the code's constants. This is heuristic but works for the given dict.
    if hasattr(f, '__code__'):
        consts = f.__code__.co_consts
        for c in consts:
            if isinstance(c, (set, frozenset)):
                return frozenset(c)
            if isinstance(c, tuple) and len(c) > 0 and all(isinstance(x, int) for x in c):
                return frozenset(c)
    return frozenset()

def _get_increasing_groups_params(f: Callable) -> List[Callable]:
    # The lambda is: lambda tokens: _strictly_increasing_groups(tokens, [mask1, mask2, ...])
    # We try to retrieve the second argument from the function's closure/defaults.
    if hasattr(f, '__defaults__') and f.__defaults__:
        args = f.__defaults__
        if len(args) > 0 and isinstance(args[0], list):
            return args[0]
    # Fallback: look inside the code's constants for a list of lambdas
    if hasattr(f, '__code__'):
        consts = f.__code__.co_consts
        for c in consts:
            if isinstance(c, tuple) and len(c) > 0 and callable(c[0]):
                return list(c)
    return []

# Build a registry: matcher -> metadata dict
MATCHER_META = {}

# For each matcher in TOKEN_INDICES_DICT (assuming it's in scope)
# We'll manually register them here. In practice you can iterate over the dict.
# This is a one-time mapping.

# Register init matchers
for gender, gender_id in [('F', 11), ('M', 12)]:
    # matcher = build_init_matcher(gender)
    matcher = TOKEN_INDICES_DICT['initf' if gender == 'F' else 'initm']
    MATCHER_META[matcher] = {
        'type': 'fixed_length',
        'length': 6,
        'first_token': 5,
        'last_token': 1,
        'middle_groups': {0: {10} | set(range(13,19)),   # group 0: A1..A6 + UNK
                          1: set(range(19,31)),          # group 1: months
                          2: set(range(31,117)),         # group 2: years
                          3: {gender_id}}               # group 3: gender
    }

# Register one-value-set matchers with their explicit sets
one_value_sets = {
    "monthone": {19},
    "monthtwo": {20},
    "monththree": {21},
    "monthfour": {22},
    "monthfive": {23},
    "monthsix": {24},
    "monthseven": {25},
    "montheight": {26},
    "monthnine": {27},
    "monthten": {28},
    "montheleven": {29},
    "monthtwelve": {30},

    "tipooneeleven": set(range(129, 140)),  # 129..139

    "durone": {117},
    "duronetwo": set(range(117, 119)),        # 117..118
    "duronethree": set(range(117, 120)),      # 117..119
    "duronefour": set(range(117, 121)),       # 117..120
    "duronefive": set(range(117, 122)),       # 117..121
    "duronesix": set(range(117, 123)),        # 117..122
    "duroneseven": set(range(117, 124)),      # 117..123
    "duroneeight": set(range(117, 125)),      # 117..124
    "duronenine": set(range(117, 126)),       # 117..125
    "duroneten": set(range(117, 127)),        # 117..126
    "duroneeleven": set(range(117, 128)),     # 117..127
    "duronetwelve": set(range(117, 129)),     # 117..128
    "durtwelve": {128},

    "eoy": {3},
    "eol": {2},
}

for name, values in one_value_sets.items():
    matcher = TOKEN_INDICES_DICT[name]
    MATCHER_META[matcher] = {"type": "one_value_set", "set": frozenset(values)}

# # Register one-value-set matchers
# one_value_names = ['monthone', 'monthtwo', 'monththree', 'monthfour', 'monthfive',
#                    'monthsix', 'monthseven', 'montheight', 'monthnine', 'monthten',
#                    'montheleven', 'monthtwelve', 'durone', 'eoy', 'eol', 'durtwelve']
# # Their sets are defined in the dict; we extract them
# # We need to evaluate the dict entries. Assume TOKEN_INDICES_DICT is defined.
# for name in one_value_names:
#     matcher = TOKEN_INDICES_DICT[name]
#     s = _get_one_value_set_param(matcher)
#     MATCHER_META[matcher] = {'type': 'one_value_set', 'set': s}

# # Special duronetwo, duronethree, ... (they are also one_value_set but with ranges)
# duron_variants = ['duronetwo', 'duronethree', 'duronefour', 'duronefive', 'duronesix',
#                   'duroneseven', 'duroneeight', 'duronenine', 'duroneten', 'duroneeleven', 'duronetwelve']
# for name in duron_variants:
#     matcher = TOKEN_INDICES_DICT[name]
#     s = _get_one_value_set_param(matcher)
#     MATCHER_META[matcher] = {'type': 'one_value_set', 'set': s}

# Also register increasing groups matchers explicitly 
increasing_sets = {
    "aftertipooneelevenf": 
        [
            lambda t: (140 <= t) & (t < 238),   # INCOME
            lambda t: (238 <= t) & (t < 247),   # WRKT
            lambda t: (247 <= t) & (t < 357),   # WRKP
            lambda t: (357 <= t) & (t < 636),   # ATE
            lambda t: (636 <= t) & (t < 641),   # FSIZE
            lambda t: (t == 641) | (t == 642),  # FULL_TIME / PART_TIME
            lambda t: (643 <= t) & (t < 649),   # WRKINT
            lambda t: (649 <= t) & (t < 655),   # SIKINT
            lambda t: (655 <= t) & (t < 661),   # MATINT
        ],

    "aftertipooneelevenm": 
        [
            lambda t: (140 <= t) & (t < 238),   # INCOME_1..INCOME_98
            lambda t: (238 <= t) & (t < 247),   # WRKT_1..5
            lambda t: (247 <= t) & (t < 357),   # WRKP
            lambda t: (357 <= t) & (t < 636),   # ATE
            lambda t: (636 <= t) & (t < 641),   # FSIZE
            lambda t: (t == 641) | (t == 642),  # FULL/PART_TIME
            lambda t: (643 <= t) & (t < 649),   # WRKINT
            lambda t: (649 <= t) & (t < 655),   # SIKINT
        ]
}

for name, list_of_lambdas in increasing_sets.items():
    matcher = TOKEN_INDICES_DICT[name]
    MATCHER_META[matcher] = {"type": "increasing_groups", "group_masks": list_of_lambdas}

# # Increasing groups matchers
# inc_names = ['aftertipooneelevenf', 'aftertipooneelevenm']
# for name in inc_names:
#     matcher = TOKEN_INDICES_DICT[name]
#     masks = _get_increasing_groups_params(matcher)
#     # masks are lambda functions that return boolean masks; we store them as is
#     MATCHER_META[matcher] = {'type': 'increasing_groups', 'group_masks': masks}

# ------------------------------------------------------------
# Helper functions for each type
# ------------------------------------------------------------

def _satisfied_one_value_set(tokens: List[int], valid_set: frozenset) -> bool:
    return all(tok in valid_set for tok in tokens)

def _can_extend_one_value_set(tokens: List[int], valid_set: frozenset) -> bool:
    # Can we add at least one more token and still satisfied?
    # Only possible if valid_set is non-empty and (if empty, no tokens anyway)
    return len(valid_set) > 0

def _satisfied_fixed_length(tokens: List[int], meta: dict) -> bool:
    if len(tokens) != meta['length']:
        return False
    if tokens[0] != meta['first_token'] or tokens[-1] != meta['last_token']:
        return False
    middle = tokens[1:-1]
    # Check that middle tokens are a permutation of the four groups
    # Group each token
    groups = []
    for t in middle:
        g = None
        for gid, valid_vals in meta['middle_groups'].items():
            if t in valid_vals:
                g = gid
                break
        if g is None:
            return False
        groups.append(g)
    # Must have exactly one of each group 0,1,2,3
    return set(groups) == {0,1,2,3} and len(groups) == 4

def _can_extend_fixed_length(tokens: List[int], meta: dict) -> bool:
    # A prefix of a valid 6-token sequence?
    # We need to check if tokens can be extended (by adding tokens at the end) to a full valid sequence.
    # The order is fixed: first token must be 5, last token must be 1.
    # Middle four can be any order, but we already have some prefix.
    # For simplicity: if len(tokens) < 6 and the current prefix is consistent with some completion.
    # We'll allow any prefix of a valid permutation.
    if len(tokens) > meta['length']:
        return False
    # Check first token if present
    if len(tokens) >= 1 and tokens[0] != meta['first_token']:
        return False
    # If last token is present, it must be the last position
    if len(tokens) == meta['length'] and tokens[-1] != meta['last_token']:
        return False
    # For middle positions, we need to ensure that we don't have duplicates of a group
    # and that all tokens belong to some group.
    middle_prefix = tokens[1:-1] if len(tokens) > 1 and tokens[-1] != meta['last_token'] else tokens[1:]
    used_groups = set()
    for t in middle_prefix:
        found = False
        for gid, vals in meta['middle_groups'].items():
            if t in vals:
                if gid in used_groups:
                    return False  # duplicate group
                used_groups.add(gid)
                found = True
                break
        if not found:
            return False
    # If we have the last token in place, the middle must be exactly 4 tokens
    if len(tokens) == meta['length']:
        return len(used_groups) == 4
    # Otherwise, we can extendable as long as the remaining groups exist
    remaining_groups = {0,1,2,3} - used_groups
    # We also need to ensure that the last token will be 1 at the end.
    # That's always possible if we have space.
    # Additional constraint: the last token cannot appear before the end.
    if len(tokens) > 0 and tokens[-1] == meta['last_token']:
        # Then the last token is already placed; we must have exactly 4 middle tokens already
        return len(middle_prefix) == 4 and len(used_groups) == 4
    # Otherwise, we can always append missing middle tokens (any order) and then the final 1.
    return True

def _satisfied_increasing_groups(tokens: List[int], group_masks: List[Callable]) -> bool:
    # Re-implement _strictly_increasing_groups logic
    t = np.asarray(tokens)
    if t.size > len(group_masks):
        return False
    if t.size == 0:
        return True
    gid = np.full(t.shape, -1, dtype=np.int64)
    for i, mask in enumerate(group_masks):
        gid[mask(t)] = i
    return np.all(gid >= 0) and np.all(gid[1:] > gid[:-1])

def _can_extend_increasing_groups(tokens: List[int], group_masks: List[Callable]) -> bool:
    # Check if tokens form a prefix of some valid increasing sequence.
    # That means the group indices must be strictly increasing so far,
    # and there must be at least one higher group with at least one possible token.
    t = np.asarray(tokens)
    if t.size == 0:
        return len(group_masks) > 0  # can extend if there is any group
    # Compute group ids for each token
    gid = np.full(t.shape, -1, dtype=np.int64)
    for i, mask in enumerate(group_masks):
        gid[mask(t)] = i
    if np.any(gid < 0):
        return False
    if not np.all(gid[1:] > gid[:-1]):
        return False
    last_group = gid[-1]
    # Check if there exists any group index > last_group that is non-empty (i.e., mask has at least one token)
    for i in range(last_group+1, len(group_masks)):
        # Test if group i has any possible token value (we can check by calling mask on a representative)
        # Since masks are defined over integers, we can try to find any token that satisfies it.
        # We'll do a heuristic: check if the mask lambda returns True for some plausible value.
        # For simplicity, we assume that if a group mask is defined as range or set, it's non-empty.
        # But to be safe, we can check the mask's closure. However, given the dictionary, all masks are non-empty.
        return True
    return False

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

def evaluate_matcher(matcher: Callable, tokens: List[int]) -> Tuple[bool, bool, bool]:
    #Returns (satisfied, extendable, fixable) for the given matcher and token list.
    
    #- satisfied: matcher(tokens) is True.
    #- extendable: satisfied is True and there exists a longer token list that also satisfies.
    #- fixable: satisfied is False but there exists a longer list that satisfies.

    meta = MATCHER_META.get(matcher)
    if meta is None:
        raise ValueError("Unknown matcher. Only matchers from TOKEN_INDICES_DICT are supported.")
    
    if meta['type'] == 'one_value_set':
        valid_set = meta['set']
        satisfied = _satisfied_one_value_set(tokens, valid_set)
        if satisfied:
            extendable = _can_extend_one_value_set(tokens, valid_set)
            fixable = False
        else:
            extendable = False
            # Can we extend to a valid sequence? That requires that all current tokens are in the set
            # and we can add more tokens (if set not empty) OR we can add tokens to fix?
            # Actually fixable means we can add tokens (possibly at the end) to become satisfieding.
            # For one_value_set, if any token is not in set, no extension can fix it because adding more tokens won't remove the invalid one.
            # So fixable only if all tokens are in set (but sequence is empty or non-empty) AND we can add more tokens? Wait, satisfied false means at least one token is not in set.
            # If a token is not in set, the sequence can never become satisfieding because the invalid token remains.
            # However, if the sequence is empty, it does not satisfied (since empty? Actually empty sequence: all([]) is True, so satisfied would be True. So empty satisfies one_value_set.)
            # So for non-empty with a bad token -> cannot become satisfieding.
            # For empty, satisfied is True (already handled).
            # Thus fixable is always False for one_value_set.
            fixable = False
        return satisfied, extendable, fixable
    
    elif meta['type'] == 'fixed_length':
        satisfied = _satisfied_fixed_length(tokens, meta)
        if satisfied:
            # Can we extend? If len(tokens) < 6, yes; if len==6, no because no longer sequence can be valid (exact length)
            extendable = (len(tokens) < meta['length'])
            fixable = False
        else:
            extendable = False
            # Check if tokens can be extended to a valid sequence (prefix)
            fixable = _can_extend_fixed_length(tokens, meta)
        return satisfied, extendable, fixable
    
    elif meta['type'] == 'increasing_groups':
        group_masks = meta['group_masks']
        satisfied = _satisfied_increasing_groups(tokens, group_masks)
        if satisfied:
            extendable = _can_extend_increasing_groups(tokens, group_masks)
            fixable = False
        else:
            extendable = False
            # Check if it's a prefix of some valid increasing sequence
            # That means tokens must be strictly increasing groups so far, and all tokens belong to some group.
            t = np.asarray(tokens)
            if t.size == 0:
                fixable = len(group_masks) > 0
            elif t.size <= len(group_masks):
                gid = np.full(t.shape, -1, dtype=np.int64)
                for i, mask in enumerate(group_masks):
                    gid[mask(t)] = i
                if np.any(gid < 0):
                    fixable = False
                else:
                    if np.all(gid[1:] > gid[:-1]):
                        # Strictly increasing so far -> can extend if there is any higher group
                        last = gid[-1]
                        fixable = any(i > last for i in range(len(group_masks)))
                    else:
                        fixable = False
            else:
                fixable = False
        return satisfied, extendable, fixable
    
    else:
        raise ValueError("Unknown matcher type")

# ------------------------------------------------------------
# Example usage (assuming TOKEN_INDICES_DICT is defined)
# ------------------------------------------------------------
#s, c, p = evaluate_matcher(matcher, toks)
#print(f"{toks} -> satisfied={s}, extendable={c}, partial={p}")

#satisfied : the token list fully satisfies the matcher.
#extendable : satisfied is True and there exists a longer list (adding tokens) that also satisfies the matcher.
#fixable : satisfied is False but the list can be extended to a satisfieding list.