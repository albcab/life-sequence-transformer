from src.llm.pda import PDA
from greenery import parse
from itertools import permutations
import sre_parse

def build_init_pattern(gender: str) -> str:
    assert gender in ("F", "M")

    area   = r"(A[1-6]|\[UNK\])"
    month  = r"(MONTH_([1-9]|1[0-2]))"
    year   = r"(YEAR_(191[3-9]|19[2-8][0-9]|199[0-8]))"
    gender_ = gender

    perms = [" ".join(p) for p in permutations([area, month, year, gender_])]
    return "<PLCH0> (" + "|".join(perms) + ") <BOL>"

INCOME = r"INCOME_([1-9]|[1-8][0-9]|9[0-8])"   # INCOME_1..INCOME_98
WRKT   = r"WRKT_[1-5]"
WRKP   = r"WRKP_[A-Z]{2}"                      # WRKP_XX = 2 caratteri alfabetici
ATE    = r"ATE_[A-Z]\d{1,3}(?:\(\d+\)|/\d+)"                 # ATE_Xxxx = 4 caratteri alfanumerici w/ extras maybe
FSIZE  = r"FSIZE_[1-5]"
TIME   = r"(FULL_TIME|PART_TIME)"
WRKINT = r"WRKINT_S(?:[0-4]|4\+)"
SIKINT = r"SIKINT_S(?:[0-4]|4\+)"
MATINT = r"MATINT_S(?:[0-4]|4\+)"

def build_after_tipo_pattern(gender: str) -> str:
    assert gender in ("F", "M")

    groups = [INCOME, WRKT, WRKP, ATE, FSIZE, TIME, WRKINT, SIKINT]
    if gender == "F":
        groups.append(MATINT)
    groups = [f"({g})" for g in groups]

    alts = []
    for i, g in enumerate(groups):
        tail = "".join(f"( {h})?" for h in groups[i + 1:])
        alts.append(g + tail)
    return "( (" + "|".join(alts) + "))?"

def regex_to_dfa(pattern: str):
    p = pattern
    if p.startswith('^'):
        p = p[1:]
    if p.endswith('$'):
        p = p[:-1]
    return parse(p).to_fsm()

def precompute_dfa_dict(pda_dict: dict) -> dict:
    input_alphabet = pda_dict.get("input_alphabet", set())
    dfa_dict = {}
    deadlock_states = {}
    for sym in input_alphabet:
        if not sym:
            continue
        try:
            dfa = regex_to_dfa(sym)
            dead = None
            for state, transitions in dfa.map.items():
                if all(next_state == state for next_state in transitions.values()) and state not in dfa.finals:
                    dead = state
                    break
            dfa_dict[sym] = (dfa, dead)
        except Exception as e:
            print(f"Warning: Could not parse regex pattern '{sym}': {e}")
            dfa_dict[sym] = None
    return dfa_dict


def create_pda(eos):
    REGEX_DICT = {
        "initf": build_init_pattern("F"),
        "initm": build_init_pattern("M"),

        # MONTHS
        "monthone": " MONTH_1",
        "monthtwo": " MONTH_2",
        "monththree": " MONTH_3",
        "monthfour": " MONTH_4",
        "monthfive": " MONTH_5",
        "monthsix": " MONTH_6",
        "monthseven": " MONTH_7",
        "montheight": " MONTH_8",
        "monthnine": " MONTH_9",
        "monthten": " MONTH_10",
        "montheleven": " MONTH_11",
        "monthtwelve": " MONTH_12",

        # TIPOS
        "tipooneeleven": " TIPO_([1-9]|1[01])",

        # DURATIONS
        "durone": " DUR_1",
        "duronetwo": " DUR_[1-2]",
        "duronethree": " DUR_[1-3]",
        "duronefour": " DUR_[1-4]",
        "duronefive": " DUR_[1-5]",
        "duronesix": " DUR_[1-6]",
        "duroneseven": " DUR_[1-7]",
        "duroneeight": " DUR_[1-8]",
        "duronenine": " DUR_[1-9]",
        "duroneten": " DUR_([1-9]|10)",
        "duroneeleven": " DUR_([1-9]|10|11)",
        "duronetwelve": " DUR_([1-9]|10|11|12)",
        "durtwelve": " DUR_12",

        # EOY / EOL
        "eoy": " <EOY>",
        "eol": eos,

        # AFTER TIPOS
        "aftertipooneelevenf": build_after_tipo_pattern("F"),
        "aftertipooneelevenm": build_after_tipo_pattern("M"),
    }

    pda = {
        "initial_state": "q0",
        "stack_bottom": "Z0",
        "final_states": {"qfinal"},
        "transitions": {

            # initial transitions for F and M
            ("q0", REGEX_DICT["initf"], "Z0"): [("qf1", ("EOS", "Z0"))],
            ("q0", REGEX_DICT["initm"], "Z0"): [("qm1", ("EOS", "Z0"))],

            # F BRANCH
            # transitions for month tokens F
            ("qf1", REGEX_DICT["monthone"], "EOS"): [("qf2", ("EOS",))],
            ("qf1", REGEX_DICT["monthtwo"], "EOS"): [("qf3", ("EOS",))],
            ("qf1", REGEX_DICT["monththree"], "EOS"): [("qf4", ("EOS",))],
            ("qf1", REGEX_DICT["monthfour"], "EOS"): [("qf5", ("EOS",))],
            ("qf1", REGEX_DICT["monthfive"], "EOS"): [("qf6", ("EOS",))],
            ("qf1", REGEX_DICT["monthsix"], "EOS"): [("qf7", ("EOS",))],
            ("qf1", REGEX_DICT["monthseven"], "EOS"): [("qf8", ("EOS",))],
            ("qf1", REGEX_DICT["montheight"], "EOS"): [("qf9", ("EOS",))],
            ("qf1", REGEX_DICT["monthnine"], "EOS"): [("qf10", ("EOS",))],
            ("qf1", REGEX_DICT["monthten"], "EOS"): [("qf11", ("EOS",))],
            ("qf1", REGEX_DICT["montheleven"], "EOS"): [("qf12", ("EOS",))],
            ("qf1", REGEX_DICT["monthtwelve"], "EOS"): [("qf13", ("EOS",))],

            # MONTH_1 F
            ("qf2", REGEX_DICT["tipooneeleven"], "EOS"): [("qf15", ("EOS",))],
            ("qf15", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf17", ("EOS",))],
            ("qf15", REGEX_DICT["duronetwelve"], "EOS"): [("qf18", ("EOS",))],
            ("qf17", REGEX_DICT["duronetwelve"], "EOS"): [("qf18", ("EOS",))],

            ("qf2", REGEX_DICT["durtwelve"], "EOS"): [("qf16", ("EOS",))],

            # MONTH_2 F
            ("qf3", REGEX_DICT["tipooneeleven"], "EOS"): [("qf20", ("EOS",))],
            ("qf20", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf21", ("EOS",))],
            ("qf20", REGEX_DICT["duroneeleven"], "EOS"): [("qf18", ("EOS",))],
            ("qf21", REGEX_DICT["duroneeleven"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_3 F
            ("qf4", REGEX_DICT["tipooneeleven"], "EOS"): [("qf22", ("EOS",))],
            ("qf22", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf23", ("EOS",))],
            ("qf22", REGEX_DICT["duroneten"], "EOS"): [("qf18", ("EOS",))],
            ("qf23", REGEX_DICT["duroneten"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_4 F
            ("qf5", REGEX_DICT["tipooneeleven"], "EOS"): [("qf24", ("EOS",))],
            ("qf24", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf25", ("EOS",))],
            ("qf24", REGEX_DICT["duronenine"], "EOS"): [("qf18", ("EOS",))],
            ("qf25", REGEX_DICT["duronenine"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_5 F
            ("qf6", REGEX_DICT["tipooneeleven"], "EOS"): [("qf26", ("EOS",))],
            ("qf26", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf27", ("EOS",))],
            ("qf26", REGEX_DICT["duroneeight"], "EOS"): [("qf18", ("EOS",))],
            ("qf27", REGEX_DICT["duroneeight"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_6 F
            ("qf7", REGEX_DICT["tipooneeleven"], "EOS"): [("qf28", ("EOS",))],
            ("qf28", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf29", ("EOS",))],
            ("qf28", REGEX_DICT["duroneseven"], "EOS"): [("qf18", ("EOS",))],
            ("qf29", REGEX_DICT["duroneseven"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_7 F
            ("qf8", REGEX_DICT["tipooneeleven"], "EOS"): [("qf30", ("EOS",))],
            ("qf30", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf31", ("EOS",))],
            ("qf30", REGEX_DICT["duronesix"], "EOS"): [("qf18", ("EOS",))],
            ("qf31", REGEX_DICT["duronesix"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_8 F
            ("qf9", REGEX_DICT["tipooneeleven"], "EOS"): [("qf32", ("EOS",))],
            ("qf32", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf33", ("EOS",))],
            ("qf32", REGEX_DICT["duronefive"], "EOS"): [("qf18", ("EOS",))],
            ("qf33", REGEX_DICT["duronefive"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_9 F
            ("qf10", REGEX_DICT["tipooneeleven"], "EOS"): [("qf34", ("EOS",))],
            ("qf34", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf35", ("EOS",))],
            ("qf34", REGEX_DICT["duronefour"], "EOS"): [("qf18", ("EOS",))],
            ("qf35", REGEX_DICT["duronefour"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_10 F
            ("qf11", REGEX_DICT["tipooneeleven"], "EOS"): [("qf36", ("EOS",))],
            ("qf36", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf37", ("EOS",))],
            ("qf36", REGEX_DICT["duronethree"], "EOS"): [("qf18", ("EOS",))],
            ("qf37", REGEX_DICT["duronethree"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_11 F
            ("qf12", REGEX_DICT["tipooneeleven"], "EOS"): [("qf38", ("EOS",))],
            ("qf38", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf39", ("EOS",))],
            ("qf38", REGEX_DICT["duronetwo"], "EOS"): [("qf18", ("EOS",))],
            ("qf39", REGEX_DICT["duronetwo"], "EOS"): [("qf18", ("EOS",))],

            # MONTH_12 F
            ("qf13", REGEX_DICT["tipooneeleven"], "EOS"): [("qf40", ("EOS",))],
            ("qf40", REGEX_DICT["aftertipooneelevenf"], "EOS"): [("qf41", ("EOS",))],
            ("qf40", REGEX_DICT["durone"], "EOS"): [("qf18", ("EOS",))],
            ("qf41", REGEX_DICT["durone"], "EOS"): [("qf18", ("EOS",))],

            # RESTART F
            ("qf18", REGEX_DICT["monthone"], "EOS"): [("qf2", ("EOS",))],
            ("qf18", REGEX_DICT["monthtwo"], "EOS"): [("qf3", ("EOS",))],
            ("qf18", REGEX_DICT["monththree"], "EOS"): [("qf4", ("EOS",))],
            ("qf18", REGEX_DICT["monthfour"], "EOS"): [("qf5", ("EOS",))],
            ("qf18", REGEX_DICT["monthfive"], "EOS"): [("qf6", ("EOS",))],
            ("qf18", REGEX_DICT["monthsix"], "EOS"): [("qf7", ("EOS",))],
            ("qf18", REGEX_DICT["monthseven"], "EOS"): [("qf8", ("EOS",))],
            ("qf18", REGEX_DICT["montheight"], "EOS"): [("qf9", ("EOS",))],
            ("qf18", REGEX_DICT["monthnine"], "EOS"): [("qf10", ("EOS",))],
            ("qf18", REGEX_DICT["monthten"], "EOS"): [("qf11", ("EOS",))],
            ("qf18", REGEX_DICT["montheleven"], "EOS"): [("qf12", ("EOS",))],
            ("qf18", REGEX_DICT["monthtwelve"], "EOS"): [("qf13", ("EOS",))],

            ("qf19", REGEX_DICT["monthone"], "EOS"): [("qf2", ("EOS",))],
            ("qf19", REGEX_DICT["monthtwo"], "EOS"): [("qf3", ("EOS",))],
            ("qf19", REGEX_DICT["monththree"], "EOS"): [("qf4", ("EOS",))],
            ("qf19", REGEX_DICT["monthfour"], "EOS"): [("qf5", ("EOS",))],
            ("qf19", REGEX_DICT["monthfive"], "EOS"): [("qf6", ("EOS",))],
            ("qf19", REGEX_DICT["monthsix"], "EOS"): [("qf7", ("EOS",))],
            ("qf19", REGEX_DICT["monthseven"], "EOS"): [("qf8", ("EOS",))],
            ("qf19", REGEX_DICT["montheight"], "EOS"): [("qf9", ("EOS",))],
            ("qf19", REGEX_DICT["monthnine"], "EOS"): [("qf10", ("EOS",))],
            ("qf19", REGEX_DICT["monthten"], "EOS"): [("qf11", ("EOS",))],
            ("qf19", REGEX_DICT["montheleven"], "EOS"): [("qf12", ("EOS",))],
            ("qf19", REGEX_DICT["monthtwelve"], "EOS"): [("qf13", ("EOS",))],

            # EOY or EOL f
            ("qf18", REGEX_DICT["eoy"], "EOS"): [("qf19", ("EOS",))],
            ("qf19", "", "EOS"): [("qfinal", ())],
            ("qf18", REGEX_DICT["eol"], "EOS"): [("qfinal", ())],
            ("qf16", REGEX_DICT["eoy"], "EOS"): [("qf19", ("EOS",))],
            ("qf16", REGEX_DICT["eol"], "EOS"): [("qfinal", ())],

            # M BRANCH
            # transitions for month tokens M
            ("qm1", REGEX_DICT["monthone"], "EOS"): [("qm2", ("EOS",))],
            ("qm1", REGEX_DICT["monthtwo"], "EOS"): [("qm3", ("EOS",))],
            ("qm1", REGEX_DICT["monththree"], "EOS"): [("qm4", ("EOS",))],
            ("qm1", REGEX_DICT["monthfour"], "EOS"): [("qm5", ("EOS",))],
            ("qm1", REGEX_DICT["monthfive"], "EOS"): [("qm6", ("EOS",))],
            ("qm1", REGEX_DICT["monthsix"], "EOS"): [("qm7", ("EOS",))],
            ("qm1", REGEX_DICT["monthseven"], "EOS"): [("qm8", ("EOS",))],
            ("qm1", REGEX_DICT["montheight"], "EOS"): [("qm9", ("EOS",))],
            ("qm1", REGEX_DICT["monthnine"], "EOS"): [("qm10", ("EOS",))],
            ("qm1", REGEX_DICT["monthten"], "EOS"): [("qm11", ("EOS",))],
            ("qm1", REGEX_DICT["montheleven"], "EOS"): [("qm12", ("EOS",))],
            ("qm1", REGEX_DICT["monthtwelve"], "EOS"): [("qm13", ("EOS",))],

            # MONTH_1 M
            ("qm2", REGEX_DICT["tipooneeleven"], "EOS"): [("qm15", ("EOS",))],
            ("qm15", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm17", ("EOS",))],
            ("qm15", REGEX_DICT["duronetwelve"], "EOS"): [("qm18", ("EOS",))],
            ("qm17", REGEX_DICT["duronetwelve"], "EOS"): [("qm18", ("EOS",))],

            ("qm2", REGEX_DICT["durtwelve"], "EOS"): [("qm16", ("EOS",))],

            # MONTH_2 M
            ("qm3", REGEX_DICT["tipooneeleven"], "EOS"): [("qm20", ("EOS",))],
            ("qm20", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm21", ("EOS",))],
            ("qm20", REGEX_DICT["duroneeleven"], "EOS"): [("qm18", ("EOS",))],
            ("qm21", REGEX_DICT["duroneeleven"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_3 M
            ("qm4", REGEX_DICT["tipooneeleven"], "EOS"): [("qm22", ("EOS",))],
            ("qm22", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm23", ("EOS",))],
            ("qm22", REGEX_DICT["duroneten"], "EOS"): [("qm18", ("EOS",))],
            ("qm23", REGEX_DICT["duroneten"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_4 M
            ("qm5", REGEX_DICT["tipooneeleven"], "EOS"): [("qm24", ("EOS",))],
            ("qm24", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm25", ("EOS",))],
            ("qm24", REGEX_DICT["duronenine"], "EOS"): [("qm18", ("EOS",))],
            ("qm25", REGEX_DICT["duronenine"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_5 M
            ("qm6", REGEX_DICT["tipooneeleven"], "EOS"): [("qm26", ("EOS",))],
            ("qm26", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm27", ("EOS",))],
            ("qm26", REGEX_DICT["duroneeight"], "EOS"): [("qm18", ("EOS",))],
            ("qm27", REGEX_DICT["duroneeight"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_6 M
            ("qm7", REGEX_DICT["tipooneeleven"], "EOS"): [("qm28", ("EOS",))],
            ("qm28", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm29", ("EOS",))],
            ("qm28", REGEX_DICT["duroneseven"], "EOS"): [("qm18", ("EOS",))],
            ("qm29", REGEX_DICT["duroneseven"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_7 M
            ("qm8", REGEX_DICT["tipooneeleven"], "EOS"): [("qm30", ("EOS",))],
            ("qm30", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm31", ("EOS",))],
            ("qm30", REGEX_DICT["duronesix"], "EOS"): [("qm18", ("EOS",))],
            ("qm31", REGEX_DICT["duronesix"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_8 M 
            ("qm9", REGEX_DICT["tipooneeleven"], "EOS"): [("qm32", ("EOS",))],
            ("qm32", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm33", ("EOS",))],
            ("qm32", REGEX_DICT["duronefive"], "EOS"): [("qm18", ("EOS",))],
            ("qm33", REGEX_DICT["duronefive"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_9 M
            ("qm10", REGEX_DICT["tipooneeleven"], "EOS"): [("qm34", ("EOS",))],
            ("qm34", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm35", ("EOS",))],
            ("qm34", REGEX_DICT["duronefour"], "EOS"): [("qm18", ("EOS",))],
            ("qm35", REGEX_DICT["duronefour"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_10 M
            ("qm11", REGEX_DICT["tipooneeleven"], "EOS"): [("qm36", ("EOS",))],
            ("qm36", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm37", ("EOS",))],
            ("qm36", REGEX_DICT["duronethree"], "EOS"): [("qm18", ("EOS",))],
            ("qm37", REGEX_DICT["duronethree"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_11 M
            ("qm12", REGEX_DICT["tipooneeleven"], "EOS"): [("qm38", ("EOS",))],
            ("qm38", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm39", ("EOS",))],
            ("qm38", REGEX_DICT["duronetwo"], "EOS"): [("qm18", ("EOS",))],
            ("qm39", REGEX_DICT["duronetwo"], "EOS"): [("qm18", ("EOS",))],

            # MONTH_12 M
            ("qm13", REGEX_DICT["tipooneeleven"], "EOS"): [("qm40", ("EOS",))],
            ("qm40", REGEX_DICT["aftertipooneelevenm"], "EOS"): [("qm41", ("EOS",))],
            ("qm40", REGEX_DICT["durone"], "EOS"): [("qm18", ("EOS",))],
            ("qm41", REGEX_DICT["durone"], "EOS"): [("qm18", ("EOS",))],

            # RESTART M
            ("qm18", REGEX_DICT["monthone"], "EOS"): [("qm2", ("EOS",))],
            ("qm18", REGEX_DICT["monthtwo"], "EOS"): [("qm3", ("EOS",))],
            ("qm18", REGEX_DICT["monththree"], "EOS"): [("qm4", ("EOS",))],
            ("qm18", REGEX_DICT["monthfour"], "EOS"): [("qm5", ("EOS",))],
            ("qm18", REGEX_DICT["monthfive"], "EOS"): [("qm6", ("EOS",))],
            ("qm18", REGEX_DICT["monthsix"], "EOS"): [("qm7", ("EOS",))],
            ("qm18", REGEX_DICT["monthseven"], "EOS"): [("qm8", ("EOS",))],
            ("qm18", REGEX_DICT["montheight"], "EOS"): [("qm9", ("EOS",))],
            ("qm18", REGEX_DICT["monthnine"], "EOS"): [("qm10", ("EOS",))],
            ("qm18", REGEX_DICT["monthten"], "EOS"): [("qm11", ("EOS",))],
            ("qm18", REGEX_DICT["montheleven"], "EOS"): [("qm12", ("EOS",))],
            ("qm18", REGEX_DICT["monthtwelve"], "EOS"): [("qm13", ("EOS",))],

            ("qm19", REGEX_DICT["monthone"], "EOS"): [("qm2", ("EOS",))],
            ("qm19", REGEX_DICT["monthtwo"], "EOS"): [("qm3", ("EOS",))],
            ("qm19", REGEX_DICT["monththree"], "EOS"): [("qm4", ("EOS",))],
            ("qm19", REGEX_DICT["monthfour"], "EOS"): [("qm5", ("EOS",))],
            ("qm19", REGEX_DICT["monthfive"], "EOS"): [("qm6", ("EOS",))],
            ("qm19", REGEX_DICT["monthsix"], "EOS"): [("qm7", ("EOS",))],
            ("qm19", REGEX_DICT["monthseven"], "EOS"): [("qm8", ("EOS",))],
            ("qm19", REGEX_DICT["montheight"], "EOS"): [("qm9", ("EOS",))],
            ("qm19", REGEX_DICT["monthnine"], "EOS"): [("qm10", ("EOS",))],
            ("qm19", REGEX_DICT["monthten"], "EOS"): [("qm11", ("EOS",))],
            ("qm19", REGEX_DICT["montheleven"], "EOS"): [("qm12", ("EOS",))],
            ("qm19", REGEX_DICT["monthtwelve"], "EOS"): [("qm13", ("EOS",))],

            # EOY or EOL M
            ("qm18", REGEX_DICT["eoy"], "EOS"): [("qm19", ("EOS",))],
            ("qm19", "", "EOS"): [("qfinal", ())],
            ("qm18", REGEX_DICT["eol"], "EOS"): [("qfinal", ())],
            ("qm16", REGEX_DICT["eoy"], "EOS"): [("qm19", ("EOS",))],
            ("qm16", REGEX_DICT["eol"], "EOS"): [("qfinal", ())],

            # add EOL self loop for qfinal
            ("qfinal", REGEX_DICT["eol"], "Z0"): [("qfinal", ("Z0",))],
        }
    }
    # extract stack alphabet and input alphabet
    stack_alphabet = set()
    input_alphabet = set()

    pda["states"] = set([k[0] for k in pda["transitions"].keys()])

    new_transtions = {}
    for (state, input_symbol, stack_top), transitions in pda["transitions"].items():
        if input_symbol == "EOS":
            new_transtions[(state, eos.replace("|", "\|"), stack_top)] = transitions
        else:
            new_transtions[(state, input_symbol, stack_top)] = transitions

    for (state, input_symbol, stack_top), transitions in pda["transitions"].items():
        for next_state, stack_replacement in transitions:
            stack_alphabet.update(stack_replacement)
        if input_symbol:
            input_alphabet.add(input_symbol)

    # update pda
    pda["stack_alphabet"] = stack_alphabet
    pda["input_alphabet"] = input_alphabet
    pda["input_alphabet_dfas"] = precompute_dfa_dict(pda)

    # create PDA 
    pda = PDA(**pda)

    return pda

def regex_characters(pattern):
    characters = set()

    def visit(parsed):
        for op, value in parsed:
            if op == sre_parse.LITERAL:
                characters.add(chr(value))

            elif op == sre_parse.IN:
                for sub_op, sub_value in value:
                    if sub_op == sre_parse.LITERAL:
                        characters.add(chr(sub_value))

                    elif sub_op == sre_parse.RANGE:
                        start, end = sub_value
                        characters.update(
                            chr(code)
                            for code in range(start, end + 1)
                        )

                    elif sub_op == sre_parse.CATEGORY:
                        if sub_value == sre_parse.CATEGORY_DIGIT:
                            characters.update("0123456789")

            elif op == sre_parse.SUBPATTERN:
                visit(value[-1])

            elif op == sre_parse.BRANCH:
                for branch in value[1]:
                    visit(branch)

            elif op in (
                sre_parse.MAX_REPEAT,
                sre_parse.MIN_REPEAT,
            ):
                visit(value[-1])

    visit(sre_parse.parse(pattern))
    return characters


def get_pda_characters(pda):
    characters = set()

    for pattern in pda.input_alphabet:
        characters.update(
            regex_characters(pattern)
        )

    return characters


def get_valid_token_ids(pda, tokenizer):
    characters = get_pda_characters(pda)

    valid_token_ids = []

    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        if set(decoded) <= characters:
            valid_token_ids.append(token_id)

    return valid_token_ids