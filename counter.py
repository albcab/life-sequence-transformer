import pandas as pd
from pathlib import Path
import json
import numpy as np
import argparse


OUTFILE = "generated"
OUTFILE = "logs"

IMPLEMENTATION = "vno12_norm_pos_100sp"
NAME = "decoder"
SUFFIX = "test_ids_clean"

OFFSET_MOTHERS = 1
OFFSET_UNEMPLOYED = 1
OFFSET_PENSION = 3

def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction modules with configurable parameters")

    # Default parameters
    parser.add_argument("-i", "--implementation", type=str, default=IMPLEMENTATION, help="Model implementation version")
    parser.add_argument("-n", "--name", type=str, default=NAME, help="Model name")  
    parser.add_argument("-s", "--suffix", type=str, default=SUFFIX, help="Suffix for saving or loading")

    parser.add_argument("-om", "--offset_mothers", type=int, default=OFFSET_MOTHERS, help="Offset for maternity prediction")
    parser.add_argument("-ou", "--offset_unemployed", type=int, default=OFFSET_UNEMPLOYED, help="Offset for unemployment prediction")
    parser.add_argument("-op", "--offset_pension", type=int, default=OFFSET_PENSION, help="Offset for pension prediction")

    # Boolean flags
    parser.add_argument("--pension", action="store_true", help="Run the pension function")
    parser.add_argument("--maternity", action="store_true", help="Run the maternity function")
    parser.add_argument("--nonmaternity", action="store_true", help="Run the non maternity function")
    parser.add_argument("--unemployment", action="store_true", help="Run the unemployment function")
    parser.add_argument("--out_of_sample", action="store_true", help="Run the out_of_sample function")

    return parser.parse_args()


def pension(implementation, name, suffix, offset_pension, logs=False):

    # Step 2: Process each row
    def compute_row_result(row, num_years):
        row = row.tolist()
        try:
            idx_138 = row.index(138)
            count_3_before = row[:idx_138].count(3)
            if count_3_before >= num_years:
                return None
            preceding = row[idx_138 - 1]
            result = (preceding - 19) + (12 * count_3_before)
            return result
        except ValueError:
            return None  # 138 not found

    def compute_results(file, num_years):
        _df = pd.read_csv(file, header=None)
        # Step 1: Remove columns with 0 in the first row
        last = np.argmax(_df.iloc[1] != 0)
        df = _df.iloc[:, last:]
        # Apply to each row
        results = df.apply(lambda r: compute_row_result(r, num_years=num_years), axis=1)
        diff = results[1:] - results[0]
        if "decoder_only" in name:
            month = _df.iloc[0, 3] - 19
            year = 1913 + _df.iloc[0, 4] - 31
            prev_month = 12 * _df.iloc[0, :last].tolist().count(3)
            # age_month = 12 * (1990 - year) + results[0] + prev_month
            age_month = 12 * (1990 - year - 1) + (12 - month) + prev_month + results[0]
            return diff.tolist(), (age_month, month + 1)
        return diff.tolist(), None

    dir_name = Path(f"generated/{implementation}/{name}/pensioner_{suffix}/{offset_pension}")
    files = list(dir_name.glob("*_index.csv"))

    file_name = f"pensioner_{suffix}.csv"
    ids_df = pd.read_csv(file_name)

    for file in files:
        idx = int(file.name.split("_")[0])
        year_to_remove = ids_df.loc[ids_df.USER_ID == idx, "year_to_remove"].tolist()[0] + offset_pension
        diff, age_month = compute_results(file, year_to_remove)
        weight_file = dir_name / f"{idx}_weights.csv"
        weights = np.loadtxt(weight_file)
        for i, (d, weight) in enumerate(zip(diff, weights)):
            ids_df.loc[ids_df.USER_ID == idx, f"diff{i}"] = d
            ids_df.loc[ids_df.USER_ID == idx, f"weight{i}"] = weight
        if "decoder_only" in name:
            ids_df.loc[ids_df.USER_ID == idx, "real_age_months"] = age_month[0]
            ids_df.loc[ids_df.USER_ID == idx, "real_month"] = age_month[1]

    if not logs:
        ids_df.to_csv(dir_name.parent / f"results_pension.csv", index=False)
    else:
        ids_df.to_csv(f"{OUTFILE}/results_pension_{name}_{suffix}_{offset_pension}.csv", index=False)
    return None


def maternity(implementation, name, suffix, offset_mothers, logs=False, prefix=""):

    if len(implementation.split("100")) > 1:
        bins = 100
        upper = 237
    else:
        bins = 300
        upper = 437
    with open(f"income_{bins}_cache.json", "r") as f:
        value_dict = json.load(f)

    def process_subrow(subrow):
        values = [value_dict[str(v)] for v in subrow]
        # if not values:
        #     return None
        return sum(values) / len(values) if len(values) > 1 else values[0]
    
    def process_subrow_year(subrow, durs):
        values = [value_dict[str(v)] for v in subrow]
        return sum([val * dur for val, dur in zip(values, durs)])

    def compute_row_result(row, num_years=None):
        row = row.tolist()

        result = []
        current = []
        current_dur = []
        split_count = 0

        for i, val in enumerate(row):
            if val == 3:
                if current:
                    # result.append(process_subrow(current))
                    result.append(process_subrow_year(current, current_dur))
                    split_count += 1
                    current = []
                    current_dur = []
                else:
                    split_count += 1
                    result.append(None)
                if num_years is not None and split_count >= num_years:
                    break
            elif 140 <= val <= upper:
                if row[i - 1] < 129 or row[i - 1] > 139:
                    print(row[i - 1], row[i])
                    continue
                if row[i - 1] > 131:
                    continue
                current.append(val)
                try:
                    for j in range(1, 10):
                        if row[i + j] >= 117 and row[i + j] <= 128:
                            current_dur.append(row[i + j] - 116)
                            break
                except:
                    current.pop()
                    print("Number of incomes pre=", len(current))

        return result

    def compute_results(file, num_years):
        df = pd.read_csv(file, header=None)

        # Separate columns
        last = np.argmax(df.iloc[1] != 0)
        static_cols = df.columns[:last]
        dynamic_cols = df.columns[last:]

        # Compute static part once from the first row
        static_result = compute_row_result(df.loc[0, static_cols], num_years=None)

        results = []
        for idx, row in df.iterrows():
            dynamic_result = compute_row_result(row[dynamic_cols], num_years=num_years)
            full_result = static_result + dynamic_result
            results.append(full_result)

        return results
    
    def compute_maternity_row(row, num_years):
        row = row.tolist()
        split_count = 0
        for i, val in enumerate(row):
            if val == 129:
                for j in range(1, 11):
                    try:
                        if row[i + j] >= 657 and row[i + j] <= 660:
                            count_3_before = row[:i].count(3)
                            preceding = row[i - 1]
                            result = (preceding - 19) + (12 * count_3_before)
                            return result
                    except:
                        print(row[:6])
                        print(row[i:])
            elif val == 3:
                split_count += 1
            if split_count >= num_years:
                break
        return None

    def compute_maternity(file, num_years):
        df = pd.read_csv(file, header=None)
        # Step 1: Remove columns with 0 in the first row
        last = np.argmax(df.iloc[1] != 0)
        passed_years = df.iloc[0, :last].tolist().count(3)
        past_result = compute_maternity_row(df.iloc[0, :last], passed_years)
        if past_result is not None:
            return df.apply(lambda r: past_result, axis=1)
        # Apply to each row
        df = df.iloc[:, last:]
        results = df.apply(lambda r: compute_maternity_row(r, num_years=num_years), axis=1)
        return results + (12 * passed_years)
    

    def compute_row_inactivity(row, num_years=None):
        row = row.tolist()
        
        result = []
        current = [1] * 12
        split_count = 0

        for i, val in enumerate(row):
            if val == 3:
                result.append(sum(current))
                split_count += 1
                current = [1] * 12

                if num_years is not None and split_count >= num_years:
                    break

            elif 129 <= val <= 139 and val != 134: #no TIPO_6

                start_month = row[i - 1] - 19
                duration = None

                try:
                    for j in range(1, 15):
                        if 117 <= row[i + j] <= 128:
                            duration = row[i + j] - 116
                            break
                except IndexError:
                    print("Row too short to find duration")

                if duration is not None:
                    end_month = start_month + duration
                    try:
                        for m in range(start_month, end_month):
                            current[m] = 0
                    except:
                        print("Duration too long")

        return result

    def compute_inactivity(file, num_years):
        df = pd.read_csv(file, header=None)

        # Separate columns
        last = np.argmax(df.iloc[1] != 0)
        static_cols = df.columns[:last]
        dynamic_cols = df.columns[last:]

        # Compute static part once from the first row
        static_result = compute_row_inactivity(df.loc[0, static_cols], num_years=None)

        results = []
        for idx, row in df.iterrows():
            dynamic_result = compute_row_inactivity(row[dynamic_cols], num_years=num_years)
            full_result = static_result + dynamic_result
            results.append(full_result)

        return results
    
    dir_name = Path(f"generated/{implementation}/{name}/{prefix}mothers_{suffix}/{offset_mothers}")
    files = list(dir_name.glob("*_index.csv"))

    file_name = f"{prefix}mothers_{suffix}.csv"
    ids_df = pd.read_csv(file_name)
    
    rows = []
    for file in files:
        idx = int(file.name.split("_")[0])
        year_to_remove = ids_df.loc[ids_df.USER_ID == idx, "year_to_remove"].tolist()[0] + offset_mothers
        life_incomes = compute_results(file, year_to_remove)
        maternity = compute_maternity(file, year_to_remove)
        inactivity = compute_inactivity(file, year_to_remove)
        meta = ids_df.loc[ids_df.USER_ID == idx].to_dict(orient="list")
        weight_file = dir_name / f"{idx}_weights.csv"
        weights = np.loadtxt(weight_file)
        weights = np.concatenate([[np.nan], np.atleast_1d(weights)])
        for values, mat, inac, weight in zip(life_incomes, maternity, inactivity, weights):
            row = {k: i[0] for k, i in meta.items()}
            row["weight"] = weight
            row["maternity_months"] = mat
            lyear = 26 - len(values)
            for i in range(lyear):
                row[f"year{i}"] = None
            for i, v in enumerate(values):
                row[f'year{i+lyear}'] = v
            _lyear = 26 - len(inac)
            assert lyear == _lyear
            for i in range(_lyear):
                row[f"year{i}_inactivity"] = None
            for i, v in enumerate(inac):
                row[f'year{i+_lyear}_inactivity'] = v
            rows.append(row)

    final_df = pd.DataFrame(rows)
    if not logs:
        final_df.to_csv(dir_name.parent / f"results_{prefix}maternity.csv", index=False)
    else:
        final_df.to_csv(f"{OUTFILE}/results_{prefix}maternity_{name}_{suffix}_{offset_mothers}.csv", index=False)
    return None


def restore_generated_prefix(frame):
    """Restore zeroed shared context, preserving the first generated token."""
    if len(frame) < 2:
        raise ValueError("Expected an original sequence and at least one particle")
    starts = []
    for row in frame.iloc[1:].to_numpy():
        nonzero = np.flatnonzero(row)
        if not len(nonzero):
            raise ValueError("Cannot locate generation boundary in an empty particle")
        starts.append(int(nonzero[0]))
    if len(set(starts)) != 1:
        raise ValueError(f"Particles have inconsistent generation boundaries: {starts}")
    start = starts[0]
    restored = frame.copy()
    restored.iloc[1:, :start] = np.broadcast_to(
        frame.iloc[0, :start].to_numpy(), (len(frame) - 1, start)
    )
    return restored, start


def unemployment(implementation, name, suffix, offset_unemployed, logs=False):

    def compute_row_month(row):
        row = row.tolist()
        for i, val in enumerate(row):
            if val == 133:# and i + 2 < len(row):
                return row[i - 1] - 19

    def compute_row_result(row, num_years):
        row = row.tolist()
        result = 0
        year_result = 0
        just_one = False
        split_count = 0
        for i, val in enumerate(row):
            if val == 133:# and i + 2 < len(row):
                for j in range(1, 11):
                    if row[i + j] >= 117 and row[i + j] <= 128:
                        result += (row[i + j] - 116)
                        year_result += (row[i + j] - 116)
                        break
                # elif row[i + 1] >= 117 and row[i + 1] <= 128:
                #     result += (row[i + 1] - 116)
                # else: 
                #     print(split_count, num_years)
                #     input()
            elif val == 3:
                if year_result == 0:
                    if just_one:
                        break
                    else:
                        just_one = True
                split_count += 1
                year_result = 0
            if split_count >= num_years:
                break
        return result

    def compute_results(file, num_years):
        _df = pd.read_csv(file, header=None)
        # Reconstruct the shared context for every sampled life.
        _df, last = restore_generated_prefix(_df)
        #make last so that it is the element right after the last 3 in _df.iloc[0] that is also not a zero in _df.iloc[1]:
        try:
            last = last - list(reversed(_df.iloc[0, :(last - 1)])).index(3) - 1
        except:
            last = last - list(reversed(_df.iloc[0, :(last - 1)])).index(1) - 1
            print(file)
        df = _df.iloc[:, last:]
        # Apply to each row
        results = df.apply(lambda r: compute_row_result(r, num_years=num_years + 1), axis=1)
        months = df.apply(lambda r: compute_row_month(r), axis=1)
        diff = results[1:] - results[0]
        if not all(months[1:] == months[0]):
            raise ValueError(f"Unemployment starting months disagree in {file}: {months.tolist()}")
        if "decoder_only" in name:
            month = _df.iloc[0, 3] - 19
            year = 1913 + _df.iloc[0, 4] - 31
            prev_month = 12 * _df.iloc[0, :last].tolist().count(3)
            age_month = 12 * (1990 - year - 1) + (12 - month) + prev_month + months[0]
            return diff.tolist(), (age_month, results[0])
        return diff.tolist(), None

    dir_name = Path(f"generated/{implementation}/{name}/unemployed_{suffix}/{offset_unemployed}")
    files = list(dir_name.glob("*_index.csv"))

    file_name = f"unemployed_{suffix}.csv"
    ids_df = pd.read_csv(file_name)

    for file in files:
        idx = int(file.name.split("_")[0])
        year_to_remove = ids_df.loc[ids_df.USER_ID == idx, "year_to_remove"].tolist()[0] + offset_unemployed
        diff, age_month = compute_results(file, year_to_remove)
        weight_file = dir_name / f"{idx}_weights.csv"
        weights = np.loadtxt(weight_file)
        for i, (d, weight) in enumerate(zip(diff, weights)):
            ids_df.loc[ids_df.USER_ID == idx, f"diff{i}"] = d
            ids_df.loc[ids_df.USER_ID == idx, f"weight{i}"] = weight
        if "decoder_only" in name:
            ids_df.loc[ids_df.USER_ID == idx, "real_age_months"] = age_month[0]
            ids_df.loc[ids_df.USER_ID == idx, "real_results_months"] = age_month[1]

    if not logs:
        ids_df.to_csv(dir_name.parent / f"results_unemployment.csv", index=False)
    else:
        ids_df.to_csv(f"{OUTFILE}/results_unemployment_{name}_{suffix}_{offset_unemployed}.csv", index=False)
    return None


def out_of_sample(implementation, name, suffix, args, logs=False):

    def compute_row_result(row, num_years):
        row = row.tolist()
        results = []
        result = 0
        split_count = 0
        next_month = False
        next_dur = False
        next_month_or_year = False

        for i, val in enumerate(row):
            if val == 3:
                split_count += 1
            if split_count < num_years:
                continue

            if val == 3:
                next_month = True
                next_month_or_year = False
                results.append(result)
                result = 0
                if split_count - num_years > 19:
                    print("-", end="")
                    break
                continue

            if val == 0:
                if len(results) > 1:
                    results.append(result)
                break

            if next_month:
                if 19 <= val <= 30:
                    next_month = False
                    next_dur = True
                    continue
                else:
                    result = 1
                    continue

            if next_dur:
                if 129 <= val <= 660:
                    continue
                elif 117 <= val <= 128:
                    next_dur = False
                    next_month_or_year = True
                    continue
                else:
                    result = 1
                    continue

            if next_month_or_year:
                if 19 <= val <= 30:
                    next_month_or_year = False
                    next_dur = True
                    continue
                elif val == 3:
                    next_month = True
                    next_month_or_year = False
                    results.append(result)
                    result = 0
                    continue
                else:
                    result = 1
                    continue

            # If none of the conditions above matched
            result = 1

        return results[1:]

    def compute_results(file, num_years):
        df = pd.read_csv(file, header=None)
        # Step 1: Remove columns with 0 in the first row
        last = np.argmax(df.iloc[1] != 0)
        df = df.iloc[:, last:]
        # Apply to each row
        results = df.apply(lambda r: compute_row_result(r, num_years=num_years), axis=1)
        assert len(results[0]) == 0, results[0]
        return results[1:]
    
    rows = []
    for datas, offset in zip(["mothers", "unemployed", "pensioner"], [args.offset_mothers, args.offset_unemployed, args.offset_pension]):

        dir_name = Path(f"generated/{implementation}/{name}/{datas}_{suffix}/{offset}")
        files = list(dir_name.glob("*_index.csv"))

        file_name = f"{datas}_{suffix}.csv"
        ids_df = pd.read_csv(file_name)
        if datas == "pensioner":
            ids_df = ids_df.drop(ids_df.columns[0], axis=1)

        for file in files:
            idx = int(file.name.split("_")[0])
            year_to_remove = ids_df.loc[ids_df.USER_ID == idx, "year_to_remove"].tolist()[0] + offset
            errors = compute_results(file, year_to_remove)
            meta = ids_df.loc[ids_df.USER_ID == idx].to_dict(orient="list")
            for values in errors:
                row = {k: i[0] for k, i in meta.items()}
                for i, v in enumerate(values):
                    row[f'postyear{i}'] = v
                rows.append(row)

    final_df = pd.DataFrame(rows)
    if not logs:
        final_df.to_csv(dir_name.parent / f"results_oosample.csv", index=False)
    else:
        final_df.to_csv(f"{OUTFILE}/results_oosample_{name}_{suffix}.csv", index=False)
    return None


if __name__ == "__main__":

    args = parse_args()

    if args.pension:
        pension(args.implementation, args.name, args.suffix, args.offset_pension, True)

    if args.maternity:
        maternity(args.implementation, args.name, args.suffix, args.offset_mothers, True)

    if args.nonmaternity:
        maternity(args.implementation, args.name, args.suffix, args.offset_mothers, True, prefix="non_")

    if args.unemployment:
        unemployment(args.implementation, args.name, args.suffix, args.offset_unemployed, True)

    if args.out_of_sample:
        out_of_sample(args.implementation, args.name, args.suffix, args, True)

    # print("Choose implementation:")
    # print("1. vno12_norm_pos_100sp (default)")
    # print("2. vno12_norm_pos")
    # impl_choice = input("Select number (1 or 2): ").strip()
    # if impl_choice == "2":
    #     implementation = "vno12_norm_pos"
    # else:
    #     implementation = "vno12_norm_pos_100sp"

    # print("\nChoose name:")
    # print("1. decoder (default)")
    # print("2. decoder_only")
    # name_choice = input("Select number (1 or 2): ").strip()
    # if name_choice == "2":
    #     name = "decoder_only"
    # else:
    #     name = "decoder_dim"

    # print("\nChoose suffix:")
    # print("1. test_ids_clean (default)")
    # print("2. test_ids")
    # suffix_choice = input("Select number (1 or 2): ").strip()
    # if suffix_choice == "2":
    #     suffix = "test_ids"
    # else:
    #     suffix = "test_ids_clean"

    # print("\nChoose data:")
    # print("1. pension")
    # print("2. maternity")
    # print("3. unemployment")
    # print("4. out of sample")
    # data_choice = input("Select number (1, 2, 3 or 4): ").strip()
    # fun_map = {
    #     "1": pension,
    #     "2": maternity,
    #     "3": unemployment,
    #     "4": out_of_sample,
    # }

    # fun_map.get(data_choice, lambda: 'Invalid')(implementation, name, suffix)
