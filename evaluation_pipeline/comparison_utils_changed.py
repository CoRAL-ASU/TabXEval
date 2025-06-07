from fuzzywuzzy import fuzz
import pandas as pd
import re
import ast


def find_first_number(s):

    # Remove commas from the string
    s = s.replace(',', '')
    # Regular expression pattern to find a number with optional decimals
    pattern = r'\d+\.\d+|\d+'
    # Search for the first match
    match = re.search(pattern, s)
    # Return the matched number or None if no match is found
    return float(match.group(0)) if match else None


def calculate_fuzzy_ratio(cell):
    if cell is None:
        return fuzz.ratio('', '')
    values = cell.split('/')
    if len(values) == 2:
        return fuzz.ratio(values[0].strip(), values[1].strip())
    return None


def table_to_dict_list(table_string, suffix=""):
    table_string = table_string.replace("markdown", "")
    # Split the table into lines
    lines = table_string.strip().split('\n')
    if lines and lines[0].startswith("|----"):
        lines = lines[1:]
    # Find the first line that contains the table header (i.e., a line with '|')
    table_start_idx = next(
        (i for i, line in enumerate(lines) if '|' in line), None)

    # If no table is found, return an empty list
    if table_start_idx is None:
        return []

    # Process the header from the detected table start line
    header = lines[table_start_idx].strip().split('|')
    header = [col.strip() for col in header if col.strip()]

    # Prepare the list to hold dictionaries
    table_as_dicts = []

    # Loop through each data row, skipping any separator rows and stopping at ``` or blank lines
    for line in lines[table_start_idx + 1:]:
        # Stop processing if the table ends
        if '```' in line or not line.strip():
            break

        # Skip lines that contain only '---'
        if '---' in line:
            continue

        row_values = line.strip().split('|')
        row_values = [val.strip() for val in row_values if val.strip()]

        # Create a dictionary for the current row, ensuring to match header order with values
        row_dict = {}
        for i in range(len(header)):
            if i < len(row_values):
                row_dict[header[i]] = row_values[i]
            else:
                row_dict[header[i]] = None  # Fill with None if data is missing

        table_as_dicts.append(row_dict)

    return table_as_dicts


def replace_em(dataf, result):
    updated_table = dataf.copy()
    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            if result.iloc[row, col] == 100:
                updated_table.iloc[row, col] = 'EM'

    return updated_table


def process_zero_ratio_cells(df, fuzzy_ratios):
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] != 'EM' and '/' in df.iloc[row, col]:
                # print(df.iloc[row, col])
                left, right = df.iloc[row, col].split('/', 1)
                if left.strip() in ['', '-']:
                    df.iloc[row, col] = 'EI'
                elif right.strip() in ['', '-']:
                    df.iloc[row, col] = 'MI'
    return df


def compare(aligned_table):
    aligned_table_list = table_to_dict_list(aligned_table)
    df = pd.DataFrame(aligned_table_list)
    mapped = df.map(calculate_fuzzy_ratio)
    values_to_delete = ['EM', 'MI', 'EI']
    df = replace_em(df, mapped)
    df_wo_em = df.copy()
    df = process_zero_ratio_cells(df, mapped)
    compare_table = df.copy()
    # Define the values to check
    values_to_check = {'EM', 'MI', 'EI'}

    # Identify rows to drop
    rows_to_drop = df.apply(lambda row: set(row) <= values_to_check, axis=1)
    # Identify columns to drop
    cols_to_drop = df.apply(lambda col: set(col) <= values_to_check, axis=0)
    # Drop the identified rows and columns
    df_cleaned = df.loc[~rows_to_drop, ~cols_to_drop]
    df_replaced = df_cleaned.replace({'EM': '-', 'MI': '-', 'EI': '-'})
    cols_em = df_wo_em.apply(lambda col: set(col) == {'EM'}, axis=0)
    rows_em = df_wo_em.apply(lambda row: set(row) == {'EM'}, axis=1)
    df_wo_em = df_wo_em.replace({'EM': '-'})
    df_wo_em = df_wo_em.loc[~rows_em, ~cols_em]
    return df_replaced, compare_table, df_wo_em


def table_to_dict_list_comparison(table_string, suffix=""):
    if table_string == []:
        return []
    table_string = table_string.replace("markdown", "")
    # Split the table into lines

    lines = table_string.strip().split('\n')
    if lines and lines[0].startswith("|----"):
        lines = lines[1:]
    # Find the first line that contains the table header (i.e., a line with '|')
    try:
        table_start_idx = next((i for i, line in enumerate(
            lines) if '|' in line and "--" in lines[i+1]), None)
    except:
        print("Error in table")
        print(table_string)

    # If no table is found, return an empty list
    if table_start_idx is None:
        return []

    # Process the header from the detected table start line
    header = lines[table_start_idx].strip().split('|')
    header = [col.strip() for col in header if col.strip()]

    # Prepare the list to hold dictionaries
    table_as_dicts = []

    # Loop through each data row, skipping any separator rows and stopping at ``` or blank lines
    for line in lines[table_start_idx + 1:]:
        # Stop processing if the table ends
        if '```' in line or not line.strip():
            break

        # Skip lines that contain only '---'
        if '---' in line:
            continue

        row_values = line.strip().split('|')
        row_values = [val.strip() for val in row_values if val.strip()]

        # Create a dictionary for the current row, ensuring to match header order with values
        row_dict = {}
        for i in range(len(header)):
            if i < len(row_values):
                row_dict[header[i]] = row_values[i]
            else:
                row_dict[header[i]] = None  # Fill with None if data is missing

        table_as_dicts.append(row_dict)

    return table_as_dicts


def calculate_average_second_part(data, column):
    """
    Calculates the average of the numeric values in the second part of a specified column.

    Args:
        data (list of dict): The dataset as a list of dictionaries.
        column (str): The column name to extract data from.

    Returns:
        float or None: The average of the numeric values, or None if no valid numbers are found.
    """

    numbers = []
    for item in data:
        # Split the column value by '/' to get the second part
        # print(type(column), column)
        # print(item)
        column_value = item.get(column, "")
        if column_value is not None and "/" in column_value:
            second_part = column_value.split('/')[0]
            # Extract the numeric value from the second part
            number = find_first_number(second_part)
            if number is not None:
                numbers.append(number)
            # print(column_value)

    # Calculate and return the average
    return sum(numbers) / len(numbers) if numbers else 0


def parse_string(input_string):
    if "list/list" in input_string.lower():
        return parse_string_to_lists(input_string)
    # Initialize the parsed data list
    parsed_data = []

    # Split the input string by ',' first
    parts = input_string.split(", ")
    if len(parts) > 5:
        parts[4] = ", ".join(parts[4:])
        parts = parts[:5]
        # print(parts)
    for part in parts:
        # print(part)
        if ":" in part:
            # Split by ':' to separate value and explanation
            parsed_data.append(part.replace(
                '[', "").replace(']', "").split(":"))
        elif "/" in part:
            # Handle components like 'Numerical/Numerical'
            parsed_data.append(part.replace(
                '[', "").replace(']', "").split("/"))
        elif part.strip().lower().replace('[', "").replace(']', "") == "none":
            # Handle the keyword 'None'
            parsed_data.append(None)
        else:
            # Handle other components
            parsed_data.append(part.strip().replace('[', "").replace(']', ""))
    return parsed_data


def parse_string_to_lists(input_string):
    parsed_list = []

    # Remove the outer square brackets if present
    if input_string.startswith('[') and input_string.endswith(']'):
        input_string = input_string[1:-1]

    # Find the "list difference" part using regex
    list_difference_match = re.search(r"list difference:(.+)", input_string)
    list_difference_data = None
    remaining_string = input_string

    if list_difference_match:
        list_difference_str = list_difference_match.group(1)
        remaining_string = input_string[:list_difference_match.start()].rstrip(
            ',')  # Remove trailing comma
        try:
            parts = list_difference_str.split(']:[')
            if len(parts) == 3:
                # Add square brackets back to each part to ensure they are proper lists
                parts[0] = parts[0] + ']'
                parts[1] = '[' + parts[1] + ']'
                parts[2] = '['+parts[2]
                # print(parts)
                list1 = ast.literal_eval(parts[0])
                list2 = ast.literal_eval(parts[1])
                list3 = ast.literal_eval(parts[2])
                list_difference_data = ['list difference', list1, list2, list3]
            else:
                print(
                    f"Warning: Unexpected format in list difference component: {list_difference_str}")
        except (SyntaxError, ValueError) as e:
            print(f"Warning: Could not parse list difference component: {e}")
            list_difference_data = None

    # Split the remaining string by comma
    items = remaining_string.split(',')

    for item in items:
        item = item.strip()
        if item:
            if '/' in item:
                key, value = item.split('/')[:2]
                parsed_list.append([key, value])
            elif item == "None":
                parsed_list.append(None)
            else:
                # Handle cases where a value might be left over without a key
                parsed_list.append(item)  # Consider how to best handle these

    if list_difference_data:
        parsed_list.append(list_difference_data)

    return parsed_list


def get_partial_cells_stats(alignments, allowed_data_types):
    idx_to_type = {0: "Data Type", 1: "NER", 2: "Unit",
                   3: "Missing Value", 4: "Numerical Difference"}
    Data_types = set()
    from copy import deepcopy
    deltas_non_column = {"bool": {"same": 0, "different": 0},
                         "Date": {"date": [], "time": []}, "List": []}
    deltas_columns = {"Numerical": {"unit_mismatch": 0, "ner_mismatch": 0, "delta": [], "MI": 0, "EI": 0}, "String": {
        "ner_mismatch": 0, "spell_errors": 0, "abbreviated_string": 0, "semantically": {"same": 0, "different": 0}, "other": 0, "MI": 0, "EI": 0}}

    for indexx, tables in enumerate(alignments):
        type_counts = {"Numerical": 0, "String": 0, "bool": 0,
                       "Date": 0, "List": 0, "Time": 0, "Others": 0}
        type_mismatch = 0
        deltas_table = deepcopy(deltas_non_column)
        empty_cells = {"MI": {}, "EI": {}, "Partial": {}}
        partial_scores = {"Numerical": 0, "Date": 0}
        compare_table = []
        if 'comparison_tuples_parsed' not in tables:
            # print("WHAAAAAAAAAAAAAAT")
            # print(tables['alignment'])
            continue
        # print(tables['comparison_tuples'])
        # print("\n\n")
        # print(tables['alignment'])
        # print("\n\n")

        for i, table in enumerate(tables['comparison_tuples_parsed']):
            differences = []
            row_updated = deepcopy(table)
            for k, v in table.items():
                # print(k, v)

                if v is None:
                    row_updated[k] = None
                    continue
                for idx, val in enumerate(v):
                    if idx > 2:
                        row_updated[k] = None
                        continue
                    if val == '-':
                        row_updated[k] = None
                        continue
                    if v is None:
                        row_updated[k] = None
                        continue
                deltas_table.setdefault(k, deepcopy(deltas_columns))
                # now we need to account for deltas
                if len(v) > 1:
                    if len(v) == 4:
                        print("WHAAAAT", v)
                    Data_types.add(v[0][0])
                    # Data_types.add(v[0][1])
                    if v[0] == '-' and len(v) > 4:
                        v = v[1:]
                    if v[0] == '-' and v[1] == '-':
                        v[0] = ['Empty', 'Empty']
                    if len(v[0]) > 1 and (v[0][0] == v[0][1]):
                        if v[0][0].lower() == "numerical" and v[1][0].lower() not in ["date", "time"]:
                            # print(v, "NumericaLLLLLLLLl")
                            type_counts["Numerical"] += 1
                            if v[1][0] != v[1][1]:
                                if isinstance(v[3], list) and "missing" in v[3][0].lower():
                                    deltas_table[k]['Numerical']['MI'] += 1
                                elif isinstance(v[3], list) and "extra" in v[3][0].lower():
                                    deltas_table[k]['Numerical']['EI'] += 1
                            if v[1][0] != v[1][1]:
                                deltas_table[k]['Numerical']['ner_mismatch'] += 1
                                empty_cells["Partial"].setdefault(v[0][0], 0)
                                empty_cells["Partial"][v[0][0]] += 1
                                continue
                            elif len(v) > 4 and v[4] is not None and 'absolute' in v[4][0] and 'difference' in v[4][0]:
                                summ = calculate_average_second_part(
                                    table_to_dict_list(tables['alignment']), k)

                                if summ is None or summ == 0:
                                    deltas_table[k]['Numerical']['delta'].append(find_first_number(
                                        v[4][1]))
                                    partial_scores["Numerical"] += min(1, find_first_number(
                                        v[4][1]))
                                else:
                                    if find_first_number(v[4][1]) is not None:
                                        deltas_table[k]['Numerical']['delta'].append(
                                            find_first_number(v[4][1])/summ)
                                        partial_scores["Numerical"] += min(1, find_first_number(
                                            v[4][1])/summ)

                            elif len(v) == 4 and v[3] is not None and 'absolute' in v[3][0] and 'difference' in v[3][0]:
                                summ = calculate_average_second_part(
                                    table_to_dict_list(tables['alignment']), k)

                                if summ is None or summ == 0:
                                    deltas_table[k]['Numerical']['delta'].append(find_first_number(
                                        v[3][1]))
                                    partial_scores["Numerical"] += min(1, find_first_number(
                                        v[3][1]))
                                else:
                                    deltas_table[k]['Numerical']['delta'].append(find_first_number(
                                        v[3][1])/summ)
                                    partial_scores["Numerical"] += min(1, find_first_number(
                                        v[3][1])/summ)
                            else:
                                print(v, "NumericaLLLLLLLLl")

                            deltas_table[k]['Numerical']['unit_mismatch'] += 1 if (
                                len(v[2]) > 1 and v[2][0].lower() != v[2][1].lower()) else 0
                        elif v[0][0].lower() in ["date", "time"] or (v[1][0].lower() in ["date", "time"] and v[0][0].lower() != "list"):

                            if v[1][0].lower() == "date" or v[0][0].lower() == "date":
                                type_counts["Date"] += 1
                                if len(v) < 5 and v[3] is not None:
                                    deltas_table['Date']['date'].append(
                                        find_first_number(v[3][1]))
                                    if find_first_number(v[3][1]) is not None and find_first_number(v[3][1]) > 0:
                                        partial_scores["Date"] += 1

                                elif v[4] is not None:
                                    print(v, "DATE")
                                    deltas_table['Date']['date'].append(
                                        find_first_number(v[4][1]))
                                    if find_first_number(v[4][1]) is not None and find_first_number(v[4][1]) > 0:
                                        print(v[4][1], "DATE",
                                              find_first_number(v[4][1]))
                                        partial_scores["Date"] += 1
                            elif v[1][0].lower() == "time" or v[0][0].lower() == "time":
                                type_counts["Time"] += 1
                                if len(v) > 4 and v[4] is not None:

                                    deltas_table['Date']['time'].append(
                                        find_first_number(v[4][1]))
                                elif v[3] is not None:
                                    deltas_table['Date']['time'].append(
                                        find_first_number(v[3][1]))
                            else:
                                print(v, "TIMEEEE")

                        elif v[0][0].lower() == "string":
                            # print(v, "String")
                            if v[1][0] != v[1][1]:
                                if isinstance(v[3], list) and "missing" in v[3][0].lower():
                                    deltas_table[k]['String']['MI'] += 1
                                elif isinstance(v[3], list) and "extra" in v[3][0].lower():
                                    deltas_table[k]['String']['EI'] += 1
                            if v[1][0] != v[1][1]:
                                deltas_table[k]['String']['ner_mismatch'] += 1

                            info_part = v[4] if len(
                                v) > 4 else v[3] if len(v) > 3 else None
                            if isinstance(info_part, list) and len(info_part) > 0:
                                info_part = info_part[0]
                            if info_part is None:
                                type_counts["String"] += 1
                                empty_cells["Partial"].setdefault(v[0][0], 0)
                                empty_cells["Partial"][v[0][0]] += 1
                                continue
                            if "abbreviated" in info_part:
                                deltas_table[k]['String']['abbreviated_string'] += 1
                            elif "spell" in info_part and "error" in info_part:
                                deltas_table[k]['String']['spell_errors'] += 1
                            elif "semantically" in info_part:

                                if "similar" in info_part:
                                    deltas_table[k]['String']['semantically']['same'] += 1
                                else:
                                    deltas_table[k]['String']['semantically']['different'] += 1
                            elif "other" in info_part:
                                deltas_table[k]['String']['other'] += 1
                            else:
                                print(v, "String")
                            type_counts["String"] += 1
                        elif v[0][0].lower() == "boolean":
                            type_counts["bool"] += 1
                            if "similar" in v[4][0]:
                                deltas_table['bool']['same'] += 1
                            else:
                                deltas_table['bool']['different'] += 1
                        elif v[0][0].lower() == "list":
                            type_counts["List"] += 1
                            # print(tables['comparison_tuples_parsed'])
                            # print(tables['comparison_tuples'])
                            # print()
                            try:
                                if len(v) > 4 and v[4] is not None:
                                    list1, list2 = v[4][1], v[4][2]
                                    list1 = [str(x).lower() for x in list1]
                                    list2 = [str(x).lower() for x in list2]
                                    mi = set(list1) - set(list2)
                                    ei = set(list2) - set(list1)
                                    em = set(list1) & set(list2)
                                    deltas_table["List"].append(
                                        {"MI": len(mi), "EI": len(ei), "EM": len(em)})
                                else:
                                    list1, list2 = v[3][1], v[3][2]
                                    list1 = [str(x).lower() for x in list1]
                                    list2 = [str(x).lower() for x in list2]
                                    mi = set(list1) - set(list2)
                                    ei = set(list2) - set(list1)
                                    em = set(list1) & set(list2)
                                    deltas_table["List"].append(
                                        {"MI": len(mi), "EI": len(ei), "EM": len(em)})
                            except:
                                pass

                        elif v[0][0].lower() == "empty":
                            row_updated[k] = None
                            continue

                        else:
                            type_counts["Others"] += 1
                            # print(v, "Data type not available")
                            # print(v[0][0].lower() == "numerical" and v[1]
                            #       [0].lower() not in ["date", "time"])

                        row_updated[k] = None
                        cell_data_type = v[0][0]
                        if v[0][0] not in allowed_data_types:
                            cell_data_type = "Others"
                        empty_cells["Partial"].setdefault(cell_data_type, 0)
                        empty_cells["Partial"][cell_data_type] += 1
                    else:
                        if v[0][0].lower() == "empty" and v[0][1].lower() != "empty":
                            cell_data_type = v[0][1]
                            if v[0][1] not in allowed_data_types:
                                cell_data_type = "Others"
                            empty_cells["EI"].setdefault(cell_data_type, 0)
                            empty_cells["EI"][cell_data_type] += 1
                            row_updated[k] = ("EI", v[0][1])
                        elif v[0][1].lower() == "empty" and v[0][0].lower() != "empty":
                            row_updated[k] = ("MI", v[0][0])
                            cell_data_type = v[0][0]
                            if v[0][0] not in allowed_data_types:
                                cell_data_type = "Others"
                            empty_cells["MI"].setdefault(cell_data_type, 0)
                            empty_cells["MI"][cell_data_type] += 1
                        type_mismatch += 1
                        # print(v, "data type mismatch")

            compare_table.append(row_updated)

        total_mi = 0
        total_ei = 0
        total_em = 0
        for delta in deltas_table["List"]:
            total_mi += delta["MI"]
            total_ei += delta["EI"]
            total_em += delta["EM"]
        deltas_table["List"] = {"MI": total_mi, "EI": total_ei, "EM": total_em}
        for k, v in deltas_table.items():
            if "Numerical" in v:
                deltas_table[k]['Numerical']['delta'] = sum(deltas_table[k]['Numerical']['delta'])/len(
                    deltas_table[k]['Numerical']['delta']) if len(deltas_table[k]['Numerical']['delta']) > 0 else 0
        deltas_table["Date"]["date"] = [
            x for x in deltas_table["Date"]["date"] if x]
        deltas_table["Date"]["time"] = [
            x for x in deltas_table["Date"]["time"] if x]
        deltas_table["Date"]["date"] = sum(deltas_table["Date"]["date"])/len(
            deltas_table["Date"]["date"]) if len(deltas_table["Date"]["date"]) > 0 else 0
        deltas_table["Date"]["time"] = sum(deltas_table["Date"]["time"])/len(
            deltas_table["Date"]["time"]) if len(deltas_table["Date"]["time"]) > 0 else 0
        alignments[indexx]['delta'] = deltas_table
        alignments[indexx]['type_counts'] = type_counts
        alignments[indexx]['empty_cells'] = empty_cells
        alignments[indexx]['ei_mi_table'] = compare_table
        alignments[indexx]['partial_scores'] = partial_scores
        # print(deltas_table)
    # print(type_counts)
    return alignments


def create_summary_table_from_df(df, allowed_data_types, default_categories=['EI', 'MI', 'Partial'], partial={}):
    # If default_categories is provided, ensure it is a list; otherwise use an empty list.
    if default_categories is None:
        default_categories = []

    # Initialize a dictionary to store counts per category.
    categories = {}

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        # For each cell in the row
        for cell in row:
            if cell is not None:
                # Check if the cell is a tuple of length 2
                if isinstance(cell, tuple) and len(cell) == 2:
                    category, dtype = cell
                    # If the data type is not allowed, classify it as "Others"
                    if dtype not in allowed_data_types:
                        dtype = "Others"
                    # Initialize the category in the dictionary if not already present
                    if category not in categories:
                        categories[category] = {
                            dt: 0 for dt in allowed_data_types}
                    # Increment the count for the corresponding data type
                    categories[category][dtype] += 1

    # Ensure that each default category is present even if it has no data
    for cat in default_categories:
        if cat not in categories:
            categories[cat] = {dt: 0 for dt in allowed_data_types}

    for a in allowed_data_types:
        if a in partial:
            categories["Partial"][a] = partial[a]
        else:
            categories["Partial"][a] = 0
    # Convert the dictionary into a DataFrame with the allowed data types as columns
    summary_df = pd.DataFrame.from_dict(categories, orient="index")
    # Ensure the column order is as specified in allowed_data_types

    return summary_df


def _get_column_type_count(table, num_rows, type):
    if len(table) < num_rows:
        return 0
    mask = table.map(lambda x: (
        isinstance(x, tuple) and x[0] == type))
    type_count = mask.all(axis=0).sum()
    # print(type_count, "COLUMN COUNT")
    # Remove columns that match the condition
    table.drop(columns=table.columns[mask.all(axis=0)], inplace=True)
    cols_to_remove = []
    for col_header in table.columns:
        parts = col_header.split("/")
        if type == "MI" and ((len(parts) == 1 and "T1" in parts[0]) or (len(parts) == 2 and parts[1].strip() in ['-', ""])):
            cols_to_remove.append(col_header)
            type_count += 1
        elif type == "EI" and ((len(parts) == 1 and "T1" in parts[0]) or (len(parts) == 2 and parts[0].strip() in ['-', ""])):
            cols_to_remove.append(col_header)
            type_count += 1
    table.drop(columns=cols_to_remove, inplace=True)
    return type_count


def _get_row_type_count(table, n_cols, type):
    current_cols_num = len(table.columns)
    if current_cols_num < n_cols:
        return 0
    mask = table.map(lambda x: (
        isinstance(x, tuple) and x[0] == type))
    # print(mask)
    type_count = mask.all(axis=1).sum()
    # print(type_count, "ROW COUNT")
    # Remove rows that match the condition
    table.drop(index=table.index[mask.all(axis=1)], inplace=True)

    return type_count


def get_row_column_statistics(ei_mi_table, df_compare, n_rows, n_cols):
    ei_column_count = _get_column_type_count(ei_mi_table, n_rows, "EI")
    mi_column_count = _get_column_type_count(ei_mi_table, n_rows, "MI")
    ei_row_count = _get_row_type_count(ei_mi_table,n_cols, "EI")
    mi_row_count = _get_row_type_count(ei_mi_table,n_cols, "MI")

    em_rows = df_compare.apply(lambda row: all(
        cell == "EM" for cell in row), axis=1).sum()
    em_columns = df_compare.apply(lambda col: all(
        cell == "EM" for cell in col), axis=0).sum()

    df_dict = {"type": "row", "MI": mi_row_count, "EI": ei_row_count, "EM": em_rows}, {
        "type": "column", "MI": mi_column_count, "EI": ei_column_count, "EM": em_columns}
    df_stats = pd.DataFrame(df_dict)

    return df_stats, ei_mi_table


def make_delta_stats_table(deltas, num_stats):
    # print(num_stats,"NUMSTATS")
    # print(deltas, "DELTAS")
    notes = "Notes: \n"
    notes += "STRING_TUPLE: (column name, ner_mismatch, spell_errors, abbreviated_string, semantically_same, semantically_different, other)\n"
    notes += "NUMERICAL_TUPLE: (column name, unit_mismatch, ner_mismatch, delta)\n"
    notes += "BOOLEAN_TUPLE: (same, different)\n"
    notes += "LIST_TUPLE: (MI, EI, EM)\n\n"
    md_table = """Type| Count |Differences|\n"""
    md_table += f"""Numerical | {num_stats.get('Numerical', 0)} |"""
    for k, v in deltas.items():
        if "Numerical" in v:
            all_zeros = all([x == 0 for x in v['Numerical'].values()])
            if all_zeros:
                continue
            md_table += f"""[{k}, {v['Numerical']['unit_mismatch']}, {v['Numerical']['ner_mismatch']}, {round(v['Numerical']['delta'])}], """
    md_table += "|\n"
    md_table += f"""String | {num_stats['String']} |"""
    for k, v in deltas.items():
        if "String" in v:
            all_zeros = v['String']['ner_mismatch'] | v['String']['spell_errors'] | v['String']['abbreviated_string'] | v[
                'String']['semantically']['same'] | v['String']['semantically']['different'] | v['String']['other']
            if not all_zeros:
                continue
            md_table += f"""[{k}, {v['String']['ner_mismatch']}, {v['String']['spell_errors']}, {v['String']['abbreviated_string']}, {v['String']['semantically']['same']}, {v['String']['semantically']['different']}, {v['String']['other']}], """
    md_table += "|\n"
    md_table += f"""Boolean | {num_stats['bool']} |{deltas['bool']['same']}, {deltas['bool']['different']}|\n"""
    md_table += f"""Date | {num_stats['Date']} |{deltas['Date']['date']}|\n"""
    md_table += f"""Time | {num_stats['Time']} |{deltas['Date']['time']}|\n"""
    md_table += f"""List | {num_stats['List']} |{deltas['List']['MI']}, {deltas['List']['EI']}, {deltas['List']['EM']}|\n"""
    md_table = notes + md_table
    return md_table


def make_empty_cells_table(data, allowed_data_types):

    # Get all unique types to use as columns
    all_types = allowed_data_types

    # Calculate the width for each column
    col_widths = {
        'Category': max(len(category) for category in data) + 2,
        **{t: max(len(t), len(str(max(types.get(t, 0) for types in data.values())))) + 2 for t in all_types}
    }

    # Create the header
    header = "| " + \
        " | ".join(f"{col:<{col_widths[col]}}" for col in [
                   'Category'] + all_types) + " |\n"
    header += "|-" + \
        "-|-".join('-' * col_widths[col]
                   for col in ['Category'] + all_types) + "-|\n"

    # Create rows for each category
    rows = ""
    for category, types in data.items():
        row = [str(types.get(t, 0)).ljust(col_widths[t]) for t in all_types]
        rows += f"| {category.ljust(col_widths['Category'])} | " + \
            " | ".join(row) + " |\n"

    # Combine header and rows
    markdown_table = header + rows
    return markdown_table


def get_row_col_statistics(table, compare_table, allowed_data_types):

    # MI row count (all cells are MI and not None, considering only the first part of the tuple)
    mi_row_count = table.map(lambda x: isinstance(
        x, tuple) and x[0] == 'MI').all(axis=1).sum()
    ei_row_count = table.map(lambda x: isinstance(
        x, tuple) and x[0] == 'EI').all(axis=1).sum()

    # EI column count (all cells are EI and not MI or None, considering only the first part of the tuple)
    ei_column_count = table.map(lambda x: x == 'EI' or (
        isinstance(x, tuple) and x[0] == 'EI')).all(axis=0).sum()
    mi_column_count = table.map(lambda x: x == 'MI' or (
        isinstance(x, tuple) and x[0] == 'MI')).all(axis=0).sum()

    em_rows = compare_table.apply(lambda row: all(
        cell == "EM" for cell in row), axis=1).sum()
    em_columns = compare_table.apply(lambda col: all(
        cell == "EM" for cell in col), axis=0).sum()

    df_dict = {"type": "row", "MI": mi_row_count, "EI": ei_row_count, "EM": em_rows}, {
        "type": "column", "MI": mi_column_count, "EI": ei_column_count, "EM": em_columns}

    df_stats = pd.DataFrame(df_dict)

    ei_mi_column_types = {}

    for data_type in allowed_data_types:
        ei_col_type = table.map(lambda x: isinstance(
            x, tuple) and x[0] == 'EI' and x[1] == data_type).all(axis=0).sum()
        mi_col_type = table.map(lambda x: isinstance(
            x, tuple) and x[0] == 'MI' and x[1] == data_type).all(axis=0).sum()
        ei_mi_column_types[data_type] = {"EI": ei_col_type, "MI": mi_col_type}

    df_ei_mi_column_types = pd.DataFrame(ei_mi_column_types)
    return df_stats, df_ei_mi_column_types
