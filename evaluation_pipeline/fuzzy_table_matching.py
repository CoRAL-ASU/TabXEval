from fuzzywuzzy import fuzz
from bert_score import score as bscore


def table_to_row_list(table_string):
    # Split the table into lines
    lines = table_string.strip().split('\n')

    # Find the first line that contains the table header (i.e., a line with '|')
    table_start_idx = next(
        (i for i, line in enumerate(lines) if '|' in line), None)

    # If no table is found, return an empty list
    if table_start_idx is None:
        return []

    # Process the header from the detected table start line
    header = lines[table_start_idx].strip().split('|')
    header = [col.strip() for col in header]

    # Initialize the output list with headers as the first row
    table_as_rows = [header]

    # Loop through each data row, skipping any separator rows and stopping at ``` or blank lines
    for line in lines[table_start_idx + 1:]:
        # Stop processing if the table ends
        if '```' in line or not line.strip():
            break

        # Skip lines that contain only '---'
        if '---' in line:
            continue

        # Split row values and handle empty cells as empty strings
        row_values = line.strip().split('|')
        row_values = [val.strip() if val.strip() else "" for val in row_values]

        # Ensure the row has the same number of columns as the header
        row = row_values + [""] * (len(header) - len(row_values))

        # Add the row to the table as rows
        table_as_rows.append(row)

    # modify the list of lists to remove all indexes with all empty strings
    table_as_rows = [row for row in table_as_rows if any(row)]
    table_struct = []
    idx_with_all_empty = set(list(range(len(table_as_rows[0]))))
    for row in table_as_rows:
        idx_with_empty = set([i for i, val in enumerate(row) if not val])
        idx_with_all_empty = idx_with_all_empty.intersection(idx_with_empty)

    for row in table_as_rows:
        table_struct.append([val for i, val in enumerate(
            row) if i not in idx_with_all_empty])
    return table_struct


def calc_fuzz_score(row1, row2):
    """Calculate fuzzy similarity score for two rows."""
    similarity_scores = sorted(
        (fuzz.ratio(str(v1), str(v2)) for v1 in row1 for v2 in row2),
        reverse=True
    )
    return sum(similarity_scores[:min(len(row1), len(row2))])/min(len(row1), len(row2)) if similarity_scores else 0


def check_for_transpose(table1, table2):
    # Check for transpose
    fuzzy_score_col_col = calc_fuzz_score(table1[0], table2[0])
    fuzzy_score_col_row = calc_fuzz_score(
        table1[0], [row[0] for row in table2])
    fuzzy_score_row_col = calc_fuzz_score(
        [row[0] for row in table1], table2[0])
    fuzzy_score_row_row = calc_fuzz_score(
        [row[0] for row in table1], [row[0] for row in table2])
    print(f"Col-Col: {fuzzy_score_col_col}")
    print(f"Col-Row: {fuzzy_score_col_row}")
    print(f"Row-Col: {fuzzy_score_row_col}")
    print(f"Row-Row: {fuzzy_score_row_row}")
    if fuzzy_score_col_col > 90 or fuzzy_score_row_row > 90:
        return True
    return False


def map_columns_fuzzy(table1, table2, column_mapping={}, row_mapping={}, threshold=100):
    fuzzy_scores = {}
    if not row_mapping:
        # print(column_mapping,"HEEEEEEEEEEEEEEE")
        for idx1, col1 in enumerate(table1[0]):
            for idx2, col2 in enumerate(table2[0]):
                score = fuzz.ratio(col1, col2)
                fuzzy_scores[(idx1, idx2)] = score
        for (idx1, idx2), score in sorted(fuzzy_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold and idx1 not in column_mapping and idx2 not in column_mapping.values():
                column_mapping[idx1] = idx2
        return column_mapping

    for idx1, col1 in enumerate(table1[0]):
        for idx2, col2 in enumerate(table2[0]):
            column_score = []
            for row1, row2 in row_mapping.items():
                if table1[row1][idx1] and table2[row2][idx2]:
                    score = fuzz.ratio(table1[row1][idx1], table2[row2][idx2])
                    column_score.append(score)
            if column_score:

                column_score = sum(column_score)/len(column_score)
            else:
                column_score = 0
            fuzzy_scores[(idx1, idx2)] = column_score
    # print(sorted(fuzzy_scores.items(), key=lambda x: x[1], reverse=True))
    for (idx1, idx2), score in sorted(fuzzy_scores.items(), key=lambda x: x[1], reverse=True):
        if score >= threshold and idx1 not in column_mapping and idx2 not in column_mapping.values():
            column_mapping[idx1] = idx2
    return column_mapping


def map_rows_fuzzy(table1, table2, column_mapping, threshold=100):
    row_mapping = {}
    row_fuzzy_scores = {}
    # print(table1, table2)
    # print(column_mapping)
    for idx1, row1 in enumerate(table1[1:]):
        for idx2, row2 in enumerate(table2[1:]):
            row_score = []
            for col1, col2 in column_mapping.items():
                # print(row1, row2, col1, col2)
                if row1[col1] and row2[col2]:
                    score = fuzz.ratio(row1[col1], row2[col2])
                    row_score.append(score)

            row_fuzzy_scores[(idx1+1, idx2+1)] = sum(row_score) / \
                len(row_score) if row_score else 0

    for (idx1, idx2), score in sorted(row_fuzzy_scores.items(), key=lambda x: x[1], reverse=True):
        # print(score, threshold)
        if score >= threshold and idx1 not in row_mapping and idx2 not in row_mapping.values():
            row_mapping[idx1] = idx2
    # print(row_mapping)
    return row_mapping


def map_columns_bert(table1, table2, column_mapping, row_mapping, threshold=100):
    bert_scores = {}
    string_pairs = {}
    for idx1, col1 in enumerate(table1[0]):
        for idx2, col2 in enumerate(table2[0]):
            string1 = f"{col1} -> {' '.join([table1[r_idx1][idx1] for r_idx1 in row_mapping.keys()])}"
            string2 = f"{col2} -> {' '.join([table2[r_idx2][idx2] for r_idx2 in row_mapping.values()])}"
            string_pairs[(idx1, idx2)] = (string1, string2)

    _, _, bert_scores = bscore([string_pairs[(col1, col2)][0] for col1, col2 in string_pairs.keys()], [string_pairs[(
        col1, col2)][1] for col1, col2 in string_pairs.keys()], lang="en", model_type="roberta-large", nthreads=4)
    bert_scores = {(col1, col2): bert_score.mean().item() for (
        col1, col2), bert_score in zip(string_pairs.keys(), bert_scores)}
    for (idx1, idx2), _score in sorted(bert_scores.items(), key=lambda x: x[1], reverse=True):
        if _score >= threshold and idx1 not in column_mapping and idx2 not in column_mapping.values():
            column_mapping[idx1] = idx2
    return column_mapping


def map_rows_bert(table1, table2, column_mapping, row_mapping, threshold=100):
    bert_scores = {}
    string_pairs = {}
    for idx1, row1 in enumerate(table1[1:]):
        for idx2, row2 in enumerate(table2[1:]):
            row1_values = [(table1[0][col], table1[idx1+1][col])
                           for col in column_mapping.keys()]
            row2_values = [(table2[0][col], table2[idx2+1][col])
                           for col in column_mapping.values()]
            string1 = " ".join(f"{col}:{val}\n" for col, val in row1_values)
            string2 = " ".join(f"{col}:{val}\n" for col, val in row2_values)
            string_pairs[(idx1+1, idx2+1)] = (string1, string2)

    _, _, bert_scores = bscore([string_pairs[(row1, row2)][0] for row1, row2 in string_pairs.keys()], [string_pairs[(
        row1, row2)][1] for row1, row2 in string_pairs.keys()], lang="en", model_type="roberta-large", nthreads=4)
    bert_scores = {(row1, row2): bert_score.mean().item() for (
        row1, row2), bert_score in zip(string_pairs.keys(), bert_scores)}

    for (idx1, idx2), _score in sorted(bert_scores.items(), key=lambda x: x[1], reverse=True):
        if _score >= threshold:
            if idx2 in row_mapping.values():
                row_mapping = {k: v for k,
                               v in row_mapping.items() if v != idx2}
            row_mapping[idx1] = idx2

    return row_mapping


def update_mappings(table_original, table_perturbation):
    """Run column and row mapping until convergence."""
    old_column_mapping, old_row_mapping = {}, {}
    column_mapping, row_mapping = {}, {}
    while True:
        # Apply fuzzy and BERT-based mapping functions in sequence
        column_mapping = map_columns_fuzzy(
            table_original, table_perturbation, column_mapping, row_mapping)
        row_mapping = map_rows_fuzzy(
            table_original, table_perturbation, column_mapping)
        column_mapping = map_columns_bert(
            table_original, table_perturbation, column_mapping, row_mapping)
        row_mapping = map_rows_bert(
            table_original, table_perturbation, column_mapping, row_mapping)
        # Break if mappings have converged
        if old_column_mapping == column_mapping and old_row_mapping == row_mapping:
            break
        old_column_mapping, old_row_mapping = column_mapping.copy(), row_mapping.copy()

    return column_mapping, row_mapping


def add_extra_rows(row_mapping, table_original, table_perturbation):
    """Add extra rows if there are unmapped rows in either table."""
    extra_row = 1

    # Add extra rows for unmapped rows in original table
    for i in range(len(table_original) - 1):
        row_no = i + 1
        if row_no not in row_mapping:
            row_mapping[row_no] = f"extra{extra_row}"
            extra_row += 1

    # Add extra rows for unmapped rows in perturbed table
    extra_row = 1
    for i in range(len(table_perturbation) - 1):
        row_no = i + 1
        if row_no not in row_mapping.values():
            row_mapping[f"extra{extra_row}"] = row_no
            extra_row += 1


def add_extra_columns(column_mapping, table_original, table_perturbation):
    """Add extra columns if there are unmapped columns in either table."""
    extra_col = 1

    # Add extra columns for unmapped columns in original table
    for i in range(len(table_original[0])):
        if i not in column_mapping:
            column_mapping[i] = f"extra{extra_col}"
            extra_col += 1

    # Add extra columns for unmapped columns in perturbed table
    extra_col = 1
    for i in range(len(table_perturbation[0])):
        if i not in column_mapping.values():
            column_mapping[f"extra{extra_col}"] = i
            extra_col += 1


def is_extra(label):
    """Check if the label is an 'extra' row or column."""
    return "extra" in str(label)


def get_cell_value(table, row, col):
    """Get the cell value, or 'none' if the row or column is 'extra'."""
    if is_extra(row) or is_extra(col):
        return "none"
    return table[row][col]


def merge_headers(table1, table2, column_mapping):
    """Merge the headers of two tables based on column mappings."""
    return [(get_cell_value(table1, 0, col1)+'.T1', get_cell_value(table2, 0, col2)+'.T2')
            for col1, col2 in column_mapping.items()]


def merge_rows(table1, table2, row1, row2, column_mapping):
    """Merge a pair of rows from two tables based on column mappings."""
    return [(get_cell_value(table1, row1, col1), get_cell_value(table2, row2, col2))
            for col1, col2 in column_mapping.items()]


def get_merged_tables(table1, table2, column_mapping, row_mapping):
    """Merge two tables based on column and row mappings."""
    # Merge headers
    merged_table = [merge_headers(table1, table2, column_mapping)]

    # Merge each row according to row mappings
    for row1, row2 in row_mapping.items():
        merged_table.append(merge_rows(
            table1, table2, row1, row2, column_mapping))

    return merged_table


def table_to_markdown(data, merged=False):
    markdown = ""
    # Loop through each row
    for idx, row in enumerate(data):
        # Convert each cell to the format: original_value (new_value)
        if idx == 1: 
            mardown_row = " | ".join("---" for _ in row)
            markdown += f"| {mardown_row} |\n"
        markdown_row = " | ".join(
            [f"{cell[0]} / {cell[1]}" for cell in row]) if merged else " | ".join(row)
        markdown += f"| {markdown_row} |\n"
    return markdown


def merge_tables(table1, table2):
    table1 = table_to_row_list(table1)
    table2 = table_to_row_list(table2)
    if not table1 or not table2:
        return None
    column_mapping, row_mapping = update_mappings(table1, table2)
    add_extra_rows(row_mapping, table1, table2)
    add_extra_columns(column_mapping, table1, table2)
    merged_table = get_merged_tables(
        table1, table2, column_mapping, row_mapping)

    return table_to_markdown(merged_table, merged=True), merged_table


def merge_tables_fuzzy(table1, table2):
    try:
        table_1 = table_to_row_list(table1)
        table_2 = table_to_row_list(table2)
        if not table_1 or not table_2:
            return None, None
        column_mapping = {}
        row_mapping = {}
        column_mapping = map_columns_fuzzy(
            table_1, table_2, column_mapping, row_mapping)
        row_mapping = map_rows_fuzzy(table_1, table_2, column_mapping)
        column_mapping = map_columns_fuzzy(
            table_1, table_2, column_mapping, row_mapping)
        row_mapping = map_rows_fuzzy(table_1, table_2, column_mapping)
        merged_table = get_merged_tables(
            table_1, table_2, column_mapping, row_mapping)
        return table_to_markdown(merged_table, merged=True), merged_table
    except:
        return None, None