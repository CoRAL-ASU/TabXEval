import Levenshtein
import difflib
import pandas as pd
import os
import json
import numpy as np
import re

def parse_table(text):
    tables = {}
    
    table_matches = re.finditer(r"###\s+(\w+)\n\|(.+?)\|\n(\|.+?\|(?:\n|$))+(?=\n###|$)", text, re.DOTALL)
    for match in table_matches:
        table_name = match.group(1).strip()
        header_line = match.group(2).strip()
        table_rows = match.group(0).split("\n")[2:]

        columns = [col.strip() if col.strip() != "" else "" for col in header_line.split("|")]

        data = []
        for line in table_rows:
            row = [cell.strip() if cell.strip() != "None" else None for cell in line.split("|") if cell.strip()]
            if len(row) == len(columns):
                data.append(row)

        df = pd.DataFrame(data, columns=columns, dtype=object)
        df.replace('None', np.nan, inplace=True)
        df = df[~df.iloc[:, 1:].isna().all(axis=1)]
        df.dropna(axis=1, how='all', inplace=True)

        tables[table_name] = df

    return tables 

def levenshtein_similarity(pred, gold):
    if not gold:
        # If gold is empty, give zero score
        return 0.0
    return 1 - Levenshtein.distance(pred, gold) / (2 * len(gold))

def difflib_similarity(pred, gold):
    sm = difflib.SequenceMatcher(None, pred, gold)
    return sm.ratio()

def flatten_table(table):
    return [x for row in table for x in row ]

def get_table_artifacts(table):
    if table is None:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "column_names": [],
            "data_rows": []
        }
    elif isinstance(table, pd.DataFrame) and table.empty:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "column_names": [],
            "data_rows": []
        }

    columns = table.columns.to_list()
    columns[0] = ''

    return {
        "num_rows": table.shape[0],
        "num_cols": table.shape[1],
        "column_names": columns,
        "data_rows": table.to_numpy().tolist()
    }


def score_artifacts(artifacts_1, artifacts_2):
    sub_scores = {}
    sub_scores["num_rows_match"] = (artifacts_1["num_rows"] == artifacts_2["num_rows"]) * 1.0
    sub_scores["num_cols_match"] = (artifacts_1["num_cols"] == artifacts_2["num_cols"]) * 1.0
    sub_scores["columns_levenshtein_score"] = levenshtein_similarity(
        artifacts_1["column_names"],
        artifacts_2["column_names"],
    )
    sub_scores["columns_difflib_score"] = difflib_similarity(
        artifacts_1["column_names"],
        artifacts_2["column_names"],
    )
    sub_scores["data_levenshtein_score"] = levenshtein_similarity(
        flatten_table(artifacts_1["data_rows"]),
        flatten_table(artifacts_2["data_rows"]),
    )
    sub_scores["data_difflib_score"] = difflib_similarity(
        flatten_table(artifacts_1["data_rows"]),
        flatten_table(artifacts_2["data_rows"]),
    )
    return sub_scores

def score_tables(gold_team, method_team):

    team_table_artifacts_pred = get_table_artifacts(method_team)
    team_table_artifacts_gold = get_table_artifacts(gold_team)

    team_artifacts = score_artifacts(team_table_artifacts_pred, team_table_artifacts_gold)

    return team_artifacts

def h_score(gold_table, pred_table):
    content_scores = 0.0
    format_scores = 0.0

    count = 0
    error_idx = []
    our_score = score_tables(gold_table, pred_table)
    content_scores = content_scores + our_score["data_levenshtein_score"] + our_score["data_difflib_score"]
    format_scores = format_scores + our_score["num_rows_match"] + our_score["num_cols_match"] + our_score["columns_levenshtein_score"] + our_score["columns_difflib_score"]
    
    count += 1


    content_scores = content_scores / count
    format_scores = format_scores / count

    print("---------------------")
    print("Strucbench score")
    print("Content_score: ", content_scores)
    print("Structure_score: ", format_scores)
    print("---------------------")

    print(error_idx)
    print(len(error_idx))

    return content_scores,format_scores


#strucbench score
# content_scores = 0.0
# format_scores = 0.0

# method_path = '/scratch/fbardoli/Text2Table/model_outputs/new/gemini/gemini_2_0_flash_exp/method_3prompt_cot/table/'
    
# gold_player_path = '/scratch/fbardoli/Text2Table/rotowire_corrected_full/player/'
# gold_team_path = '/scratch/fbardoli/Text2Table/rotowire_corrected_full/team/'

# count = 0
# error_idx = []
# for res_key in range(728):
#     # if int(res_key) != 200:
#     #     continue
#     # print('Idx: ', res_key)
#     try:

#         res_key = str(res_key)


#         # with open(os.path.join(method_path, f'{res_key}.txt'), 'r') as f:
#         #     text = f.read().strip()
#         # tables = parse_table(text)
#         # team, player = tables.get('Team'), tables.get('Player')

#         # if os.path.getsize(os.path.join(gold_player_path, f'{res_key}.csv')) == 0:
#         #     gold_player = pd.DataFrame()
#         # else:
#         #     gold_player = pd.read_csv(gold_player_path + f'{int(res_key)}.csv', keep_default_na=False, na_values=[''],engine='python', dtype=object)
        
#         # if os.path.getsize(os.path.join(gold_team_path, f'{res_key}.csv')) == 0:
#         #     gold_team = pd.DataFrame()
#         # else:
#         #     gold_team = pd.read_csv(gold_team_path + f'{int(res_key)}.csv', keep_default_na=False, na_values=[''],engine='python', dtype=object)
        


#         our_score = score_tables(gold_team, team)
#         content_scores = content_scores + our_score["data_levenshtein_score"] + our_score["data_difflib_score"]
#         format_scores = format_scores + our_score["num_rows_match"] + our_score["num_cols_match"] + our_score["columns_levenshtein_score"] + our_score["columns_difflib_score"]
        
#         count += 1
    
#     except Exception as e:
#         print(e)
#         error_idx.append(int(res_key))


# content_scores = content_scores / count
# format_scores = format_scores / count

# print("---------------------")
# print("Strucbench score")
# print("Content_score: ", content_scores)
# print("Structure_score: ", format_scores)
# print("---------------------")

# print(error_idx)
# print(len(error_idx))