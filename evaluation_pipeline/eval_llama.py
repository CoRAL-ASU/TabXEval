import logging
import pickle as pkl
from fuzzy_table_matching import merge_tables_fuzzy
from comparison_utils import *
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import sys
from argparse import ArgumentParser
import json
load_dotenv()


def ask_chatgpt(text: str, prompt_path=None):
    # Check if a prompt path is provided and read prompt text
    _ = load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(base_url="https://model-4w5z66pw.api.baseten.co/environments/production/sync/v1",
                    api_key="UDKtz4ud.RFYYtPraNu7BhFQxfN4Nivo9klis31de",
                    )
    if prompt_path and text:
        with open(prompt_path, 'r') as file:
            prompt = file.read().strip()
    else:
        return "Error: no data given"

    # Model configuration - replace 'gpt-4' with the specific model if needed
    # or "gpt-3.5-turbo" if you want a different model
    model_name = "meta-llama/llama-3.3-70b-instruct"
    # Send the prompt to the model
    response = client.chat.completions.create(
        model="placeholder",
        messages=[
            {'role': "system", "content": prompt},
            {'role': "user", "content": text}
        ]
    )

    # Return the response content
    return response.choices[0].message.content


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    argument_parser = ArgumentParser()
    argument_parser.add_argument('--align_prompt', type=str, default=None)
    argument_parser.add_argument('--compare_prompt', type=str, default=None)
    argument_parser.add_argument('--input_tables', type=str, default=None)
    argument_parser.add_argument('--output_path', type=str, default=None)
    args = argument_parser.parse_args()

    logger.info("Starting table alignment pipeline")
    logger.info(f"Arguments: {vars(args)}")

    # Read the input tables
    logger.info("Reading input tables")
    input_tables = json.load(open(args.input_tables))
    gpt_alignments = []

    logger.info(f"Processing {len(input_tables)} unique table pairs")
    for unique_id in input_tables:
        logger.info(f"Processing unique_id: {unique_id}")
        table1 = input_tables[unique_id][0]
        table2 = input_tables[unique_id][1]

        logger.debug("Running fuzzy merge")
        partial_alignment = merge_tables_fuzzy(table1, table2)[0]

        prompt = f"Align the following tables:\n\n{table1}\n\n{table2}\n\n"
        if partial_alignment is not None:
            prompt += f"Partially Aligned Table:{partial_alignment}"

        logger.info("Requesting ChatGPT alignment")
        alignment = ask_chatgpt(prompt, args.align_prompt)
        logger.debug(f"Alignment received for {unique_id}")

        gpt_alignments.append({
            "unique_id": unique_id,
            "alignment": alignment,
            "partial_alignment": partial_alignment,
            "table1": table1,
            "table2": table2
        })
    pkl.dump(gpt_alignments, open(args.output_path+"step1", "wb"))
    # gpt_alignments = pkl.load(open(args.output_path+"step1", "rb"))
    gpt_alignments_parsed = []
    logger.info("Parsing alignments")
    for alignment in gpt_alignments:
        aligned_table = alignment["alignment"]
        df_replaced, df_compare, df_wo_em = compare(aligned_table)
        gpt_alignments_parsed.append(
            {**alignment, "df_replaced": df_replaced, "df_compare": df_compare, "df_wo_em": df_wo_em})

    # Get comparison tuples from GPT
    logger.info("Generating comparison tuples")
    for idx, tables in enumerate(gpt_alignments_parsed):
        table = tables['df_wo_em'].to_markdown(index=False)
        try:
            logger.debug(f"Comparing table {idx}")
            comparison_tuples = ask_chatgpt(table, args.compare_prompt)
        except Exception as e:
            logger.error(f"Error in table {idx}: {str(e)}", exc_info=True)
            logger.error(f"Problematic table:\n{table}")
            comparison_tuples = []
        gpt_alignments_parsed[idx]['comparison_tuples'] = comparison_tuples
    pkl.dump(gpt_alignments_parsed, open(args.output_path+"step2", "wb"))
    # gpt_alignments_parsed = pkl.load(open(args.output_path+"step2", "rb"))
    logger.info("Parsing comparison tuples")
    for idx, tables in enumerate(gpt_alignments_parsed):
        compare_tuples_list = table_to_dict_list_comparison(
            tables['comparison_tuples'])
        table_list_updated = []
        for row in compare_tuples_list:
            row_updated = {}
            for key, value in row.items():
                if value is None:
                    row_updated[key] = None
                    continue
                row_updated[key] = parse_string(value)
            table_list_updated.append(row_updated)
        gpt_alignments_parsed[idx]['comparison_tuples_parsed'] = table_list_updated

    allowed_data_types = ["Numerical", "String",
                          "Bool", "Date", "List", "Time", "Others"]
    logger.info("Calculating statistics")
    gpt_alignment_parsed = get_partial_cells_stats(
        gpt_alignments_parsed, allowed_data_types)

    for idx, tables in enumerate(gpt_alignment_parsed):
        type_counts = tables['type_counts']
        delta = tables['delta']
        delta_stats = make_delta_stats_table(delta, type_counts)
        gpt_alignment_parsed[idx]['partial_cell_delta_stats'] = delta_stats
        ei_mi_table = pd.DataFrame(tables['ei_mi_table'])
        dfs = get_row_col_statistics(
            ei_mi_table, tables['df_compare'], allowed_data_types)
        gpt_alignment_parsed[idx]['ei_mi_column_types'] = dfs[1]
        gpt_alignment_parsed[idx]['row_col_statistics'] = get_row_column_statistics(
            ei_mi_table, tables['df_compare'])
        gpt_alignment_parsed[idx]['cell_stats'] = create_summary_table_from_df(
            ei_mi_table, allowed_data_types, default_categories=['EI', 'MI', 'Partial'], partial=tables['empty_cells']['Partial'])

    logger.info(f"Saving results to {args.output_path}")
    with open(args.output_path, 'wb') as f:
        pkl.dump(gpt_alignment_parsed, f)

    logger.info("Pipeline completed successfully")
