import logging
import pickle as pkl
from fuzzy_table_matching import merge_tables_fuzzy
from comparison_utils import compare, table_to_dict_list_comparison, parse_string, find_first_number, get_partial_cells_stats, make_delta_stats_table, make_empty_cells_table, get_row_col_statistics, get_row_column_statistics, create_summary_table_from_df
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import sys
from argparse import ArgumentParser
import json
import asyncio
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

DEFAULT_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


async def get_output(model, prompt, img=None, safety_settings=DEFAULT_SAFETY_SETTINGS, system_instruction=None):
    try:

        r = await model.generate_content_async(
            prompt, safety_settings=safety_settings
        )

        return r.text
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None


async def process_job(model, args, prompt, key, index, results):
    await asyncio.sleep(index * args.delay)  # Introduce delay between jobs
    try:
        result = await get_output(model, prompt)
        if result:
            results.append({"key": key, "result": result})
            print(f"Processed: {key} - {result[:50]}...")
    except Exception as e:
        print(f"Error processing job {index}: {e}")


async def run_gemini(system_instruction, prompts, generation_config, args):
    results = []
    tasks = []
    model = genai.GenerativeModel(
        model_name="gemini-2.0-pro-exp-02-05",
        system_instruction=system_instruction,
        generation_config=generation_config,
    )
    for index, (key, prompt) in enumerate(prompts.items()):
        tasks.append(
            asyncio.create_task(process_job(
                model, args, prompt, key, index, results))
        )
    await asyncio.gather(*tasks)
    return results


if __name__ == '__main__':
    # Configure logging
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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
    argument_parser.add_argument(
        "--delay", type=float, default=15, help="Delay between processing each prompt."
    )
    args = argument_parser.parse_args()

    logger.info("Starting table alignment pipeline")
    logger.info(f"Arguments: {vars(args)}")

    # Read the input tables
    logger.info("Reading input tables")
    input_tables = json.load(open(args.input_tables))
    # take top 100 tables
    input_tables = {k: input_tables[k] for k in list(input_tables.keys())}
    # gpt_alignments = []

    # logger.info(f"Processing {len(input_tables)} unique table pairs")

    # gpt_alignment_prompts = {}
    # for unique_id in input_tables:
    #     logger.info(f"Processing unique_id: {unique_id}")
    #     table1 = input_tables[unique_id][0]
    #     table2 = input_tables[unique_id][1]

    #     logger.debug("Running fuzzy merge")
    #     partial_alignment = merge_tables_fuzzy(table1, table2)[0]

    #     prompt = f"Align the following tables:\n\n{table1}\n\n{table2}\n\n"
    #     if partial_alignment is not None:
    #         prompt += f"Partially Aligned Table:{partial_alignment}"

    #     gpt_alignment_prompts[unique_id] = prompt
    #     logger.debug(f"Alignment received for {unique_id}")
    #     # log first line of alignment
    #     gpt_alignments.append({
    #         "unique_id": unique_id,
    #         "alignment_prompt": prompt,
    #         "partial_alignment": partial_alignment,
    #         "table1": table1,
    #         "table2": table2
    #     })
    # aligmment_system_instruction = open(args.align_prompt, "r").read()
    # alignment_results = loop.run_until_complete(
    #     run_gemini(aligmment_system_instruction, gpt_alignment_prompts,
    #                generation_config=GENERATION_CONFIG, args=args)
    # )
    # for idx, result in enumerate(alignment_results):
    #     for ii, alignment in enumerate(gpt_alignments):
    #         if alignment["unique_id"] == result["key"]:
    #             gpt_alignments[ii]["alignment"] = result["result"]
    #             break
    # pkl.dump(gpt_alignments, open(args.output_path+"step1", "wb"))
    # gpt_alignments_parsed = []
    # logger.info("Parsing alignments")
    # for alignment in gpt_alignments:
    #     aligned_table = alignment["alignment"]
    #     df_replaced, df_compare, df_wo_em = compare(aligned_table)
    #     gpt_alignments_parsed.append(
    #         {**alignment, "df_replaced": df_replaced, "df_compare": df_compare, "df_wo_em": df_wo_em})

    # # Get comparison tuples from GPT
    # logger.info("Generating comparison tuples")
    # gpt_comparison_prompts = {}
    # for idx, tables in enumerate(gpt_alignments_parsed):
    #     table = tables['df_wo_em'].to_markdown(index=False)
    #     try:
    #         logger.debug(f"Comparing table {idx}")
    #         gpt_comparison_prompts[tables['unique_id']] = table
    #     except Exception as e:
    #         logger.error(f"Error in table {idx}: {str(e)}", exc_info=True)
    #         logger.error(f"Problematic table:\n{table}")
    #         comparison_tuples = []
    # comp_system_instruction = open(args.compare_prompt, "r").read()
    # comparison_results = loop.run_until_complete(
    #     run_gemini(comp_system_instruction, gpt_comparison_prompts,
    #                generation_config=GENERATION_CONFIG, args=args)
    # )
    # for idx, result in enumerate(comparison_results):
    #     for ii, alignment in enumerate(gpt_alignments_parsed):
    #         if alignment["unique_id"] == result["key"]:
    #             gpt_alignments_parsed[ii]["comparison_tuples"] = result["result"]
    #             break
    # for idx, tables in enumerate(gpt_alignments_parsed):
    #     if "comparison_tuples" not in tables:
    #         gpt_alignments_parsed[idx]['comparison_tuples'] = ""
    # pkl.dump(gpt_alignments_parsed, open(args.output_path+"step2", "wb"))
    gpt_alignments_parsed = pkl.load(open(args.output_path+"step2", "rb"))
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
    logger.info("Generating final metrics")
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
