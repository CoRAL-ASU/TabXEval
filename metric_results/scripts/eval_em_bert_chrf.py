import numpy as np
import bert_score
from sacrebleu import sentence_chrf
from g_rouge import rouge 
# from utils.utils import *
import torch

def evaluate_table(model_output, gold_label, eval_exact_match=False, eval_bert=False, eval_chrf=False, eval_rougeL=False):
    """
    Evaluate the model output table against the gold label table.
    Parameters:
        model_output: The model-generated table (as a NumPy array).
        gold_label: The ground truth table (as a NumPy array).
        eval_exact_match: Boolean flag to compute Exact Match metric.
        eval_bert: Boolean flag to compute BERTScore metric.
        eval_chrf: Boolean flag to compute chrF metric.
    Returns:
        A dictionary containing the computed metrics.
    """
    # Initialize BERTScorer only if needed
    bert_scorer = None
    if eval_bert:
        bert_scorer = bert_score.BERTScorer(model_type='roberta-large', lang='en', rescale_with_baseline=True, device='cuda')
    metric_cache = dict()  # Cache similarities to avoid redundant computations

    # Helper function to convert data types to standard Python strings
    def to_python_str(s):
        if isinstance(s, np.ndarray):
            # Flatten array and join elements into a string
            return ' '.join(map(str, s.flatten()))
        elif isinstance(s, (np.str_, np.generic)):
            return s.item()
        elif s is None:
            return ''
        else:
            return str(s)

    # Helper function to calculate similarity between two items
    def calc_similarity(tgt, pred, metric):
        # Handle tuples by computing element-wise similarity
        if isinstance(tgt, tuple) and isinstance(pred, tuple):
            if len(tgt) != len(pred):
                return 0.0  # If tuples are of different lengths, similarity is zero
            sim = 1.0
            for t_elem, p_elem in zip(tgt, pred):
                elem_sim = calc_similarity(t_elem, p_elem, metric)
                sim *= elem_sim
            return sim
        else:
            # Convert to strings
            tgt_str = to_python_str(tgt)
            pred_str = to_python_str(pred)

            # Use string representations as cache keys
            cache_key = (tgt_str, pred_str, metric)
            if cache_key in metric_cache:
                return metric_cache[cache_key]

            if metric == 'exact_match':
                sim = float(tgt_str.strip() == pred_str.strip())
            elif metric == 'chrf':
                sim = sentence_chrf(pred_str, [tgt_str]).score / 100  # chrF score ranges from 0 to 1
            elif metric == 'BERT_score':
                if bert_scorer is None:
                    raise ValueError("BERTScore evaluation is not enabled.")
                P, R, F1 = bert_scorer.score([pred_str], [tgt_str])
                sim = F1.item()  # Scaled BERTScore
                # Optional: Clip the score between 0 and 1
                sim = max(0.0, min(sim, 1.0))
            elif metric == 'rougeL':
                rouge_score = rouge([pred_str],[tgt_str])
                sim = rouge_score['rouge_l/f_score']
                sim = max(0.0, min(sim, 1.0))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            metric_cache[cache_key] = sim
            return sim

    # Function to extract data from the table
    def parse_table_to_data(table):
        """
        Extract row headers, column headers, and relations from the table.
        """
        # Check if table is at least 2D
        if table.ndim != 2 or table.size == 0:
            return set(), set(), set()

        num_rows, num_cols = table.shape

        # Initialize sets
        row_headers = set()
        col_headers = set()
        relations = set()

        # Extract row headers
        if num_cols > 0 and num_rows > 1:
            for i in range(1, num_rows):
                row_header = to_python_str(table[i, 0])
                row_headers.add(row_header)

        # Extract column headers
        if num_rows > 0 and num_cols > 1:
            for j in range(1, num_cols):
                col_header = to_python_str(table[0, j])
                col_headers.add(col_header)

        # Extract relations
        for i in range(1, num_rows):
            for j in range(1, num_cols):
                cell_value = table[i, j]
                # Skip if cell_value is empty or NaN
                if cell_value in ['', None]:
                    continue
                if isinstance(cell_value, float) and np.isnan(cell_value):
                    continue

                row_header = to_python_str(table[i, 0]) if num_cols > 0 else ''
                col_header = to_python_str(table[0, j]) if num_rows > 0 else ''
                relation = (row_header, col_header, to_python_str(cell_value))
                relations.add(relation)
        return row_headers, col_headers, relations

    # Extract data from model_output and gold_label tables
    gold_row_headers, gold_col_headers, gold_relations = parse_table_to_data(gold_label)
    model_row_headers, model_col_headers, model_relations = parse_table_to_data(model_output)

    # Initialize metrics dictionary
    metrics = {}

    # Build list of metrics to compute based on flags
    metric_names = []
    if eval_exact_match:
        metric_names.append('exact_match')
    if eval_bert:
        metric_names.append('BERT_score')
    if eval_chrf:
        metric_names.append('chrf')
    if eval_rougeL:
        metric_names.append('rougeL')


    # Evaluate cells (relations)
    if metric_names:
        cells_metrics = {}
        for metric_name in metric_names:
            precision, recall, f1 = metrics_by_sim(
                gold_relations, model_relations, metric_name, calc_similarity
            )
            cells_metrics[metric_name + '(%)'] = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
            }
        metrics['cells'] = cells_metrics

        # Evaluate row headers
        row_header_metrics = {}
        for metric_name in metric_names:
            precision, recall, f1 = metrics_by_sim(
                gold_row_headers, model_row_headers, metric_name, calc_similarity
            )
            row_header_metrics[metric_name + '(%)'] = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
            }
        metrics['row_header'] = row_header_metrics

        # Evaluate column headers
        col_header_metrics = {}
        for metric_name in metric_names:
            precision, recall, f1 = metrics_by_sim(
                gold_col_headers, model_col_headers, metric_name, calc_similarity
            )
            col_header_metrics[metric_name + '(%)'] = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
            }
        metrics['col_header'] = col_header_metrics

    return metrics

def metrics_by_sim(tgt_data, pred_data, metric_name, calc_similarity_func):
    """
    Compute precision, recall, and F1 score based on similarity.
    """
    if not tgt_data and not pred_data:
        return 1.0, 1.0, 1.0  # Both are empty, perfect match
    if not pred_data:
        return 0.0, 0.0, 0.0  # No predictions
    if not tgt_data:
        return 0.0, 0.0, 0.0  # No target data

    tgt_data_list = list(tgt_data)
    pred_data_list = list(pred_data)

    # Build similarity matrix
    sim_matrix = np.zeros((len(tgt_data_list), len(pred_data_list)), dtype=float)
    for i, tgt_item in enumerate(tgt_data_list):
        for j, pred_item in enumerate(pred_data_list):
            sim = calc_similarity_func(tgt_item, pred_item, metric_name)
            sim_matrix[i, j] = sim

    # Precision: average of max similarities for each predicted item
    max_sim_pred = np.max(sim_matrix, axis=0)
    precision = np.mean(max_sim_pred)

    # Recall: average of max similarities for each target item
    max_sim_tgt = np.max(sim_matrix, axis=1)
    recall = np.mean(max_sim_tgt)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def calculate_macro_avg_per_category(results, category, metric_names):
    """
    Calculate the macro average for a specific category (cells, row_header, col_header)
    across all test points by first averaging over subcategories within each test point.
    """
    test_point_averages = {}
    
    # Loop through each test point
    for test_key, test_val in results.items():
        # Initialize totals and counts per test point
        totals = {metric: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for metric in metric_names}
        counts = {metric: 0 for metric in metric_names}
        
        # Loop over subcategories
        for subcategory_key, subcategory_val in test_val.items():
            # subcategory_val should have 'cells', 'row_header', 'col_header'
            if category in subcategory_val:
                category_metrics = subcategory_val[category]
                for metric in metric_names:
                    metric_key = metric + '(%)'
                    if metric_key in category_metrics:
                        precision = category_metrics[metric_key].get('precision', 0.0)
                        recall = category_metrics[metric_key].get('recall', 0.0)
                        f1 = category_metrics[metric_key].get('f1', 0.0)
                        totals[metric]['precision'] += precision
                        totals[metric]['recall'] += recall
                        totals[metric]['f1'] += f1
                        counts[metric] += 1
        
        # Compute averages per test point
        test_point_avg = {}
        for metric in metric_names:
            if counts[metric] > 0:
                test_point_avg[metric] = {
                    'precision': totals[metric]['precision'] / counts[metric],
                    'recall': totals[metric]['recall'] / counts[metric],
                    'f1': totals[metric]['f1'] / counts[metric]
                }
            else:
                test_point_avg[metric] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        test_point_averages[test_key] = test_point_avg
    
    # Now compute macro average over test points
    totals = {metric: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for metric in metric_names}
    counts = {metric: 0 for metric in metric_names}
    
    for test_key, metrics in test_point_averages.items():
        for metric in metric_names:
            if metric in metrics:
                totals[metric]['precision'] += metrics[metric]['precision']
                totals[metric]['recall'] += metrics[metric]['recall']
                totals[metric]['f1'] += metrics[metric]['f1']
                counts[metric] += 1  # Should be incremented per test point
    
    # Compute final macro averages
    macro_avg = {}
    for metric in metric_names:
        if counts[metric] > 0:
            macro_avg[metric + '(%)'] = {
                'precision': totals[metric]['precision'] / counts[metric],
                'recall': totals[metric]['recall'] / counts[metric],
                'f1': totals[metric]['f1'] / counts[metric]
            }
        else:
            macro_avg[metric + '(%)'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    return macro_avg


def calculate_all_macro_avgs(results, eval_exact_match=True, eval_bert=True, eval_chrf=True, eval_rougeL=True):
    """
    Calculate macro averages for cells, row_header, and col_header across all test points.
    Parameters:
        results: Dictionary containing evaluation metrics for each test point.
        eval_exact_match: Boolean flag indicating whether 'exact_match' metric was evaluated.
        eval_bert: Boolean flag indicating whether 'BERT_score' metric was evaluated.
        eval_chrf: Boolean flag indicating whether 'chrf' metric was evaluated.
    Returns:
        A dictionary containing the macro-averaged metrics for 'cells', 'row_header', and 'col_header'.
    """
    # Build list of metrics to compute based on flags
    metric_names = []
    if eval_exact_match:
        metric_names.append('exact_match')
    if eval_bert:
        metric_names.append('BERT_score')
    if eval_chrf:
        metric_names.append('chrf')
    if eval_rougeL:
        metric_names.append('rougeL')

    # Handle the case where no metrics are selected
    if not metric_names:
        print("No metrics selected for macro average calculation.")
        return {}

    # Calculate macro averages for each category
    cells_avg = calculate_macro_avg_per_category(results, 'cells', metric_names)
    row_header_avg = calculate_macro_avg_per_category(results, 'row_header', metric_names)
    col_header_avg = calculate_macro_avg_per_category(results, 'col_header', metric_names)

    # Return the results as a dictionary
    return {
        'Cells': cells_avg,
        'Row Header': row_header_avg,
        'Column Header': col_header_avg
    }

