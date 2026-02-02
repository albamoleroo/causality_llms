"""
PC Algorithm for Causal Discovery on Cardiac Disease Dataset

This module implements the PC (Peter-Clark) algorithm for constraint-based
causal discovery, a traditional data-driven approach to compare with LLM-based methods.
"""

import pandas as pd
import numpy as np
from pgmpy.estimators import PC
from pgmpy.base import DAG
import networkx as nx
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def run_pc_algorithm(
    data: pd.DataFrame,
    alpha: float = 0.05,
    ci_test: str = 'chi_square',
    max_cond_vars: int = 3,
    return_type: str = 'both'
) -> Dict:
    
    pc_estimator = PC(data=data)
    
    estimated_dag = pc_estimator.estimate(
        variant='stable',
        ci_test=ci_test,
        significance_level=alpha,
        max_cond_vars=max_cond_vars,
        return_type='dag'
    )
    
    skeleton = estimated_dag.to_undirected()
    
    edge_dict = {(edge[0], edge[1]): 1 for edge in estimated_dag.edges()}
    variables = list(data.columns)
    
    n = len(variables)
    adj_matrix = np.zeros((n, n))
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if estimated_dag.has_edge(var1, var2):
                adj_matrix[i, j] = 1
    
    results = {
        'estimated_dag': estimated_dag,
        'skeleton': skeleton,
        'edge_dict': edge_dict,
        'adjacency_matrix': adj_matrix,
        'n_edges': len(edge_dict),
        'variables': variables,
        'parameters': {
            'alpha': alpha,
            'ci_test': ci_test,
            'max_cond_vars': max_cond_vars,
            'n_samples': len(data)
        }
    }
    
    return results


def compare_with_baseline(
    pc_results: Dict,
    baseline_graph: Dict[Tuple[str, str], int],
    ground_truth_edges: Optional[int] = None
) -> Dict:
    
    pc_edges = set(pc_results['edge_dict'].keys())
    baseline_edges = set(baseline_graph.keys())
    
    true_positives = pc_edges.intersection(baseline_edges)
    false_positives = pc_edges - baseline_edges
    false_negatives = baseline_edges - pc_edges
    
    n_vars = len(pc_results['variables'])
    total_possible_edges = n_vars * (n_vars - 1)
    true_negatives = total_possible_edges - len(pc_edges) - len(baseline_edges) + len(true_positives)
    
    precision = len(true_positives) / len(pc_edges) if len(pc_edges) > 0 else 0
    recall = len(true_positives) / len(baseline_edges) if len(baseline_edges) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (len(true_positives) + true_negatives) / total_possible_edges if total_possible_edges > 0 else 0
    shd = len(false_positives) + len(false_negatives)
    
    comparison = {
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'true_negatives': true_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'shd': shd,
        'pc_edges': len(pc_edges),
        'baseline_edges': len(baseline_edges),
        'total_possible_edges': total_possible_edges,
        'edge_agreement': list(true_positives),
        'pc_only': list(false_positives),
        'baseline_only': list(false_negatives)
    }
    
    if ground_truth_edges:
        comparison['ground_truth_edges'] = ground_truth_edges
    
    return comparison


def comparison(comp: Dict):
    
    print("\n" + "="*70)
    print("pc algorithm vs. llm baseline")
    print("="*70)
    
    print("\nedge counts:")
    print(f"  pc: {comp['pc_edges']}")
    print(f"  llm: {comp['baseline_edges']}")
    print(f"  common: {comp['true_positives']}")
    
    print("\nkey metrics:")
    print(f"  accuracy:  {comp['accuracy']:.1%}")
    print(f"  shd:       {comp['shd']}")
    print(f"  fp:        {comp['false_positives']}")
    print(f"  fn:        {comp['false_negatives']}")
    
    if comp['pc_only']:
        print(f"\n  pc only ({len(comp['pc_only'])})")
        for edge in sorted(comp['pc_only']):
            print(f"    {edge[0]} → {edge[1]}")
    
    if comp['baseline_only']:
        print(f"\n  llm only ({len(comp['baseline_only'])})")
        for edge in sorted(comp['baseline_only']):
            print(f"    {edge[0]} → {edge[1]}")
    
    print("\n" + "="*70)


def visualize_pc_graph(pc_results: Dict, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    
    dag = pc_results['estimated_dag']
    plt.figure(figsize=(12, 8))
    G = nx.DiGraph(dag.edges())
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                           node_size=3000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, arrowstyle='->', width=2)
    
    plt.title(f"pc algorithm dag\n{len(dag.edges())} edges", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("pc algorithm module")
    print("from pc_algorithm import run_pc_algorithm, compare_with_baseline, comparison")
