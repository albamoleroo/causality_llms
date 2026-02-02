import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_coverage_comparison(df_kg, df_no_graph, output_filename='coverage_by_target_variable_comparison.png'):
    """
    Create coverage comparison boxplots for Knowledge Graph vs No-Graph models.
    
    Parameters:
    df_kg : pandas.DataFrame
        DataFrame containing Knowledge Graph model results
    df_no_graph : pandas.DataFrame
        DataFrame containing No-Graph model results
    output_filename : str, optional
        Filename for saving the plot (default: 'coverage_by_target_variable_comparison.png')
    
    Returns:
    summary_stats : pandas.DataFrame
        Summary statistics grouped by Target Variable, Model, and Method
    """
    sns.set_theme(style="whitegrid")
    
    # Add model labels
    df_kg = df_kg.copy()
    df_no_graph = df_no_graph.copy()
    df_kg['Model'] = 'Knowledge Graph'
    df_no_graph['Model'] = 'No-Graph'
    
    # Combine datasets
    df_combined = pd.concat([df_kg, df_no_graph], ignore_index=True)
    
    # Reshape data for plotting
    coverage_data = []
    for _, row in df_combined.iterrows():
        coverage_data.append({
            'Target Variable': row['Target Variable'],
            'Model': row['Model'],
            'Method': 'Gaussian',
            'Coverage': row['Coverage Gaussian 95%'] * 100
        })
        coverage_data.append({
            'Target Variable': row['Target Variable'],
            'Model': row['Model'],
            'Method': 'Bootstrap',
            'Coverage': row['Coverage Bootstrap 95%'] * 100
        })
    
    df_coverage_long = pd.DataFrame(coverage_data).dropna()
    
    # Piecewise transformation functions
    def forward_piecewise(y, split=80, frac=0.35, lower=-1, upper=101):
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        arr = np.clip(arr, lower, upper)
        mask = arr <= split
        out = np.empty_like(arr)
        out[mask] = (frac / (split - lower)) * (arr[mask] - lower)
        out[~mask] = frac + (arr[~mask] - split) * (1 - frac) / (upper - split)
        return out[0] if np.isscalar(y) else out
    
    def inverse_piecewise(y, split=80, frac=0.35, lower=-1, upper=101):
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        mask = arr <= frac
        out = np.empty_like(arr)
        out[mask] = lower + (arr[mask] * (split - lower) / frac)
        out[~mask] = split + (arr[~mask] - frac) * (upper - split) / (1 - frac)
        return out[0] if np.isscalar(y) else out
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    titles = ['Gaussian Coverage by Target Variable', 'Bootstrap Coverage by Target Variable']
    methods = ['Gaussian', 'Bootstrap']
    
    for ax, method, title in zip(axes, methods, titles):
        data = df_coverage_long[df_coverage_long['Method'] == method]
        sns.boxplot(
            data=data,
            x='Target Variable',
            y='Coverage',
            hue='Model',
            ax=ax
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Variable', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-1, 101)
        ax.set_yscale('function', functions=(forward_piecewise, inverse_piecewise))
        ax.set_yticks([0, 20, 40, 60, 80, 85, 90, 95, 100])
        ax.set_yticklabels([0, 20, 40, 60, 80, 85, 90, 95, 100])
        ax.margins(y=0.02)
        ax.set_ylabel('Coverage (%)', fontsize=12)
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
    
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='lower right', title='Model')
    axes[1].legend_.remove()
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # # Calculate summary statistics
    # summary_stats = df_coverage_long.groupby(['Target Variable', 'Model', 'Method'])['Coverage'].agg(['mean', 'std', 'median', 'min', 'max'])
    # print("\nCoverage Statistics by Target Variable:")
    # print(summary_stats)
    
    return summary_stats


def plot_single_target_coverage_bars(df_kg, df_no_graph, target_var, output_filename=None):
    sns.set_theme(style="whitegrid")
    
    # Filter by target variable
    df_kg_filtered = df_kg[df_kg['Target Variable'] == target_var].copy()
    df_no_graph_filtered = df_no_graph[df_no_graph['Target Variable'] == target_var].copy()
    
    # Add model labels
    df_kg_filtered['Model'] = 'Knowledge Graph'
    df_no_graph_filtered['Model'] = 'No-Graph'
    
    df_combined = pd.concat([df_kg_filtered, df_no_graph_filtered], ignore_index=True)
    
    df_combined['Coverage Gaussian 95%'] = df_combined['Coverage Gaussian 95%'] * 100
    df_combined['Coverage Bootstrap 95%'] = df_combined['Coverage Bootstrap 95%'] * 100
    
    academic_colors = ['#1f77b4', "#eaa466"]  # Blue, Orange
    
    # Piecewise transformation functions
    def forward_piecewise(y, split=80, frac=0.35, lower=-1, upper=101):
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        arr = np.clip(arr, lower, upper)
        mask = arr <= split
        out = np.empty_like(arr)
        out[mask] = (frac / (split - lower)) * (arr[mask] - lower)
        out[~mask] = frac + (arr[~mask] - split) * (1 - frac) / (upper - split)
        return out[0] if np.isscalar(y) else out
    
    def inverse_piecewise(y, split=80, frac=0.35, lower=-1, upper=101):
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        mask = arr <= frac
        out = np.empty_like(arr)
        out[mask] = lower + (arr[mask] * (split - lower) / frac)
        out[~mask] = split + (arr[~mask] - frac) * (upper - split) / (1 - frac)
        return out[0] if np.isscalar(y) else out
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Coverage for Target: {target_var}', fontsize=16, fontweight='bold', y=0.98)
    
    titles = ['Gaussian Coverage', 'Bootstrap Coverage']
    columns = ['Coverage Gaussian 95%', 'Coverage Bootstrap 95%']
    
    for ax, column, title in zip(axes, columns, titles):
        # Create barplot
        sns.barplot(
            data=df_combined,
            x='Intervention Variable',
            y=column,
            hue='Model',
            palette=academic_colors,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Intervention Variable', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-1, 101)
        ax.set_yscale('function', functions=(forward_piecewise, inverse_piecewise))
        ax.set_yticks([0, 20, 40, 60, 80, 85, 90, 95, 100])
        ax.set_yticklabels([0, 20, 40, 60, 80, 85, 90, 95, 100])
        ax.margins(y=0.02)
        ax.set_ylabel('Coverage (%)', fontsize=12)
    
    # Legend only on first plot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='lower right', title='Model', fontsize=10)
    axes[1].legend_.remove()
    
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    plt.show()

    return df_combined


# # Example Usage (commented out - uncomment to use):
# target_variable = 'major acute cardiovascular event'
# coverage_data = plot_single_target_coverage_bars(
#     df_kg=df1, 
#     df_no_graph=df2, 
#     target_var=target_variable,
#     output_filename=f'coverage_{target_variable.replace(" ", "_")}.png'
# )


def plot_coverage_comparison(df_kg, df_no_graph, df_third=None, third_model_name='No graph random order', 
                            output_filename='coverage_by_target_variable_comparison.png'):
    """
    Create coverage comparison boxplots for Knowledge Graph vs No-Graph models (and optionally a third model).
    
    Parameters:
    df_kg : pandas.DataFrame
        DataFrame containing Knowledge Graph model results
    df_no_graph : pandas.DataFrame
        DataFrame containing No-Graph model results
    df_third : pandas.DataFrame, optional
        DataFrame containing third model results
    third_model_name : str, optional
        Name for the third model (default: 'Third Model')
    output_filename : str, optional
        Filename for saving the plot 
    
    Returns:
    summary_stats : pandas.DataFrame
        Summary statistics grouped by Target Variable, Model, and Method
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    sns.set_theme(style="whitegrid")
    
    df_kg = df_kg.copy()
    df_no_graph = df_no_graph.copy()
    df_kg['Model'] = 'Knowledge Graph'
    df_no_graph['Model'] = 'No-Graph Ground Truth order'
    
    dataframes = [df_kg, df_no_graph]
    
    if df_third is not None:
        df_third = df_third.copy()
        df_third['Model'] = third_model_name
        dataframes.append(df_third)
    
    df_combined = pd.concat(dataframes, ignore_index=True)
    
    # Reshape data for plotting
    coverage_data = []
    for _, row in df_combined.iterrows():
        coverage_data.append({
            'Target Variable': row['Target Variable'],
            'Model': row['Model'],
            'Method': 'Gaussian',
            'Coverage': row['Coverage Gaussian 95%'] * 100
        })
        coverage_data.append({
            'Target Variable': row['Target Variable'],
            'Model': row['Model'],
            'Method': 'Bootstrap',
            'Coverage': row['Coverage Bootstrap 95%'] * 100
        })
    
    df_coverage_long = pd.DataFrame(coverage_data).dropna()
    
    academic_colors = ['#1f77b4', "#eaa466", '#D9D9D9']  # Blue, Dark Gray, Light Gray    

    def forward_piecewise(y, split=80, frac=0.35, lower=-1, upper=101):
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        arr = np.clip(arr, lower, upper)
        mask = arr <= split
        out = np.empty_like(arr)
        out[mask] = (frac / (split - lower)) * (arr[mask] - lower)
        out[~mask] = frac + (arr[~mask] - split) * (1 - frac) / (upper - split)
        return out[0] if np.isscalar(y) else out
    
    def inverse_piecewise(y, split=80, frac=0.35, lower=-1, upper=101):
        arr = np.atleast_1d(np.asarray(y, dtype=float))
        mask = arr <= frac
        out = np.empty_like(arr)
        out[mask] = lower + (arr[mask] * (split - lower) / frac)
        out[~mask] = split + (arr[~mask] - frac) * (upper - split) / (1 - frac)
        return out[0] if np.isscalar(y) else out
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    titles = ['Gaussian Coverage by Target Variable', 'Bootstrap Coverage by Target Variable']
    methods = ['Gaussian', 'Bootstrap']
    
    for ax, method, title in zip(axes, methods, titles):
        data = df_coverage_long[df_coverage_long['Method'] == method]
        sns.boxplot(
            data=data,
            x='Target Variable',
            y='Coverage',
            hue='Model',
            palette=academic_colors,
            ax=ax
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Variable', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-1, 101)
        ax.set_yscale('function', functions=(forward_piecewise, inverse_piecewise))
        ax.set_yticks([0, 20, 40, 60, 80, 85, 90, 95, 100])
        ax.set_yticklabels([0, 20, 40, 60, 80, 85, 90, 95, 100])
        ax.margins(y=0.02)
        ax.set_ylabel('Coverage (%)', fontsize=12)
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
    
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='lower right', title='Model')
    axes[1].legend_.remove()
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    # summary_stats = df_coverage_long.groupby(['Target Variable', 'Model', 'Method'])['Coverage'].agg(['mean', 'std', 'median', 'min', 'max'])
    # print("\nCoverage Statistics by Target Variable:")
    # print(summary_stats)
    
    return summary_stats