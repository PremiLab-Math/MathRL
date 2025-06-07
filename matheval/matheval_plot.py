import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def extract_model_name(base_path):
    folder_name = os.path.basename(base_path.rstrip('/'))
    if '-' in folder_name:
        last_dash_index = folder_name.rfind('-')
        model_name = folder_name[:last_dash_index]
    else:
        model_name = folder_name
    return model_name


def load_models_data(base_paths, datasets):
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    all_data = []
    for base_path in base_paths:
        model_name = extract_model_name(base_path)
        step_dirs = [d for d in os.listdir(base_path)
                     if d.startswith("global_step_") and os.path.isdir(os.path.join(base_path, d))]
        steps = sorted(int(d.split("_")[-1]) for d in step_dirs)

        matheval_path = os.path.join(base_path, step_dirs[0], "matheval")
        subfolders = [f for f in os.listdir(matheval_path)
                      if os.path.isdir(os.path.join(matheval_path, f))]

        valid = ['overall', 'average'] + subfolders
        bad = [d for d in datasets if d not in valid]
        if bad:
            print(f"Invalid datasets for {model_name}: {bad}")
            continue

        data = {sf: [] for sf in subfolders}
        for step in steps:
            for sf in subfolders:
                fp = os.path.join(base_path,
                                  f"global_step_{step}",
                                  "matheval", sf,
                                  "test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1_metrics.json")
                try:
                    acc = json.load(open(fp)).get('acc', np.nan)
                except:
                    acc = np.nan
                data[sf].append(acc)

        all_data.append((data, model_name, steps, base_path))
    return all_data


def compute_averages(metrics, steps, selected_keys=None):
    keys_all = list(metrics.keys())
    all_deepseek = []
    sel_deepseek = []

    for i in range(len(steps)):
        ds_all = [
            metrics[sf]['deepseek'][i]
            for sf in keys_all
            if 'deepseek' in metrics[sf] and not np.isnan(metrics[sf]['deepseek'][i])
        ]
        all_deepseek.append(sum(ds_all) / len(ds_all) if ds_all else np.nan)

        if selected_keys:
            ds_sel = [
                metrics[sf]['deepseek'][i]
                for sf in selected_keys
                if sf in metrics and not np.isnan(metrics[sf]['deepseek'][i])
            ]
            sel_deepseek.append(sum(ds_sel) / len(ds_sel) if ds_sel else np.nan)

    result = {'all_deepseek': all_deepseek}
    if selected_keys:
        result['sel_deepseek'] = sel_deepseek
    return result


def add_value_labels(steps, values, ax, color='dimgrey', rotation=45,
                     show_values='none', fontsize=16):
    valid_data = [(s, v) for s, v in zip(steps, values) if not np.isnan(v)]
    if not valid_data or show_values == 'none':
        return

    valid_steps, valid_values = zip(*valid_data)
    max_val = max(valid_values)
    max_index = next(i for i, v in enumerate(valid_values) if v == max_val)

    if show_values == 'max':
        s_max = valid_steps[max_index]
        ax.text(
            s_max, max_val + 0.5,
            f'{max_val:.1f}%',
            ha='center', va='bottom',
            rotation=rotation,
            fontsize=fontsize,
            color=color
        )
    elif show_values == 'min_max':
        min_val = min(valid_values)
        min_index = next(i for i, v in enumerate(valid_values) if v == min_val)
        s_min = valid_steps[min_index]
        ax.text(
            s_min, min_val + 0.5,
            f'{min_val:.1f}%',
            ha='center', va='bottom',
            rotation=rotation,
            fontsize=fontsize,
            color=color
        )
        s_max = valid_steps[max_index]
        ax.text(
            s_max, max_val + 0.5,
            f'{max_val:.1f}%',
            ha='center', va='bottom',
            rotation=rotation,
            fontsize=fontsize,
            color=color
        )
    elif show_values == 'all':
        for s, v in valid_data:
            ax.text(
                s, v + 0.5,
                f'{v:.1f}%',
                ha='center', va='bottom',
                rotation=rotation,
                fontsize=fontsize,
                color=color
            )


def setup_plot(title, xlabel, ylabel, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    plt.title(title, fontsize=26, pad=25, fontweight='bold')
    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    plt.grid(True, linestyle=':', alpha=0.7)
    return ax


def plot_common(ax, steps, datasets, data, model_name, show_values, 
                is_combined=False, color=None, marker=None):
    colors = plt.cm.tab20.colors
    markers = ['o','s','D','^','v','<','>','p','*','h','H','+','x','X','d','|','_']
    
    sel = [d for d in datasets if d in data]
    avg = compute_averages({sf: {'deepseek': data[sf]} for sf in data},
                           steps, selected_keys=sel if sel else None)
    all_avg = avg.get('all_deepseek', None)
    sel_avg = avg.get('sel_deepseek', None)

    line_color = color if color else '#2c3e50'
    marker_style = marker if marker else 'D'

    if 'overall' in datasets and all_avg is not None:
        label = f'{model_name}' if is_combined else 'Overall Avg.'
        plt.plot(steps, all_avg, '-', linewidth=4, marker=marker_style, markersize=8,
                 color=line_color, label=label)
        add_value_labels(steps, all_avg, ax,
                         color=line_color, rotation=45,
                         show_values=show_values)

    if 'average' in datasets and sel_avg is not None:
        label = f'{model_name} - Selected Avg.' if is_combined else 'Selected Avg.'
        plt.plot(steps, sel_avg, '-', linewidth=3, marker='o', markersize=8,
                 color='#e74c3c', label=label)
        add_value_labels(steps, sel_avg, ax,
                         color='#e74c3c', rotation=45,
                         show_values=show_values)

    for idx, ds in enumerate(datasets):
        if ds in ['overall', 'average'] or ds not in data:
            continue
        
        ds_color = colors[idx*2] if not is_combined else line_color
        ds_marker = markers[idx % len(markers)] if is_combined else 'o'
        label = f'{model_name} - {ds.upper()}' if is_combined else ds.upper()
        
        plt.plot(steps, data[ds], '-', linewidth=2, marker=ds_marker, markersize=6,
                 color=ds_color, label=label)
        add_value_labels(steps, data[ds], ax,
                         color=ds_color, rotation=45,
                         show_values=show_values)

    plt.xticks(steps, rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(0, 55)

    handles, labels = ax.get_legend_handles_labels()
    n_legend = len(handles)
    
    if n_legend >= 8:
        plt.legend(loc='lower right', fontsize=20, prop={'weight':'bold', 'size': 20}, 
                  markerscale=1.5, ncol=2)
    else:
        plt.legend(loc='lower right', fontsize=20, prop={'weight':'bold', 'size': 20}, 
                  markerscale=1.5)


def main():
    parser = argparse.ArgumentParser(description='Plot mathematical evaluation metrics.')
    parser.add_argument('--base_paths', nargs='+', required=True, help='Base directory paths')
    parser.add_argument('--datasets', nargs='+', default=['overall'],
                        help='Datasets to plot (e.g. overall average aime24)')
    parser.add_argument('--plot_mode', choices=['single', 'combined', 'both'], default='both',
                        help='Plot mode: single (individual plots), combined (one combined plot), or both')
    parser.add_argument('--show_values', choices=['none', 'all', 'min_max', 'max'], default='none',
                        help='Value display mode: none (no labels), all (all values), min_max (only min and max), max (only max)')
    args = parser.parse_args()

    all_data = load_models_data(args.base_paths, args.datasets)

    if args.plot_mode in ['single', 'both']:
        for data, model_name, steps, base_path in all_data:
            suffix = f"_{model_name}" if len(all_data) > 1 else ""
            ax = setup_plot(f"{model_name} MathEval Progression", "Training Steps", "Accuracy (%)")
            plot_common(ax, steps, args.datasets, data, model_name, args.show_values)
            out = os.path.join(base_path, f"matheval_progression{suffix}.svg")
            plt.savefig(out, bbox_inches='tight', dpi=300, format='svg')
            plt.close()
            print(f"Saved to {out}")

    if args.plot_mode in ['combined', 'both'] and len(all_data) > 1:
        output_dir = os.path.dirname(args.base_paths[0]) or '.'
        combined_path = os.path.join(output_dir, "matheval_progression_combined.svg")
        ax = setup_plot("Model Performance Dynamics - Overall Avg.", "Training Steps", "Accuracy (%)")
        colors = plt.cm.tab20.colors
        for idx, (data, model_name, steps, _) in enumerate(all_data):
            plot_common(ax, steps, args.datasets, data, model_name, args.show_values,
                        is_combined=True, color=colors[idx], marker='D')
        plt.savefig(combined_path, bbox_inches='tight', dpi=300, format='svg')
        plt.close()
        print(f"Saved combined plot to {combined_path}")


if __name__ == '__main__':
    main()
