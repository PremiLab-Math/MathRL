import argparse
import glob
import json
import os
import sys

from tqdm import tqdm

from grader import math_equal


def process_file(input_path: str, output_path: str) -> None:
    with open(input_path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    with open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(lines, desc=f"Processing {os.path.basename(input_path)}", unit="lines"):
            data = json.loads(line)
            preds = data.get('pred', [])
            scores = data.get('score', [])

            unique_preds = []
            unique_scores = []
            seen_exact = set()

            for pred, score in zip(preds, scores):
                if pred in seen_exact:
                    continue

                is_dup = False
                for upred in unique_preds:
                    try:
                        if math_equal(pred, upred, timeout=True):
                            is_dup = True
                            break
                    except Exception as e:
                        print(f"Error comparing {pred} vs {upred}: {e}", file=sys.stderr)
                if is_dup:
                    continue

                unique_preds.append(pred)
                unique_scores.append(score)
                seen_exact.add(pred)

            data['pred_set'] = unique_preds
            data['score_set'] = unique_scores

            fout.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Process math prediction files.')
    parser.add_argument('--input_dir', type=str, default='input',
                        help='Input directory containing JSONL files')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for processed results')
    args = parser.parse_args()

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, '*pass16*.jsonl')
    files = glob.glob(pattern)

    for filepath in tqdm(files, desc="Overall progress", unit="file"):
        base = os.path.basename(filepath)
        name, ext = os.path.splitext(base)
        out_file = os.path.join(args.output_dir, f"{name}_processed{ext}")
        print(f"Processing {filepath} -> {out_file}")
        process_file(filepath, out_file)


if __name__ == '__main__':
    main()
