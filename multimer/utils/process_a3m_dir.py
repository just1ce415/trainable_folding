import json
from tqdm import tqdm

def read_second_line(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        return lines[1].strip() if len(lines) > 1 else ""

def process_samples(input_file, output_file, a3m_dir):
    seq_lens = []
    results = []

    with open(input_file, 'r') as sample_file:
        for line in tqdm(sample_file):
            sample_id = line.strip()
            filepath = f"{a3m_dir}/{sample_id}/mmseqs/aggregated.a3m"
            second_line = read_second_line(filepath)
            seq_lens.append(len(second_line))
            results.append({
                "sample_id": sample_id,
                "sequences": {
                    'A': second_line
                },
                "seq_len": len(second_line),
                "dataset": "predict",
                "seed": 0,
                "is_val": 1,
                "is_inference": 1,
                "crop_size": 512

            })

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=2)

    print(f"Max seq len: {max(seq_lens)}")
    print(f"Min seq len: {min(seq_lens)}")
    print(f"Mean seq len: {sum(seq_lens) / len(seq_lens)}")
    print(f"Median seq len: {sorted(seq_lens)[len(seq_lens) // 2]}")

if __name__ == "__main__":
    input_file = "/storage/erglukhov/new_residue/human/samples"
    output_file = "/storage/erglukhov/new_residue/human/data.json"
    a3m_dir = "/storage/erglukhov/new_residue/human/a3m"
    process_samples(input_file, output_file, a3m_dir)
