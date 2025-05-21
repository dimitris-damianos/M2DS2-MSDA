import argparse

from datasets import Dataset


def load_dataset(dataset_name):
    dataset = Dataset.load_from_disk(f"datafast/{dataset_name}.train.hf")
    return dataset


def remove_long_utterances(
    dataset, max_seconds=12, min_seconds=1, sr=16_000, num_proc=24
):
    max_length = max_seconds * sr
    min_length = min_seconds * sr
    dataset = dataset.filter(
        lambda x: min_length < len(x["input_values"]) < max_length,
        batch_size=1,
        num_proc=num_proc,
    )
    return dataset


def trim_dataset(
    dataset,
    dataset_name,
    max_seconds: float = 12,
    min_seconds: float = 0.5,
    num_proc: int = 24,
):
    print(f"Len before removing long utts {len(dataset)}")
    data_location = f"datafast/{dataset_name}.train.trim_{int(max_seconds)}.hf"
    dataset = remove_long_utterances(
        dataset, max_seconds=max_seconds, min_seconds=min_seconds, num_proc=num_proc
    )
    dataset.save_to_disk(data_location)
    print(f"Len after removing long utts {len(dataset)}")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser("Domain adaptive pretraining")
    parser.add_argument("--dataset-name", type=str, help="dataset to trim")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=12,
        help="keep utterances under max seconds",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=24,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.5,
        help="keep utterances over max seconds",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name)
    trim_dataset(dataset, args.dataset_name, max_seconds=args.max_seconds, min_seconds=args.min_seconds, num_proc=args.nj)


if __name__ == "__main__":
    main()
