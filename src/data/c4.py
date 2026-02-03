import os
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
from tqdm import tqdm

tknzr = tiktoken.get_encoding("gpt2")
SEED = 2357
TOTAL_SAMPLES = 4_000_000
TEST_RATIO = 0.0001


def get_c4_data(datasets_base_dir, num_proc=4):
    C4_DATA_PATH = os.path.join(datasets_base_dir, "c4/")
    train_file = os.path.join(C4_DATA_PATH, "train.bin")
    val_file = os.path.join(C4_DATA_PATH, "val.bin")

    if os.path.exists(train_file) and os.path.exists(val_file):
        return {"train": train_file, "val": val_file}

    os.makedirs(C4_DATA_PATH, exist_ok=True)

    raw_dataset_path = os.path.join(C4_DATA_PATH, "raw_4M")
    if not os.path.exists(raw_dataset_path):
        print(f"Downloading first {TOTAL_SAMPLES} samples from C4...")
        dataset_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        raw_list = list(dataset_stream.take(TOTAL_SAMPLES))
        raw_dataset = Dataset.from_list(raw_list)
        raw_dataset.save_to_disk(raw_dataset_path)
        del raw_list, raw_dataset
    else:
        print("Raw dataset already downloaded.")

    print("Loading raw dataset for train/val split...")
    full_dataset = Dataset.load_from_disk(raw_dataset_path)
    split_dataset = full_dataset.train_test_split(
        test_size=TEST_RATIO,
        seed=SEED,
        shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")
    del full_dataset

    def process(example):
        ids = tknzr.encode_ordinary(example["text"])
        ids.append(tknzr.eot_token)
        return {"ids": ids, "len": len(ids)}

    print("Tokenizing splits...")
    tokenized = split_dataset.map(
        process,
        remove_columns=["text", "timestamp", "url"],
        desc="tokenizing",
        num_proc=num_proc,
        batched=False,
    )
    del split_dataset

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"])
        filename = os.path.join(C4_DATA_PATH, f"{split}.bin")
        dtype = np.uint16
        print(f"Writing {filename} (total tokens: {arr_len:,})...")

        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        idx = 0

        total_batches = min(1024, len(dset))
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {split}"):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
            del batch, arr_batch

        arr.flush()
        del arr

    return {"train": train_file, "val": val_file}


if __name__ == "__main__":
    result = get_c4_data("./datasets/", num_proc=4)
    print("Done:", result)