import os
import time
import glob
import requests
import pandas as pd
import numpy as np
import tiktoken
from datasets import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# === 配置 ===
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/run01/scw6c4c/HT_Noise_EXP/cache/tiktoken"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 2357
TOTAL_SAMPLES = 1_000_000
TEST_RATIO = 0.0001

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DATASET_ID = "wikimedia/wikipedia"
SUBDIR = "20231101.en"  # 英文数据集子目录
HF_MIRROR = "https://hf-mirror.com"
SAVE_DIR_BASE = "/HOME/scw6c4c/run/datasets/"
WIKI_DATA_PATH = os.path.join(SAVE_DIR_BASE, "wikipedia")
RAW_PARQUET_DIR = os.path.join(WIKI_DATA_PATH, "parquet_raw")
os.makedirs(RAW_PARQUET_DIR, exist_ok=True)

tknzr = tiktoken.get_encoding("gpt2")


def download_parquet_files_if_needed(max_files=200):
    """下载前 max_files 个 parquet 分片（使用正确URL）"""
    print("🔍 获取 Parquet 文件列表...")
    tree_url = f"{HF_MIRROR}/api/datasets/{DATASET_ID}/tree/{REVISION}/{SUBDIR}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(tree_url, headers=headers)
    resp.raise_for_status()

    # 获取文件列表（API返回的path已包含20231101.en/前缀）
    files = [
        f["path"] for f in resp.json()
        if f["type"] == "file" and f["path"].endswith(".parquet")
    ]
    files = sorted(files)
    target_files = files[:max_files]
    print(f"Found {len(files)} files, will download {len(target_files)}")

    def download_one(filename):
        local_path = os.path.join(RAW_PARQUET_DIR, os.path.basename(filename))
        if os.path.exists(local_path):
            print(f"✅ Already downloaded locally: {filename}")
            return

        # ✅ 修复点：直接使用API返回的完整路径（不添加SUBDIR）
        url = f"{HF_MIRROR}/datasets/{DATASET_ID}/resolve/{REVISION}/{filename}"

        for attempt in range(5):
            try:
                r = requests.get(url, headers=headers, stream=True, timeout=120)
                if r.status_code in (403, 404):
                    print(f"⚠️ {r.status_code} for {filename} (attempt {attempt + 1}/5)")
                    time.sleep(5)
                    continue
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                print(f"✅ Downloaded: {filename}")
                return
            except Exception as e:
                print(f"❌ Failed ({attempt + 1}/5): {filename} - {str(e)}")
                time.sleep(10)
        raise RuntimeError(f"Failed to download {filename} after 5 attempts")

    print(f"📥 Downloading {len(target_files)} files...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        list(tqdm(executor.map(download_one, target_files), total=len(target_files), desc="Downloading Parquet"))


def load_first_n_samples(n):
    """从本地 Parquet 文件中流式读取前 n 条样本"""
    parquet_files = sorted(glob.glob(os.path.join(RAW_PARQUET_DIR, "*.parquet")))
    samples = []
    count = 0

    print(f"📚 Reading first {n:,} samples from Parquet files...")
    for file in tqdm(parquet_files, desc="Processing Parquet"):
        if count >= n:
            break
        df = pd.read_parquet(file)
        remaining = n - count
        take = min(len(df), remaining)
        batch = df.iloc[:take].to_dict(orient="records")
        samples.extend(batch)
        count += take
        del df

    print(f"✅ Loaded {len(samples)} samples")
    return samples


def get_wiki_data(datasets_base_dir, num_proc=10):
    WIKI_DATA_PATH = os.path.join(datasets_base_dir, "wikipedia")
    train_file = os.path.join(WIKI_DATA_PATH, "train.bin")
    val_file = os.path.join(WIKI_DATA_PATH, "val.bin")

    if os.path.exists(train_file) and os.path.exists(val_file):
        return {"train": train_file, "val": val_file}

    os.makedirs(WIKI_DATA_PATH, exist_ok=True)

    # # === Step 1: 下载 Parquet 文件（正确路径）===
    # download_parquet_files_if_needed(max_files=200)

    # === Step 2: 加载前 TOTAL_SAMPLES 条 ===
    raw_list = load_first_n_samples(TOTAL_SAMPLES)
    raw_dataset = Dataset.from_list(raw_list)
    del raw_list

    # === Step 3: 划分 train/val ===
    split_dataset = raw_dataset.train_test_split(
        test_size=TEST_RATIO,
        seed=SEED,
        shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")
    del raw_dataset

    # === Step 4: Tokenize ===
    def process(example):
        ids = tknzr.encode_ordinary(example["text"])
        ids.append(tknzr.eot_token)
        return {"ids": ids, "len": len(ids)}

    print("Tokenizing splits...")
    tokenized = split_dataset.map(
        process,
        remove_columns=["url", "title"],
        desc="tokenizing",
        num_proc=num_proc,
        batched=False,
    )

    del split_dataset

    # === Step 5: 写入 .bin 文件 ===
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"])
        filename = os.path.join(WIKI_DATA_PATH, f"{split}.bin")
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
    result = get_wiki_data("/HOME/scw6c4c/run/datasets/", num_proc=4)
    print("Done:", result)