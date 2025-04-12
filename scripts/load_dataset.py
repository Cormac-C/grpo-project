import os
import sys
import logging
from torch.utils.data import DataLoader
from datasets import load_dataset

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)


def custom_collate_fn(batch):
    return batch


# TODO: Probably get rid of this script eventually


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Loading dataset from HF...")
    # Load the dataset from Hugging Face
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    logger.info("Dataset loaded.")

    if dataset is None:
        logger.error("Failed to load dataset from HF")
        return
    logger.info("Dataset loaded successfully.")
    # Split dataset into train and test sets
    train_test = dataset.train_test_split(test_size=0.1)
    dataset = train_test["train"]
    test_dataset = train_test["test"]
    logger.info("Dataset split into train and test sets.")

    # Log some examples
    logger.info("Logging some examples from the dataset...")
    for i in range(5):
        logger.info(f"Example {i}: {dataset[i]}")
    logger.info("Finished logging examples.")

    batch_size = 16
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    logger.info("DataLoader created with batch size: %d", batch_size)

    # log some examples
    for i, batch in enumerate(train_dataloader):
        if i >= 5:
            break
        logger.info(f"Batch {i}: {batch}")
    logger.info("Finished logging batches.")


if __name__ == "__main__":
    main()
