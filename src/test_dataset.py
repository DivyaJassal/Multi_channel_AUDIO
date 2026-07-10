from dataset import SpeechSeparationDataset
from config import load_config

config = load_config()

dataset = SpeechSeparationDataset(
    root_dir=config["data_root"],
    split="train",
    sample_rate=config["sample_rate"],
    segment_seconds=config["segment_seconds"],
    max_sources=config["max_sources"],
    return_metadata=True,   
                           
                            
                            
)

sample = dataset[0]

print("Mixture:", sample["mixture"].shape)
print("Sources:", sample["sources"].shape)
print("Number of Speakers:", sample["num_sources"])
print("Speaker IDs:", sample["speaker_ids"])
print("Overlap:", sample["overlap_ratio"])
print("Duration:", sample["duration"])