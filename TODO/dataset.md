# Dataset Preparation

## Dataset Creation

load the dataset by the following code:

```python
import webdataset as wds
shards = "/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/data/cholec80/shards/shard-{000001..000046}.tar"
dataset = wds.WebDataset(shards, shardshuffle=42).decode("pil").shuffle(4000).to_tuple("jpg") or dataset = wds.WebDataset(shards, shardshuffle=42).decode("rgb").shuffle(4000).to_tuple("jpg")
```


## Custom on dinov3 dataset

Create a similar file cholec80.py similar to other datasets under the folder dinov3/dinov3/data/datasets