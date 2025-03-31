# DLIO Data Generator

## Usage

### Config Files

`data_generator_config.json` 

Args:
- "data_path": The path of the processed data in csv format.
- "num_files": Number of synthetic files to generate. Based on the algorithm used to generate the synthetic data, the files used to generate is sequential to the provided dataset. If *null* or *"all"*, then will generate synthetic data for the entire dataset.
- "compression_type": in string format the file extension of the compressed data
- "save_path": path where the synthetic data is saved
- "analyze": whether an analysis of the synthetic data is desired


The example config provided in the `./config/data_generator_config.json` is as follows:

```
{
  "data_path": "./data/processed/inputs/20240605T052200--imagenet-rand-300000-processed__jpg_npz.csv",
  "num_files": 10,
  "compression_type": "npz",
  "save_path": "./outputs/imagenet-deflate/",
  "analyze": {
    "analyze": false,
    "analysis_save_dir": null,
    "analysis_fname": null
  }
}
```

