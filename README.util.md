# FSL Util Code

[![WesleyCh3n - FSL](https://img.shields.io/badge/WesleyCh3n-FSL-2ea44f?logo=github)](https://github.com/WesleyCh3n/FSL)
[![hackmd-github-sync-badge](https://hackmd.io/nQElH4AyS3SF9ZijfrdSSA/badge)](https://hackmd.io/nQElH4AyS3SF9ZijfrdSSA)


## `util_convert_format.py`

Convert `checkpoint` to `h5` or `tflite`

```bash
python3 util_convert_format.py <path/to/params.py>
```

## `util_export_embedding.py`

Export data to embeddings.

```bash
python3 util_export_embedding.py <path/to/params.py>
```

This will export `params['test_ds']` into embeddings.

## `util_export_reference.py`

Average embeddings with each class as supported set.

```bash
python3 util_export_reference.py <path/to/output_dir_name/>
```
