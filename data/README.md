# Data Folder

Place the downloaded datasets or preprocessed data here. The repository does
not redistribute data files.

Do not commit raw datasets, preprocessed dataframes, recovered predictions,
trained model weights, or access credentials. The VR goalkeeper dataset and
ForDigitStress dataset have separate access conditions and citation
requirements. ForDigitStress is an external dataset that must be requested from
the dataset administrators and used according to its end-user license agreement
(EULA), including scientific non-commercial use only and no redistribution of
the dataset.

If you use the VR goalkeeper dataset or code derived from this repository,
please cite the dataset record and the accompanying paper. If you use
ForDigitStress, please cite the corresponding ForDigitStress publication and
follow the acknowledgement requirements in its EULA.

## Data access procedures

- **VR goalkeeper dataset**:
  - download the dataset from the Zenodo record referenced in the repository
    root `README.md`,
  - place the raw files in `data/vr_goalkeeper/raw/`,
  - preprocess them with `scripts/preprocessing/preprocess_vr_goalkeeper.py`,
  - minimal example of reading one raw file:

    ```python
    import pandas as pd
    df = pd.read_csv("data/vr_goalkeeper/raw/LogID_0_base1.csv", sep=";")
    ```

- **ForDigitStress dataset**:
  - request access from the dataset administrators at
    [hcai.eu/fordigitstress/](https://hcai.eu/fordigitstress/),
  - place the authorized raw files in `data/fordigitstress/raw/`,
  - preprocess them with `scripts/preprocessing/preprocess_fordigitstress.py`,
  - minimal example of reading one raw file:

    ```python
    import pandas as pd
    df = pd.read_csv("data/fordigitstress/raw/VP1/stress.csv", sep=";")
    ```

The preprocessing scripts document the expected input files and the dataset-
specific processing steps used in the manuscript.

Suggested layout:

```text
data/
├── vr_goalkeeper/
│   ├── raw/
│   └── dataframes/
│       ├── DL_out.pkl
│       └── features_out.pkl
└── fordigitstress/
    ├── raw/
    └── dataframes/
        └── DL_out.pkl
```

The example configuration files in `configs/examples/` assume this layout.
