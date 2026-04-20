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
