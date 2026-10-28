# Hillstrom MineThatData Dataset

The dataset is not redistributed in this repository because the original
publication does not state an explicit reuse license.

Download the CSV from the original
[MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
and save it as:

```text
dataset/Kevin_Hillstrom_MineThatData.csv
```

Direct publisher URL:

```text
https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
```

Expected SHA-256 for the dataset used in this analysis:

```text
9bd6dd74628d14b652cef843df5e52c07f9024706dc00ece13855a85c4543b41
```


The file served by the publisher today is not byte-identical to the copy used in
the analysis: it writes `spend` as `0` where the analysis copy has `0.0`. Its
SHA-256 is:

```text
0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece
```

Both load as identical DataFrames (64,000 rows, 12 columns), and re-serializing the
publisher file with pandas (CRLF line endings, `index=False`) reproduces the
analysis copy byte for byte. The pre-registered fingerprints in `artifacts/` refer to
that copy. To fetch and rebuild it, verifying both hashes:

```bash
python scripts/fetch_hillstrom.py
```

The project license applies to the repository's code and documentation, not to
the third-party dataset.
