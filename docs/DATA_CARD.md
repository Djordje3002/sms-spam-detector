# Data card

UCI SMS Spam Collection: this archive parses to 5,574 labeled lines. After normalized duplicate and conflict handling, 5,159 messages remain. The original corpus combines several historical SMS sources, so it is not a current or geographically representative messaging sample.

## Source and citation

[Original dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

## License

UCI lists the dataset under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the creators and source when reusing data or derived figures. The repository's MIT license applies to its code, not to the dataset.

## Integrity and storage

```text
{
  "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
  "sha256": "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
}
```

For downloaded datasets, every run verifies archive bytes against this SHA-256 before parsing. The downloader has a 60-second timeout and 20 MB response cap. It reads named members without extracting filesystem paths. Raw files stay in the ignored `data/raw/` cache. To recover from a corrupted cache, remove `data/raw/source.zip`, verify the upstream source, then rerun training. Do not silently replace the expected hash after a source change.

## Processing and sampling

Casefolded, whitespace-normalized messages define duplicates. Conflicting duplicate-label groups are discarded; one representative of every remaining normalized message is retained. Seeded stratified splits allocate 60%/20%/20% to train/validation/test. Vectorizers learn vocabulary and IDF only on training text. Each candidate gets the validation threshold with maximum F1 subject to precision ≥ 0.98; the model with highest feasible validation F1 is selected. With no feasible threshold, the model predicts no spam.

## Representation and privacy

Near-duplicate templates can survive exact normalized deduplication. Sender IDs, campaign IDs and reliable chronology are unavailable, so this is not an unseen-campaign or temporal deployment estimate. Non-English text, modern phishing campaigns and adversarial obfuscation require fresh evaluation. Do not automatically delete user messages based on this demonstration.

No private account data, API credentials, or scraped personal profiles are used. Synthetic fixtures exist only in tests and manually constructed examples; published metrics come from the cited dataset.
