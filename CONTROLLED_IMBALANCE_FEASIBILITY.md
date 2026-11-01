# Controlled Imbalance Feasibility

This audit checks whether target training ratios can be formed without duplicating minority examples.

## baf_base

- Natural train rows: 33784
- Positive train rows: 338
- Negative train rows: 33446
- Natural majority-per-minority ratio: 98.95
- Maximum common fixed train size across audited ratios: 1352

| target_ratio | sampling_mode | minimal_drop_train_rows | minimal_drop_positive_n | minimal_drop_negative_n | fixed_common_positive_n | fixed_common_negative_n | positive_count_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:3 | majority_downsample_only | 1352 | 0 | 32432 | 338 | 1014 | False |
| 1:5 | majority_downsample_only | 2028 | 0 | 31756 | 225 | 1127 | False |
| 1:10 | majority_downsample_only | 3718 | 0 | 30066 | 122 | 1230 | False |
| 1:20 | majority_downsample_only | 7098 | 0 | 26686 | 64 | 1288 | False |
| 1:50 | majority_downsample_only | 17238 | 0 | 16546 | 26 | 1326 | False |

## online_shoppers_purchasing_intention

- Natural train rows: 7398
- Positive train rows: 1145
- Negative train rows: 6253
- Natural majority-per-minority ratio: 5.46
- Maximum common fixed train size across audited ratios: 4580

| target_ratio | sampling_mode | minimal_drop_train_rows | minimal_drop_positive_n | minimal_drop_negative_n | fixed_common_positive_n | fixed_common_negative_n | positive_count_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:3 | majority_downsample_only | 4580 | 0 | 2818 | 1145 | 3435 | False |
| 1:5 | majority_downsample_only | 6870 | 0 | 528 | 763 | 3817 | False |
| 1:10 | minority_downsample_required | 6878 | 520 | 0 | 416 | 4164 | False |
| 1:20 | minority_downsample_required | 6565 | 833 | 0 | 218 | 4362 | False |
| 1:50 | minority_downsample_required | 6378 | 1020 | 0 | 89 | 4491 | False |

## secom

- Natural train rows: 751
- Positive train rows: 50
- Negative train rows: 701
- Natural majority-per-minority ratio: 14.02
- Maximum common fixed train size across audited ratios: 200

| target_ratio | sampling_mode | minimal_drop_train_rows | minimal_drop_positive_n | minimal_drop_negative_n | fixed_common_positive_n | fixed_common_negative_n | positive_count_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1:3 | majority_downsample_only | 200 | 0 | 551 | 50 | 150 | False |
| 1:5 | majority_downsample_only | 300 | 0 | 451 | 33 | 167 | False |
| 1:10 | majority_downsample_only | 550 | 0 | 201 | 18 | 182 | True |
| 1:20 | minority_downsample_required | 736 | 15 | 0 | 9 | 191 | True |
| 1:50 | minority_downsample_required | 715 | 36 | 0 | 3 | 197 | True |

## Recommendation

- Main common controlled ratios: `1:3`, `1:5`, `1:10`.
- BAF can support `1:20` and `1:50` as dataset-specific extreme-ratio checks.
- Online Shoppers can support extreme ratios only by discarding positive examples when the target ratio is more imbalanced than the natural training distribution.
- SECOM should not be the main evidence for extreme controlled ratios because fixed-ratio cells leave very few positive examples.
