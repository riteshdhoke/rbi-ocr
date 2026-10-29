# Corpus evaluation — 2026-08-17, validators only

**Run:** dev split (variants 0-2), 177 documents, 0 failures. **Code:** `1e50be7`.
**Mode:** `--no-slm` — every remote provider method refused, so this is the deterministic
layer alone. **Supersedes** [`REPORT-2026-08-08-preSLM.md`](REPORT-2026-08-08-preSLM.md),
which was the same pipeline in the same mode at code `bcd85dd`. That is the controlled
comparison: same corpus, same seed, same harness, SLM absent in both.

## The SLM path is NOT measured here

This run refused every remote provider method, so nothing below describes the SLM. The last
SLM measurement is [`REPORT-2026-08-08-slmlive.md`](REPORT-2026-08-08-slmlive.md), and **its
numbers are stale** — it predates the 15 Aug recognition fixes and everything in `1e50be7`.
Two figures from it are still the best available and should be treated as *upper bounds on
what is known*, not as current measurements:

* **SLM field F1 0.119** (P 0.675, R 0.065) against 147 eligible documents — a **miss**
  against the PRD target, and the only time that target has ever been measured;
* `bank_statement` recognition CER **0.211** under both heads, cause still unknown.

Re-running with the SLM live needs a working `HF_TOKEN` and credits. Until someone does that,
**no current number exists for the SLM path at all** — do not quote the 0.119 as if it
described this code.

## What moved

| Metric | 2026-08-08 | 2026-08-17 | |
|---|---|---|---|
| Recognition CER, overall | 0.483 | **0.427** | -11.6% relative |
| `bank_name` F1 | 0.747 | **0.789** | P 0.862 → 0.857, R 0.659 → **0.732** |
| `bank_name` tp/fp/fn | 81/13/42 | **90/15/33** | |
| LID accuracy (matched) | — | 0.877 | |
| Detection line recall, clean | — | 0.955 | |

The CER move is the width-bucket fix being measured for the first time. Recognition was
never re-evaluated after 15 Aug, so the improvement had been real and unrecorded for nine days.

## What this run does NOT measure

**The bank-name defect this code fixed is invisible here, by construction.** The corpus
renders five banks and all five were already in the old table, so the widening from 5 to
~70 banks cannot show up in these numbers. The `bank_name` gain above is recognition
improving, not coverage improving. The live-data effect — `City Union Bank` no longer
published as *Union Bank of India*, 35 real banks no longer silently missing — is measured
by the sweep in `tests/unit/test_s5_validators.py::TestBankName`, not by this corpus.

A corpus built from five banks cannot tell you what a system does with the sixth.

Recognition on **real handwriting** is also unmeasured; the corpus renders handwriting with
fonts. The caveat at the foot of this report is not boilerplate.

---

```
==============================================================================
RBI OCR corpus evaluation — harness 1.1.0
==============================================================================
corpus       E:\pythonprojects\python_ocr\corpus  (seed 1729, generator 1.2.0)
predictions  docs\eval\predictions-2026-08-17
split        dev  final=False  locked variants [3, 4]
documents    177 scored / 177 selected / 118 withheld

Recognition — CER / WER by script x modality x profile
-----------------------------------------------------
cell               lines  ref_chars    CER    WER
-------------------------------------------------
Deva|hand|bounce      51       1524  1.000  1.000
Deva|hand|clean       51       1524  0.041  0.192
Deva|hand|heavy       51       1527  0.622  0.918
Deva|hand|light       51       1518  0.056  0.265
Deva|hand|photo       51       1521  0.526  0.865
Deva|hand|scan        51       1511  0.509  0.731
Deva|print|bounce     54       1035  1.000  1.000
Deva|print|clean      54       1035  0.044  0.249
Deva|print|heavy      54       1035  0.262  0.692
Deva|print|light      54       1035  0.090  0.423
Deva|print|photo      54       1035  0.626  0.930
Deva|print|scan       54       1035  0.143  0.502
Latn|hand|bounce      69       1827  1.000  1.000
Latn|hand|clean       69       1824  0.002  0.019
Latn|hand|heavy       69       1820  0.726  1.015
Latn|hand|light       69       1810  0.018  0.141
Latn|hand|photo       69       1824  0.930  1.163
Latn|hand|scan        69       1830  0.480  0.748
Latn|print|bounce    588       8110  1.000  1.000
Latn|print|clean     543       7698  0.015  0.023
Latn|print|heavy     588       8106  0.432  0.648
Latn|print|light     588       8145  0.113  0.244
Latn|print|photo     588       8123  0.605  0.852
Latn|print|scan      588       8143  0.258  0.488

Recognition — rollups
--------------------------------------------------
cell            lines  ref_chars    CER    WER
----------------------------------------------
script:Deva       630      15335  0.419  0.649
script:Latn      3897      59260  0.429  0.574
profile:bounce    762      12496  1.000  1.000
profile:clean     717      12081  0.019  0.072
profile:heavy     762      12488  0.484  0.745
profile:light     762      12508  0.090  0.251
profile:photo     762      12503  0.645  0.909
profile:scan      762      12519  0.311  0.562
OVERALL          4527      74595  0.427  0.593

Detection — IoU and line recall by profile
----------------------------------------------------------------
profile  gt_lines  matched  spurious  mean_IoU  line_recall
-----------------------------------------------------------
bounce        762        0         0       n/a        0.000
clean         717      717         9     0.767        0.955
heavy         762      679        93     0.358        0.287
light         762      757         8     0.543        0.577
photo         762      673       105     0.266        0.123
scan          762      741        39     0.451        0.450
OVERALL      4527     3567       254     0.482        0.393

Fields — precision / recall / F1 by field
-------------------------------------------------------------
field                  TP  FP   FN      P      R     F1
-------------------------------------------------------
account_number         65  14   61  0.823  0.516  0.634
bank_name              90  15   33  0.857  0.732  0.789
card_last_four         23   1   10  0.958  0.697  0.807
complainant_address     0   0   54    n/a  0.000    n/a
complainant_contact     0   0   54    n/a  0.000    n/a
complainant_name        0   0  108    n/a  0.000    n/a
complaint_summary       0   0   90    n/a  0.000    n/a
disputed_amount        50  30   91  0.625  0.355  0.452
ifsc_code              50  11   40  0.820  0.556  0.662
micr_code              11   3    7  0.786  0.611  0.688
prior_complaint_ref     0   0   72    n/a  0.000    n/a
relief_sought           0   0   36    n/a  0.000    n/a
transaction_date       91  48   32  0.655  0.740  0.695
transaction_reference  61  10   62  0.859  0.496  0.629

Fields — by source mechanism (R = share of ALL reference fields this source captured)
----------------------------------------------------------
source            TP   FP    GT      P      R     F1
----------------------------------------------------
lookup_table      90   15  1191  0.857  0.076  0.139
regex_validator  351  117  1191  0.750  0.295  0.423

Fields — field x source (R = share of that field's reference this source captured)
-----------------------------------------------------------------------------
field|source                           TP  FP   GT      P      R     F1
-----------------------------------------------------------------------
account_number|regex_validator         65  14  126  0.823  0.516  0.634
bank_name|lookup_table                 90  15  123  0.857  0.732  0.789
card_last_four|regex_validator         23   1   33  0.958  0.697  0.807
disputed_amount|regex_validator        50  30  141  0.625  0.355  0.452
ifsc_code|regex_validator              50  11   90  0.820  0.556  0.662
micr_code|regex_validator              11   3   18  0.786  0.611  0.688
transaction_date|regex_validator       91  48  123  0.655  0.740  0.695
transaction_reference|regex_validator  61  10  123  0.859  0.496  0.629

Fields — bilingual slots (fields whose native spelling differs from its English)
---------------------------------
metric                     count
--------------------------------
gt occurrences               198
claimed                       22
value = page's own string      8
value = its translation        0
value_en correct               0
value_en wrong                 0
value_en absent               22
    `value` is contractually the page's own string; a translation there is a contract violation this table counts but precision/recall cannot see.

Schema validity
---------------------
metric         value
--------------------
checked          177
valid            177
validity_rate  1.000

Escalation vs reference rule 'quality_or_hand'
-----------------------------------------
TP  FP  FN  precision  recall     F1
------------------------------------
36   6  96      0.857   0.273  0.414

Quality gate — confusion (rows: quality_expectation, cols: verdict)
--------------------------------------
expectation  accept  repair  bounce
-----------------------------------
clean            42      15       0
degraded         15      75       0
bounce            0       0      30
    expectation-satisfied rate: 0.915 over 177 documents

Language ID — LINE level
----------------------------------------------------------------------------
cell        gt_lines  matched  correct  acc_matched  acc_all  coverage
----------------------------------------------------------------------
Deva|hand        306      243      101        0.416    0.330     0.794
Deva|print       324      261      209        0.801    0.645     0.806
Latn|hand        414      322      293        0.910    0.708     0.778
Latn|print      3483     2741     2524        0.921    0.725     0.787
OVERALL         4527     3567     3127        0.877    0.691     0.788

Latency (ms) by stage
------------------------------------------
stage             n       p50       p95
---------------------------------------
assemble         46     0.790     3.629
e2e_total       177  1072.124  2273.951
s0_read         177     1.150    40.614
s15_quality     177   208.818   234.478
s15_quality:p1  177   208.818   234.478
s1_ingest       177    19.580    54.168
s2b_lid         147     1.087     5.365
s2b_lid:p1      147     1.087     5.365
s3_classify     147   160.722   329.131
s3_classify:p1  147   160.722   329.131
s4_ocr          147   595.175  1626.566
s4_ocr:p1       147   595.175  1626.566
s4b_vlm          46     0.111     0.149
s4b_vlm:p1       46     0.111     0.149
s5_slm          147     0.014     0.024
s5_validators   147    35.528   199.888

Latency (ms) by stage x profile
------------------------------------------------
stage|profile           n       p50       p95
---------------------------------------------
assemble|clean          3     0.245     0.255
assemble|heavy         12     0.894     3.472
assemble|light          9     0.681     1.054
assemble|photo         11     1.010     3.662
assemble|scan          11     0.761     3.154
e2e_total|bounce       30   152.166   174.922
e2e_total|clean        27  1290.180  2680.505
e2e_total|heavy        30  1067.834  2224.373
e2e_total|light        30  1255.504  2582.187
e2e_total|photo        30  1022.574  2165.319
e2e_total|scan         30  1137.059  2308.174
s0_read|bounce         30     0.885     1.144
s0_read|clean          27     1.165    40.515
s0_read|heavy          30    40.115    40.576
s0_read|light          30     2.329    40.854
s0_read|photo          30     1.004    40.295
s0_read|scan           30    40.126    40.767
s15_quality|bounce     30   142.177   161.796
s15_quality|clean      27   214.775   244.536
s15_quality|heavy      30   212.321   230.238
s15_quality|light      30   218.794   246.208
s15_quality|photo      30   212.896   224.984
s15_quality|scan       30   210.317   231.625
s15_quality:p1|bounce  30   142.177   161.796
s15_quality:p1|clean   27   214.775   244.536
s15_quality:p1|heavy   30   212.321   230.238
s15_quality:p1|light   30   218.794   246.208
s15_quality:p1|photo   30   212.896   224.984
s15_quality:p1|scan    30   210.317   231.625
s1_ingest|bounce       30     8.012     9.515
s1_ingest|clean        27    49.864    54.237
s1_ingest|heavy        30    24.906    32.275
s1_ingest|light        30    50.047    60.882
s1_ingest|photo        30    17.684    20.177
s1_ingest|scan         30    36.724    43.840
s2b_lid|clean          27     1.087     6.151
s2b_lid|heavy          30     1.038     5.038
s2b_lid|light          30     1.109     4.868
s2b_lid|photo          30     1.095     5.361
s2b_lid|scan           30     1.073     5.158
s2b_lid:p1|clean       27     1.087     6.151
s2b_lid:p1|heavy       30     1.038     5.038
s2b_lid:p1|light       30     1.109     4.868
s2b_lid:p1|photo       30     1.095     5.361
s2b_lid:p1|scan        30     1.073     5.158
s3_classify|clean      27   299.111   335.706
s3_classify|heavy      30   146.149   181.282
s3_classify|light      30   302.694   358.838
s3_classify|photo      30   128.330   171.181
s3_classify|scan       30   225.703   254.356
s3_classify:p1|clean   27   299.111   335.706
s3_classify:p1|heavy   30   146.149   181.282
s3_classify:p1|light   30   302.694   358.838
s3_classify:p1|photo   30   128.330   171.181
s3_classify:p1|scan    30   225.703   254.356
s4_ocr|clean           27   689.777  1887.464
s4_ocr|heavy           30   593.826  1521.310
s4_ocr|light           30   595.438  1812.122
s4_ocr|photo           30   582.214  1568.570
s4_ocr|scan            30   584.511  1620.959
s4_ocr:p1|clean        27   689.777  1887.464
s4_ocr:p1|heavy        30   593.826  1521.310
s4_ocr:p1|light        30   595.438  1812.122
s4_ocr:p1|photo        30   582.214  1568.570
s4_ocr:p1|scan         30   584.511  1620.959
s4b_vlm|clean           3     0.108     0.112
s4b_vlm|heavy          12     0.110     0.133
s4b_vlm|light           9     0.105     0.122
s4b_vlm|photo          11     0.121     0.148
s4b_vlm|scan           11     0.114     0.163
s4b_vlm:p1|clean        3     0.108     0.112
s4b_vlm:p1|heavy       12     0.110     0.133
s4b_vlm:p1|light        9     0.105     0.122
s4b_vlm:p1|photo       11     0.121     0.148
s4b_vlm:p1|scan        11     0.114     0.163
s5_slm|clean           27     0.014     0.019
s5_slm|heavy           30     0.014     0.027
s5_slm|light           30     0.013     0.025
s5_slm|photo           30     0.014     0.017
s5_slm|scan            30     0.013     0.020
s5_validators|clean    27    31.331   130.276
s5_validators|heavy    30    35.043   224.324
s5_validators|light    30    37.328   112.177
s5_validators|photo    30    34.172   231.138
s5_validators|scan     30    35.963   176.860

CAVEAT  Font-rendered handwriting is not real handwriting. This corpus validates plumbing, schema, routing and throughput. It does NOT validate handwriting accuracy — measure that on iiit-indic-hw-words and real complaint pages.
NOTE    variants [3, 4] are the locked test split and were not scored. Pass --split all --final to include them.
```
