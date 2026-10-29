# models/ — local model weights (gitignored)

This directory holds CPU model weights loaded by the `local` provider. It is empty in git;
in air-gapped deployments it is populated offline from approved artifacts.

Expected layout (see `app/providers/local.py` for the authoritative loader paths):

```
models/
  ocr/            # PP-OCRv5 det + rec ONNX (Latin + Devanagari heads), provenance in OPEN-QUESTIONS Q1
  mt/             # IndicTrans2 indic-en dist-200M, CTranslate2 INT8
  lid/lid.176.ftz # fasttext language ID (917 KB compressed model)
```

Dev-laptop policy (owner decision D1): only models under ~300 MB each may be downloaded here.
SLM/VLM weights are never stored locally — those route through the HF Inference API under the
`laptop` profile, and through approved offline artifacts in production profiles.
