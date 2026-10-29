# Fonts for the synthetic corpus

Not vendored — licences must be reviewed per deployment. All are OFL from Google Fonts.

Place in `tools/assets/fonts/`:

| File | Role |
|---|---|
| NotoSans-Regular.ttf / NotoSans-Bold.ttf | Latin print |
| NotoSansDevanagari-Regular.ttf / -Bold.ttf | Devanagari print |
| Kalam-Regular.ttf | Handwriting — covers **both** Devanagari and Latin |
| Caveat-Regular.ttf | Handwriting — Latin |
| NotoSansMono-Regular.ttf | Statement / MICR lines |

Latin falls back to DejaVuSans if Noto is absent.
Devanagari has **no fallback**: the generator exits with an explanatory error rather than
rendering tofu boxes. A corpus full of `□□□□` that silently trains a recogniser is worse
than a corpus that failed to build.
