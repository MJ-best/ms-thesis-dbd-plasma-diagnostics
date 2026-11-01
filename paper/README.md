# Paper draft: control-device study of electrical DBD degradation diagnosis

Draft manuscript (path A of [docs/research-outlook.ko.md](../docs/research-outlook.ko.md)), including the supply–load analysis (path C).

| File | Content |
|---|---|
| [manuscript.md](manuscript.md) | Manuscript draft v0.1 (English) |
| `figures/fig*.png`, `figures/fig*.pdf` | Figures 1–6 |
| [make_figures.py](make_figures.py) | Regenerates all figures from the raw-data archive |

Regenerate the figures (the archive is not distributed; see [docs/raw-data-verification.md](../docs/raw-data-verification.md)):

```bash
pip install -e ".[dev]" pillow
python paper/make_figures.py "<archive root>" paper/figures
```

`<archive root>` is the folder that contains `전류전압 측정`, `전압에 따른 QV 변화` and `현미경사진`. The numbers in the tables come from `examples/verify_raw_archive.py`.

Open items before submission are listed at the end of the manuscript.
