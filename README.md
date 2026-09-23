# RYXTHO002_MEC4128S

FINAL YEAR PROJECT, SEA ICE FLOE SEGMENTATION

## Overview

This repository holds the code for a UCT Mechanical and Mechatronic Engineering final year thesis benchmarking four methods for segmenting sea ice floes from ship based video footage, GVF snake, YOLOE11, SAM 2, and a reproduction of Liss's (2023) original thresholding and contouring method, so the newer three methods can be compared fairly against the project's predecessor work. Every method reports the same two metrics per frame, sea ice concentration as a percentage and floe count, so results line up directly across methods.

All four pipelines run on the same raw footage, video01.mp4, and share one preprocessing module, sea_ice_preprocessing.py, which handles perspective correction and, where enabled, gray level enhancement. The perspective correction is a homography based proxy for full orthorectification, since camera intrinsic and extrinsic calibration data isn't available for this camera, a limitation carried through from the predecessor project rather than a new gap introduced here.

## Methods compared

- GVF snake, gvf_pipeline.py, matched to Passerotti et al. (2026), Section 4.1, tiled Otsu binarization, distance transform seeding, erosion based separation of merged floes, and GVF diffusion following Xu and Prince (1998)
- YOLOE11, yoloe_pipeline.py, an open vocabulary detector, run in visual prompt mode by default, since hand drawn example boxes substantially outperform text prompts on this footage
- SAM 2, sam2_pipeline.py, run in prompt based mode (SAM-P), reusing GVF's own seed generation stage directly, with shape based filters, convexity, circularity, and eccentricity, applied afterward to reject masks that swallow open water rather than a single floe
- Liss benchmark, benchmark_pipeline.py, a faithful reproduction of Liss's original threshold and contour method from SEA_ICE_Functions.py, run on the shared preprocessing pipeline rather than his original hardcoded setup, so it's directly comparable, kept exactly as he had it rather than retuned

Supporting scripts, comparison_plots.py and threshold_sweep.py, handle cross method comparison plots and threshold tuning, and are not standalone pipelines in their own right.

## Repository structure

Scripts sit flat at the root of this repository rather than split into per method folders, since several of them import from each other, sea_ice_preprocessing.py is shared by all four, and sam2_pipeline.py additionally imports seed generation functions directly from gvf_pipeline.py. Keeping everything in one folder avoids fragile cross folder imports while the project is still under active development.

Debug scripts, output videos, CSV logs, and images are prefixed by the method they belong to, gvf_, yoloe_, and sam2_, to keep them distinguishable within the flat structure. venv and __pycache__ are excluded from version control via .gitignore.

## Usage

Each of the four pipeline scripts is run directly.

```
python gvf_pipeline.py
python yoloe_pipeline.py
python sam2_pipeline.py
python benchmark_pipeline.py
```

Running any of them opens an interactive window on the first frame of the video for perspective calibration, drag the trackbars until the correction looks right, then press Esc to lock it in and let the rest of the video process automatically. yoloe_pipeline.py, when set to visual prompt mode, opens a second window afterward for boxing a handful of example floes by hand.

Each script writes an annotated output video and a per frame CSV log of floe count and sea ice concentration, named for the method that produced it.

## Notes

yoloe_pipeline.py needs its model weights file, yoloe-11l-seg.pt, present in this folder. sam2_pipeline.py needs a SAM 2 checkpoint downloaded separately per Meta's setup instructions, with SAM2_CHECKPOINT near the top of that file pointed at wherever it's installed locally, this path is machine specific and needs updating on a new setup.

Several configuration constants, methodology decisions, and known limitations specific to each method are documented directly in that script's own module docstring and inline comments, rather than repeated here.