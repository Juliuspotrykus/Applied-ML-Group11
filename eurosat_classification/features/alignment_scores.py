"""
Main and secondary bands were chosen based on literature. 
Main bands were those identified by both sources, secondary bands are those identified by only one of the sources.

AnnualCrop
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture
- B4 [2] -> soil contrast 

Forest
- B5, B6, B7, B8 [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture
- B4 [2] -> soil contrast 

HerbaceousVegetation
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture

Highway
- B11, B12, B8 [1]
- B2 [2] -> man-made object detection
- B4 [2] -> urban and soil separation
- B11, B12 [2] -> moisture contrast

Industrial
- B11, B12, B8 [1]
- B2 [2] -> man-made object detection
- B4 [2] -> urban and soil separation
- B11, B12 [2] -> moisture contrast

Pasture
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture

PermanentCrop
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture
- B4 [2] -> soil contrast 

Residential
- B11, B12, B8 [1]
- B2 [2] -> man-made object detection
- B4 [2] -> urban and soil separation
- B11, B12 [2] -> moisture contrast

River
- B3, B8, B11 [1]
- B3 [2] -> water contrast
- B8 [2] -> shoreline mapping
- B4 [2] -> land water separation

SeaLake
- B3, B8, B11 [1]
- B3 [2] -> water contrast
- B8 [2] -> shoreline mapping
- B4 [2] -> land water separation

[1] https://www.mdpi.com/2071-1050/17/22/10324
[2] https://custom-scripts.sentinel-hub.com/sentinel-2/bands/

"""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.band_names import MS_BAND_NAMES
from ..data.label_map import label_map
from ..data.datasets import EuroSATMSDataset
from ..data.download import get_dataset_path
from ..data.preprocessors import normalize_MS_img
from ..data.split import get_train_val_test_splits
from .integrated_gradients import band_attribution_totals


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
	parser = argparse.ArgumentParser(
			description="Alignment of attribution scores with literature.",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		)
	parser.add_argument("--model_path", required=True, help="Path to saved model (.pkl)")
	parser.add_argument("--n_steps", type=int, default=50,
						help="IG interpolation steps per image.")
	parser.add_argument("--max_samples", type=int, default=None,
						help="Process only this many images (useful for quick smoke-tests).")
	parser.add_argument("--device", default=None,
						help="Torch device (cuda/mps/cpu). Auto-detected if not set.")
	parser.add_argument("--alpha", type=float, default=0.5,
						help="Weighting given to secondary bands.")
	args = parser.parse_args()


	device = args.device or _auto_device()
	print(f"Device:     {device}")

	model = torch.load(args.model_path, map_location=device, weights_only=False)
	model.eval()

	data_root = Path(get_dataset_path())
	train_csv, _, _ = get_train_val_test_splits()

	dataset = EuroSATMSDataset(
            root=data_root / "EuroSATallBands",
            csv_path=train_csv,
            transform=normalize_MS_img,
        )
	band_names = MS_BAND_NAMES

	n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

	expect_main_class_to_band = {
		0: ["B5 - Red Edge 1", "B6 - Red Edge 2", "B7 - Red Edge 3", "B8A - Narrow NIR"],
		1: ["B5 - Red Edge 1", "B6 - Red Edge 2", "B7 - Red Edge 3", "B8 - NIR"],
		2: ["B5 - Red Edge 1", "B6 - Red Edge 2", "B7 - Red Edge 3", "B8A - Narrow NIR"],
		3: ["B11 - SWIR 1", "B12 - SWIR 2"],
		4: ["B11 - SWIR 1", "B12 - SWIR 2"],
		5: ["B5 - Red Edge 1", "B6 - Red Edge 2", "B7 - Red Edge 3", "B8A - Narrow NIR"],
		6: ["B5 - Red Edge 1", "B6 - Red Edge 2", "B7 - Red Edge 3", "B8A - Narrow NIR"],
		7: ["B11 - SWIR 1", "B12 - SWIR 2"],
		8: ["B3 - Green", "B8 - NIR"],
		9: ["B3 - Green", "B8 - NIR"],
	}

	expect_secondary_class_to_band = {
		0: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2", "B4 - Red"],
		1: ["B8A - Narrow NIR", "B11 - SWIR 1", "B12 - SWIR 2", "B4 - Red"],
		2: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2"],
		3: ["B8 - NIR", "B2 - Blue", "B4 - Red"],
		4: ["B8 - NIR", "B2 - Blue", "B4 - Red"],
		5: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2"],
		6: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2", "B4 - Red"],
		7: ["B8 - NIR", "B2 - Blue", "B4 - Red"],
		8: ["B11 - SWIR 1", "B4 - Red"],
		9: ["B11 - SWIR 1", "B4 - Red"],
	}

	alignment = {}

	for class_idx, _ in label_map.items():
		results = band_attribution_totals(
			model=model,
			dataset=dataset,
			n_steps=args.n_steps,
			target_class=class_idx,
			max_samples=args.max_samples,
			device=device,
			verbose=True,
		)

		positive = results["positive"]
		negative = results["negative"]

		importance = np.abs(positive) + np.abs(negative)
		total = importance.sum()

		if total == 0:
			alignment[class_idx] = 0.0
			continue

		main_mask = np.isin(band_names, expect_main_class_to_band[class_idx])
		secondary_mask = np.isin(band_names, expect_secondary_class_to_band[class_idx])

		attr_main = importance[main_mask].sum() / total
		attr_secondary = importance[secondary_mask].sum() / total

		alignment[class_idx] = float(attr_main + args.alpha * attr_secondary)

	return alignment

if __name__ == "__main__":
    main()
