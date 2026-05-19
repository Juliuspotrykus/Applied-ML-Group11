import torch
from torchvision import transforms


def normalize_MS_img(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize all bands of a given image,
    knowing their means and standard deviations
    """
    # Precomputed 2nd and 98th percentile pixel values
    # of each band, based on the training set.
    clip_mins = torch.tensor(
        [
            1000.0,
            718.0,
            483.0,
            257.0,
            220.0,
            202.0,
            187.0,
            150.0,
            60.0,
            6.0,
            27.0,
            15.0,
            140.0,
        ]
    )[:, None, None]

    clip_maxs = torch.tensor(
        [
            1964.0,
            1961.0,
            2071.0,
            2528.0,
            2587.0,
            3680.0,
            4579.0,
            4531.0,
            1702.0,
            23.0,
            3953.0,
            2895.0,
            5004.0,
        ]
    )[:, None, None]

    # Precomputed means and standard deviations of
    # each band, based on the training set.
    means = torch.tensor(
        [
            1350.6204966130333,
            1108.56131338614,
            1033.6290657422908,
            937.3560028625166,
            1191.3386558185557,
            1996.7863281120824,
            2365.4366610346397,
            2292.8702064990493,
            727.79170901021,
            11.951646567046957,
            1814.140218899843,
            1111.522513485863,
            2591.064080429481,
        ]
    )

    stdevs = torch.tensor(
        [
            228.87170028962095,
            288.2377114365647,
            353.65416293463716,
            556.0305161306454,
            538.2618318700565,
            846.6016300939549,
            1066.9121517697647,
            1099.1289321615109,
            391.73180417327205,
            3.758520342333439,
            985.3114209351224,
            738.2927133046666,
            1212.4142852989894,
        ]
    )

    # First clip per band
    img = torch.clip(img, clip_mins, clip_maxs)

    # Then normalize per band
    normalize = transforms.Normalize(means, stdevs)
    return normalize(img)
