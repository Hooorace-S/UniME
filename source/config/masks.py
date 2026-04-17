import numpy as np

MASKS = [
    [False, False, False, True],
    [False, True, False, False],
    [False, False, True, False],
    [True, False, False, False],

    [False, True, False, True],
    [False, True, True, False],
    [True, False, True, False],
    [False, False, True, True],
    [True, False, False, True],
    [True, True, False, False],

    [True, True, True, False],
    [True, False, True, True],
    [True, True, False, True],
    [False, True, True, True],

    [True, True, True, True]
]

MASK_NAMES = [
    'T2', 'T1ce', 'T1', 'FLAIR',

    'T1ce-T2', 'T1ce-T1', 'FLAIR-T1',
    'T1-T2', 'FLAIR-T2', 'FLAIR-T1ce',

    'FLAIR-T1ce-T1', 'FLAIR-T1-T2', 'FLAIR-T1ce-T2', 'T1ce-T1-T2',

    'ALL'
]

MASK_ARRAY = np.array(MASKS)
MASK_MODALITY_MAP = {
    name: np.array([i]) for i, name in enumerate(MASK_NAMES)
}
VALID_MASKS = [True, True, True, True]
