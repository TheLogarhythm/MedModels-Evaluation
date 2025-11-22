"""Configuration management utilities"""

# Path configurations
DATASET_META_CLS = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json"
META_DATA_BASE = "/jhcnas4/Generalist/shebd/Generalist_Meta_data"
DATASET_BASE = "/jhcnas4/Generalist/Medical_Data_2025"
LOCAL_DATASET_BASE = "/home/sunanhe/luoyi/model_eval/datasets"

SAVING_BASE_DIR = "/home/sunanhe/luoyi/model_eval/eval_results"
# /{Model}_{Scale}_{Dataset}/
# checkpoints: best.pth, last.pth
# json: best_results.json, last_results.json (prediction and ground_truth of all instances)

# Dataset preparation state file
TRAINING_STATE_FILE = "/home/sunanhe/luoyi/model_eval/logs/dataset_preparation/preparation_state.json"

# Datasets list: first run prepare_datasets.py to check dataset availability
ALL_DATASETS = [
        "010_RSNA",
        "011_SIIM-ACR",
        "016_CBIS_DDSM_CALC",
        "016_CBIS_DDSM_MASS",
        "017_MedFMC_ENDO",
        "017_MedFMC_COLON",
        "021_PADUFES20",
        "025_Dermnet",
        "027_NLMTB",
        "034_ODIR",
        "037_WCE",
        "043_UBIBC",
        "044_BUSI",
        "046_BUSBRA",
        "051_Derm7PT_Derm",
        "051_Derm7PT_Clinic",
        "075_PCam200",
        "077_RetOCT",
        "136_ChestXRay2017",
        "137_OCT2017"
    ]


# Model paths (relative to trainer directory)
MODEL_PATHS = {
    'medvit': '../../../MedViT',
    'medvitv2': '../../../MedViTV2',
    'medmamba': '../../../MedMamba'
}

# Image normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEDIMAGE_MEAN = [0.5, 0.5, 0.5]
MEDIMAGE_STD = [0.5, 0.5, 0.5]