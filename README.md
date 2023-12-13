# Boosting Object Detection Ensembles with Error Diversity
實驗方法與github原始code相同方法在/core/oden.py中修改

## Introduction
Object detection has played a pivotal role in numerous mission-critical applications. This paper presents a focal error diversity framework, called EDI, for strengthening the robustness of object detection ensembles under benign and adversarial scenarios. We introduce an ensemble pruning method for object detection using a novel focal error diversity measure as the robustness synergy indicator. Given a base model pool, it recommends top sub-ensembles with a smaller ensemble size yet achieving equivalent or even better mAP performance than using all available object detection models as a large ensemble. This is made possible by our negative sampling methods for object detection to capture the degree of negative correlations and the focal error diversity to measure the failure independence of component detection models in an ensemble. Extensive experiments on three object detection benchmark datasets validate that EDI effectively selects space-time efficient object detection ensembles with high mAP performance.
This repository contains the source code of the following paper:
* Ka-Ho Chow and Ling Liu. "Boosting Object Detection Ensembles with Error Diversity." IEEE International Conference on Data Mining (ICDM), Orlando, FL, USA, Nov. 28 - Dec. 1, 2022.

## Installation and Dependencies
This project runs on Python 3.6. It is lightweight and requires only two external libraries, which can be installed through:
```bash
pip install scikit-learn
pip install numpy
```

## Instruction
1. Prepare a directory containing ground-truth objects. Each file in the directory corresponds to a test image with the following format for each row, which represents an object included in the image:
```
class_name xmin ymin xmax ymax
```
2. Prepare a set of directories containing objects detected by different detectors. Each file in the directory corresponds to a test image with the following format for each row, which represents an object detected in the image:
```
class_name confidence_score xmin ymin xmax ymax
```
3. Modify `main.py`: update `DATASET` to either `voc`, `coco`, and `openimages`, point `GT_DIR` to the directory created in Step 1 and include all directories from Step 2 in `TEAM`
4. Run the following script to initiate, train, and evaluate EDI.
```bash
python main.py
```

We provide an example with three member models trained on MS COCO, where you can simply run the script in Step 4 without changing any parameters to get the following outputs:
```bash
YOLOv3-D53: 65.60%
SSD512-R50: 57.45%
FRCNN-M3: 60.40%
=====
EDI: 70.62%
```

## Acknowledgement
This research is partially sponsored by the National Science Foundation CISE grants 2038029, 2026945, 1564097, an IBM faculty award, and a Cisco grant on Edge Computing.
