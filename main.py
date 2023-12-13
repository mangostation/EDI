from core.oden import EDI
from tools.io import load_detection_results
from tools.evaluate import evaluate
import random
random.seed(1993)
DATASET = 'coco'  # supported: [voc|coco|openimages]
TEAM = ['./dataset/coco/detection/YOLOv3-D53',
        './dataset/coco/detection/SSD512-R50',
        './dataset/coco/detection/FRCNN-M3']
GT_DIR = './dataset/coco/groundtruths'

####################################################################################################################
# Load ground truth labels and detection results
groundtruth = load_detection_results(dataset=DATASET, dr_dir=GT_DIR)
team_det = {model: load_detection_results(dataset=DATASET, dr_dir=model) for model in TEAM}

####################################################################################################################
# Split the training and validation sets
fnames = list(groundtruth.keys())
random.shuffle(fnames)
split = int(len(fnames) * 0.20)
fnames_val, fnames_test = fnames[:split], fnames[split:]


if __name__ == '__main__':
    edi = EDI(dataset=DATASET, team=TEAM)
    X = {fname: [team_det[model][fname] for model in TEAM] for fname in fnames_val}
    y = {fname: groundtruth[fname] for fname in fnames_val}
    edi.fit(X, y)
    det_EDI = {fname: edi.detect(x=[team_det[model][fname] for model in TEAM]) for fname in fnames_test}
    for model in TEAM:
        print('%s: %.2f%%' % (model.split('/')[-1], evaluate(team_det[model], groundtruth=groundtruth)['map'] * 100))
    print('=====')
    print('EDI: %.2f%%' % (evaluate(det_EDI, groundtruth=groundtruth)['map'] * 100))
