from tools.constants import VOC_BBOX_LABEL_NAMES, COCO_BBOX_LABEL_NAMES, OPENIMAGES_BBOX_LABEL_NAMES
import numpy as np
import os


class Object(object):
    def __init__(self, dataset, class_name, xmin, ymin, xmax, ymax, confidence=None, model=None):
        assert dataset in ['voc', 'coco', 'openimages'], 'Supported dataset: voc, coco, openimages'
        self.model = model
        self.class_name = class_name
        self.class_id = None
        if dataset == 'coco':
            self.class_id = COCO_BBOX_LABEL_NAMES.index(self.class_name)
        elif dataset == 'openimages':
            self.class_id = OPENIMAGES_BBOX_LABEL_NAMES.index(self.class_name)
        elif dataset == 'voc':
            self.class_id = VOC_BBOX_LABEL_NAMES.index(self.class_name)
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)
        self.confidence = float(confidence) if confidence is not None else confidence

    def iou(self, o):
        xA = max(self.xmin, o.xmin)
        yA = max(self.ymin, o.ymin)
        xB = min(self.xmax, o.xmax)
        yB = min(self.ymax, o.ymax)
        interArea = max(0., xB - xA) * max(0., yB - yA)
        boxAArea = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        boxBArea = (o.xmax - o.xmin) * (o.ymax - o.ymin)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    @property
    def bbox(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    @staticmethod
    def generic_iou(objects):
        venn = np.zeros(shape=(int(max(obj.xmax for obj in objects)) + 1, int(max(obj.ymax for obj in objects)) + 1))
        for obj in objects:
            venn[int(obj.xmin):int(obj.xmax), int(obj.ymin):int(obj.ymax)] += 1
        return np.sum(venn == len(objects)) / np.sum(venn != 0)


def load_detection_results(dr_dir, dataset, min_confidence=0.05, whitelist=None):
    assert os.path.exists(dr_dir), 'Detection results not found (%s)' % dr_dir
    detected_objects = {}
    for fname in filter(lambda fn: '.txt' in fn, os.listdir(dr_dir)):
        fid = fname.replace('.txt', '')
        if whitelist is not None and fid not in whitelist:
            continue
        detected_objects[fid] = []
        with open(os.path.join(dr_dir, fname), 'r') as f:
            for line in f.readlines():
                values = line.strip().split(' ')
                confidence = 1.00
                if len(values) == 6:
                    class_name, confidence, xmin, ymin, xmax, ymax = values
                else:
                    class_name, xmin, ymin, xmax, ymax = values
                if float(confidence) < min_confidence or (float(xmax) - float(xmin)) * (float(ymax) - float(ymin)) <= 0:
                    continue
                detected_objects[fid].append(Object(dataset=dataset, model=dr_dir, class_name=class_name,
                                                    confidence=confidence, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
    return detected_objects
