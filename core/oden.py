from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest
from tools.io import Object
import itertools
import copy


class EDI:
    def __init__(self, dataset, team):
        self.dataset = dataset
        self.team = team
        self.clf = None

    def fit(self, X, y):
        _X, _y = [], []
        for fname in X.keys():
            clusters = fusion(dataset=self.dataset, team=self.team, ensemble=X[fname])
            for cluster in clusters:
                _X.append(cluster.vectorize())
                _y.append(find_match(y[fname], cluster.repr))
        # print(len(_X[0]))
        self.sl = SelectKBest(k=3)
        _X = self.sl.fit_transform(_X, _y)
        print(_y)
        self.clf = LogisticRegression().fit(_X, _y) # 70.54%
        # self.clf = GaussianNB().fit(_X, _y) # 70.51%
        # clf1 = LogisticRegression()
        # clf2 = RandomForestClassifier()
        # clf3 = GaussianNB()
        # self.clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft').fit(_X, _y) # 70.33%
        # self.clf = GradientBoostingClassifier().fit(_X, _y) # 70.53%
        # self.clf = RandomForestClassifier().fit(_X, _y) # 69.07%
        return self

    def detect(self, x):
        res = []
        clusters = fusion(dataset=self.dataset, team=self.team, ensemble=x)
        for cluster in clusters:
            Xt = self.sl.transform([cluster.vectorize()])
            cluster_objectness = self.clf.predict_proba(Xt)[0, 1]
            # cluster_objectness = self.clf.predict_proba([cluster.vectorize()])[0, 1]
            detected_object = copy.deepcopy(cluster.repr)
            detected_object.confidence = detected_object.confidence * cluster_objectness
            res.append(detected_object)
        return res


class Cluster:
    def __init__(self, team, first_member, dataset):
        self.members = []
        self.team = team
        self.models = set()
        self.repr = None
        self.dataset = dataset
        self.add(first_member)

    def add(self, new_member):
        self.members.append(new_member)
        self.models.add(new_member.model)
        self.update_repr()

    def update_repr(self):
        if len(self.members) == 1:
            self.repr = Object(class_name=self.members[0].class_name, confidence=self.members[0].confidence,
                               xmin=self.members[0].xmin, ymin=self.members[0].ymin,
                               xmax=self.members[0].xmax, ymax=self.members[0].ymax, dataset=self.dataset)
        else:
            sum_of_confidence = sum(member.confidence for member in self.members)
            self.repr.xmin = sum(member.xmin * member.confidence / sum_of_confidence for member in self.members)
            self.repr.ymin = sum(member.ymin * member.confidence / sum_of_confidence for member in self.members)
            self.repr.xmax = sum(member.xmax * member.confidence / sum_of_confidence for member in self.members)
            self.repr.ymax = sum(member.ymax * member.confidence / sum_of_confidence for member in self.members)
            self.repr.confidence = sum_of_confidence / len(self.members)

    def vectorize(self):
        x = [0. for _ in range(len(self.team) * 2)]
        for member in self.members:
            ind = self.team.index(member.model)
            x[ind*2+0] = member.confidence
            x[ind*2+1] = member.iou(self.repr)
        return x


def find_matching_cluster(clusters, obj, match_iou):
    best_iou = match_iou
    best_index = -1
    for cluster_id, cluster in enumerate(clusters):
        if obj.class_name != cluster.repr.class_name or obj.model in cluster.models:
            continue
        iou = obj.iou(cluster.repr)
        if iou > best_iou:
            best_index = cluster_id
            best_iou = iou
    return best_index, best_iou


def fusion(dataset, team, ensemble, iou_thr=0.50):
    grouped_objects = {}
    for obj in itertools.chain(*ensemble):
        if obj.class_name not in grouped_objects:
            grouped_objects[obj.class_name] = []
        grouped_objects[obj.class_name].append(obj)

    constructed_clusters = []
    for label, objs in grouped_objects.items():
        clusters = []
        # Clusterize boxes
        for obj in sorted(objs, key=lambda x: -x.confidence):
            index, best_iou = find_matching_cluster(clusters, obj, iou_thr)
            if index == -1:  # create a new cluster:
                clusters.append(Cluster(team=team, first_member=obj, dataset=dataset))
            else:
                clusters[index].add(obj)
        # Rescale confidence based on number of models and boxes
        for cluster in clusters:
            cluster.repr.confidence = cluster.repr.confidence * len(cluster.members) / len(ensemble)
        constructed_clusters += clusters
    return constructed_clusters


def find_match(matchers, matchee):
    for matcher in matchers:
        if matcher.class_name == matchee.class_name and matcher.iou(matchee) >= 0.50:
            return True
    return False
