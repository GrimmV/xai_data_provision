import pandas as pd
import numpy as np
import math
from collections import defaultdict
from typing import Union, List, Dict
from sklearn.metrics import confusion_matrix
from utils import combine_lists_uniquely

from DecisionRuling import DecisionRuling


class DataHandler:

    def __init__(self, db, stage="selection") -> None:
        self.db = db
        self.data_collection = self.db["data"]
        self.model_collection = self.db["model"]

        self.model_id = self.model_collection.find_one({"stage": stage})["_id"]

        self.datapoint_collection = self.db["datapoint"]
        self.predictions_meta_collection = self.db["predictions_meta"]
        self.predictions_collection = self.db["predictions"]
        self.feature_collection = self.db["feature"]
        self.label_collection = self.db["label"]
        self.feature_stats_collection = self.db["feature_stats"]
        self.label_stats_collection = self.db["label_stats"]
        self.model_scores = self.db["model_scores"]
        self.anchor_collection = self.db["anchor"]
        self.anchor_meta_collection = self.db["anchor_meta"]
        self.shap_collection = self.db["shap"]
        self.shap_meta_collection = self.db["shap_meta"]
        self.trustscore_collection = self.db["trustscore"]
        self.trustscore_meta_collection = self.db["trustscore_meta"]
        self.ice_collection = self.db["ice_collection"]
        self.partial_dependence_collection = self.db["partial_dependence_collection"]
        self.pd_meta_collection = self.db["pd_meta"]
        self.pd_values_collection = self.db["pd_values_collection"]

        self.label_name = self._get_label_name()

    def get_confusion_matrix(self, kind="test"):
        matrix = []
        predictions = self._get_predictions(kind)
        labels = self._get_labels(kind)

        unique_classes = combine_lists_uniquely(labels, predictions)
        cm = confusion_matrix(labels, predictions, labels=unique_classes)

        for i in range(len(unique_classes)):
            for j in range(len(unique_classes)):
                matrix.append(
                    {
                        "label": unique_classes[i],
                        "prediction": unique_classes[j],
                        "count": int(cm[i][j]),
                    }
                )

        return matrix

    def get_model_scores(self, kind="test"):
        model_scores = self.model_scores.find_one({'model_id': self.model_id})

        return self.transform_metrics(model_scores["general"], kind)


    def get_learning_curves(self, kinds=["train", "test"]) -> dict[str, list]:
        model_scores = self.model_scores.find_one({"model_id": self.model_id})
        curves = {}

        for score in list(model_scores["learning_curve"]):
            for kind in kinds:
                if kind not in curves:
                    curves[kind] = [{"n": score["n"], "score": score[f"{kind}_score"]}]
                else:
                    curves[kind].append(
                        {"n": score["n"], "score": score[f"{kind}_score"]}
                    )

        return curves

    def get_relevances(self, individual_class=False):
        all_shaps = self._get_all_shaps()

        relevances = {}

        if individual_class:
            for elem in all_shaps:
                tmp_class = elem["class"]
                if tmp_class not in relevances:
                    relevances[tmp_class] = {}
                    for key in elem["values"]:
                        relevances[tmp_class][key] = 0
                for key, value in elem["values"].items():
                    relevances[tmp_class][key] = relevances[tmp_class][key] + abs(value)
        else:
            for elem in all_shaps:
                for key in elem["values"]:
                    if key not in relevances:
                        relevances[key] = 0
                for key, value in elem["values"].items():
                    relevances[key] = relevances[key] + abs(value)

        return relevances

    def get_statistics(
        self,
        kind: str = "train",
        class_list: List[str] = ["all"],
        feature_list: List[str] = ["all"],
    ):
        label_name = self.label_name
        datapoints = self._get_data(kind)
        if class_list == ["all"]:
            classes = list(map(lambda x: x["values"][label_name], datapoints))
            classes = np.unique(classes).astype(int).astype(str)
        else:
            classes = class_list

        if feature_list == ["all"]:
            feature_names = datapoints[0]["values"].keys()
        else:
            feature_names = feature_list

        # Step 1: Aggregate data based on index, excluding label_name
        aggregated_data = {}
        for dp in datapoints:
            for key, value in dp["values"].items():
                class_label = str(int(dp["values"][label_name]))
                if class_label not in aggregated_data:
                    aggregated_data[class_label] = {}
                if key != label_name:  # Dynamically exclude the label_name
                    if key in aggregated_data[class_label]:
                        aggregated_data[class_label][key].append(
                            value
                        )
                    else:
                        aggregated_data[class_label][key] = [value]

        # Step 2 & 3: Calculate statistics for each chosen prediction and feature
        statistics = defaultdict(dict)
        for label, features in aggregated_data.items():
            if str(label) not in classes:
                continue
            for feature, values in features.items():
                # print(feature)
                if str(feature) not in feature_names:
                    continue
                values_np = np.array(values)
                stats = {
                    "count": len(values),
                    "mean": np.mean(values_np),
                    "std": np.std(
                        values_np, ddof=1
                    ),  # Use ddof=1 for sample standard deviation
                    "median": np.median(values_np),
                    "min": np.min(values_np),
                    "max": np.max(values_np),
                }
                statistics[str(int(label))][feature] = stats

        return statistics

    def get_histogram(
        self,
        feature: str,
        class_list: List[str] = ["all"],
        kind: str = "train",
        bins: Union[int, str] = "auto",
    ) -> Dict:

        datapoints = self._get_data(kind)
        n_datapoints = len(datapoints)
        all_feature_values = [
            feature_value["values"][feature] for feature_value in datapoints
        ]

        feature_values = self._get_hist_feature_values(
            class_list, feature, datapoints, all_feature_values
        )

        if bins == "auto":
            # "The Square Root Choice" for bin_number
            n_bins = int(math.ceil(math.sqrt(n_datapoints)))
        else:
            n_bins = bins
        limits = {"min": min(all_feature_values), "max": max(all_feature_values)}
        rng = limits["max"] - limits["min"]
        bin_width = rng / n_bins

        print(class_list)

        if class_list == ["all"]:
            print(feature_values, limits, bin_width, n_bins)
            bins = self._get_hist_bins(feature_values, limits, bin_width, n_bins)
            hist = {
                "n_bins": n_bins,
                "start": limits["min"],
                "end": limits["max"],
                "bin_width": bin_width,
                "bins": bins,
            }
        else:
            hist = {}
            for key in feature_values:
                values = feature_values[key]
                print(values)
                bins = self._get_hist_bins(values, limits, bin_width, n_bins)
                hist[key] = {
                    "n_bins": n_bins,
                    "start": limits["min"],
                    "end": limits["max"],
                    "bin_width": bin_width,
                    "bins": bins,
                }

        return hist
    
    def get_scatterplot(self, feature1: str, feature2: str, kind: str = "train"):
        label_name = self.label_name
        data_id = self.data_collection.find_one({'type': kind})["_id"]
        datapoints = self.datapoint_collection.find({'data_id': data_id})
        pairs = [[[dp["values"][feature1], dp["values"][feature2]], str(int(dp["values"][label_name]))] for dp in datapoints]
        
        return list(zip(*pairs))

    def get_scatter_patterns(self, X, y, feature_list: list[str], complexity: int):

        decision_ruler = DecisionRuling()
        rules = decision_ruler.get_rules(X, y, feature_list, complexity)

        return rules

    def get_datapoint(self, index, kind="test", with_impact=True):
        datapoint = self._get_datapoint(index, kind=kind)
        prediction = self._get_prediction(index, kind=kind)
        if with_impact:
            shap_values = self._get_shap(index, prediction["prediction"])
        
        return {
            "datapoint": datapoint,
            "prediction": prediction.prediction,
            "shap_values": shap_values
        }
    
    def get_probabilities(self, index, kind="test"):
        prediction = self._get_prediction(index, kind=kind)
        
        return prediction["probs"]
    
    def get_trustscore(self, index):
        trustscore_meta = self.trustscore_meta_collection.find_one(
            {"model_id": self.model_id}
        )
        if not trustscore_meta:
            return None
        trustscore = self.trustscore_collection.find_one(
            {"trustscores_meta_id": trustscore_meta["_id"], "index": int(index)}
        )
        return {
            "extreme": trustscore["extreme"],
            "score": trustscore["score"],
            "percentile": trustscore["percentile"],
            "neighbour": trustscore["neighbour"],
        }

    def get_context(self, index, feature, classname="auto"):
        prediction = self._get_prediction(index)
        tmp_classname = classname if classname != "auto" else prediction.prediction
        datapoint = self._get_datapoint(index)
        feature_value = datapoint[feature]
        shap_values = self._get_shap(index, tmp_classname)
        shap_feature_value = shap_values[feature]
        anchor = self._get_anchor(index)
        distribution = self.get_histogram(feature, tmp_classname, kind="test", bins=20)
        overall_distribution = self.get_histogram(feature, "all", kind="test", bins=20)

        return {
            "feature": feature,
            "class": tmp_classname,
            "feature_value": feature_value,
            "anchor": anchor,
            "distribution": distribution,
            "overall_distribution": overall_distribution,
            "shap_value": shap_feature_value
        }

    def transform_metrics(self, metrics, kind: str):
        result = {}

        for key, value in metrics.items():
            if kind == "train" and "train" in key:
                new_key = key.replace("_train", "")
                result[new_key] = value
            elif kind == "test" and "train" not in key:
                result[key] = value

        return result

    def _get_datapoint(self, index, kind="test"):
        data_id = self.data_collection.find_one({'type': kind})["_id"]
        datapoint = self.datapoint_collection.find_one({'data_id': data_id, "index": int(index)})
        datapoint['values'].pop("quality", None)
        return datapoint['values']
    
    def _get_prediction(self, index, kind="test"):
        data_id = self.data_collection.find_one({'type': kind})["_id"]
        predictions_meta = self.predictions_meta_collection.find_one({'data_id': data_id})
        prediction = self.predictions_collection.find_one({"predictions_meta_id": predictions_meta["_id"], "index": int(index)})
        return {'prediction': prediction['prediction'], 'probs': prediction['probs']}
    
    def _get_shap(self, index, classname):
        shap_meta = self.shap_meta_collection.find_one({"model_id": self.model_id})
        if not shap_meta:
            return None
        shap = self.shap_collection.find_one(
            {
                "shap_meta_id": shap_meta["_id"],
                "index": int(index),
                "class": int(classname),
            }
        )
        return shap["values"]
    
    def _get_hist_feature_values(
        self, class_list, feature, datapoints, all_feature_values
    ):
        label_name = self.label_name
        if class_list != ["all"]:
            feature_values = defaultdict(dict)
            for elem in class_list:
                feature_values[elem] = [
                    feature_value["values"][feature]
                    for feature_value in datapoints
                    if str(int(feature_value["values"][label_name])) == elem
                ]
                feature_values[elem].sort()
        else:
            feature_values = all_feature_values
            feature_values.sort()

        return feature_values

    def _get_hist_bins(self, feature_values, limits, bin_width, n_bins):
        bins = []
        active_bin = 0
        for val in feature_values:
            while val >= limits["min"] + bin_width * (active_bin + 1) and val >= limits[
                "min"
            ] + bin_width * (active_bin + 2):
                active_bin += 1
                # make sure active_bin is never larger than n_bins - 1
                if active_bin >= n_bins:
                    active_bin = n_bins - 1
            while len(bins) < active_bin + 1:
                bins.append(0)
            bins[active_bin] = bins[active_bin] + 1

        return bins

    def _get_data(self, kind="test"):
        # TODO: Hack. Datapoints should point to full also.
        if kind == "full": kind = "train"
        data_id = self.data_collection.find_one({"type": kind})["_id"]
        datapoints = self.datapoint_collection.find({"data_id": data_id})
        return [
            {"index": datapoint["index"], "values": datapoint["values"]}
            for datapoint in datapoints
        ]

    
    def _get_anchor(self, index):
        anchor_meta = self.anchor_meta_collection.find_one({"model_id": self.model_id})
        if not anchor_meta:
            return None
        anchor = self.anchor_collection.find_one(
            {"anchor_meta_id": anchor_meta["_id"], "index": int(index)}
        )

        print([anchor_parser(elem) for elem in anchor["anchor"]])

        return {
            "anchor": anchor["anchor"],
            "precision": anchor["precision"],
            "coverage": anchor["coverage"],
        }

    def _get_all_shaps(self):
        shap_meta = self.shap_meta_collection.find_one({"model_id": self.model_id})
        if not shap_meta:
            return None
        shaps = self.shap_collection.find({"shap_meta_id": shap_meta["_id"]})
        return [
            {"class": shap["class"], "index": shap["index"], "values": shap["values"]}
            for shap in shaps
        ]

    def _get_labels(self, kind="test") -> list[str]:
        data_id = self.data_collection.find_one({"type": kind})["_id"]
        datapoints = self.datapoint_collection.find({"data_id": data_id})

        return [int(datapoint["values"][self.label_name]) for datapoint in datapoints]

    def _get_predictions(self, kind="test") -> list[str]:
        data_id = self.data_collection.find_one({"type": kind})["_id"]
        predictions_meta = self.predictions_meta_collection.find_one(
            {"data_id": data_id}
        )
        predictions = self.predictions_collection.find(
            {"predictions_meta_id": predictions_meta["_id"]}
        )

        return [int(prediction["prediction"]) for prediction in predictions]

    def _get_label_name(self):
        label = self.label_collection.find_one()
        label_name = label["name"]
        return label_name
