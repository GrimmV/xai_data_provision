from cors_handling import _corsify_actual_response, _build_cors_preflight_response
from flask import request, make_response
from config import base_url
from typing import Union

from DataHandler import DataHandler

allFeatures = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

default_error = lambda method: "Weird - don't know how to handle method {}".format(
    method
)


def endpoints(app, handler: DataHandler) -> None:
    @app.route("{}/model/scores".format(base_url), methods=["GET", "OPTIONS"])
    def get_model_scores():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            kind = request.args.get("kind", default="test", type=str)
            response = make_response(handler.get_model_scores(kind))
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/model/confusion".format(base_url), methods=["GET", "OPTIONS"])
    def get_confusion_matrix():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            kind = request.args.get("kind", default="test", type=str)
            response = make_response(handler.get_confusion_matrix(kind))
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/model/learning_curve".format(base_url), methods=["GET", "OPTIONS"])
    def get_learning_curve():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            kinds = request.args.getlist("kinds", type=str)
            # define default
            if not kinds:
                kinds = ["train", "test"]
            response = make_response(handler.get_learning_curves(kinds))
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    def is_it_true(value):
        return value.lower() == "true"

    @app.route(
        "{}/model/feature_relevance".format(base_url), methods=["GET", "OPTIONS"]
    )
    def get_feature_relevance():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            individual_class = request.args.get(
                "individual_class", default=False, type=is_it_true
            )
            response = make_response(handler.get_relevances(individual_class))
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/data/statistics".format(base_url), methods=["GET", "OPTIONS"])
    def get_statistics():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            kind = request.args.get("kind", default="train", type=str)
            class_list = request.args.getlist("class_list", type=str)
            # define default
            if not class_list:
                class_list = ["all"]
            feature_list = request.args.getlist("feature_list", type=str)
            # define default
            if not feature_list:
                feature_list = ["all"]
            response = make_response(
                handler.get_statistics(kind, class_list, feature_list)
            )
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/data/correlation".format(base_url), methods=["GET", "OPTIONS"])
    def get_correlation():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            response = make_response(handler.get_correlation())
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/data/distribution".format(base_url), methods=["GET", "OPTIONS"])
    def get_distribution():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            feature = request.args.get("feature", type=str)
            kind = request.args.get("kind", default="train", type=str)
            class_list = request.args.getlist("class_list", type=str)
            # define default
            if not class_list:
                class_list = ["all"]
            bins = request.args.get("bins", default="auto", type=Union[str, int])
            response = make_response(
                handler.get_histogram(feature, class_list, kind, bins)
            )
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/data/distribution/full".format(base_url), methods=["GET", "OPTIONS"])
    def get_all_distributions():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            kind = request.args.get("kind", default="train", type=str)
            bins = 50
            histograms = {}
            for feature in allFeatures:
                histograms[feature] = handler.get_histogram(
                    feature, ["all"], kind, bins
                )
            response = make_response(histograms)
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/data/datapoint".format(base_url), methods=["GET", "OPTIONS"])
    def get_datapoint():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            index = request.args.get("index", type=int)
            with_impact = request.args.get(
                "with_impact", default=False, type=is_it_true
            )
            response = make_response(
                handler.get_datapoint(index, with_impact=with_impact)
            )
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/xai/probabilities".format(base_url), methods=["GET", "OPTIONS"])
    def get_probabilities():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            index = request.args.get("index", type=int)
            response = make_response(handler.get_probabilities(index))
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/xai/trustscore".format(base_url), methods=["GET", "OPTIONS"])
    def get_trustscore():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            index = request.args.get("index", type=int)
            response = make_response(handler.get_trusscore(index))
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))

    @app.route("{}/xai/context".format(base_url), methods=["GET", "OPTIONS"])
    def get_context():
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        elif request.method == "GET":
            index = request.args.get("index", type=int)
            feature = request.args.get("feature", type=str)
            classname = request.args.get("classname", type=str)
            response = make_response(handler.get_context(index, feature, classname))
            return _corsify_actual_response(response)
        else:
            raise RuntimeError(default_error(request.method))
