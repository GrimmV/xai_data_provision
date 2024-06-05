
from sklearn.tree import DecisionTreeClassifier, export_text
import json
import numpy as np
from sklearn.tree import _tree

class DecisionRuling:

    def get_rules(self, X, y, feature_names: list[str] = ['Feature 1', 'Feature 2'], complexity: int = 5):

        clf = DecisionTreeClassifier(max_depth=complexity, random_state=42)
        clf.fit(X, y)
        
        paths = self._get_rule_paths(clf, feature_names)
        rules_metrics = self._get_coverage_and_precision(clf, X, y, paths)

        # Convert to JSON
        rules_json = json.dumps(rules_metrics, indent=4)
        print(rules_json)


    def _get_rule_paths(self, tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        
        paths = []
        
        def recurse(node, path, rules):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = round(tree_.threshold[node], 4)
                path_left = path.copy()
                path_right = path.copy()
                path_left.append(f"{name} <= {threshold}")
                path_right.append(f"{name} > {threshold}")
                recurse(tree_.children_left[node], path_left, rules)
                recurse(tree_.children_right[node], path_right, rules)
            else:
                rule = " and ".join(path)
                paths.append((rule, node))
        
        recurse(0, [], paths)
        
        return paths

    def _get_coverage_and_precision(self, tree, X, y, paths):
        tree_ = tree.tree_
        results = []
        
        for rule, node in paths:
            node_samples = tree_.n_node_samples[node]
            node_value = tree_.value[node]
            node_class = np.argmax(node_value)
            precision = node_value[0, node_class] / node_samples
            coverage = node_samples / len(y)
            if (precision > 0.9):
                results.append({
                    "rule": rule,
                    "coverage": coverage,
                    "precision": precision,
                    "class": int(node_class)
                })
            
        return results