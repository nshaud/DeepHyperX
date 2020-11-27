import sklearn

from models import save_model

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [
    {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [3], "gamma": [1e-1, 1e-2, 1e-3]},
]

SKLEARN_MODELS = ["SVM", "SGD", "nearest"]


def fit_sklearn_model(
    model, X_train, y_train, exp_name="", class_balancing=False, n_jobs=0
):
    class_weight = "balanced" if class_balancing else None
    if model not in SKLEARN_MODELS:
        raise NotImplementedError(
            "sklearn-based model {} is not supported.".format(model)
        )
    if model == "SVM_grid":
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        # TODO: use a better hyperparam search
        clf = sklearn.model_selection.GridSearchCV(
            sklearn.svm.SVC(class_weight=class_weight),
            SVM_GRID_PARAMS,
            verbose=5,
            n_jobs=n_jobs,
        )
    elif model == "SVM":
        # TODO: hyperparameter search, at least for C
        clf = sklearn.svm.SVC(class_weight=class_weight)
    elif model == "SGD":
        clf = sklearn.linear_model.SGDClassifier(
            class_weight=class_weight,
            learning_rate="optimal",
            tol=1e-3,#why?
            average=10,#why?
            n_jobs=n_jobs,
        )
    elif model == "nearest":
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train) #why?
        clf = sklearn.model_selection.GridSearchCV(
            sklearn.neighbors.KNeighborsClassifier(weights="distance"),
            {"n_neighbors": [1, 3, 5, 10, 20]},
            verbose=5,
            n_jobs=n_jobs,
        )
        # TODO: implement SAM classifier
    clf.fit(X_train, y_train)
    save_model(clf, model, exp_name)
    return clf


def infer_from_sklearn_model(clf, hsi_image):
    n_bands = hsi_image.shape[-1]
    prediction = clf.predict(hsi_image.reshape(-1, n_bands))
    prediction = prediction.reshape(hsi_image.shape[:2])
    return prediction
