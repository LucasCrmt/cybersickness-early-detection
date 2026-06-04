from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Conv1D, Dense, Dropout, LSTM, Bidirectional, Input, Flatten,
        concatenate, Activation, BatchNormalization, GlobalAveragePooling1D,
        TimeDistributed, MaxPooling1D, Reshape, Lambda,
    )
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

_VALID_MODEL_TYPES = {
    "random_forest", "xgboost",
    "cnn_1d", "inception_time", "bilstm", "cnn_lstm", "td_cnn_lstm",
    "multistream",   # fusion end-to-end deux branches (Islam et al. ISMAR 2021)
}


class _XGBClassifierWrapper:
    """XGBClassifier avec encodage automatique des labels string → int.

    XGBoost n'accepte que des labels numériques en classification.
    Ce wrapper encode y à l'entraînement et décode les prédictions,
    rendant le modèle transparent pour le reste de la pipeline.
    """

    def __init__(self, **kwargs):
        self._model = XGBClassifier(**kwargs)
        self._le = LabelEncoder()

    def fit(self, X, y):
        self._model.fit(X, self._le.fit_transform(y))
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._model.predict(X))

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    @property
    def classes_(self):
        return self._le.classes_

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


def _get_model_type(model_profile):
    mt = model_profile.get("model_type", "random_forest").lower()
    if mt not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type invalide: '{mt}'. Valeurs acceptées: {_VALID_MODEL_TYPES}")
    return mt


def get_search_space(task_type, model_profile=None):
    mt = _get_model_type(model_profile or {})

    if mt == "xgboost":
        return {
            "n_estimators": [100, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

    if mt == "random_forest":
        common = {
            "n_estimators": [200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }
        if task_type == "classification":
            class_weight = "balanced" if model_profile is None else model_profile.get("class_weight", "balanced")
            common["class_weight"] = [class_weight]
        return common

    # Modèles approche B - séries temporelles
    if mt == "cnn_1d":
        return {
            "filters": [2, 3],
            "kernel_size": [3, 5],
            "dropout_rate": [0.2, 0.4],
            "learning_rate": [0.001, 0.01],
            "batch_size": [16, 32],
        }
    
    if mt == "inception_time":
        return {
            "filters": [2, 3],
            "depth": [2, 3],
            "dropout_rate": [0.2, 0.3],
            "learning_rate": [0.001, 0.005],
            "batch_size": [16, 32],
        }
    
    if mt == "bilstm":
        return {
            "units": [4, 8],
            "dropout_rate": [0.2, 0.3],
            "learning_rate": [0.001, 0.01],
            "batch_size": [16, 32],
        }
    
    if mt == "cnn_lstm":
        return {
            "cnn_filters": [2, 3],
            "cnn_kernel": [3, 5],
            "lstm_units": [32, 64],
            "dropout_rate": [0.2, 0.3],
            "learning_rate": [0.001, 0.01],
            "batch_size": [16, 32],
        }

    if mt == "td_cnn_lstm":
        return {
            "filters":      [32, 64],
            "kernel_size":  [3, 5],
            "lstm_units":   [64, 128],
            "dropout_rate": [0.2, 0.5],
            "learning_rate": [0.001, 0.01],
            "batch_size":   [32, 64],
            "n_subseq":     [4],
        }

    if mt == "multistream":
        return {
            "cnn_filters":    [32, 64],
            "lstm_units":     [64, 128],
            "dropout_rate":   [0.2, 0.3],
            "learning_rate":  [0.001, 0.01],
            "batch_size":     [16, 32],
            # n_eye_features est fixe par les donnees, pas recherche ici
        }

    if mt == "svm":
        return {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }

    # par défaut, random_forest
    common = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }


def _build_cnn_1d(input_shape, output_shape, is_classif, params):
    """Construit un modèle CNN 1D pour les séries temporelles."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras n'est pas installé. Installez tensorflow pour utiliser CNN 1D.")
    
    filters = params.get("filters", 3)
    kernel_size = params.get("kernel_size", 3)
    dropout_rate = params.get("dropout_rate", 0.2)
    
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        Conv1D(filters * 2, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(dropout_rate),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(output_shape, activation='softmax' if is_classif else 'linear')
    ])
    return model


def _build_inception_time(input_shape, output_shape, is_classif, params):
    """Construit un modèle InceptionTime pour les séries temporelles."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras n'est pas installé. Installez tensorflow pour utiliser InceptionTime.")
    
    filters = params.get("filters", 3)
    depth = params.get("depth", 2)
    dropout_rate = params.get("dropout_rate", 0.2)
    
    input_tensor = Input(shape=input_shape)
    x = input_tensor
    
    # Empile plusieurs blocs InceptionTime
    for _ in range(depth):
        # Branche 1 : Conv 1x1
        b1 = Conv1D(filters, 1, padding='same', activation='relu')(x)
        
        # Branche 2 : Conv 3x3
        b2 = Conv1D(filters, 3, padding='same', activation='relu')(x)
        
        # Branche 3 : Conv 5x5
        b3 = Conv1D(filters, 5, padding='same', activation='relu')(x)
        
        # Concaténation et normalisation
        x = concatenate([b1, b2, b3])
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(output_shape, activation='softmax' if is_classif else 'linear')(x)
    
    model = Model(inputs=input_tensor, outputs=output)
    return model


def _build_bilstm(input_shape, output_shape, is_classif, params):
    """Construit un modèle BiLSTM pour les séries temporelles."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras n'est pas installé. Installez tensorflow pour utiliser BiLSTM.")
    
    units = params.get("units", 32)
    dropout_rate = params.get("dropout_rate", 0.2)
    # recurrent_dropout > 0 force une implementation LSTM non-CuDNN,
    # necessaire pour eviter l'op CudnnRNN indisponible sous DirectML.
    recurrent_dropout_rate = params.get("recurrent_dropout", 0.1)
    
    model = Sequential([
        Bidirectional(
            LSTM(
                units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
            ),
            input_shape=input_shape,
        ),
        Bidirectional(
            LSTM(
                units,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
            )
        ),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(output_shape, activation='softmax' if is_classif else 'linear')
    ])
    return model


def _build_cnn_lstm(input_shape, output_shape, is_classif, params):
    """Construit un modèle CNN-LSTM hybride pour les séries temporelles."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras n'est pas installé. Installez tensorflow pour utiliser CNN-LSTM.")
    
    cnn_filters = params.get("cnn_filters", 3)
    cnn_kernel = params.get("cnn_kernel", 3)
    lstm_units = params.get("lstm_units", 32)
    dropout_rate = params.get("dropout_rate", 0.2)
    # recurrent_dropout > 0 force une implementation LSTM non-CuDNN,
    # necessaire pour eviter l'op CudnnRNN indisponible sous DirectML.
    recurrent_dropout_rate = params.get("recurrent_dropout", 0.1)
    
    model = Sequential([
        Conv1D(cnn_filters, cnn_kernel, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        Conv1D(cnn_filters * 2, cnn_kernel, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(
            lstm_units,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
        ),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(output_shape, activation='softmax' if is_classif else 'linear')
    ])
    return model


def _build_td_cnn_lstm(input_shape, output_shape, is_classif, params):
    """Time-Distributed CNN-LSTM inspiré de Islam et al. (ISMAR 2021).

    La séquence (T, n_features) est découpée en n_subseq sous-séquences.
    Chaque sous-séquence est traitée indépendamment par un CNN (TimeDistributed),
    puis l'LSTM encode les dépendances temporelles entre les blocs.

    Adapté de l'architecture du papier :
    - Input (T, n_features) → Reshape (n_subseq, T//n_subseq, n_features)
    - TimeDistributed(Conv1D) → TimeDistributed(MaxPool) → TimeDistributed(Flatten)
    - LSTM → Dense(256) → Output
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras requis pour td_cnn_lstm.")

    T, n_features = input_shape
    n_subseq     = params.get("n_subseq", 4)
    filters      = params.get("filters", 32)
    kernel_size  = params.get("kernel_size", 3)
    lstm_units   = params.get("lstm_units", 64)
    dropout_rate = params.get("dropout_rate", 0.2)

    # T doit être divisible par n_subseq
    subseq_len = T // n_subseq

    inp = Input(shape=(T, n_features))
    x = Reshape((n_subseq, subseq_len, n_features))(inp)

    x = TimeDistributed(Conv1D(filters, kernel_size, activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(Flatten())(x)

    x = LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(output_shape, activation="softmax" if is_classif else "linear")(x)

    return Model(inputs=inp, outputs=out)


def _build_multistream_fusion(input_shape, output_shape, is_classif, params):
    """Fusion end-to-end deux branches (eye et head) inspirée de Islam et al. (ISMAR 2021).

    Paramètres architecturaux (fixes par hypothèse, injectés depuis model_profile) :
      branch_type      (str)  : "cnn_lstm" (défaut) ou "inception".
      n_eye_features   (int)  : colonnes oculaires en tête du tenseur (défaut 3).
      eye_cnn_filters  (int)  : filtres Conv1D branche eye  (défaut = cnn_filters).
      head_cnn_filters (int)  : filtres Conv1D branche head (défaut = cnn_filters).
      eye_lstm_units   (int)  : unités LSTM branche eye     (défaut = lstm_units).
      head_lstm_units  (int)  : unités LSTM branche head    (défaut = lstm_units).

    Hyperparamètres searchés :
      cnn_filters   (int)  : filtres Conv1D partagés si pas de surcharge (défaut 32).
      lstm_units    (int)  : unités LSTM partagées si pas de surcharge (défaut 64).
      dropout_rate  (float): dropout dans les branches et la fusion (défaut 0.2).
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow requis pour multistream.")

    T, n_features = input_shape
    branch_type   = str(params.get("branch_type", "cnn_lstm")).lower()
    n_eye         = int(params.get("n_eye_features", 3))
    n_head        = n_features - n_eye
    cnn_filters   = int(params.get("cnn_filters",   32))
    lstm_units    = int(params.get("lstm_units",    64))
    dropout_rate  = float(params.get("dropout_rate", 0.2))
    eye_cnn_f     = int(params.get("eye_cnn_filters",  cnn_filters))
    head_cnn_f    = int(params.get("head_cnn_filters", cnn_filters))
    eye_lstm_u    = int(params.get("eye_lstm_units",   lstm_units))
    head_lstm_u   = int(params.get("head_lstm_units",  lstm_units))

    if n_eye <= 0 or n_head <= 0:
        raise ValueError(
            f"n_eye_features={n_eye} invalide pour n_features={n_features}. "
            "Verifier que les features eye sont en premier dans include_features."
        )

    def _cnn_lstm_branch(x_in, name, n_filt, lstm_u):
        """Branche CNN-LSTM (architecture du papier Islam et al.)."""
        x = Conv1D(n_filt, 3, activation="relu", padding="same",     name=f"conv1_{name}")(x_in)
        x = BatchNormalization(name=f"bn1_{name}")(x)
        x = Dropout(dropout_rate, name=f"drop1_{name}")(x)
        x = Conv1D(n_filt * 2, 3, activation="relu", padding="same", name=f"conv2_{name}")(x)
        x = BatchNormalization(name=f"bn2_{name}")(x)
        x = Dropout(dropout_rate, name=f"drop2_{name}")(x)
        x = LSTM(lstm_u, dropout=dropout_rate, recurrent_dropout=0.2, name=f"lstm_{name}")(x)
        x = Dense(256, activation="relu", name=f"dense_{name}")(x)
        x = BatchNormalization(name=f"bn3_{name}")(x)
        x = Dropout(0.5, name=f"drop3_{name}")(x)
        return x

    def _inception_branch(x_in, name, n_filt):
        """Branche InceptionTime: 2 blocs de convolutions parallèles k=1/3/5 + GAP + Dense."""
        x = x_in
        for d in range(2):
            c1 = Conv1D(n_filt, 1, activation="relu", padding="same", name=f"inc_k1_{name}_{d}")(x)
            c3 = Conv1D(n_filt, 3, activation="relu", padding="same", name=f"inc_k3_{name}_{d}")(x)
            c5 = Conv1D(n_filt, 5, activation="relu", padding="same", name=f"inc_k5_{name}_{d}")(x)
            x  = concatenate([c1, c3, c5], name=f"inc_cat_{name}_{d}")
            x  = BatchNormalization(name=f"inc_bn_{name}_{d}")(x)
            x  = Dropout(dropout_rate, name=f"inc_drop_{name}_{d}")(x)
        x = GlobalAveragePooling1D(name=f"inc_gap_{name}")(x)
        x = Dense(256, activation="relu", name=f"inc_dense_{name}")(x)
        x = BatchNormalization(name=f"inc_bnd_{name}")(x)
        x = Dropout(0.5, name=f"inc_dropd_{name}")(x)
        return x

    inp    = Input(shape=(T, n_features), name="input_all")
    x_eye  = Lambda(lambda z: z[:, :, :n_eye],  name="slice_eye")(inp)
    x_head = Lambda(lambda z: z[:, :, n_eye:],  name="slice_head")(inp)

    if branch_type == "inception":
        feat_eye  = _inception_branch(x_eye,  "eye",  eye_cnn_f)
        feat_head = _inception_branch(x_head, "head", head_cnn_f)
    else:  # "cnn_lstm" (défaut)
        feat_eye  = _cnn_lstm_branch(x_eye,  "eye",  eye_cnn_f,  eye_lstm_u)
        feat_head = _cnn_lstm_branch(x_head, "head", head_cnn_f, head_lstm_u)

    fused = concatenate([feat_eye, feat_head], name="fusion")
    fused = Dense(256, activation="relu", name="fusion_dense")(fused)
    fused = BatchNormalization(name="fusion_bn")(fused)
    fused = Dropout(dropout_rate, name="fusion_drop")(fused)
    out   = Dense(output_shape,
                  activation="softmax" if is_classif else "linear",
                  name="output")(fused)

    return Model(inputs=inp, outputs=out, name="multistream_fusion")


class KerasSklearnWrapper:
    """Wrapper pour rendre les modèles Keras compatibles avec l'interface sklearn."""
    
    def __init__(self, model_builder, input_shape, output_shape, is_classif, params, epochs=50, verbose=0):
        self.model_builder = model_builder
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.is_classif = is_classif
        self.params = params
        self.epochs = epochs
        self.verbose = verbose
        self.model = None
        self.learning_rate = params.get("learning_rate", 0.001)
        self.batch_size = params.get("batch_size", 32)
        self.classes_ = None
        self.class_to_index_ = None
    
    def fit(self, X, y):
        """Entraîne le modèle avec poids de classe équilibrés."""
        from tensorflow.keras.optimizers import Adam
        from sklearn.utils.class_weight import compute_class_weight

        y_train = y
        class_weight_dict = None
        if self.is_classif:
            # Keras sparse_categorical_crossentropy attend des labels entiers.
            y_series = pd.Series(y)
            self.classes_ = list(pd.unique(y_series.dropna()))
            self.class_to_index_ = {label: i for i, label in enumerate(self.classes_)}
            y_train = y_series.map(self.class_to_index_).to_numpy(dtype=np.int32)

            # Poids de classe balancés pour gérer le déséquilibre (comme class_weight="balanced" sklearn)
            cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            class_weight_dict = {int(i): float(w) for i, w in enumerate(cw)}

        self.model = self.model_builder(self.input_shape, self.output_shape, self.is_classif, self.params)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy' if self.is_classif else 'mse',
            metrics=['accuracy' if self.is_classif else 'mse']
        )
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
        ]
        self.model.fit(
            X, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=class_weight_dict,
            validation_split=0.1,
            callbacks=callbacks,
        )
        return self
    
    def predict(self, X):
        """Prédit sur les données."""
        predictions = self.model.predict(X, verbose=0)
        if self.is_classif:
            pred_idx = np.argmax(predictions, axis=1)
            if self.classes_ is not None:
                return np.asarray([self.classes_[int(i)] for i in pred_idx], dtype=object)
            return pred_idx
        return predictions.flatten()

    def predict_proba(self, X):
        """Retourne les probabilités softmax par classe."""
        return self.model.predict(X, verbose=0)


def build_model(params, model_profile):
    mt = _get_model_type(model_profile)
    is_classif = model_profile["task_type"] == "classification"

    if mt == "xgboost":
        if is_classif:
            return _XGBClassifierWrapper(random_state=model_profile["random_state"], n_jobs=-1, eval_metric="logloss", **params)
        return XGBRegressor(random_state=model_profile["random_state"], n_jobs=-1, **params)
    
    if mt == "random_forest":
        if is_classif:
            return RandomForestClassifier(random_state=model_profile["random_state"], n_jobs=-1, **params)
        return RandomForestRegressor(random_state=model_profile["random_state"], n_jobs=-1, **params)
    
    # Modèles d'approche B - séries temporelles
    # Note: input_shape et output_shape seront définis à l'appel selon les données
    if mt == "cnn_1d":
        return KerasSklearnWrapper(
            _build_cnn_1d,
            input_shape=(params.get("sequence_length", 100), params.get("n_features", 1)),
            output_shape=model_profile.get("n_classes", 2) if is_classif else 1,
            is_classif=is_classif,
            params=params
        )
    
    if mt == "inception_time":
        return KerasSklearnWrapper(
            _build_inception_time,
            input_shape=(params.get("sequence_length", 100), params.get("n_features", 1)),
            output_shape=model_profile.get("n_classes", 2) if is_classif else 1,
            is_classif=is_classif,
            params=params
        )
    
    if mt == "bilstm":
        return KerasSklearnWrapper(
            _build_bilstm,
            input_shape=(params.get("sequence_length", 100), params.get("n_features", 1)),
            output_shape=model_profile.get("n_classes", 2) if is_classif else 1,
            is_classif=is_classif,
            params=params
        )
    
    if mt == "cnn_lstm":
        return KerasSklearnWrapper(
            _build_cnn_lstm,
            input_shape=(params.get("sequence_length", 100), params.get("n_features", 1)),
            output_shape=model_profile.get("n_classes", 2) if is_classif else 1,
            is_classif=is_classif,
            params=params
        )

    if mt == "td_cnn_lstm":
        T = params.get("sequence_length", 60)
        n_subseq = params.get("n_subseq", 4)
        T = (T // n_subseq) * n_subseq
        return KerasSklearnWrapper(
            _build_td_cnn_lstm,
            input_shape=(T, params.get("n_features", 1)),
            output_shape=model_profile.get("n_classes", 2) if is_classif else 1,
            is_classif=is_classif,
            params={**params, "sequence_length": T},
        )

    if mt == "multistream":
        T  = params.get("sequence_length", 300)
        nf = params.get("n_features", 12)
        _ARCH_KEYS = (
            "branch_type", "n_eye_features",
            "eye_cnn_filters", "head_cnn_filters",
            "eye_lstm_units",  "head_lstm_units",
            # Hyperparamètres fixes (utilisés quand use_hyperparam_search=False)
            "cnn_filters", "lstm_units", "dropout_rate", "learning_rate", "batch_size",
        )
        injected = {k: model_profile[k] for k in _ARCH_KEYS
                    if k in model_profile and k not in params}
        merged_params = {**injected, **params}
        return KerasSklearnWrapper(
            _build_multistream_fusion,
            input_shape=(T, nf),
            output_shape=model_profile.get("n_classes", 2) if is_classif else 1,
            is_classif=is_classif,
            params=merged_params,
        )

    if mt == "svm":
        class_weight = model_profile.get("class_weight", None) if is_classif else None
        if is_classif:
            return SVC(class_weight=class_weight, **params)
        return SVR(**params)

    raise ValueError(f"Type de modèle non reconnu: {mt}")



def run_hyperparam_search(X_train_imp, y_train, X_val_imp, y_val, model_profile):
    search_space = get_search_space(model_profile["task_type"], model_profile=model_profile)

    results = []
    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(search_space):
        model = build_model(params, model_profile)
        model.fit(X_train_imp, y_train)
        pred_val = model.predict(X_val_imp)

        row = deepcopy(params)
        if model_profile["task_type"] == "classification":
            row["val_accuracy"] = accuracy_score(y_val, pred_val)
            row["val_f1_weighted"] = f1_score(y_val, pred_val, average="weighted", zero_division=0)
            score = row["val_f1_weighted"]
        else:
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            row["val_rmse"] = rmse
            row["val_r2"] = r2_score(y_val, pred_val)
            score = -rmse

        results.append(row)
        if score > best_score:
            best_score = score
            best_params = deepcopy(params)

    results_df = pd.DataFrame(results)
    sort_col = "val_f1_weighted" if model_profile["task_type"] == "classification" else "val_rmse"
    ascending = model_profile["task_type"] != "classification"
    results_df = results_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
    return best_params, results_df


def fit_final_model(X_train_imp, y_train, X_val_imp, y_val, best_params, model_profile):
    X_train_val = np.vstack([X_train_imp, X_val_imp])
    y_train_val = np.concatenate([y_train, y_val])

    final_model = build_model(best_params, model_profile)
    final_model.fit(X_train_val, y_train_val)
    return final_model
