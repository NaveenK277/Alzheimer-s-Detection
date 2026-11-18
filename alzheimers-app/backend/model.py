import os
import cv2
import numpy as np
import joblib

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, InceptionV3, MobileNetV2
from tensorflow.keras.applications.densenet    import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble          import RandomForestClassifier, VotingClassifier
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.model_selection   import train_test_split, StratifiedKFold
from sklearn.metrics           import accuracy_score, f1_score

SEED             = 42
IMG_SIZE         = 224
BATCH_SIZE       = 16
EXPECTED_CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

MODEL_DIR       = os.path.join(os.path.dirname(__file__), 'saved_model')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.joblib')
SCALER_PATH     = os.path.join(MODEL_DIR, 'scaler.joblib')
SELECTOR_PATH   = os.path.join(MODEL_DIR, 'selector.joblib')

np.random.seed(SEED)
tf.random.set_seed(SEED)

_densenet   = None
_inception  = None
_mobilenet  = None
_scaler     = None
_selector   = None
_classifier = None

def _build_extractors():
    global _densenet, _inception, _mobilenet
    if _densenet is not None:
        return
    print("Loading CNN extractors...")
    _densenet  = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE,IMG_SIZE,3))
    _inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE,IMG_SIZE,3))
    _mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE,IMG_SIZE,3))
    print("CNN extractors ready.")

def load_images_from_dataset(data_dir, categories=EXPECTED_CLASSES, img_size=IMG_SIZE, max_per_class=None):
    X, y, paths = [], [], []
    for idx, cat in enumerate(categories):
        folder = os.path.join(data_dir, cat)
        if not os.path.isdir(folder):
            print(f"Folder not found: {folder} (skipped)")
            continue
        files = sorted(os.listdir(folder))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            fp = os.path.join(folder, f)
            try:
                img = cv2.imread(fp)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(idx)
                paths.append(fp)
            except Exception:
                continue
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32), paths

def _extract_batch(model, preprocess_fn, X_arr, batch_size=BATCH_SIZE):
    out = []
    for i in range(0, X_arr.shape[0], batch_size):
        batch = X_arr[i:i+batch_size].astype('float32')
        batch = preprocess_fn(batch)
        feats = model.predict(batch, verbose=0)
        out.append(feats)
    return np.vstack(out)

def _fuse_features(X_arr):
    _build_extractors()
    fd = _extract_batch(_densenet,  densenet_preprocess,  X_arr)
    fi = _extract_batch(_inception, inception_preprocess, X_arr)
    fm = _extract_batch(_mobilenet, mobilenet_preprocess, X_arr)
    return np.concatenate([fd, fi, fm], axis=1)

def train_and_save(data_dir: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    _build_extractors()

    print("\n📂 Loading dataset...")
    X, y, _ = load_images_from_dataset(data_dir)
    print(f"   {X.shape[0]} images loaded.")

    print("\n🔬 Extracting features (this will take a while)...")
    X_fused = _fuse_features(X)
    print(f"   Fused shape: {X_fused.shape}")

    selector = VarianceThreshold(threshold=0.01)
    X_sel    = selector.fit_transform(X_fused)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    print(f"   Processed shape: {X_scaled.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=SEED)

    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1, class_weight='balanced'),
        'SVM':          SVC(kernel='rbf', probability=True, random_state=SEED, class_weight='balanced'),
        'KNN':          KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }

    print("\n📊 Cross-validation...")
    skf        = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_summary = {}
    for name, clf in classifiers.items():
        accs, f1s = [], []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[val_idx])
            accs.append(accuracy_score(y_train[val_idx], preds))
            f1s.append(f1_score(y_train[val_idx], preds, average='macro'))
        cv_summary[name] = {'cv_acc_mean': np.mean(accs), 'cv_f1_mean': np.mean(f1s)}
        print(f"   {name} CV Acc: {np.mean(accs):.4f}  F1: {np.mean(f1s):.4f}")

    print("\n🏋️  Final training...")
    trained, final_results = {}, {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        final_results[name] = {
            'test_acc':      accuracy_score(y_test, preds),
            'test_f1_macro': f1_score(y_test, preds, average='macro')
        }
        trained[name] = clf
        print(f"   {name} Test Acc: {final_results[name]['test_acc']:.4f}")

    top2   = sorted(cv_summary, key=lambda x: cv_summary[x]['cv_acc_mean'], reverse=True)[:2]
    estims = [(n.lower(), trained[n]) for n in top2]
    ens    = VotingClassifier(estimators=estims, voting='hard')
    ens.fit(X_train, y_train)
    y_ens  = ens.predict(X_test)
    final_results['Ensemble'] = {
        'test_acc':      accuracy_score(y_test, y_ens),
        'test_f1_macro': f1_score(y_test, y_ens, average='macro')
    }
    trained['Ensemble'] = ens
    print(f"   Ensemble Test Acc: {final_results['Ensemble']['test_acc']:.4f}")

    best_name  = max(final_results, key=lambda x: final_results[x]['test_acc'])
    best_model = trained[best_name]
    print(f"\n🏆 Best: {best_name} (Acc: {final_results[best_name]['test_acc']:.4f})")

    joblib.dump(best_model, CLASSIFIER_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    joblib.dump(selector,   SELECTOR_PATH)
    print(f"💾 Saved to {MODEL_DIR}")

    return best_name, final_results[best_name]

def _load_artifacts():
    global _classifier, _scaler, _selector
    if _classifier is not None:
        return
    missing = [p for p in [CLASSIFIER_PATH, SCALER_PATH, SELECTOR_PATH] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "Model not trained yet. Run: python3 train.py"
        )
    print("Loading saved model artifacts...")
    _classifier = joblib.load(CLASSIFIER_PATH)
    _scaler     = joblib.load(SCALER_PATH)
    _selector   = joblib.load(SELECTOR_PATH)
    print("Artifacts loaded.")

def predict_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    _build_extractors()
    _load_artifacts()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img     = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_arr = np.expand_dims(img, axis=0).astype('float32')

    fd    = _densenet.predict(densenet_preprocess(img_arr.copy()),   verbose=0)
    fi    = _inception.predict(inception_preprocess(img_arr.copy()), verbose=0)
    fm    = _mobilenet.predict(mobilenet_preprocess(img_arr.copy()), verbose=0)
    fused = np.concatenate([fd, fi, fm], axis=1)

    fused_sel    = _selector.transform(fused)
    fused_scaled = _scaler.transform(fused_sel)

    pred_idx  = _classifier.predict(fused_scaled)[0]
    return EXPECTED_CLASSES[pred_idx]
