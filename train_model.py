hfrom pathlib import Path
import cv2
import numpy as np
import joblib

from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_PATH = Path("dataset")
MODEL_PATH = Path("liver_model.pkl")

CATEGORIES = ["Normal", "Benign", "Malignant"]
IMAGE_SIZE = (128, 128)

EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


# ==========================================
# 2. EXTRACT FEATURES FROM AN IMAGE
# ==========================================

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, IMAGE_SIZE)

    gray = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    pixels = gray.astype(np.float64)

    features = [
        float(np.mean(pixels)),
        float(np.std(pixels)),
        float(np.var(pixels)),
        float(np.median(pixels)),
        float(np.min(pixels)),
        float(np.max(pixels)),
        float(np.percentile(pixels, 25)),
        float(np.percentile(pixels, 75))
    ]

    histogram = cv2.calcHist(
        [gray], [0], None, [256], [0, 256]
    ).flatten()

    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    features.append(float(entropy))

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True
    )

    for property_name in [
        "contrast",
        "homogeneity",
        "energy",
        "correlation",
        "dissimilarity"
    ]:
        values = graycoprops(glcm, property_name)
        features.append(float(np.mean(values)))

    return features


# ==========================================
# 3. LOAD DATASET
# ==========================================

X = []
y = []

print("\nLoading ultrasound images...")

if not DATASET_PATH.exists():
    raise SystemExit(
        "Dataset folder not found. Create the dataset folder "
        "beside train_model.py."
    )

for category in CATEGORIES:

    category_path = DATASET_PATH / category

    if not category_path.exists():
        print("Missing category folder:", category_path)
        continue

    image_paths = sorted(
        path for path in category_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
    )

    print(f"{category}: {len(image_paths)} images")

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            print("Skipping unreadable image:", image_path)
            continue

        try:
            feature_vector = extract_features(image)

            X.append(feature_vector)
            y.append(category)

        except Exception as error:
            print("Could not process:", image_path, error)


# ==========================================
# 4. CHECK DATA
# ==========================================

if len(X) == 0:
    raise SystemExit("No usable images were found.")

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y)

unique_classes, class_counts = np.unique(
    y, return_counts=True
)

print("\nUsable images:", len(X))
print("Features per image:", X.shape[1])

print("\nClass counts:")
for category, count in zip(unique_classes, class_counts):
    print(f"{category}: {count}")

if len(unique_classes) < 2:
    raise SystemExit(
        "At least two populated classes are required for training."
    )


# ==========================================
# 5. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

# Stratify only when each class has enough examples.
if len(X) >= 10 and np.min(class_counts) >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
else:
    raise SystemExit(
        "There are too few images in at least one class for a "
        "reliable stratified split. Add more images to each class."
    )


# ==========================================
# 6. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining the classifier...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=CATEGORIES,
        zero_division=0
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=CATEGORIES
    )
)


# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH.resolve())
print("Training completed!")