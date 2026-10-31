
from pathlib import Path
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog
from skimage.feature import graycomatrix, graycoprops


# ==========================================
# 1. SETTINGS
# ==========================================

MODEL_PATH = Path("liver_model.pkl")
RESULTS_PATH = Path("results")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)


# ==========================================
# 2. FEATURE EXTRACTION
# Must match train_model.py
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

    return features, gray


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():
    print("Trained model not found!")
    print("Run this first: python train_model.py")
    raise SystemExit

model = joblib.load(MODEL_PATH)


# ==========================================
# 4. CHOOSE IMAGE
# ==========================================

root = Tk()
root.withdraw()

image_file = filedialog.askopenfilename(
    title="Select a Liver Ultrasound Image",
    filetypes=[
        ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")
    ]
)

root.destroy()

if not image_file:
    print("No image selected.")
    raise SystemExit

original = cv2.imread(image_file)

if original is None:
    print("Unable to read the selected image.")
    raise SystemExit

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)


# ==========================================
# 5. PREPROCESS IMAGE
# ==========================================

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

denoised = cv2.medianBlur(gray, 5)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(denoised)


# ==========================================
# 6. BASIC THRESHOLD SEGMENTATION
# ==========================================

_, segmented = cv2.threshold(
    enhanced,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ==========================================
# 7. CLASSIFY IMAGE
# ==========================================

feature_vector, model_input_gray = extract_features(original)

feature_array = np.asarray(
    feature_vector, dtype=np.float64
).reshape(1, -1)

predicted_category = model.predict(feature_array)[0]

probabilities = model.predict_proba(feature_array)[0]

class_probabilities = dict(
    zip(model.classes_, probabilities)
)

print("\n====================================")
print("LIVER ULTRASOUND CLASSIFICATION")
print("====================================")

print("Image:", Path(image_file).name)
print("Predicted category:", predicted_category)

print("\nModel scores:")
for category, score in class_probabilities.items():
    print(f"{category}: {score * 100:.2f}%")


# ==========================================
# 8. DISPLAY OUTPUT IMAGES
# ==========================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(denoised, cmap="gray")
plt.title("Noise Reduced")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(enhanced, cmap="gray")
plt.title("Contrast Enhanced")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(segmented, cmap="gray")
plt.title("Basic Threshold Segmentation")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.axis("off")

score_text = "\n".join(
    f"{category}: {score * 100:.1f}%"
    for category, score in class_probabilities.items()
)

plt.text(
    0.05,
    0.65,
    f"Predicted category:\n{predicted_category}\n\n"
    f"Model scores:\n{score_text}",
    fontsize=13,
    verticalalignment="center"
)

plt.suptitle(
    "Liver Ultrasound Image Analysis",
    fontsize=16
)

plt.tight_layout()

comparison_path = RESULTS_PATH / "classification_comparison.png"

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 9. SAVE PROCESSED IMAGES
# ==========================================

cv2.imwrite(
    str(RESULTS_PATH / "grayscale.png"),
    gray
)

cv2.imwrite(
    str(RESULTS_PATH / "denoised.png"),
    denoised
)

cv2.imwrite(
    str(RESULTS_PATH / "enhanced.png"),
    enhanced
)

cv2.imwrite(
    str(RESULTS_PATH / "segmented.png"),
    segmented
)


# ==========================================
# 10. SAVE CLASSIFICATION RESULT
# ==========================================

result_file = RESULTS_PATH / "classification_result.txt"

with open(result_file, "w", encoding="utf-8") as file:
    file.write(f"Image: {Path(image_file).name}\n")
    file.write(f"Predicted category: {predicted_category}\n\n")
    file.write("Model scores:\n")

    for category, score in class_probabilities.items():
        file.write(f"{category}: {score * 100:.2f}%\n")

print("\nComparison image saved to:", comparison_path.resolve())
print("Classification result saved to:", result_file.resolve())

print("\nAnalysis completed!")
print("These predictions are experimental, not a medical diagnosis.")