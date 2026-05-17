from PIL import Image
import pytesseract
import re
from recommendations import get_recommendations
from predict import predict_diabetes


# Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load image
image = Image.open("images/ocr test 4.png.jpeg")

# OCR text extraction
text = pytesseract.image_to_string(image)

print("OCR TEXT:")
print(text)

# Extract glucose
glucose_match = re.search(
    r'Glucose[: ]+(\d+)',
    text,
    re.IGNORECASE
)

# Extract BMI
bmi_match = re.search(
    r'BMI[: ]+([\d.]+)',
    text,
    re.IGNORECASE
)

# Extract age
age_match = re.search(
    r'Age[: ]+(\d+)',
    text,
    re.IGNORECASE
)

# Default values
glucose = 100
bmi = 22
age = 30

# Update if found
if glucose_match:
    glucose = int(glucose_match.group(1))

if bmi_match:
    bmi = float(bmi_match.group(1))

if age_match:
    age = int(age_match.group(1))

print("\nExtracted Parameters:")
print("Glucose:", glucose)
print("BMI:", bmi)
print("Age:", age)

# AI prediction
result = predict_diabetes(glucose, bmi, age)

print("\nAI Prediction:")
print(result)
recommendations = get_recommendations(
    result["risk_level"]
)

print("\nRecommendations:")

for r in recommendations:
    print("-", r)