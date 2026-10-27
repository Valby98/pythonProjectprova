import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import matplotlib.pyplot as plt

# Ensure Tesseract is installed on your system and the path is set
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

def save_image(title, image):
    cv2.imwrite(title, image)

def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image("grayscale_image.png", gray)

    # Invert the grayscale image
    inverted_gray = cv2.bitwise_not(gray)
    save_image("inverted_grayscale_image.png", inverted_gray)

    # Threshold the image to get binary image
    _, binary_image = cv2.threshold(inverted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_image("binary_image.png", binary_image)

    # Dilate the image to connect components
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    save_image("dilated_image.png", dilated_image)

    return dilated_image, image

def find_text_bounding_boxes(dilated_image, original_image):
    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # OCR configuration
    config = "--psm 6"  # Set Page Segmentation Mode to 6: Assume a single uniform block of text.

    # Draw bounding boxes around contours and extract text
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) and apply OCR
        roi = dilated_image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config=config)

        # Print the recognized text
        print(f"Recognized Text: {text.strip()}")

        # Draw the recognized text on the original image
        cv2.putText(original_image, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    save_image("image_with_text_bounding_boxes.png", original_image)
    display_image(original_image, title="Bounding Boxes and Recognized Text")



def find_and_draw_text_regions1(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return

    # Perform OCR using Tesseract
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    # Draw bounding boxes
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        text = data['text'][i]
        if text.strip() != '':
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save and display the image with bounding boxes
    save_image("image_with_text_bounding_boxes1.png", image)

    display_image1(image, title="Image with Text Bounding Boxes1")

def display_image1(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def main():
    input_image_path = '002_02.jpg'  # Ensure the image is in the same directory

    # Preprocess the image
    dilated_image, original_image = preprocess_image(input_image_path)

    find_and_draw_text_regions1(input_image_path)

    if dilated_image is not None and original_image is not None:
        # Find and draw text regions
        find_text_bounding_boxes(dilated_image, original_image)

if __name__ == "__main__":
    main()
