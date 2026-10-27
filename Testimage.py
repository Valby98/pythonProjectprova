import cv2
import numpy as np
import os


def preprocess_handwritten_text(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image: {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted = cv2.bitwise_not(gray)

    # Threshold the image to get binary image
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilate the image to connect text components
    kernel = np.ones((5, 100), np.uint8)  # Adjust kernel size as necessary
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each contour to create bounding boxes and save line images
    line_images = []
    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the line from the original image
        line_image = image[y:y + h, x:x + w]
        line_images.append(line_image)
        # Save the line image
        line_image_path = os.path.join(output_dir, f"line_{i + 1}.png")
        cv2.imwrite(line_image_path, line_image)

    print(f"Line images saved to {output_dir}")

    return line_images


def segment_words_from_lines(line_images, word_output_dir):
    os.makedirs(word_output_dir, exist_ok=True)

    word_images = []
    for i, line_image in enumerate(line_images):
        # Convert to grayscale
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image
        inverted = cv2.bitwise_not(gray)

        # Threshold the image to get binary image
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate the image to connect text components
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as necessary
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for j, contour in enumerate(contours):
            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Extract the word from the line image
            word_image = line_image[y:y + h, x:x + w]
            word_images.append(word_image)
            # Save the word image
            word_image_path = os.path.join(word_output_dir, f"line_{i + 1}_word_{j + 1}.png")
            cv2.imwrite(word_image_path, word_image)

    print(f"Word images saved to {word_output_dir}")

    return word_images


# Example usage
if __name__ == "__main__":
    image_path = '002_02.jpg'
    line_output_dir = 'output/lines'
    word_output_dir = 'output/words'

    line_images = preprocess_handwritten_text(image_path, line_output_dir)
    if line_images:
        segment_words_from_lines(line_images, word_output_dir)
