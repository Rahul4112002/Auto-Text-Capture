from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import imutils
import easyocr
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Function to preprocess the uploaded image and extract text
def process_image(image_path):
    # Read the uploaded image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Filter and Find Edges For Localization
    bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find the Contours and Apply Mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Loop through each contour to find Number Plate Location
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # Creating a Blank Mask
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Cropping the image to the segment which has the number plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+3, y1:y2+3]

    # Use Easy OCR to Read Text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    # Overlay Results on the Original Image
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Calculate y-coordinate for text placement

    text_height = cv2.getTextSize(text, font, 1, 5)[0][1]
    text_y = location[1][0][1] + 60 + text_height + 10

    # Calculate text size
    text_size = cv2.getTextSize(text, font, 1, 5)[0]

    # Calculate x-coordinate for text placement
    text_x = location[0][0][0] - text_size[0] - 10  # Adjusted to place text on the left side of the x-axis

    res = cv2.putText(img.copy(), text=text, org=(text_x, text_y), fontFace=font, fontScale=0, color=(0, 0, 0), thickness=0)
    res = cv2.rectangle(res, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

    # Save the processed image
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, res)

    # Convert processed image to base64 format
    with open(processed_image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    return processed_image_path, text, encoded_string

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            processed_image_path, extracted_text, encoded_image = process_image(filepath)
            return render_template('upload.html', filename=filename, processed_image=encoded_image, extracted_text=extracted_text)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
