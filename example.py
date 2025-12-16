import cv2
import pytesseract
import numpy as np
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



img_source = cv2.imread('images/coffee.jpg')


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




gray = get_grayscale(img_source)
#opening = opening(gray)
#canny = canny(gray)


for img in [img_source, gray]:
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['text'])


    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for i in range(n_boxes):
        #if int(d['conf'][i]) > 60:
        if d['conf'][i] != '-1' and int(d['conf'][i]) >60:
            (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # don't show empty text
            if text and text.strip() != "":
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('img', img)
    cv2.waitKey(0)


    #
#
#
#


#import cv2
#import numpy as np
#import pytesseract

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#img = cv2.imread("images/coffee.jpg")
#img = cv2.resize(img, None, fx=0.5, fy=0.5)


#text = pytesseract.image_to_string(img)
#print(text)


#cv2.imshow("Img", img)
#cv2.waitKey(0)

#
#
#
#


#import cv2
#import pytesseract


#def ocr_core(img):
#    text = pytesseract.image_to_string(img)
#    return text


#img = cv2.imread('images/coffee.jpg')

#makes image gray
#def get_grayscale(image):
#    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#noise remover
#def remove_noise(image):
#    return cv2.medianBlur(image, 5)


# thresholding
#def thresholding(image):
#    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#img  = get_grayscale(img)
#img = thresholding(img)
#img = remove_noise(img)


#print(ocr_core(img))