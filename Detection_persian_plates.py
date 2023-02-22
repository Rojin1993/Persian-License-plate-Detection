import numpy as np
import torch
import torchvision
import cv2 as cv
from math import sqrt, atan, degrees
import imutils

#constant of class of characters
classes = ('a','b','ch','d','ein','f', 'g', 'ghaf', 'ghein', 'h2','hj', 'j', 'k', 'kh','l','m','n','p','r','s','sad','sh','t','ta','th','v','y','z','za','zad','zal','zh')

# definition of required functions
def find_longest_line(plate_img_gr):
    low_threshold = 150
    high_threshold = 200

    edges = cv.Canny(plate_img_gr, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    len1 = 0
    x1_main, y1_main, x2_main, y2_main = 0, 0, 0, 0
    for _, line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            line_length = sqrt((x2-x1)**2 + (y2-y1)**2)
            if line_length > len1:
                len1 = line_length
                x1_main = x1
                x2_main = x2
                y1_main = y1
                y2_main = y2
    return x1_main, y1_main, x2_main, y2_main

def find_line_angle(x1_main, y1_main, x2_main, y2_main):
    x1,y1,x2,y2 = x1_main, y1_main, x2_main, y2_main
    angle = degrees(atan(((y2-y1)/(x2-x1))))
    return angle

def rotate_image(image, angle):
    rotated = imutils.rotate(image, angle)
    return rotated


tr = torchvision.transforms.ToPILImage()
tr1 = torchvision.transforms.Resize([90, 90])
tr2 = torchvision.transforms.ToTensor()
tr3 = torchvision.transforms.Resize([39,195])
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model of predicting persian numbers
model1 = torch.load('R:/horoof/Augdensenet-numbers90.pth').to(d)
model1.eval()
# model of predicting persian characters
model2 = torch.load('R:/horoof/Augdensenet-characters90.pth').to(d)
model2.eval()

# definition of image
image1 = cv.imread("R:/horoof/cropped_license/2.jpg")
image2 = cv.resize(image1, (195, 35))

# rotation and cropping of cropped license plate
l1, l2, l3, l4 = find_longest_line(image2)
angle = find_line_angle(l1, l2, l3, l4)
image3 = rotate_image(image2, angle)

image4 = np.transpose(image3, (2,0,1))
# c, h, w = image4.shape

# find bounds of cropping plates to detect characters and numbers
x = np.sin(angle)*6
crop = [20+x,40+x,60+x,95+x,115+x,135+x,155+x,175+x,195+x]


i = 0
predictions = []
image4 = torch.from_numpy(image4)
image4 = tr3(image4)

while i < 8:

    # crop the plate to 8 characters
    w1 = int(crop[i])
    w2 = int(crop[i+1])
    img = image4[:, :, w1:w2]
    # tr(img).show()

    # change size, type and dimension of image
    img = tr1(img)
    img = tr(img)
    img = tr2(img)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.type(torch.FloatTensor)

    img = torchvision.transforms.functional.rgb_to_grayscale(img, 3)

    img = img.to(d)

    # if i == 2 implement model of predicting elements else implement model of predicting numbers
    if i == 2:
        output = model2(img)
    else:
        output = model1(img)

    _, prediction = torch.max(output.data, 1)

    if i == 2:
        predictions.append(classes[prediction[0]])
    else:
        predictions.append(int(prediction[0]))
    i += 1


print(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4], predictions[5], predictions[6], predictions[7])
