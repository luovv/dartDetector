import cv2
import numpy as np
import math
import sys
import argparse

OPENCV_HOUGH = 0
MY_HOUGH = 1

class DartDetector:
    clf_path = 'dartcascade/cascade.xml'

    # constructor
    def __init__(self, filename=None):
        self.image = cv2.imread(filename,cv2.IMREAD_COLOR)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.clf = cv2.CascadeClassifier(self.clf_path)

    # detect the display result
    def detect(self):
        gray = self.gray
        return self.clf.detectMultiScale(gray, 1.1, 3, 3, (60, 60))

    # detect and display the detection result
    def detect_and_display(self):
        image = self.image
        items = self.detect()
        for i in items:
            cv2.rectangle(image, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 255, 0), 2)
        return image

    # return gradient and angles of images
    def gradient(self, image=None):
        if image is None:
            gray = self.gray
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = cv2.phase(gx, gy)
        gradient = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
        gradient = cv2.convertScaleAbs(gradient)
        return gradient, angles

    # self-implemented Hough circles
    def my_hough_circles(self, image=None):
        if image is None:
            image = self.image
        gradient, angles = self.gradient(image)
        height, width = gradient.shape
        maxr = int(max(width, height)/2)
        h = np.zeros((width, height, maxr))
        for y in range(height):
            for x in range(width):
                if gradient[y, x] > 210:
                    for r in range(30, maxr):
                        xzero = x + r * math.cos(angles[y, x])
                        yzero = y + r * math.sin(angles[y, x])
                        if 0 < xzero < width-1 and 0 < yzero < height-1:
                            h[round(xzero), round(yzero), r] += 1

                        xzero = x - r * math.cos(angles[y, x])
                        yzero = y - r * math.sin(angles[y, x])
                        if 0 < xzero < width-1 and 0 < yzero < height-1:
                            h[round(xzero), round(yzero), r] += 1
        circle_list = []
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                for k in range(h.shape[2]):
                    value = h[i, j, k]
                    th = round(k/2.8)
                    if value > th:
                        circle_list.append([i, j, k])

        if circle_list:
            return np.array([circle_list])
        else:
            return None

    # self-implemented Hough lines
    def my_hough_lines(self, image=None):
        if image is None:
            image = self.image
        gradient, angles = self.gradient(image)
        height, width = gradient.shape
        maxRho = int(math.sqrt(width**2+height**2))
        h = np.zeros((maxRho, 180))
        for y in range(height):
            for x in range(width):
                if gradient[y, x] > 210:
                    for theta in range(1, 180):
                        rho = int(x * math.cos(np.pi/180*theta) + y * math.sin(np.pi/180*theta))
                        if rho > 0:
                            h[rho, theta] += 1
        line_list = []

        th = max(int(image.shape[0]/2), 80)
        for rho in range(h.shape[0]):
            for theta in range(h.shape[1]):
                value = h[rho, theta]
                if value > th:
                    line_list.append([[rho, np.pi/180*theta]])

        if line_list:
            return np.array(line_list)
        else:
            return None

    # Build in Opencv Hough circles
    def opencv_hough_circles(self, image=None):
        if image is None:
            image = self.image
            gray = self.gray
        else:
             gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        max_radius = math.ceil(min(image.shape[0], image.shape[1])/2)
        return cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 0.0001,
                                param1=200,
                                param2=120,
                                minRadius=15,
                                maxRadius=max_radius)

    # Build in Opencv Hough line method
    def opencv_hough_lines(self, image=None):
        if image is None:
            gray = self.gray
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 220, apertureSize=3)
        return cv2.HoughLines(edges, 1, np.pi/180, max(int(image.shape[0]/2.8), 75))

    # Given two points, return distance
    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    # find if there exist concentric circles
    def find_concentric_circles(self, circles):
        for i in circles:
            for j in circles:
                large_r = max(i[2], j[2])
                if self.distance(i, j) < large_r/10 and abs(i[2]-j[2]) > large_r/3:
                    return [i, j]
        return [circles[0]]

    # Given two lins, find cross point
    def find_cross_point(self, i, j):
        try:
            a = np.array([[1 / math.tan(i[1]), 1], [1 / math.tan(j[1]), 1]])
        except:
            a = np.array([[sys.float_info.max, 1], [sys.float_info.max, 1]])
        b = np.array([i[0] / math.sin(i[1]), j[0] / math.sin(j[1])])
        return np.linalg.solve(a, b)

    # Try to eliminate similar lines and find if there exist cross lines
    def find_cross_lines(self, candidate_lines):
        result = set()
        lines = []
        for i in range(len(candidate_lines)):
            flag = True
            for j in range(i+1, len(candidate_lines)):
                a = candidate_lines[i]
                b = candidate_lines[j]
                if abs(a[1] - b[1]) < math.pi / 15:
                    flag = False
            if flag:
                lines.append(candidate_lines[i])

        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                for k in range(j+1, len(lines)):
                    a = lines[i]
                    b = lines[j]
                    c = lines[k]

                    ab = self.find_cross_point(a, b)
                    bc = self.find_cross_point(b, c)
                    ac = self.find_cross_point(a, c)

                    if self.distance(ab, bc) < 16 and self.distance(ab, ac) < 12 and self.distance(bc, ac) < 12:
                        result.add("%s %s" % (str(a[0]), str(a[1])))
                        result.add("%s %s" % (str(b[0]), str(b[1])))
                        result.add("%s %s" % (str(c[0]), str(c[1])))
        return result

    # check if two squares are overlapped
    def is_overlapped(self, i, j):
        if j[0] < i[0] < j[0]+j[2] and j[1] < i[1] < j[1]+j[3]:
            return True
        if i[0] < j[0] < i[0]+i[2] and i[1] < j[1] < i[1]+i[3]:
            return True

        if j[0] < i[0]+i[2] < j[0]+j[2] and j[1] < i[1] < j[1]+j[3]:
            return True
        if i[0] < j[0]+j[2] < i[0]+i[2] and i[1] < j[1] < i[1]+i[3]:
            return True

        if j[0] < i[0] < j[0]+j[2] and j[1] < i[1]+i[3] < j[1]+j[3]:
            return True
        if i[0] < j[0] < i[0]+i[2] and i[1] < j[1]+j[3] < i[1]+i[3]:
            return True

        if j[0] < i[0]+i[2] < j[0]+j[2] and j[1] < i[1]+i[3] < j[1]+j[3]:
            return True
        if i[0] < j[0]+j[2] < i[0]+i[2] and i[1] < j[1]+j[3] < i[1]+i[3]:
            return True

        return False

    def run(self, fnc, threshold):
        image = self.image
        candidates = self.detect()
        weight = [3]*len(candidates)
        for index, i in enumerate(candidates):
            padding = 50
            if i[1]-padding > 0 and i[1]+i[3]+padding < image.shape[0] and i[0]-padding > 0 and i[0]+i[2]+padding < image.shape[1]:
                candidate_img = image[i[1]-padding:i[1]+i[3]+padding, i[0]-padding:i[0]+i[2]+padding]
            else:
                padding = 25
                if i[1] - padding > 0 and i[1] + i[3] + padding < image.shape[0] and i[0] - padding > 0 and i[0] + i[2] + padding < image.shape[1]:
                    candidate_img = image[i[1] - padding:i[1] + i[3] + padding, i[0] - padding:i[0] + i[2] + padding]
                else:
                    padding = 0
                    candidate_img = image[i[1] - padding:i[1] + i[3] + padding, i[0] - padding:i[0] + i[2] + padding]

            if fnc == 0:
                circles = self.opencv_hough_circles(candidate_img)
            elif fnc == 1:
                circles = self.my_hough_circles(candidate_img)

            if circles is not None:
                weight[index] += 3
                circles = np.round(circles[0, :]).astype("int")
                circles = self.find_concentric_circles(circles)

                for (x, y, r) in circles:
                    cv2.circle(candidate_img, (x, y), r, (55, 245, 234), 3)
                if len(circles) == 2:
                    weight[index] += 3

            if fnc == 0:
                lines = self.opencv_hough_lines(candidate_img)
            elif fnc == 1:
                lines = self.my_hough_lines(candidate_img)

            if lines is not None:
                lines = [[float(j) for j in i.split(' ')] for i in self.find_cross_lines(lines[:, 0])]
                if lines:
                    weight[index] += len(lines)
                    for rho, theta in lines:
                        a = math.cos(theta)
                        b = math.sin(theta)
                        cv2.line(candidate_img, (int(a * rho + 1000 * (-b)), int(b * rho + 1000 * a)),
                                 (int(a * rho - 1000 * (-b)), int(b * rho - 1000 * a)), (55, 245, 234), 2)

        # detect overlapped:
        # pick the one with higher weight
        # if same, pick the larger one
        for i in range(len(weight)):
            for j in range(i+1, len(weight)):
                rec1 = candidates[i]
                rec2 = candidates[j]
                if not rec1[0] == -1 and not rec2[0] == -1:
                    if self.is_overlapped(rec1, rec2):
                        if weight[i] > weight[j]:
                            candidates[j] = [-1, -1, -1, -1]
                        elif weight[i] < weight[j]:
                            candidates[i] = [-1, -1, -1, -1]
                        else:
                            if candidates[i, 2]*candidates[i, 3] > candidates[j, 2]*candidates[j, 3]:
                                candidates[j] = [-1, -1, -1, -1]
                            else:
                                candidates[i] = [-1, -1, -1, -1]

        for index, i in enumerate(candidates):
            if not i[0] == -1:
                if weight[index] < 5:
                    color = (200, 200, 200)
                    border = 1
                elif weight[index] < 7:
                    color = (200, 255, 200)
                    border = 1
                elif weight[index] < 9:
                    color = (120, 255, 120)
                    border = 2
                else:
                    color = (50, 255, 50)
                    border = 2

                if weight[index] > threshold:
                    cv2.putText(image, 'Weight: %s' % str(round(weight[index], 2)), (i[0] + 5, i[1] + i[3] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.6, color, border)
                    cv2.rectangle(image, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), color, border)

        cv2.imwrite('detected.jpg', image)
        return image

parser = argparse.ArgumentParser(description='Dart Detector')
parser.add_argument('filename')
parser.add_argument('--si', dest='si', help='Use self implemented Hough algorithm', default=0)
parser.add_argument('--th', dest='th', help='threshold', default=5)

args = parser.parse_args()
dd = DartDetector(args.filename)
result = dd.run(int(args.si), int(args.th))


