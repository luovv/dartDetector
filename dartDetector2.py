import cv2
import numpy as np
import math
import sys

OPENCV_HOUGH_CIRCLES = 0
MY_HOUGH_CIRCLES = 1

class DartDetector:
    clf_path = 'dartcascade/cascade.xml'

    def __init__(self, filename=None):
        self.image = cv2.imread(filename,cv2.IMREAD_COLOR)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.clf = cv2.CascadeClassifier(self.clf_path)

    def detect(self):
        gray = self.gray
        return self.clf.detectMultiScale(gray, 1.1, 3, 3, (60, 60))

    def detect_and_display(self):
        image = self.image
        items = self.detect()
        for i in items:
            cv2.rectangle(image, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 255, 0), 2)
        return image

    def gradient(self, image=None):
        if image is None:
            gray = self.gray
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = cv2.phase(dx, dy)
        gradient = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        gradient = cv2.convertScaleAbs(gradient)
        return gradient, angles

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
                    for r in range(25, maxr):
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
                    th = round(k/2.5)
                    if value > th:
                        circle_list.append([i, j, k])
                        print(i, '/', j, '/', k, '/', h[i, j, k])

        if circle_list:
            return np.array([circle_list])
        else:
            return None

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

    def hough_lines(self, image=None):
        if image is None:
            gray = self.gray
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 220, apertureSize=3)
        return cv2.HoughLines(edges, 1, np.pi/180, max(int(image.shape[0]/2.8), 75))

    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    def find_concentric_circles(self, circles):
        for i in circles:
            for j in circles:
                large_r = max(i[2], j[2])
                if self.distance(i, j) < large_r/10 and abs(i[2]-j[2]) > large_r/3:
                    return [i, j]
        return [circles[0]]

    def find_cross_point(self, i, j):
        try:
            a = np.array([[1 / math.tan(i[1]), 1], [1 / math.tan(j[1]), 1]])
        except:
            a = np.array([[sys.float_info.max, 1], [sys.float_info.max, 1]])
        b = np.array([i[0] / math.sin(i[1]), j[0] / math.sin(j[1])])
        return np.linalg.solve(a, b)

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

    def run(self, fnc):
        image = self.image
        candidates = self.detect()
        prob = [0.3]*len(candidates)
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
                prob[index] += 0.3
                circles = np.round(circles[0, :]).astype("int")
                circles = self.find_concentric_circles(circles)

                for (x, y, r) in circles:
                    cv2.circle(candidate_img, (x, y), r, (55, 245, 234), 3)
                if len(circles) == 2:
                    prob[index] += 0.3

            lines = self.hough_lines(candidate_img)
            if lines is not None:
                lines = [[float(j) for j in i.split(' ')] for i in self.find_cross_lines(lines[:, 0])]
                if lines:
                    prob[index] += len(lines)/10
                    for rho, theta in lines:
                        a = math.cos(theta)
                        b = math.sin(theta)
                        cv2.line(candidate_img, (int(a * rho + 1000 * (-b)), int(b * rho + 1000 * a)),
                                 (int(a * rho - 1000 * (-b)), int(b * rho - 1000 * a)), (55, 245, 234), 2)

        # detect overlapped:
        # pick the one with higher prob
        # if same, pick the larger one
        for i in range(len(prob)):
            for j in range(i+1, len(prob)):
                rec1 = candidates[i]
                rec2 = candidates[j]
                print(rec1)
                print(rec2)
                if not rec1[0] == -1 and not rec2[0] == -1:
                    if self.is_overlapped(rec1, rec2):
                        if prob[i] > prob[j]:
                            candidates[j] = [-1, -1, -1, -1]
                        elif prob[i] < prob[j]:
                            candidates[i] = [-1, -1, -1, -1]
                        else:
                            if candidates[i, 2]*candidates[i, 3] > candidates[j, 2]*candidates[j, 3]:
                                candidates[j] = [-1, -1, -1, -1]
                            else:
                                candidates[i] = [-1, -1, -1, -1]

        # print(candidates)
        for index, i in enumerate(candidates):
            if not i[0] == -1:
                if prob[index] >= 1:
                    prob[index] = 0.99

                if prob[index] < 0.5:
                    color = (200, 200, 200)
                    border = 1
                elif prob[index] < 0.7:
                    color = (200, 255, 200)
                    border = 1
                elif prob[index] < 0.9:
                    color = (120, 255, 120)
                    border = 2
                else:
                    color = (50, 255, 50)
                    border = 2


                cv2.putText(image, 'Prob: %s' % str(prob[index]), (i[0] + 5, i[1] + i[3] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.6, color, border)
                cv2.rectangle(image, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), color, border)

            # if lines is not None:

        print(prob)

        cv2.imshow('a', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image



for i in range(14,15):
    dd = DartDetector('images/dart%s.jpg'%i)
    # result = dd.run(OPENCV_HOUGH_CIRCLES)
    result = dd.run(MY_HOUGH_CIRCLES)
    # dd.detect_circle()
    # cv2.imshow('a', result)
    # cv2.waitKey(0)
# cv2.destroyAllWindows()
#     [[float(j) for j in i.split(' ')] for i in arr if i]
# for i in range(15):
#     dd = DartDetector('images/dart%s.jpg'%i)
#     darts = dd.detect()
#     [i[0], i[0] + i[2] : i[1], i[1] + i[3]]
#     result = dd.detect_and_display()
#     cv2.imshow('result', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# class DartDetectorTest:
#     def __init__(self):
#         self.images = []
#
#     def read_images(self, name_list):
#         if type(name_list) == str:
#             name_list = [name_list]
#         for name in name_list:
#             self.images.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))


# opencv_createsamples -img dart.bmp -bg negatives.dat -info info/info.lst -pngoutput info -num 950
# opencv_createsamples -img dart.bmp -vec dart.vec -neg negatives.dat -w 20 -h 20 -num 1000 -maxidev 80 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.2
# opencv_traincascade -data dartcascade -vec dart.vec -bg negatives.dat -numPos 1000 -numNeg 1000 -numStages 3 -maxDepth 1 -w 20 -h 20 -minHitRate 0.999 -maxFalseAlarmRate 0.05 -mode ALL

