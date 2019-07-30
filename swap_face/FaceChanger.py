# coding=utf-8
import os
import time

import cv2
import dlib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TooManyFaces(Exception):
    pass


class NoFace(Exception):
    pass


class NoJpg(Exception):
    pass


class FaceChanger(object):

    def __init__(self):
        self.errorMsg = ''
        self.resultUrl = ''
        self.current_path = BASE_DIR + "/swap_face"
        predictor_68_points_path = BASE_DIR + '/model/shape_predictor_68_face_landmarks.dat'
        self.predictor_path = predictor_68_points_path
        self.face_path = ''
        self.face_list = ''
        # some parameters
        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11

        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))

        # Points used to line up the images.
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                             self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)

        # Points from the second image to overlay on the first. The convex hull of each
        # element will be overlaid.
        self.OVERLAY_POINTS = [
            self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS,
        ]

        self.COLOUR_CORRECT_BLUR_FRAC = 0.6

        # load in models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.image1 = None
        self.image2 = None
        self.landmarks1 = None
        self.landmarks2 = None

    def load_images(self, image1_name, image2_name):
        if not image1_name.strip().split('.')[-1] == 'jpg':
            self.errorMsg = "文件必须是jpg格式"
            raise NoJpg
        if not image2_name.strip().split('.')[-1] == 'jpg':
            self.errorMsg = "文件必须是jpg格式"
            raise NoJpg

        image1_path = os.path.join(self.face_path, image1_name)
        image2_path = os.path.join(self.face_path, image2_name)

        self.image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
        self.image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

        self.landmarks1 = self.get_landmark(self.image1)
        self.landmarks2 = self.get_landmark(self.image2)

    def run(self):
        if self.image1 is None or self.image2 is None:
            self.errorMsg = '至少需要两张图片'
            raise NoFace
        M = self.transformation_from_points( \
            self.landmarks1[self.ALIGN_POINTS], self.landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(self.image2, self.landmarks2)

        warped_mask = self.warp_image(mask, M, self.image1.shape)

        combined_mask = np.max([self.get_face_mask(self.image1, self.landmarks1), \
                                warped_mask], axis=0)

        warped_img2 = self.warp_image(self.image2, M, self.image1.shape)

        warped_corrected_img2 = self.correct_colours(self.image1, warped_img2, self.landmarks1)
        warped_corrected_img2_temp = np.zeros(warped_corrected_img2.shape, dtype=warped_corrected_img2.dtype)
        cv2.normalize(warped_corrected_img2, warped_corrected_img2_temp, 0, 1, cv2.NORM_MINMAX)

        output = self.image1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask
        output_show = np.zeros(output.shape, dtype=output.dtype)
        cv2.normalize(output, output_show, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
        timestamp = str(int(time.time()))
        file_url = BASE_DIR + '/static/img/out/%s.%s' % (timestamp, 'jpg')
        cv2.imwrite(file_url, output)
        self.resultUrl = file_url

    def get_landmark(self, image):
        face_rect = self.detector(image, 1)

        if len(face_rect) > 1:
            self.errorMsg = '图片中脸太多了。。。'
            raise TooManyFaces
        elif len(face_rect) == 0:
            self.errorMsg = '图片中没有脸。。。。'
            raise NoFace
        else:
            return np.matrix([[p.x, p.y] for p in self.predictor(image, face_rect[0]).parts()])

    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    def warp_image(self, image, M, dshape):
        output_image = np.zeros(dshape, dtype=image.dtype)
        cv2.warpAffine(image, M[:2], (dshape[1], dshape[0]), dst=output_image, flags=cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_TRANSPARENT)
        return output_image

    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                im2_blur.astype(np.float64))

    def draw_convex_hull(self, img, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(img, points, color)

    def get_face_mask(self, img, landmarks):
        img = np.zeros(img.shape[:2], dtype=np.float64)
        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(img, landmarks[group], color=1)

        img = np.array([img, img, img]).transpose((1, 2, 0))

        img = (cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        img = cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

        return img


def swap_face(img1, img2):
    ctx = {}
    try:
        fc = FaceChanger()
        fc.load_images(img1, img2)
        fc.run()
    except(TooManyFaces):
        ctx['code'] = 0
        ctx['errMsg'] = "图片中脸太多了"
        return ctx
    except(NoFace):
        ctx['code'] = 0
        ctx['errMsg'] = "图片中未检测到脸"
        return ctx
    except(NoJpg):
        ctx['code'] = 0
        ctx['errMsg'] = "文件必须jpg"
        return ctx
    if fc.errorMsg == "":
        ctx['code'] = 1
        ctx['imgUrl'] = fc.resultUrl.split(BASE_DIR)[1]
    else:
        ctx['code'] = 0
        ctx['errMsg'] = fc.errorMsg
    return ctx
