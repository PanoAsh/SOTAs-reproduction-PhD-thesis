from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import copy
import cv2
import numpy as np
from math import pi, cos, sin, tan, atan, atan2, acos, sqrt, asin, floor, ceil


class ERPConfig(object):
    ERPHEIGHT = 512
    ERPWIDTH = 1024


class Geometry(object):
    @staticmethod
    def setRotationMat(p, t, r):
        phi = p * pi / 180.0
        tht = -t * pi / 180.0
        gam = r * pi / 180.0

        G = np.array([
            [cos(tht) * cos(gam) - sin(tht) * sin(phi) * sin(gam), cos(tht) * sin(gam) + sin(tht) * sin(phi) * cos(gam),
             sin(tht) * cos(phi)],
            [sin(gam) * cos(phi), cos(phi) * cos(gam), -sin(phi)],
            [-sin(tht) * cos(gam) - sin(gam) * cos(tht) * sin(phi),
             -sin(tht) * sin(gam) + cos(tht) * sin(phi) * cos(gam), cos(tht) * cos(phi)]
        ])
        return G

    @staticmethod
    def setIntrRotationMat(p, t, r):
        G = Geometry.setRotationMat(p, t, r)
        return np.linalg.inv(G)

    @staticmethod
    def setunIntrMat(dw, dh, fvx, fvy):
        fovx = pi * fvx / 180.0
        fovy = pi * fvy / 180.0
        fx = dw / 2.0 * 1 / tan(fovx / 2)
        fy = dh / 2.0 * 1 / tan(fovy / 2)
        K = np.array([
            [fx, 0, dw / 2.0],
            [0, -fy, dh / 2.0],
            [0, 0, 1]
        ])
        return K

    @staticmethod
    def setIntrMat(dw, dh, fvx, fvy):
        K = Geometry.setunIntrMat(dw, dh, fvx, fvy)
        return np.linalg.inv(K)

    @staticmethod
    def view2sph(v, u, G, K):
        x2, y2, z2 = np.dot(K, np.array([u, v, 1]))
        c = np.array([x2, y2, 1])
        z1 = 1.0 / sqrt(np.dot(c, c))
        x1, y1 = x2 * z1, y2 * z1
        p1 = np.array([x1, y1, -z1])
        p = np.dot(G, p1)
        return p

    @staticmethod
    def sph2erp(w, h, v):
        phi = acos(v[1])
        theta = atan2(v[0], -v[2])
        return h * (phi / pi), w * (0.5 + theta / pi / 2.0)

    @staticmethod
    def view2erp(v, u, w, h, G, K):
        p = Geometry.view2sph(v, u, G, K)
        return Geometry.sph2erp(w, h, p)

    @staticmethod
    def erp2sph(lat, lon, h, w):
        A = -tan((1.0 * lon / w - 0.5) * 2 * pi)
        y2 = cos(pi * lat / h)
        if 3 * w / 4 >= lon >= w / 4:
            z2 = -sqrt(1 - y2 * y2) / sqrt(1 + A * A)
        else:
            z2 = sqrt(1 - y2 * y2) / sqrt(1 + A * A)
        x2 = A * z2
        return np.array([x2, y2, z2])

    @staticmethod
    def sph2view(p2, G, K):
        p1 = np.dot(G, p2)
        if p1[2] > 0:
            return -1, -1
        p1[0] /= p1[2]
        p1[1] /= p1[2]
        p1[2] = 1
        p = np.dot(K, p1)
        return p[1], p[0]

    @staticmethod
    def erp2view(lat, lon, h, w, G, K):
        p2 = Geometry.erp2sph(lat, lon, h, w)
        return Geometry.sph2view(p2, G, K)

    @staticmethod
    def bilinear(x, y, f1, f2, f3, f4):
        return f1 * (ceil(x) - x) * (ceil(y) - y) + f2 * (x - floor(x)) * (ceil(y) - y) + f3 * (ceil(x) - x) * (
            y - floor(y)) + f4 * (x - floor(x)) * (y - floor(y))

    @staticmethod
    def get_viewport(erpImg, bbox, vpHeight, vpWidth):
        """
        Given the position tuple and the ERP image, extract the viewport in the given size.
        :param erpImg: ERP image.
        :param bbox: position tuple.
        :param vpHeight: The height of the viewport.
        :param vpWidth: The width of the viewport.
        :return: The extracted viewport image.
        """
        pitchAngle, YawAngle, FoVx, FoVy = bbox
        G = Geometry.setRotationMat(pitchAngle, YawAngle, 0)
        invK = Geometry.setIntrMat(vpWidth, vpHeight, FoVx, FoVy)
        erpHeight, erpWidth, _ = erpImg.shape
        viewport = np.zeros((vpHeight, vpWidth, 3), np.uint8)
        for i in range(vpHeight):
            for j in range(vpWidth):
                ni, nj = Geometry.view2erp(i, j, erpWidth, erpHeight, G, invK)
                # viewport[i, j] = erpImg[int(floor(ni)) % erpHeight, int(floor(nj)) % erpWidth]
                viewport[i, j] = Geometry.bilinear(ni, nj,
                                                   erpImg[(int(floor(ni)) + erpHeight) % erpHeight, (
                                                       int(floor(nj)) + erpWidth) % erpWidth],
                                                   erpImg[(int(ceil(ni)) + erpHeight) % erpHeight, (
                                                       int(floor(nj)) + erpWidth) % erpWidth],
                                                   erpImg[(int(floor(ni)) + erpHeight) % erpHeight, (
                                                       int(ceil(nj)) + erpWidth) % erpWidth],
                                                   erpImg[(int(ceil(ni)) + erpHeight) % erpHeight, (
                                                       int(ceil(nj)) + erpWidth) % erpWidth])
        return viewport

    @staticmethod
    def drawAnnotation(img, pos):
        """
        Given a position tuple, draw the corresponding bounding box.
        :param img: ERP image.
        :param pos: Position tuple.
        :return: The ERP image with bounding box.
        """
        centerPitch, centerYaw, FoVx, FoVy = pos
        retImg = copy.deepcopy(img)
        height, width, _ = retImg.shape
        annotationEdgeLength = 500
        G = Geometry.setRotationMat(centerPitch, centerYaw, 0)
        invK = Geometry.setIntrMat(annotationEdgeLength, annotationEdgeLength, FoVx, FoVy)
        for i in range(annotationEdgeLength):
            ni, nj = Geometry.view2erp(i, 0, width, height, G, invK)
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 0] = 255
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 1] = 0
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 2] = 255
            ni, nj = Geometry.view2erp(i, annotationEdgeLength - 1, width, height, G, invK)
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 0] = 255
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 1] = 0
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 2] = 255
            ni, nj = Geometry.view2erp(0, i, width, height, G, invK)
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 0] = 255
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 1] = 0
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 2] = 255
            ni, nj = Geometry.view2erp(annotationEdgeLength - 1, i, width, height, G, invK)
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 0] = 255
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 1] = 0
            retImg[int(floor(ni)) % height, int(floor(nj)) % width, 2] = 255
        return retImg