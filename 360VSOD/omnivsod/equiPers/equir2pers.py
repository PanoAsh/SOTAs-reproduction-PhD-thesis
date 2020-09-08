import os
import cv2 
import equiPers.lib.Equirec2Perspec as E2P
import equiPers.lib.Perspec2Equirec as P2E
import equiPers.lib.multi_Perspec2Equirec as m_P2E
import glob
import argparse



def equir2pers(input_img, FOV, theta, phi, height=256, width=256):

    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #
    equ = E2P.Equirectangular(input_img)    # Load equirectangular image

    img = equ.GetPerspective(FOV, theta, phi, height, width)  # Specify parameters(FOV, theta, phi, height, width)

    return img


if __name__ == '__main__':
    equir2pers()