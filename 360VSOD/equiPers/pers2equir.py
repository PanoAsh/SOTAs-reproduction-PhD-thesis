import os
import cv2 
import equiPers.lib.Equirec2Perspec as E2P
import equiPers.lib.Perspec2Equirec as P2E
import equiPers.lib.multi_Perspec2Equirec as m_P2E
import glob
import argparse



def pers2equir(input_img, FOV, theta, phi, width=4096, height=2048):
    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #

    # this can turn cube to panorama
    equ = m_P2E.Perspective([input_img],
                            [[FOV, theta, phi]])
    
    
    img = equ.GetEquirec(height,width)  

    return img



if __name__ == '__main__':
    pers2equir()