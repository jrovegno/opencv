from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
import cv2
import imutils

# https://matplotlib.org/users/colors.html
def color(c):
    return tuple(int(x*255) for x in to_rgb(c))

def df_contours(cnts):
    df = pd.DataFrame(columns=['cx','cy', 'left', 'right', 'bottom', 'top', 'w', 'h'])
    for i, c in enumerate(cnts):
        M = cv2.moments(c)
        if np.isclose(M["m00"], 0.0):
            cx, cy = np.nan, np.nan
        else:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        df.loc[i] = pd.Series({'cx': cx, 'cy': cy, 'left': x, 'top': y, 
                               'w': w, 'h': h, 'bottom': y+h, 'right': x+w})
    return df

def df_image(image):
    df = pd.DataFrame(columns=['cx','cy', 'left', 'right', 'bottom', 'top', 'w', 'h'])
    

def center(df, i=0, offset=0):
    return tuple(df.loc[i, ['cx','cy']] + offset)

def lt(df, i=0, offset=0):
    '''Left Top'''
    return tuple(df.loc[i, ['left','top']] + offset)

def rb(df, i=0, offset=0):
    '''Right Bottom'''
    return tuple(df.loc[i, ['right','bottom']] + offset)

def plot_images(images):
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    for ax, im in zip(axes.flat, images):
        if len(im.shape) == 3:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        elif len(im.shape) == 2:
            ax.imshow(im, cmap='gray')
    plt.show()

