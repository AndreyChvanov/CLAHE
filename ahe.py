import numpy as np
import cv2


def load_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.uint8(img_gray)


def get_clipping_limit(hist, c=100):
    top = c
    bottom = 0
    while top - bottom > 1:
        middle = (top + bottom) / 2
        s = np.sum(hist[hist > middle] - middle)
        if s > (c - middle) * 255:
            top = middle
        else:
            bottom = middle
    new_hist = np.zeros_like(hist)
    new_hist[hist < bottom] = hist[hist < bottom] + (c - bottom)
    new_hist[hist >= bottom] = c
    return new_hist


def get_img_histogram(image):
    p = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            p[image[i, j]] += 1
    p = get_clipping_limit(p)
    return p / p.sum()


def get_cdf(p):
    cdf = np.cumsum(p)
    cdf_ = cdf * float(p.max()) / cdf.max()
    return np.uint8(cdf * 255)


def get_lookup_table(cur_view):
    p = get_img_histogram(cur_view)
    cdf = get_cdf(p)
    return cdf


def interpolation(A_0, B_0, C_0, f1, f2):
    t = (C_0 - B_0) / (A_0 - B_0)
    new_pixel_value = f1 * t + (1 - t) * f2
    return new_pixel_value


def red_zone_processing(image, new_image, cdfs, center):
    h, w = image.shape[0], image.shape[1]
    for x in [0, 7]:
        for y in [0, 7]:
            nn0, cdf0 = cdfs[x, y][1], cdfs[x, y][0]
            if x == 0 and y == 0:
                top, bottom, left, right = 0, center[0], 0, center[1]
            if x == 7 and y == 0:
                top, bottom, left, right = nn0[0], h, 0, center[1]
            if x == 0 and y == 7:
                top, bottom, left, right = 0, center[0], nn0[1], w
            if x == 7 and y == 7:
                top, bottom, left, right = nn0[0], h, nn0[1], w
            for i in range(top, bottom):
                for j in range(left, right):
                    new_image[i, j] = np.uint8(cdf0[image[i, j]])


def green_zone_processing(image, new_image, cdfs):
    h, w = image.shape[0], image.shape[1]
    for x in [0, 7]:
        for y in range(0, 7):
            nn0, cdf0 = cdfs[x, y][1], cdfs[x, y][0]
            nn1, cdf1 = cdfs[x, y + 1][1], cdfs[x, y + 1][0]
            if x == 0:
                top, bottom, left, right = 0, nn0[0], nn0[1], nn1[1]
            elif x == 7:
                top, bottom, left, right = nn0[0], h, nn0[1], nn1[1]
            for i in range(top, bottom):
                for j in range(left, right):
                    new_pixel_value = interpolation(nn0[1], nn1[1], j, cdf0, cdf1)
                    new_image[i, j] = np.uint8(new_pixel_value[image[i, j]])
    for y in [0, 7]:
        for x in range(0, 7):
            nn0, cdf0 = cdfs[x, y][1], cdfs[x, y][0]
            nn1, cdf1 = cdfs[x + 1, y][1], cdfs[x + 1, y][0]
            if y == 0:
                top, bottom, left, right = nn0[0], nn1[0], 0, nn0[1]
            elif y == 7:
                top, bottom, left, right = nn0[0], nn1[0], nn0[1], w
            for i in range(top, bottom):
                for j in range(left, right):
                    new_pixel_value = interpolation(nn0[0], nn1[0], i, cdf0, cdf1)
                    new_image[i, j] = np.uint8(new_pixel_value[image[i, j]])


def blue_zone_processing(image, new_image, cdfs):
    for x in range(0, 7):
        for y in range(0, 7):
            nn0, cdf0 = cdfs[x, y][1], cdfs[x, y][0] # a
            nn1, cdf1 = cdfs[x + 1, y][1], cdfs[x + 1, y][0] # d
            nn2, cdf2 = cdfs[x + 1, y + 1][1], cdfs[x + 1, y + 1][0] #c
            nn3, cdf3 = cdfs[x, y + 1][1], cdfs[x, y + 1][0] # b
            for i in range(nn0[0], nn2[0]):
                for j in range(nn0[1], nn2[1]):
                    r2 = interpolation(nn0[1], nn3[1], j, cdf0, cdf3)
                    r1 = interpolation(nn0[1], nn3[1], j, cdf1, cdf2)
                    new_pixel_value = interpolation(nn0[0], nn1[0], i, r2, r1)
                    new_image[i, j] = np.uint8(new_pixel_value[image[i, j]])


def AHE(image, c=8):
    h, w = image.shape[0], image.shape[1]
    grid_size = (h // c, w // c)
    center = (grid_size[0] // 2, grid_size[1] // 2)
    cdfs = {}
    x, y = 0, 0
    new_image = np.zeros_like(image)
    for i in range(center[0], h, grid_size[0]):
        for j in range(center[1], w, grid_size[1]):
            cur_view = image[i - center[0]:i + center[0], j - center[1]:j + center[1]]
            cdf = get_lookup_table(cur_view)
            cdfs[(x, y)] = [cdf, (i, j)]
            y += 1
        x += 1
        y = 0
    blue_zone_processing(image, new_image, cdfs)
    green_zone_processing(image, new_image, cdfs)
    red_zone_processing(image, new_image, cdfs, center)

    return new_image


