IMAGE_SIZE = 1/2.8


import cv2
import numpy as np
import colorsys

# Define the size of the segments
segment_size = 3

# Read the image
img = cv2.imread('sample.jpeg')
img_grayscale = cv2.cvtColor(cv2.resize(img, (int(img.shape[1]*IMAGE_SIZE), int(img.shape[0]*IMAGE_SIZE)), interpolation=cv2.INTER_NEAREST if IMAGE_SIZE < 1 else cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY) 
img = cv2.resize(img, (int(img.shape[1]*IMAGE_SIZE/segment_size)+1, int(img.shape[0]*IMAGE_SIZE/segment_size)+1), interpolation=cv2.INTER_AREA if IMAGE_SIZE < 1 else cv2.INTER_CUBIC)


def choose_block(segment):
    flat = np.array([float(i) for i in segment.flatten()])

    if len(flat) < 9:
        return 0

    p1, p2, p3, p4, p5, p6, p7, p8, p9 = flat
    sum = flat.sum()

    # this is terrible please dont use this anywhere else and please, whoever is looking at this pls change it
    sums = [ # ngl i have no idea if this even works properly but it looks good so thats nice
        sum-(p1+p3+p7+p9)/4,

        sum-(p1+p2+p3)/3,
        sum-(p7+p8+p9)/3,
        sum-(p1+p4+p7)/3,
        sum-(p3+p6+p9)/3,

        sum-(p1+p2+p3+p4+p7)/5,
        sum-(p1+p2+p3+p6+p9)/5,
        sum-(p7+p8+p9+p1+p4)/5,
        sum-(p7+p8+p9+p3+p6)/5,

        sum-p1,
        sum-p3,
        sum-p7,
        sum-p9,

        sum/2, # because its a bit darker or something, i wonder if it will ever get chosen though lmao

        sum-(p1+p2+p3+p4+p6+p7+p9)/7,
        sum-(p1+p2+p3+p6+p7+p8+p9)/7,
        sum-(p1+p3+p4+p6+p7+p8+p9)/7,
        sum-(p1+p2+p3+p4+p7+p8+p9)/7,

        sum-(p1+p3+p4+p6+p7+p9)/6,
        sum-(p1+p2+p3+p7+p8+p9)/6,

        sum-(p1+p3)/2,
        sum-(p3+p9)/2,
        sum-(p7+p9)/2,
        sum-(p1+p7)/2,
    ]
    
    return sums.index(max(sums))


# Split the image into segments and extract the average color of each segment
segments = []
for i in range(0, img_grayscale.shape[0], segment_size):
    segments.append([])
    for j in range(0, img_grayscale.shape[1], segment_size):
        color_segment = img[i//segment_size, j//segment_size]
        h,s,v = colorsys.rgb_to_hsv(color_segment[2], color_segment[1], color_segment[0]) # using this lib because opencv outputs weird hsv values
        gray_segment = img_grayscale[i:i+segment_size, j:j+segment_size]

        block = choose_block(gray_segment)
        segments[len(segments)-1].append([block, int(h*360), int(round(s, 3)*1000), int(round(v/255, 3)*1000)])

with open("output.txt", "w") as f:
    f.write(str(segments).replace(" ", ""))
print("completed succesfully")
print("x:", img_grayscale.shape[1], " y:", img_grayscale.shape[0], " objects:", img_grayscale.shape[1] * img_grayscale.shape[0] // 9)
