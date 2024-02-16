import numpy as np
from PIL import Image
import cv2
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

w = 2192
band = 19

lambda_list = [375, 405, 435, 450, 470, 505, 525, 570, 590, 630, 645, 660, 700, 780, 850, 870, 890, 940, 970]

def create_raw_cube(path):
    cube = np.zeros((2192, 2192, 19))


    for j in range(1, 20): 
        if j <10:
            im =  Image.open(path + '_0' + str(j) + '.png').convert("L")
        else: 
            im =  Image.open(path + '_' + str(j) + '.png').convert("L")
           
        slice = np.array(im.getdata())
        slice = np.reshape(slice, (w,w))

        cube[:,:,j-1] = slice
    return cube

def background_removal(path, raw_cube_of_interest):
    img= cv2.imread(path + '_15.png') # band where we can easily remove background

    ## Convert to Gray
    imgGray =255- cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Show the different
    blur = cv2.GaussianBlur(imgGray,(5,5),0)
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(blur, kernel, iterations=4)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=4)
    # imshow(img_erosion)

    closing = cv2.morphologyEx(255-img_erosion,cv2.MORPH_CLOSE,kernel, iterations = 3)
    # imshow(closing)

    thresh = np.where(closing >80, 255, 0)
    threshcopy = np.uint8(thresh)
    
    ## Find edges
    img0 = np.zeros((w,w,3))
    contours,hierarchy =cv2.findContours(threshcopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    
    ## Loop through contours and find the biggest area.
    contour_list = []
    for cont in contours:
        area=cv2.contourArea(cont)

        if area > 10000:
            # print(area)
            contour_list.append(cont)

    cv2.drawContours(img0,contour_list,-1,(255,0,0), thickness=cv2.FILLED)
    mask = np.where(img0[:,:,0] !=0, 1, 0)
    cropped_cube = np.zeros((w,w,band))
    for k in range(band):
        cropped_cube[:,:,k] = np.multiply(raw_cube_of_interest[:,:,k], mask)
    
    return cropped_cube, mask

def find_left_right_borders_to_crop(background_mask):
    counter_begin = 0
    leaf_begin = 0
    leaf_end = 0    
    for k in range(w): 
        if background_mask[1096, k] == 255: 
            counter_begin+=1
            if counter_begin >30:
                leaf_begin = k-counter_begin
                break
        if background_mask[1096, k ]==0:
            counter_begin = 0
    counter_end = 0
    for k in range(w-1, 0, -1):
        if background_mask[1096, k] == 255: 
            counter_end+=1
            if counter_end >30:
                leaf_end = k+counter_end
                break
        if background_mask[1096, k]==0:
            counter_end = 0     
    return leaf_begin - 100, leaf_end+100

def find_window_to_crop(cube_of_interest, background_mask):
    left_border, right_border = find_left_right_borders_to_crop(background_mask)

    index_1= cube_of_interest[:,left_border:right_border,1]

    back_ground_mask_0 = np.where(index_1 >100, 255, 0)

    dim_mask_0 = np.shape(back_ground_mask_0)
    imgGray = np.reshape(back_ground_mask_0, (dim_mask_0[0], dim_mask_0[1], 1))
    im0 = np.zeros((dim_mask_0[0], dim_mask_0[1], 3))
    slice1Copy = np.uint8(imgGray) 

    contours,_ =cv2.findContours(slice1Copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for cont in contours:
        area=cv2.contourArea(cont)
        cv2.drawContours(slice1Copy,cont,-1,(255,0,0),5)

        if area > 200:
            cv2.drawContours(im0,cont,-1,(255,0,0),5, )
            
    mask_tack = np.where(im0[:,:,0] !=0, 1, 0)

    # imshow(mask_tack)
    tack_pixels_top = np.nonzero(mask_tack[:1096,:])[0]
    tack_pixels_bottom = np.nonzero(mask_tack[1096:,:])[0]
    top_border = np.max(tack_pixels_top) + 20
    bottom_border = np.min(tack_pixels_bottom) - 20 + 1096
    

    return  top_border, bottom_border, left_border, right_border

def create_smooth_cube_Savitzky_Golay(pixel_spectrum):
    f1 = interp1d(lambda_list, pixel_spectrum)
    
    xnew = np.concatenate((np.linspace(lambda_list[0], lambda_list[-1], 300), lambda_list))
    xnew.sort()
    
    
    # print(xnew)
    list_index = [1, 17, 33, 41, 52, 71, 82, 105, 117, 138, 146, 155, 176, 217, 253, 264, 275, 301, 317]
    fnew1 = f1(xnew)
    
    smooth_curve_1 = savgol_filter(fnew1, 31, 3)
    return smooth_curve_1

def create_final_cube (path):
    cube = create_raw_cube(path)
    bg_removed_cube, background_mask = background_removal(path, cube)
    top_border, bottom_border, left_border, right_border = find_window_to_crop(cube, background_mask*255)

    cropped_cube = bg_removed_cube[top_border:bottom_border, left_border:right_border,:]

    return cropped_cube