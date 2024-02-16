from spectral import *
from select_wavelength import return_average_spectra
import pickle
from preprocess_img import create_final_cube, create_raw_cube, create_smooth_cube_Savitzky_Golay
import numpy as np
import os
import xml.etree.ElementTree as ET
import math
import matplotlib.pyplot as plt

lambda_list = [375, 405, 435, 450, 470, 505, 525, 570, 590, 630, 645, 660, 700, 780, 850, 870, 890, 940, 970]


cube = create_final_cube('YR/24_14_days/high_leaf_1.raw')
view_cube(cube, bands = [8, 10, 12])

area_zones_yr, area_zones_mild, area_zones_yr_mild, area_zones_sept_YR, area_zones_sept = [], [], [], [], []

######### Mildew ####################


list_average_spectrum_mild = []
file = open('dic_zone_Mild', 'rb') 
dic_zones_Mild = pickle.load(file) # the pixels spectra for infected mildew zones 

for k in range(1, 11): # we look at 11 leaves
    cube = create_raw_cube('..\Camera_2\Mild\Mild_29_March_Vuka_Rep'+str(k))
    for zone in dic_zones_Mild[k]: # infected zones on the leaf k
       
        spectrum_zone = return_average_spectra(zone,cube=cube)
        list_average_spectrum_mild.append(spectrum_zone)
        area_zones_mild.append((zone[0]-zone[2])*(zone[1]-zone[3])) # area of the zone
 


######## Sept ############
        
list_average_spectrum_sept = []

for file in os.listdir('../Camera_2/Septoria_YR/label_YR_Sept(field)'): # these leaves present mainly Sept symptoms
    mytree = ET.parse('../Camera_2/Septoria_YR/label_YR_Sept(field)/' + file)
    cube = create_raw_cube('../Camera_2/Septoria_YR/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        zone = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[zone[0]:zone[1], zone[2]: zone[3],:]
        
        if label == 'Sept':
            list_average_spectrum_sept.append(return_average_spectra(ROI))
            area_zones_sept.append((zone[1]-zone[0])*(zone[2]-zone[3]))

##### YR #########
            
list_average_spectrum_yr = []


for file in os.listdir('../Camera_2/YR/yr_label'):
    mytree = ET.parse('../Camera_2/YR/yr_label/' + file)
    cube = create_raw_cube('../Camera_2/YR/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        zone = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[zone[0]:zone[1], zone[2]: zone[3],:]
        
        if label == 'YR':
            list_average_spectrum_yr.append(return_average_spectra(ROI))
            area_zones_yr.append((zone[1]-zone[0])*(zone[2]-zone[3]))

########### YR_Sept #########

list_average_spectrum_sept_yr = []
 
for file in os.listdir('../Camera_2/Septoria_YR/label_sept_yr'):
    mytree = ET.parse('../Camera_2/Septoria_YR/label_sept_yr/' + file)
    cube = create_raw_cube('../Camera_2/Septoria_YR/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        zone = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[zone[0]:zone[1], zone[2]: zone[3],:]
        
        if label == 'Sept_YR':
            list_average_spectrum_sept_yr.append(return_average_spectra(ROI))
            area_zones_sept_YR.append((zone[1]-zone[0])*(zone[2]-zone[3]))


###### YR_Mild ######
list_average_spectrum_yr_mild = []


for file in os.listdir('../Camera_2/YR_Mild/yr_mild_label'):
    
    mytree = ET.parse('../Camera_2/YR_Mild/yr_mild_label/' + file)
    cube = create_raw_cube('../Camera_2/YR_Mild/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        zone = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[zone[0]:zone[1], zone[2]: zone[3],:]
        
        if label == 'YR_Mild':
            list_average_spectrum_yr_mild.append(return_average_spectra(ROI))
            area_zones_yr_mild.append((zone[1]-zone[0])*(zone[2]-zone[3]))




## Healthy, we just extract random zones from leaves
        
def extract_ROI(final_cube, M=30,N=30):
    # final cube has no tack and no background
    im = np.uint8(final_cube[:,:,[10,15,17]])
   
    # im = np.uint8(np.reshape(im0, (np.shape(im0)[0],np.shape(im0)[1],1)))
    im2 = im.copy()
    # gray2 = np.uint8(np.reshape(im, (np.shape(im)[0],np.shape(im)[1],1)))

    tiles = [im2[x:x+M,y:y+N,2] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    index = [[x,x+M, y, y+N,2]for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    # print(len(tiles), len(index))
    # print(index)
    index_non_zero=[]
    non_zeros_tiles = []
    for k in range(len(tiles)): 
        
        tile = tiles[k]
        if np.count_nonzero(tile) == M*N:
            
            index_non_zero = index[k]
            non_zeros_tiles.append(final_cube[index_non_zero[0]: index_non_zero[1],index_non_zero[2]: index_non_zero[3],:])
            # print(index_non_zero)
        
            
            im[int(index_non_zero[0]):int(index_non_zero[1]), int(index_non_zero[2]):int(index_non_zero[3])] = [255,255,255]
            im[index_non_zero[0]:index_non_zero[1], index_non_zero[2]] = [255,0,0]
            im[index_non_zero[0]:index_non_zero[1], index_non_zero[3]] = [255,0,0]
            im[index_non_zero[0], index_non_zero[2]: index_non_zero[3]] = [255,0,0]
            im[index_non_zero[1], index_non_zero[2]: index_non_zero[3]] = [255,0,0]
            # imshow(im3)

    return non_zeros_tiles

file = open('cube_healthy', 'rb')
cube_healthy = pickle.load(file)

list_ROI = extract_ROI(cube_healthy)
# print(len(list_ROI))
spectrum_healthy = []
for j in range(594):
    # print(int(math.floor(j)))
    subcube_healthy = list_ROI[int(math.floor(j))]
    im1 = np.mean(subcube_healthy, axis = 0)
    im2 = np.mean(im1, axis = 0)
    spectrum_healthy.append(im2)
list_average_spectrum_healthy = np.array(spectrum_healthy)



list_error_mild = np.std(list_average_spectrum_mild, axis = 0)
list_error_yr_mild = np.std(list_average_spectrum_yr_mild, axis = 0)
list_error_yr = np.std(list_average_spectrum_yr, axis = 0) 
list_error_sept = np.std(list_average_spectrum_sept, axis = 0) 
list_error_sept_yr = np.std(list_average_spectrum_sept_yr, axis = 0) 
list_error_healthy = np.std(list_average_spectrum_healthy, axis = 0) 


average_spectrum_healthy = np.mean(list_average_spectrum_healthy, axis=0)
average_spectrum_sept = np.average(list_average_spectrum_sept, weights = area_zones_sept, axis=0)
avg_spectrum_mild = np.average(list_average_spectrum_mild, axis = 0)
avg_spetrum_YR_sept = np.average(list_average_spectrum_sept_yr, weights = area_zones_sept_YR, axis=0)
avg_spectrum_yr = np.average(list_average_spectrum_yr, weights = area_zones_yr, axis=0)
avg_spectrum_yr_mild = np.average(list_average_spectrum_yr_mild, weights = area_zones_yr_mild, axis=0)

smooth_curve_1 = create_smooth_cube_Savitzky_Golay(average_spectrum_sept)
smooth_curve_2 = create_smooth_cube_Savitzky_Golay(average_spectrum_healthy)
smooth_curve_3 = create_smooth_cube_Savitzky_Golay(avg_spectrum_yr_mild)
smooth_curve_4 = create_smooth_cube_Savitzky_Golay(avg_spectrum_mild)
smooth_curve_5 = create_smooth_cube_Savitzky_Golay(avg_spectrum_yr)
smooth_curve_6 = create_smooth_cube_Savitzky_Golay(avg_spetrum_YR_sept)


xnew = np.concatenate((np.linspace(lambda_list[0], lambda_list[-1], 300), lambda_list))


list_color = ['b', 'g', 'r', 'c', 'm', 'y']
# list_color_2 = ['faded blue', 'light salmon', 'pale green', 'ice', 'powder pink', 'pale gold'] 
f = plt.figure()
f.set_figwidth(10)

plt.plot(xnew, smooth_curve_2/255, color = list_color[1], label = 'Healthy')
plt.plot(xnew, smooth_curve_1/255, color = list_color[0], label = 'Septoria')
plt.plot(xnew, smooth_curve_4/255,color = list_color[3],  label = 'Mildew')
plt.plot(xnew, smooth_curve_5/255, color = list_color[4], label = 'YR')
plt.plot(xnew, smooth_curve_3/255, color = list_color[2], label = 'YR + mildew')
plt.plot(xnew, smooth_curve_6/255,color = list_color[5],  label = 'YR + septoria')


plt.fill_between(xnew,smooth_curve_2/255 - np.mean(list_error_healthy)/(255), smooth_curve_2/255 + np.mean(list_error_healthy)/(255),alpha=0.1, color = list_color[1])
plt.fill_between(xnew,smooth_curve_1/255 - np.mean(list_error_sept)/(255), smooth_curve_1/255 + np.mean(list_error_sept)/(255),alpha=0.1, color = list_color[0])
plt.fill_between(xnew,smooth_curve_4/255 - np.mean(list_error_mild)/(255), smooth_curve_4/255 + np.mean(list_error_mild)/(255),alpha=0.1, color = list_color[3])
plt.fill_between(xnew,smooth_curve_5/255 - np.mean(list_error_yr)/(255), smooth_curve_5/255 + np.mean(list_error_yr)/(255),alpha=0.1, color = list_color[4])
plt.fill_between(xnew,smooth_curve_3/255 - np.mean(list_error_yr_mild)/(255), smooth_curve_3/255 + np.mean(list_error_yr_mild)/(255),alpha=0.1, color = list_color[2])
plt.fill_between(xnew,smooth_curve_6/255 - np.mean(list_error_sept_yr)/(255), smooth_curve_6/255 + np.mean(list_error_sept_yr)/(255),alpha=0.1, color = list_color[5])


plt.title('Average spectra of infected and healthy regions')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalised Reflectance')
plt.legend(loc='center right', bbox_to_anchor=(0.22, 0.79), prop={'size': 12})
plt.show(block=False)
plt.savefig('../paper/average reflectance.png')