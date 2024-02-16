import pickle
import numpy as np
import scipy

band = 19

def return_average_spectra(dic_ROI):

    list_average_spectra = []
    for k in dic_ROI:
        mean_spectra_line = np.mean(dic_ROI[k], axis = 0) # average of all pixels spectra in a region
        mean_spectra = np.mean(mean_spectra_line, axis = 0)
        list_average_spectra.append(mean_spectra)
    return np.array(list_average_spectra)


# Example for the comparison between YR and healthy spectra
fichierini_YR = "ROIs_YR_24_14_days.txt"
fichierSauvegarde_YR = open(fichierini_YR,"rb")
all_ROI_YR_dic = pickle.load(fichierSauvegarde_YR)

fichierini_healthy = "ROIs_healthy_14_3_days.txt"
fichierSauvegarde_healthy = open(fichierini_healthy,"rb")
all_ROI_healthy_dic = pickle.load(fichierSauvegarde_healthy)

list_average_spectra_YR = return_average_spectra(all_ROI_YR_dic)
list_average_spectra_healthy = return_average_spectra(all_ROI_healthy_dic)

ttest_list = []
pvalue_list = []
for k in range(band):
    ttest_list.append(scipy.stats.ttest_ind(list_average_spectra_YR[:,k], list_average_spectra_healthy[:,k], nan_policy='omit')[0])
    pvalue_list.append(scipy.stats.ttest_ind(list_average_spectra_YR[:,k], list_average_spectra_healthy[:,k], nan_policy='omit')[1])