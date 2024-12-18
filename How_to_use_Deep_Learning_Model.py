###From Jingjing Tian#### 
###If any questions, please feel free to contact###



import warnings
warnings.simplefilter('ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')


import numpy as np  
import xarray as xr 
import pandas as pd  # type: ignore
import glob
import copy 
import datetime
import os
from scipy import ndimage
from scipy.interpolate import griddata
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"  


import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, concatenate, LeakyReLU, Normalization
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import categorical_crossentropy


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib as mpl
import matplotlib.gridspec as gridspec


# cloudtype:flag_values = 1, 2, 3, 4, 5, 6, 7 ;
# cloudtype:flag_meanings = "low_cloud congestus deep_convection altocumulus altostratus cirrostratus/anvil cirrus" ;    
  
def read_cldtype(date_target):
    cldtype_pwd='/ena/enacldtypeC1.c1/'
    cldtype_file=f'{cldtype_pwd}/*{date_target}*'
    output_cldtype_file=glob.glob(cldtype_file)
    dsout_cldtype =xr.open_dataset(output_cldtype_file[0]) #,decode_times=False
    
    height_cldtype=dsout_cldtype.height.values/1000.
    cld_base=dsout_cldtype.cloud_layer_base_height.values/1000.
    cld_top=dsout_cldtype.cloud_layer_top_height.values/1000.

    ntimes_cldtype=len(dsout_cldtype.time)
    time_utc_cldtype=np.empty(ntimes_cldtype)
    for i in np.arange(ntimes_cldtype):
        t=dsout_cldtype.time.values[i]
        t1 = pd.Timestamp(t)
        time_utc_cldtype[i]=t1.hour+t1.minute/60.+t1.second/3600.

    cloudtype=dsout_cldtype.cloudtype.values
    qc_cloudtype=dsout_cldtype.qc_cloudtype.values
    
    return time_utc_cldtype,cloudtype,qc_cloudtype,height_cldtype,cld_base,cld_top


def read_arscl_data(date_target):

    arscl_pwd='/ena/enaarsclkazr1kolliasC1.c0/'
    arscl_file=f'{arscl_pwd}*{date_target}*'
    output_arscl_file=glob.glob(arscl_file)
    dsout_arscl =xr.open_dataset(output_arscl_file[0]) #,decode_times=False

    ntimes_arscl=len(dsout_arscl.time)

    time_utc_arscl=np.empty(ntimes_arscl)
    for i in np.arange(ntimes_arscl):
        t=dsout_arscl.time.values[i]
        t1 = pd.Timestamp(t)
        time_utc_arscl[i]=t1.hour+t1.minute/60.+t1.second/3600.

    arscl_height=dsout_arscl.height.values
    reflectivity_best_estimate=dsout_arscl.reflectivity_best_estimate.values

    return time_utc_arscl,arscl_height,reflectivity_best_estimate
    # 4 second
    
# average to 1 mins 
def mins_array(array, avg_time_interval ):
    # Calculate the number of 4-second intervals in 60 seconds
    intervals = int(avg_time_interval / 4)  # 60 seconds divided by 4 seconds

    # Reshape the array to group data into 60-second intervals
    reshaped_array = array.reshape( -1, intervals, 95)

    # Calculate the mean along the new interval dimension
    average_array = np.nanmean(reshaped_array, axis=1)
    
    return average_array

def preprocess_image(image):
    image = tf.expand_dims(image, -1)  # Adds a third dimension for channels
    # Resize image
    image = tf.image.resize(image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image



def weighted_categorical_crossentropy(weights):
    """A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.constant(weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        # scale predictions so that the class probabilities of each pixel sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calculate the weighted loss
        loss = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, -1)
        return loss
    
    return loss


def input_hour_increment_get_prediction_4hr(time_utc_cldtype,reflectivity_best_estimate1):
    
    hour_increment=4
    hours=np.array([0,4,8,12,16,20,24]) # this can change based on interest


    arscl_loc_1=np.where( (time_utc_arscl1>=0) & (time_utc_arscl1<4)) 
    arscl_loc_2=np.where( (time_utc_arscl1>=4) & (time_utc_arscl1<8)) 
    arscl_loc_3=np.where( (time_utc_arscl1>=8) & (time_utc_arscl1<12)) 
    arscl_loc_4=np.where( (time_utc_arscl1>=12) & (time_utc_arscl1<16)) 
    arscl_loc_5=np.where( (time_utc_arscl1>=16) & (time_utc_arscl1<20)) 
    arscl_loc_6=np.where( (time_utc_arscl1>=20) & (time_utc_arscl1<24)) 


    reflectivity_best_estimate1_6hr_04=reflectivity_best_estimate1[arscl_loc_1[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_48=reflectivity_best_estimate1[arscl_loc_2[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_812=reflectivity_best_estimate1[arscl_loc_3[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_1216=reflectivity_best_estimate1[arscl_loc_4[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_1620=reflectivity_best_estimate1[arscl_loc_5[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_2024=reflectivity_best_estimate1[arscl_loc_6[0],:][:,loc_height[0]]


    refl_6hr_1min_04=mins_array(reflectivity_best_estimate1_6hr_04, avg_time_interval)
    refl_6hr_1min_48=mins_array(reflectivity_best_estimate1_6hr_48, avg_time_interval)
    refl_6hr_1min_812=mins_array(reflectivity_best_estimate1_6hr_812, avg_time_interval)
    refl_6hr_1min_1216=mins_array(reflectivity_best_estimate1_6hr_1216, avg_time_interval)
    refl_6hr_1min_1620=mins_array(reflectivity_best_estimate1_6hr_1620, avg_time_interval)
    refl_6hr_1min_2024=mins_array(reflectivity_best_estimate1_6hr_2024, avg_time_interval)


    refl_6hr_1min_04[np.isnan(refl_6hr_1min_04)] = -99
    refl_6hr_1min_48[np.isnan(refl_6hr_1min_48)] = -99
    refl_6hr_1min_812[np.isnan(refl_6hr_1min_812)] = -99
    refl_6hr_1min_1216[np.isnan(refl_6hr_1min_1216)] = -99
    refl_6hr_1min_1620[np.isnan(refl_6hr_1min_1620)] = -99
    refl_6hr_1min_2024[np.isnan(refl_6hr_1min_2024)] = -99
    
    
    # Calculate the zoom factor for each dimension
    zoom_factor = (
        240 / (60*hour_increment),  # Zoom factor along the second dimension
        1,  # No zoom along the third dimension (95)
    )

    resized_refl_6hr_1min_04 = ndimage.zoom(refl_6hr_1min_04, zoom=zoom_factor)
    resized_refl_6hr_1min_04[resized_refl_6hr_1min_04<-60] = np.nan

    resized_refl_6hr_1min_48 = ndimage.zoom(refl_6hr_1min_48, zoom=zoom_factor)
    resized_refl_6hr_1min_48[resized_refl_6hr_1min_48<-60] = np.nan

    resized_refl_6hr_1min_812 = ndimage.zoom(refl_6hr_1min_812, zoom=zoom_factor)
    resized_refl_6hr_1min_812[resized_refl_6hr_1min_812<-60] = np.nan

    resized_refl_6hr_1min_1216 = ndimage.zoom(refl_6hr_1min_1216, zoom=zoom_factor)
    resized_refl_6hr_1min_1216[resized_refl_6hr_1min_1216<-60] = np.nan
    
    ####
    resized_refl_6hr_1min_1620 = ndimage.zoom(refl_6hr_1min_1620, zoom=zoom_factor)
    resized_refl_6hr_1min_1620[resized_refl_6hr_1min_1620<-60] = np.nan

    resized_refl_6hr_1min_2024 = ndimage.zoom(refl_6hr_1min_2024, zoom=zoom_factor)
    resized_refl_6hr_1min_2024[resized_refl_6hr_1min_2024<-60] = np.nan



    # Scale 
    resized_refl_6hr_1min_04[resized_refl_6hr_1min_04<=refl_min]=refl_min
    resized_refl_6hr_1min_04[resized_refl_6hr_1min_04>=refl_max]=refl_max
    resized_refl_6hr_1min_04_norm = (resized_refl_6hr_1min_04-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_48[resized_refl_6hr_1min_48<=refl_min]=refl_min
    resized_refl_6hr_1min_48[resized_refl_6hr_1min_48>=refl_max]=refl_max
    resized_refl_6hr_1min_48_norm = (resized_refl_6hr_1min_48-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_812[resized_refl_6hr_1min_812<=refl_min]=refl_min
    resized_refl_6hr_1min_812[resized_refl_6hr_1min_812>=refl_max]=refl_max
    resized_refl_6hr_1min_812_norm = (resized_refl_6hr_1min_812-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_1216[resized_refl_6hr_1min_1216<=refl_min]=refl_min
    resized_refl_6hr_1min_1216[resized_refl_6hr_1min_1216>=refl_max]=refl_max
    resized_refl_6hr_1min_1216_norm = (resized_refl_6hr_1min_1216-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_1620[resized_refl_6hr_1min_1620<=refl_min]=refl_min
    resized_refl_6hr_1min_1620[resized_refl_6hr_1min_1620>=refl_max]=refl_max
    resized_refl_6hr_1min_1620_norm = (resized_refl_6hr_1min_1620-refl_min) / (refl_max-refl_min)
    
    resized_refl_6hr_1min_2024[resized_refl_6hr_1min_2024<=refl_min]=refl_min
    resized_refl_6hr_1min_2024[resized_refl_6hr_1min_2024>=refl_max]=refl_max
    resized_refl_6hr_1min_2024_norm = (resized_refl_6hr_1min_2024-refl_min) / (refl_max-refl_min)


    resized_refl_6hr_1min_norm=np.empty((int(24/hour_increment),240,95))
    resized_refl_6hr_1min_norm[0,:,:]=resized_refl_6hr_1min_04_norm 
    resized_refl_6hr_1min_norm[1,:,:]=resized_refl_6hr_1min_48_norm 
    resized_refl_6hr_1min_norm[2,:,:]=resized_refl_6hr_1min_812_norm 
    resized_refl_6hr_1min_norm[3,:,:]=resized_refl_6hr_1min_1216_norm 
    resized_refl_6hr_1min_norm[4,:,:]=resized_refl_6hr_1min_1620_norm 
    resized_refl_6hr_1min_norm[5,:,:]=resized_refl_6hr_1min_2024_norm 

    resized_refl_6hr_1min_norm[np.isnan(resized_refl_6hr_1min_norm)]=0.

    return resized_refl_6hr_1min_norm,hour_increment,hours


def input_hour_increment_get_prediction_6hr(time_utc_cldtype,reflectivity_best_estimate1):
    
    hour_increment=6
    hours=np.array([0,6,12,18,24]) # this can change based on interest


    arscl_loc_1=np.where( (time_utc_arscl1>=0) & (time_utc_arscl1<6)) 
    arscl_loc_2=np.where( (time_utc_arscl1>=6) & (time_utc_arscl1<12)) 
    arscl_loc_3=np.where( (time_utc_arscl1>=12) & (time_utc_arscl1<18)) 
    arscl_loc_4=np.where( (time_utc_arscl1>=18) & (time_utc_arscl1<24)) 


    reflectivity_best_estimate1_6hr_06=reflectivity_best_estimate1[arscl_loc_1[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_612=reflectivity_best_estimate1[arscl_loc_2[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_1218=reflectivity_best_estimate1[arscl_loc_3[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_1824=reflectivity_best_estimate1[arscl_loc_4[0],:][:,loc_height[0]]


    refl_6hr_1min_06=mins_array(reflectivity_best_estimate1_6hr_06, avg_time_interval)
    refl_6hr_1min_612=mins_array(reflectivity_best_estimate1_6hr_612, avg_time_interval)
    refl_6hr_1min_1218=mins_array(reflectivity_best_estimate1_6hr_1218, avg_time_interval)
    refl_6hr_1min_1824=mins_array(reflectivity_best_estimate1_6hr_1824, avg_time_interval)


    refl_6hr_1min_06[np.isnan(refl_6hr_1min_06)] = -99
    refl_6hr_1min_612[np.isnan(refl_6hr_1min_612)] = -99
    refl_6hr_1min_1218[np.isnan(refl_6hr_1min_1218)] = -99
    refl_6hr_1min_1824[np.isnan(refl_6hr_1min_1824)] = -99

    # Calculate the zoom factor for each dimension
    zoom_factor = (
        240 / (60*hour_increment),  # Zoom factor along the second dimension
        1,  # No zoom along the third dimension (95)
    )

    resized_refl_6hr_1min_06 = ndimage.zoom(refl_6hr_1min_06, zoom=zoom_factor)
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06<-60] = np.nan

    resized_refl_6hr_1min_612 = ndimage.zoom(refl_6hr_1min_612, zoom=zoom_factor)
    resized_refl_6hr_1min_612[resized_refl_6hr_1min_612<-60] = np.nan

    resized_refl_6hr_1min_1218 = ndimage.zoom(refl_6hr_1min_1218, zoom=zoom_factor)
    resized_refl_6hr_1min_1218[resized_refl_6hr_1min_1218<-60] = np.nan

    resized_refl_6hr_1min_1824 = ndimage.zoom(refl_6hr_1min_1824, zoom=zoom_factor)
    resized_refl_6hr_1min_1824[resized_refl_6hr_1min_1824<-60] = np.nan


    # Scale 
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06<=refl_min]=refl_min
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06>=refl_max]=refl_max
    resized_refl_6hr_1min_06_norm = (resized_refl_6hr_1min_06-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_612[resized_refl_6hr_1min_612<=refl_min]=refl_min
    resized_refl_6hr_1min_612[resized_refl_6hr_1min_612>=refl_max]=refl_max
    resized_refl_6hr_1min_612_norm = (resized_refl_6hr_1min_612-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_1218[resized_refl_6hr_1min_1218<=refl_min]=refl_min
    resized_refl_6hr_1min_1218[resized_refl_6hr_1min_1218>=refl_max]=refl_max
    resized_refl_6hr_1min_1218_norm = (resized_refl_6hr_1min_1218-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_1824[resized_refl_6hr_1min_1824<=refl_min]=refl_min
    resized_refl_6hr_1min_1824[resized_refl_6hr_1min_1824>=refl_max]=refl_max
    resized_refl_6hr_1min_1824_norm = (resized_refl_6hr_1min_1824-refl_min) / (refl_max-refl_min)


    resized_refl_6hr_1min_norm=np.empty((4,240,95))
    resized_refl_6hr_1min_norm[0,:,:]=resized_refl_6hr_1min_06_norm 
    resized_refl_6hr_1min_norm[1,:,:]=resized_refl_6hr_1min_612_norm 
    resized_refl_6hr_1min_norm[2,:,:]=resized_refl_6hr_1min_1218_norm 
    resized_refl_6hr_1min_norm[3,:,:]=resized_refl_6hr_1min_1824_norm 


    resized_refl_6hr_1min_norm[np.isnan(resized_refl_6hr_1min_norm)]=0.


    return resized_refl_6hr_1min_norm,hour_increment,hours

def input_hour_increment_get_prediction_12hr(time_utc_cldtype,reflectivity_best_estimate1):
    
    hour_increment=12
    hours=np.array([0,12,24])

    arscl_loc_1=np.where( (time_utc_arscl1>=0) & (time_utc_arscl1<12)) 
    arscl_loc_2=np.where( (time_utc_arscl1>=12) & (time_utc_arscl1<24)) 

    reflectivity_best_estimate1_6hr_06=reflectivity_best_estimate1[arscl_loc_1[0],:][:,loc_height[0]]
    reflectivity_best_estimate1_6hr_612=reflectivity_best_estimate1[arscl_loc_2[0],:][:,loc_height[0]]


    refl_6hr_1min_06=mins_array(reflectivity_best_estimate1_6hr_06, avg_time_interval)
    refl_6hr_1min_612=mins_array(reflectivity_best_estimate1_6hr_612, avg_time_interval)


    refl_6hr_1min_06[np.isnan(refl_6hr_1min_06)] = -99
    refl_6hr_1min_612[np.isnan(refl_6hr_1min_612)] = -99

    # Calculate the zoom factor for each dimension
    zoom_factor = (
        240 / (60*hour_increment),  # Zoom factor along the second dimension
        1,  # No zoom along the third dimension (95)
    )

    resized_refl_6hr_1min_06 = ndimage.zoom(refl_6hr_1min_06, zoom=zoom_factor)
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06<-60] = np.nan

    resized_refl_6hr_1min_612 = ndimage.zoom(refl_6hr_1min_612, zoom=zoom_factor)
    resized_refl_6hr_1min_612[resized_refl_6hr_1min_612<-60] = np.nan


    # Scale 
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06<=refl_min]=refl_min
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06>=refl_max]=refl_max
    resized_refl_6hr_1min_06_norm = (resized_refl_6hr_1min_06-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_612[resized_refl_6hr_1min_612<=refl_min]=refl_min
    resized_refl_6hr_1min_612[resized_refl_6hr_1min_612>=refl_max]=refl_max
    resized_refl_6hr_1min_612_norm = (resized_refl_6hr_1min_612-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_norm=np.empty((int(24/hour_increment),240,95))
    resized_refl_6hr_1min_norm[0,:,:]=resized_refl_6hr_1min_06_norm 
    resized_refl_6hr_1min_norm[1,:,:]=resized_refl_6hr_1min_612_norm 


    resized_refl_6hr_1min_norm[np.isnan(resized_refl_6hr_1min_norm)]=0.


    return resized_refl_6hr_1min_norm,hour_increment,hours



def input_hour_increment_get_prediction_24hr(time_utc_cldtype,reflectivity_best_estimate1):
    
    hour_increment=24
    hours=np.array([0,24])

    arscl_loc_1=np.where( (time_utc_arscl1>=0) & (time_utc_arscl1<24)) 
    
    reflectivity_best_estimate1_6hr_06=reflectivity_best_estimate1[arscl_loc_1[0],:][:,loc_height[0]]

    refl_6hr_1min_06=mins_array(reflectivity_best_estimate1_6hr_06, avg_time_interval)


    refl_6hr_1min_06[np.isnan(refl_6hr_1min_06)] = -99

    # Calculate the zoom factor for each dimension
    zoom_factor = (
        240 / (60*hour_increment),  # Zoom factor along the second dimension
        1,  # No zoom along the third dimension (95)
    )

    resized_refl_6hr_1min_06 = ndimage.zoom(refl_6hr_1min_06, zoom=zoom_factor)
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06<-60] = np.nan



    # Scale
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06<=refl_min]=refl_min
    resized_refl_6hr_1min_06[resized_refl_6hr_1min_06>=refl_max]=refl_max
    resized_refl_6hr_1min_06_norm = (resized_refl_6hr_1min_06-refl_min) / (refl_max-refl_min)

    resized_refl_6hr_1min_norm=np.empty((int(24/hour_increment),240,95))
    resized_refl_6hr_1min_norm[0,:,:]=resized_refl_6hr_1min_06_norm 


    resized_refl_6hr_1min_norm[np.isnan(resized_refl_6hr_1min_norm)]=0.


    return resized_refl_6hr_1min_norm,hour_increment,hours


def post_process(resized_refl_6hr_1min_norm,hour_increment,hours):
    
    
    with tf.device("CPU"):
        X_test_F = tf.convert_to_tensor(resized_refl_6hr_1min_norm)
        X_test_FF=preprocess_image(X_test_F)

    y_pred = model.predict(X_test_FF)
    # Assuming the predictions are probabilities, convert to class labels
    y_pred_labels = np.argmax(y_pred, axis=-1)

    num_images, num_time, num_height = y_pred_labels.shape


    # Initialize y_pred_labels_post with the same shape
    y_pred_labels_post = np.copy(y_pred_labels)

    # Iterate through each image and time slice
    for i in range(num_images):
        for t in range(num_time):
            column = y_pred_labels[i, t, :]
            mask_not_one = column != 1
            values_not_one = column[mask_not_one]
            if values_not_one.size > 0:
                counts = np.bincount(values_not_one, minlength=4)  # Ensure there are counts for 0 to 3
                counts[1] = -1  # Ignore occurrences of 1 by setting its count to -1
                most_frequent = np.argmax(counts)
                y_pred_labels_post[i, t, mask_not_one] = most_frequent
                
    # process for plotting            
    new_time_data = []  # This will store the concatenated time data
    X_test_FF_merge = np.empty((0, 128))  # This will eventually be (6*128, 128)
    y_test_labels_merge = np.empty((0, 128))  # This will eventually be (6*128, 128)
    y_pred_labels_post_merge = np.empty((0, 128))  # This will eventually be (6*128, 128)


    # Iterate through the selected indices to process each case
    for i in range(int(24/hour_increment)):

        num_points_128 = 128
        time_128 = np.linspace(hours[i],hours[i+1], num_points_128)

        # Append the new time data
        new_time_data.extend(time_128)

        # Concatenate the image data along the first axis
        y_pred_labels_post_merge = np.concatenate((y_pred_labels_post_merge, y_pred_labels_post[i]), axis=0)


    # Convert new_time_data to an array if necessary
    y_pred_labels_post_merge = np.array(y_pred_labels_post_merge)


    return new_time_data,y_pred_labels_post_merge


#########some pre-set parameters######

refl_min=-60.
refl_max=40.

avg_time_interval=60. # seconds
ratio=(4./24)

class_weights = [3.42532998, 0.32800352 ,5.42562766 ,2.10526705]  # Assuming 4 classes with varying representations (pre-calculated) 

height_95=np.arange(95)*0.03+0.16
num_points_128 = 128
start=np.nanmin(height_95)
end_95=np.nanmax(height_95)
height_128 = np.linspace(start, end_95, num_points_128)

#### Load the model####
custom_loss = weighted_categorical_crossentropy(class_weights)
from tensorflow.keras.models import load_model
model = load_model('./v1_best_model_bs16_dropout_GridSearch.keras', custom_objects={'loss': custom_loss})


###loop through cldtype data###

cldtype_pwd='/ena/enacldtypeC1.c1/'
cldtype_file=f'{cldtype_pwd}/*'
output_cldtype_file=glob.glob(cldtype_file)

num_files=len(output_cldtype_file)

outdir=f'./'

for iii in np.arange(num_files):
      
    date_target=output_cldtype_file[iii].split('enacldtypeC1.c1.')[-1][0:8]

    # Read cldtype
    time_utc_cldtype,cloudtype,qc_cloudtype,height_cldtype,cld_base,cld_top=read_cldtype(date_target)
    
    loc=np.where( (time_utc_cldtype>=0) & (time_utc_cldtype<24)) 
    ct0=np.where( (cloudtype[loc[0],:]>=2) & (cloudtype[loc[0],:]<=5) )[0].shape[0] # "mid level"
    ct5=np.where(  cloudtype[loc[0],:]==1)[0].shape[0] # "low level"


    ### simple screening, could ignore this "if" step ####
    if (ct0<1440*ratio) & (ct5>=0) :

        print(iii,date_target)
        
        if not any(date_target in f for f in os.listdir(outdir)):
            
            print(iii,date_target,'saving...')

            time_utc_arscl1,arscl_height1,reflectivity_best_estimate1=read_arscl_data(date_target)


            loc_height=np.where(arscl_height1< 3000)
            arscl_height=arscl_height1[loc_height]


            resized_refl_4hr_1min_norm,hour_increment_4hr,hours_4hr=input_hour_increment_get_prediction_4hr(time_utc_cldtype,reflectivity_best_estimate1)
            resized_refl_6hr_1min_norm,hour_increment_6hr,hours_6hr=input_hour_increment_get_prediction_6hr(time_utc_cldtype,reflectivity_best_estimate1)
            resized_refl_12hr_1min_norm,hour_increment_12hr,hours_12hr=input_hour_increment_get_prediction_12hr(time_utc_cldtype,reflectivity_best_estimate1)

            #print('start prediction and post-processing')

            new_time_data_4hr,y_pred_labels_post_merge_4hr=post_process(resized_refl_4hr_1min_norm,hour_increment_4hr,hours_4hr)
            new_time_data_6hr,y_pred_labels_post_merge_6hr=post_process(resized_refl_6hr_1min_norm,hour_increment_6hr,hours_6hr)
            new_time_data_12hr,y_pred_labels_post_merge_12hr=post_process(resized_refl_12hr_1min_norm,hour_increment_12hr,hours_12hr)





