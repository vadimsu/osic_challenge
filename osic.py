import numpy as np
import pandas as pd

import os

#import matplotlib.pyplot as plt
#from matplotlib import cm
import pydicom
import scipy.ndimage
from skimage import measure
from plotly import figure_factory as FF
from plotly.offline import iplot
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer, InputLayer, Dense, Conv3D, Flatten , Conv3DTranspose, LayerNormalization,MaxPooling3D
#from tensorflow.python.keras.layers import Activation, Add, MaxPooling2D,Dropout,AveragePooling2D
from tensorflow.python.keras.initializers import GlorotUniform
from tensorflow.python.keras.layers import Activation
from sklearn.preprocessing import OneHotEncoder
print(tf.config.experimental.list_physical_devices('GPU'))
#A dataset is here.
#The structure is
#train/
#     train.csv
#     DICOM folders (per patient)
#test/
#     test.csv
#     DICOM folders (per patient)
#TRAIN_ROOT = #Path to your training folder
#TEST_ROOT = #Path to your testing folder
TRAIN_ROOT = '../input/osic-pulmonary-fibrosis-progression/train'
TEST_ROOT = '../input/osic-pulmonary-fibrosis-progression/test'
TRAIN_DATASET_PATH = TRAIN_ROOT + '.csv'

tf.keras.backend.set_floatx('float64')

min_scan_count = 0
max_scan_count = 0
def get_scan_count_min():
    global min_scan_count
    if min_scan_count == 0:
        subfolders= [f.path for f in os.scandir(TRAIN_ROOT) if f.is_dir()]
        scans_counts = []
        for subf in subfolders:
            scans_counts.append(len([f.path for f in os.scandir(subf) if f.is_file()]))
        min_scan_count = int(round(np.min(scans_counts)))
    return min_scan_count

def get_scan_count_max():
    global max_scan_count
    if max_scan_count == 0:
        subfolders= [f.path for f in os.scandir(TRAIN_ROOT) if f.is_dir()]
        scans_counts = []
        for subf in subfolders:
            scans_counts.append(len([f.path for f in os.scandir(subf) if f.is_file()]))
        max_scan_count = int(round(np.min(scans_counts)))
    return max_scan_count

get_scan_count_min()
get_scan_count_max()

#DICOM Scans processing routines
cum_slice_thickness = 0
cum_slice_thickness_cnt = 0
def load_scan(path):
    global cum_slice_thickness
    global cum_slice_thickness_cnt
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        cum_slice_thickness += slice_thickness
        cum_slice_thickness_cnt += 1
    except:
        try:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            cum_slice_thickness_cnt += slice_thickness
            cum_slice_thickness_cnt += 1
        except:
            if cum_slice_thickness_cnt > 0:
                slice_thickness = cum_slice_thickness/cum_slice_thickness_cnt
            else:
                slice_thickness = 1
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    slices_min = 0
    slices_max = len(slices)
    if len(slices) > max_scan_count:
        slices_min = int((len(slices) - max_scan_count)/2)
        slices_max = int(len(slices) - max_scan_count)
    return np.stack([s.pixel_array for s in slices[slices_min:slices_max]]), slices[0]

def bounding_box(image):
    mid_img = image[int(image.shape[0] / 2)]
    same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
    same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
    if same_first_col and same_first_row:
        return True
    else:
        return False
    
def crop_bounding_box(image):
    mid_img = image[int(image.shape[0] / 2)]
    r_min, r_max = None, None
    c_min, c_max = None, None
    for row in range(mid_img.shape[0]):
        if not (mid_img[row, :] == mid_img[0, 0]).all() and r_min is None:
            r_min = row
        if (mid_img[row, :] == mid_img[0, 0]).all() and r_max is None and r_min is not None:
            r_max = row
            break

    for col in range(mid_img.shape[1]):
        if not (mid_img[:, col] == mid_img[0, 0]).all() and c_min is None:
            c_min = col
        if (mid_img[:, col] == mid_img[0, 0]).all() and c_max is None and c_min is not None:
            c_max = col
            break

    return image[:, r_min:r_max, c_min:c_max]

def get_pixels_hu(image, metadata):
    image = image.astype(np.int16)
    # Convert to Hounsfield units (HU)
    intercept = metadata.RescaleIntercept
    slope = metadata.RescaleSlope
    
#    if slope != 1:
#        image = slope * image.astype(np.float64)
#        image = image.astype(np.int16)
    image = (image*slope).astype(np.int16)        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.float)

def resample(image, metadata, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([metadata.SliceThickness] + list(metadata.PixelSpacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def get_morphological_mask(image):
    m = image < -500
    m = clear_border(m)
    m = label(m)
    areas = [r.area for r in regionprops(m)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(m):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    m[coordinates[0], coordinates[1]] = 0
    return m > 0

def make_lungmask(image):
    for slice_id in range(image.shape[0]):
        m = get_morphological_mask(image[slice_id])
        image[slice_id][m == False] = image[slice_id].min()
    return image

def normalize_and_center_image(image):
    #return (image - np.mean(image))/np.std(image)
    return image/255.0

def make_mesh(image, threshold=-300, step_size=1):

    print("Transposing surface")
    p = image.transpose(2,1,0)
    
    print ("Calculating surface")
#    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 
    verts, faces = measure.marching_cubes_classic(p) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print("Drawing")
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)
    

quantiles = [0.2, 0.5, 0.8]

bad_ids = ['ID00128637202219474716089', 'ID00052637202186188008618', 'ID00026637202179561894768','ID00011637202177653955184', 'train.csv', 'test.csv']

#every patient has a different number of scans
#get median
def get_scan_count_median():
    subfolders= [f.path for f in os.scandir(TRAIN_ROOT) if f.is_dir()]
    scans_counts = []
    for subf in subfolders:
        scans_counts.append(len([f.path for f in os.scandir(subf) if f.is_file()]))
    return int(round(np.median(scans_counts)))

#data generator to get train/test batches
class DataGenerator(Sequence):
    def __init__(self, original_dim, path, is_train, sex_cat_encoder, smoking_status_cat_encoder, use_cache=True):
        super(DataGenerator,self).__init__()
        self.image_cache = {}
        self.use_cache = use_cache
        self.path = path
        if is_train:
            self.dataset = pd.read_csv(path+'.csv')
        else:
            self.dataset = pd.read_csv(path+'.csv')
        #create initial week feature
        uniques = self.dataset.Patient.unique()
        min_weeks = self.dataset.groupby('Patient')['Weeks'].min()
        for i in range(len(uniques)):
            initial_fvc = self.dataset.loc[(self.dataset.Patient == uniques[i]) & (self.dataset.Weeks == min_weeks[i]),'FVC']
            initial_percent = self.dataset.loc[(self.dataset.Patient == uniques[i]) & (self.dataset.Weeks == min_weeks[i]),'Percent']
            self.dataset.loc[self.dataset.Patient == uniques[i],'InitialPercent'] = initial_percent.iloc[0]
            self.dataset.loc[self.dataset.Patient == uniques[i],'InitialFVC'] = initial_fvc.iloc[0]
        #normalize the numeric data
        self.dataset['Age'] = (self.dataset['Age'] - self.dataset['Age'].min())/(self.dataset['Age'].max() - self.dataset['Age'].min())
        self.dataset['Weeks'] = (self.dataset['Weeks'] - self.dataset['Weeks'].min())/(self.dataset['Weeks'].max() - self.dataset['Weeks'].min())
        #self.dataset['InitialPercent'] = (self.dataset['InitialPercent'] - self.dataset['InitialPercent'].min())/(self.dataset['InitialPercent'].max() - self.dataset['InitialPercent'].min())
        #self.dataset['InitialFVC'] = (self.dataset['InitialFVC'] - self.dataset['InitialFVC'].min())/(self.dataset['InitialFVC'].max() - self.dataset['InitialFVC'].min())
        #encode categorical features
        self.sex_enc = sex_cat_encoder
        if is_train:
            self.sex = pd.DataFrame(self.sex_enc.fit_transform(self.dataset.Sex.to_numpy().reshape(-1, 1)),index=self.dataset['Patient'])
        else:
            self.sex = pd.DataFrame(self.sex_enc.transform(self.dataset.Sex.to_numpy().reshape(-1, 1)),index=self.dataset['Patient'])
        self.smoking_enc = smoking_status_cat_encoder
        if is_train:
            self.smoking_status = pd.DataFrame(self.smoking_enc.fit_transform(self.dataset.SmokingStatus.to_numpy().reshape(-1, 1)),index=self.dataset['Patient'])
        else:
            self.smoking_status = pd.DataFrame(self.smoking_enc.transform(self.dataset.SmokingStatus.to_numpy().reshape(-1, 1)),index=self.dataset['Patient'])
        #list all patients
        self.patients = [dirname for dirname in os.listdir(path) if dirname != '.' and dirname != '..']
        self.idx = 0
        self.image_shape = original_dim
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset.Patient.unique())
    def _get_target(self,patient_id):
        y = self.dataset.loc[self.dataset['Patient'] == patient_id]['FVC'].to_numpy().reshape(-1,1).astype('float64')
        return y
    def _get_tab_data(self,patient_id):
        patient_tab = self.dataset.index[self.dataset['Patient'] == patient_id].to_list()
        sex_np = self.sex.iloc[patient_tab].to_numpy()
        smoking_np = self.smoking_status.iloc[patient_tab].to_numpy()
        weeks = self.dataset.loc[self.dataset['Patient'] == patient_id]['Weeks'].to_numpy().reshape(-1,1)
        #print('weeks ',weeks)
        age = self.dataset.loc[self.dataset['Patient'] == patient_id]['Age'].to_numpy().reshape(-1,1)
        #percent = self.dataset.loc[self.dataset['Patient'] == patient_id]['Percent'].to_numpy().reshape(-1,1)
        initial_percent = self.dataset.loc[self.dataset['Patient'] == patient_id]['InitialPercent'].to_numpy().reshape(-1,1)
        initial_fvc = self.dataset.loc[self.dataset['Patient'] == patient_id]['InitialFVC'].to_numpy().reshape(-1,1)
        tab_data = np.concatenate([sex_np, smoking_np, weeks, age, initial_fvc, initial_percent],axis=1)
        return tab_data
    
    def _get_tab_data_test(self,patient_id):
        prev_item = None
        for i in range(-12,134,1):
            patient_tab = self.dataset.index[self.dataset['Patient'] == patient_id].to_list()
            sex_np = self.sex.iloc[patient_tab].to_numpy()
            smoking_np = self.smoking_status.iloc[patient_tab].to_numpy()
            weeks = np.array([i]).reshape(-1,1)
         #   print('weeks ',weeks)
            age = self.dataset.loc[self.dataset['Patient'] == patient_id]['Age'].to_numpy().reshape(-1,1)
            percent = self.dataset.loc[self.dataset['Patient'] == patient_id]['Percent'].to_numpy().reshape(-1,1)
            initial_percent = self.dataset.loc[self.dataset['Patient'] == patient_id]['InitialPercent'].to_numpy().reshape(-1,1)
            initial_fvc = self.dataset.loc[self.dataset['Patient'] == patient_id]['InitialFVC'].to_numpy().reshape(-1,1)
            tab_data = np.concatenate([sex_np, smoking_np, weeks, age, initial_fvc, initial_percent],axis=1)
            if prev_item is not None:
                prev_item = np.vstack((prev_item,tab_data))
            else:
                prev_item = tab_data
        #print('prev_item ',prev_item)
        return prev_item
    
    def _prepare_image_data(self, dirname):
        patient,metadata = load_scan(dirname)
        patient = crop_bounding_box(patient)
        imgs = get_pixels_hu(patient,metadata)
        #imgs_after_resamp, spacing = resample(imgs, metadata, [1,1,1])
        imgs_after_resamp = resize(imgs,self.image_shape,anti_aliasing = True)
        if imgs.shape[0] > get_scan_count_max():
            slice_count = get_scan_count_max()
        else:
            slice_count = imgs.shape[0]
        imgs = resize(imgs,
                      (slice_count, #imgs.shape[0],
                       self.image_shape[0],
                       self.image_shape[1],
                       self.image_shape[2]),anti_aliasing = True)
        imgs_after_resamp = make_lungmask(imgs_after_resamp)
        imgs_after_resamp = normalize_and_center_image(imgs_after_resamp)
        return imgs_after_resamp

    def __getitem__(self,idx):
        if not self.is_train:
            if self.idx >= len(self.dataset.Patient.unique()):
                raise StopIteration
            
        while self.patients[self.idx] in bad_ids:
            self.idx += 1
            if self.idx == len(self.patients):
                self.idx = 0
        patient_id = self.patients[self.idx]
        dirname = os.path.join(self.path, patient_id)
        if self.use_cache and patient_id in self.image_cache:
            imgs_data = self.image_cache[patient_id]
        else:
            #prepare image data
            imgs_data = self._prepare_image_data(dirname)
            if self.use_cache:
                self.image_cache[patient_id] = imgs_data
            
        #prepare tabular data
        if self.is_train:
            tab_data = self._get_tab_data(patient_id)
        else:
            tab_data = self._get_tab_data_test(patient_id)
        #print('tab_data ',tab_data)
        if self.is_train:
            y = self._get_target(patient_id)
        else:
            y = patient_id
        self.idx = self.idx + 1
        if self.is_train:
            if self.idx == len(self.dataset.Patient.unique()):
                self.idx = 0
        return imgs_data.reshape((imgs_data.shape[0],imgs_data.shape[1],imgs_data.shape[2],1,1)), tab_data, y, patient_id

#model's outer layer        
class OSIC_outer(Layer):
    def __init__(self, intermediate_dim=64, tab_dim=6, name="outer", **kwargs):
        super(OSIC_outer, self).__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim+tab_dim, kernel_initializer='normal')
        self.dense_out = [Dense(1,kernel_initializer='normal') for i in range(len(quantiles))]

    def call(self, inputs,training=False):
        x = tf.reshape(inputs, (1,-1))
        #print('shape=',inputs.shape, 'x=',x.shape)
        x = self.dense_proj(x,training=training)
        output = [out(x,training=training) for out in self.dense_out]
        return output
    
    def print_weigths(self):
        print(self.dense_out.weights)

#model's image processing layer
class OSIC_Image(Layer):
    def __init__(self, original_dim, intermediate_dim=64, name="image", **kwargs):
        super(OSIC_Image, self).__init__(name=name, **kwargs)
        self.layers = []
        self.layers.append(InputLayer(input_shape=original_dim))
        self.layers.append(Conv3D(filters=8,kernel_size=5,strides=3,padding="same", kernel_initializer=GlorotUniform(seed=0),input_shape=original_dim))
        self.layers.append(LayerNormalization())
        self.layers.append(Activation('elu'))
        self.layers.append(Conv3D(filters=16,kernel_size=2,strides=2,padding="same", kernel_initializer=GlorotUniform(seed=0)))
        self.layers.append(LayerNormalization())
        self.layers.append(Activation('elu'))
        self.layers.append(Conv3D(filters=32,kernel_size=2,strides=1,padding="same", kernel_initializer=GlorotUniform(seed=0)))
        self.layers.append(LayerNormalization())
        self.layers.append(Activation('elu'))
        self.layers.append(Conv3D(filters=64,kernel_size=2,strides=1,padding="same", kernel_initializer=GlorotUniform(seed=0)))
        self.layers.append(LayerNormalization())
        self.layers.append(Activation('elu'))
#        self.layers.append(Conv3DTranspose(32, 2, 1))
#        self.layers.append(LayerNormalization())
#        self.layers.append(Activation('elu'))
#        self.layers.append(Conv3DTranspose(16, 2, 1))
#        self.layers.append(LayerNormalization())
        #self.layers.append(Conv3D(filters=1,kernel_size=5,strides=4,kernel_initializer=GlorotUniform(seed=0)))
#        self.layers.append(Conv3D(filters=2,kernel_size=1,activation="softmax", kernel_initializer=GlorotUniform(seed=0)))
        self.layers.append(Dense(64))
        self.layers.append(LayerNormalization())

    def call(self, inputs,training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x,training=training)
        return x
    def print_weights(self):
        print(self.dense_output.weights)

#model
class OSIC_Model(Model):

    def __init__(self, original_dim, intermediate_dim, name='OSIC_Model', **kwargs):
        super(OSIC_Model, self).__init__(name=name, **kwargs)
        #tab dimensions default is 5 which is all tabular features (Age, Sex, Smoking status, percent, weeks)
        self.outer = OSIC_outer(intermediate_dim=intermediate_dim, tab_dim=6)
        #self.after_image = Dense(4,kernel_initializer='normal',activation='linear')
        #self.after_image = Dense(4,kernel_initializer='normal')
        self.image = OSIC_Image(original_dim, intermediate_dim=intermediate_dim)
        
    def quantile_loss(self, preds, target, total_loss, all_logits):
        for q, quantile in enumerate(quantiles):
            error = tf.subtract(target, preds[q])
            loss = tf.reduce_mean(tf.maximum(quantile*error, (quantile-1)*error), axis=-1)
            total_loss.append(loss)
            all_logits.append(preds[q])
    def call(self, inputs, training=False):
        #feed image features layer
        image_features = self.image(inputs[0],training)
        image_features = tf.reshape(image_features, (1,-1))
        #image_features = self.after_image(image_features)
        all_logits = []
        total_loss = []
        #iterate patient's weeks
        for i in range(inputs[1].shape[0]):
            #concatenate features with tabular
            outer_input = tf.concat([image_features, tf.reshape(inputs[1][i],(1,-1))],axis=1)
            #feed outer layer
            logits = self.outer(outer_input,training)
            if training:
                #get quatile loss and logits
                self.quantile_loss(logits, inputs[2][i], total_loss, all_logits)
            else:
                #prediction - get logits only
                for q, quantile in enumerate(quantiles):
                    all_logits.append(logits[q])
        #calculate loss mean
        if training:
            combined_loss = tf.reduce_mean(tf.add_n(total_loss))
            self.add_loss(combined_loss)
        self.all_logits = np.concatenate(all_logits,axis=1)
        #shape as (batch size, number of quantiles)
        return self.all_logits.reshape((inputs[1].shape[0], -1))
    def print_weights(self):
        self.outer.print_weigths()
        self.image.print_weights()

#experimenting with different image sizes
#the bigger is the image, the slower is the training
#so far cannot see a difference in accuracy between
#512 and 128
    
#original_dim = (get_scan_count_median(),512,512)
original_dim = (get_scan_count_median(),64,64,1)

#create a model
model = OSIC_Model(original_dim, 1)

#learning rate scheduler
initial_learning_rate = 1e-3
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate,
#    decay_steps=100000,
#    decay_rate=0.96,
#    staircase=True)
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate,20)
#optimizer = tf.keras.optimizers.RMSprop()
#optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
optimizer = tf.keras.optimizers.Adamax()
mse_loss_fn = tf.keras.losses.MeanSquaredError()

_metric = tf.keras.metrics.Mean()

def osic_metric(preds, targets):
    sigma = preds[:, 2] - preds[:, 0]
    sigma[sigma < 70] = 70
    delta = np.absolute(preds[:, 1] - targets)
    delta[delta > 1000] = 1000
    return -np.sqrt(2) * delta / sigma - np.log(np.sqrt(2) * sigma)

def quantile_loss(preds, target):
    total_loss = []
    for i in range(preds.shape[0]):
        for q, quantile in enumerate(quantiles):
            error = tf.subtract(target[i], preds[i][q])
            loss = tf.reduce_mean(tf.maximum(quantile*error, (quantile-1)*error), axis=-1)
            total_loss.append(loss)
    combined_loss = tf.reduce_mean(tf.add_n(total_loss))
    return combined_loss

epochs = 100
#uncomment to run from pre-trained, comment to run from scratch
model.load_weights('/kaggle/input/weights20/3_conv_layers1_weights')
sex_cat_encoder = OneHotEncoder(sparse=False)
smoking_status_cat_encoder = OneHotEncoder(sparse=False)
def model_train(name):
    train_dataset = DataGenerator(original_dim,TRAIN_ROOT, True,sex_cat_encoder,smoking_status_cat_encoder)
    # Iterate over epochs.
    history = []
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        epoch_start = datetime.now()

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                #feed the model
                logits = model(x_batch_train, training=True)
                print('logits ',logits)
                print('true patient ',x_batch_train[3], x_batch_train[2])
                print('model losses ',model.losses)
                #get the loss
                loss = quantile_loss(logits, x_batch_train[2]) + sum(model.losses) +mse_loss_fn(logits,x_batch_train[2])
                print('loss ',loss)
                with tape.stop_recording():
                    #calculate gradients
                    grads = tape.gradient(loss, model.trainable_variables)
            #apply gradients
            optimizer.apply_gradients(
                (grad, var) 
                for (grad, var) in zip(grads, model.trainable_variables) 
                if grad is not None
            )
            #calculate metric
            metric = osic_metric(logits, x_batch_train[2])
            _metric(metric)

            #if step % 100 == 0:
            print("step %d: metric = %.4f" % (step, _metric.result()))
            #history.append(_metric.result())
        model.save_weights(name+'_weights')
        tf.keras.backend.clear_session()
        epoch_finish = datetime.now()
        print("start =", epoch_start, ' finish = ',epoch_finish)
                
def model_predict():
    test_dataset = DataGenerator(original_dim,TEST_ROOT, False, sex_cat_encoder,smoking_status_cat_encoder)
    predict = pd.DataFrame()
    weeks = []
    fvc = []
    confidence = []
    for _, x_batch_test in enumerate(test_dataset):
        logits = model(x_batch_test)
        #import pdb;pdb.set_trace()
        #logits = logits[0]
        #print('logits ',logits)
        #print('x_batch_test[2] ',x_batch_test[2], ' week ',' ',str(week))
        for week in range(logits.shape[0]):
            weeks.append(x_batch_test[2] + '_'+ str(week-12))
            fvc.append(logits[week,1])
            confidence.append(logits[week,2] - logits[week,0])
    predict['Patient_Week'] = weeks
    predict['FVC'] = fvc
    predict['Confidence'] = confidence
    predict.to_csv('submission.csv', index=False)
    
def explore_dataset(path):
    cols = ['Age', 'Sex', 'SmokingStatus', 'Percent', 'FVC']
    ds = pd.read_csv(path)
    print(ds)
    for col in cols:
        print(col,': ')
        if np.issubdtype(ds[col].dtype, np.number):
            print('max=', ds[col].max(), ' min=',ds[col].min(), ' mean=',ds[col].mean(),' stdev=',ds[col].std())
        else:
            for unique in ds[col].unique():
                print(unique, ' count=',ds[col].loc[ds[col] == unique].count())
        print('nan count',ds[col].isna().sum())

explore_dataset(TRAIN_DATASET_PATH)
from datetime import datetime
# current date and time
start = datetime.now()
model_train('3_conv_layers1')
finish = datetime.now()
print("start =", start, ' finish = ',finish)
model_predict()