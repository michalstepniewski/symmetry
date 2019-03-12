#usage python3.6 interpolation.py
# options
#  --ipath file_with_input_mask.nii.gz file in nii.gz format \
#  --opath file_to_output_interpolated_mask.nii
#  --itype contour or full_simplified
#   I thought it best for the input to be
#   (when transformed to numpy array) to be 3D array with
#   None (or np. nan values in slices where )
#  --direction 0,1,2 (depending on which axis to interpolate)
# python3.6 interpolation.py --ipath  ../../data/001_25_mask.nii --opath file_with_broken_mask.nii.gz --itype break_mask --direction 1
# python3.6 interpolation.py --ipath file_with_broken_mask.nii.gz  --opath file_to_output_interpolated_mask.nii.gz --itype contour --direction 1
import argparse
import os, numpy as np, nibabel as nib
from nibabel.testing import data_path
from scipy.interpolate import griddata
from tqdm import tqdm
import copy
import pandas as pd



#print(args.__dict__)

def prawidlowa_interpolacja_2D(img_mask_data, step=2):
    #nie uzywamy tego
    x_max, y_max, z_max = img_mask_data.shape
    grid_x, grid_y, grid_z = np.mgrid[0:x_max:1, 0:y_max:1,0:z_max:1]
    #points = np.random.rand(1000, 2)
    point_list = []
    for i in tqdm(range(x_max)):
        for j in range(0,y_max,2):
            for k in range(img_mask_data.shape[2]):
                point_list.append([i,j,k])
    points = np.array(point_list)
    values = img_mask_data[points[:,0],points[:,1],points[:,2]]
    grid_z2 = griddata(points, values, (grid_x, grid_y, grid_z), method='linear')
    #values = img_mask_data[points]
    #img_mask_data
    #del point_list
    grid_num = np.nan_to_num(grid_z2)
    show(grid_num[:,1,:])
    return grid_num

def simple_interpolation(img_mask_data):
    length_range = range(img_mask_data.shape[direction])
    filled = []
    for x in length_range:
        if direction ==0: slice_i = np.isfinite(img_mask_data[x,:,:]).any()
        elif direction ==1: slice_i = np.isfinite(img_mask_data[:,x,:]).any()
        elif direction ==2: slice_i = np.isfinite(img_mask_data[:,:,x]).any()
        filled.append(slice_i)
    #x, y, z = np.where(np.isfinite(img_mask_data_broken_1))
    filled = np.array(filled)
    interps = []
    for x in range(img_mask_data.shape[direction]):
        if not filled[x]:
            #finc closest filled neighbours
            base_img0 = np.zeros_like(img_mask_data[:,0,:])
            left_x = 0
            if x>0:
                left_xs = np.where(filled[:x])[0]
                if left_xs.shape[0]>0:
                    left_x = left_xs.max()
                    base_img0 = img_mask_data[:,left_x,:]
            base_img1 = np.zeros_like(img_mask_data[:,0,:])
            right_x = filled.shape[0]
            if x<filled.shape[0]:
                right_xs = np.where(filled[x:])[0]
                if right_xs.shape[0]>0:
                    right_x = right_xs.min()+x
                    base_img1 = img_mask_data[:,right_x,:]
            diff = (base_img1 - base_img0)
            diff_x = float(right_x-left_x)
            diff_atomic = diff/diff_x
            interp = (base_img0 + ((diff_atomic)*(x-left_x))
                      ).astype(int)
            interps.append(interp)
        else:
            interps.append(img_mask_data[:,x,:])
    return np.transpose(np.dstack(interps),(0,2,1))


def fill_interp_contours(img_mask_data, contours={},
                         offset=50):
    '''
    
    '''
    import copy
    interps = []
    for y in range(img_mask_data.shape[1]):
        slice_i = copy.deepcopy(img_mask_data[:,y,:])
        for mask in contours.keys():
         gorny_grid_z2, dolny_grid_z2 = [contours[mask][i] for i in [
                  'up','down']]
         zvals = gorny_grid_z2[:,y]
         zvals_d = dolny_grid_z2[:,y]
         for x in range(gorny_grid_z2.shape[0]):
            #print(zvals[x])
            if np.isfinite(zvals[x]):
                slice_i[int(round(zvals[x])), x+offset] = mask
            if np.isfinite(zvals_d[x]):
                slice_i[int(round(zvals_d[x])), x+offset] =mask
            if np.isfinite(zvals[x]) and np.isfinite(zvals_d[x]):
                for middle_z in range(int(round(zvals_d[x])),int(round(zvals[x]))):
                    slice_i[middle_z, x+offset] =mask
        interps.append(slice_i)
    return np.transpose(np.dstack(interps),(0,2,1))

def break_mask(img_mask_data, direction=1):
    img_mask_data_broken_1 = copy.deepcopy(img_mask_data)
    #zepsuc maske
    for y in tqdm(range(0,img_mask_data_broken_1.shape[direction],5)):
        for i in range(1,4+1):
            if direction==1:
                img_mask_data_broken_1[:,y+i,:] = np.nan
            elif direction==0:
                img_mask_data_broken_1[y+i,:,:] = np.nan
            elif direction==2:
                img_mask_data_broken_1[:,:,y+i] = np.nan
    return img_mask_data_broken_1


def prawidlowa_interpolacja_3D(img_mask_data, step=2):
    x_max, y_max, z_max = img_mask_data.shape
    grid_x, grid_y, grid_z = np.mgrid[0:x_max:1, 0:y_max:1,0:z_max:1]
    #points = np.random.rand(1000, 2)
    point_list = []
    for i in tqdm(range(x_max)):
        for j in range(0,y_max,2):
            for k in range(img_mask_data.shape[2]):
                point_list.append([i,j,k])
    points = np.array(point_list)
    values = img_mask_data[points[:,0],points[:,1],points[:,2]]
    grid_z2 = griddata(points, values, (grid_x, grid_y, grid_z), method='linear')
    #values = img_mask_data[points]
    #img_mask_data
    #del point_list
    grid_num = np.nan_to_num(grid_z2)
    show(grid_num[:,1,:])
    return grid_num


def prawidlowa_interpolacja_konturow_2D(img_mask_data, direction=1, x_offset=50,
                                        step=2, mask=2):
    '''
    uzywam teraz tego
    '''
    dolne_kontury = []
    gorne_kontury = []
    for z in tqdm(range(img_mask_data.shape[direction])):
        coronal_slice = img_mask_data[:,z,:]
        df = pd.DataFrame()
        df['y'], df['x'] = np.where(coronal_slice==mask)
        dolny_kontur = df.groupby('x').agg({'y':np.max}).reset_index()
        gorny_kontur = df.groupby('x').agg({'y':np.min}).reset_index()
        dolny_kontur['z'] = z
        gorny_kontur['z'] = z
        dolne_kontury.append(dolny_kontur)
        gorne_kontury.append(gorny_kontur)
        # teraz tak; dolny kontur:
        #    dla dolnego kontura y jest funkcja x i zeta (ktory jest druga wspolrzedna niepostrzezenie)
        # no i teraz z jest co piaty tylko# i musimy znalezc dla reszty zetow wartosci
    dolne_kontury = pd.concat(dolne_kontury)
    gorne_kontury = pd.concat(gorne_kontury)
    x_max = dolne_kontury['x'].max()
    z_max = img_mask_data.shape[direction]
    #jest pewien problem ze wzgledu na te kropki
    #grid musi byc w miare ciagly 'chyba'
    grid_x, grid_z = np.mgrid[x_offset:x_max:1, 0:z_max:1]
    points = gorne_kontury[['x','z']].values[::step]
    values = gorne_kontury['y'].values[::step]
    dolny_grid_z2 = griddata(points, values, (grid_x, grid_z),
                             method='linear')
    #pd.DataFrame(grid_z2[:,93],columns=['y']).reset_index().plot()
    x_max = gorne_kontury['x'].max()
    z_max = img_mask_data.shape[direction]
    #jest pewien problem ze wzgledu na te kropki
    #grid musi byc w miare ciagly 'chyba'
    grid_x, grid_z = np.mgrid[x_offset:x_max:1, 0:z_max:1]
    points = dolne_kontury[['x','z']].values[::step]
    values = dolne_kontury['y'].values[::step]
    gorny_grid_z2 = griddata(points, values, (grid_x, grid_z),
                             method='linear')
    #pd.DataFrame(grid_z2[:,93],columns=['y']).reset_index().plot()
    return dolny_grid_z2, gorny_grid_z2


def main(ipath='img_mask_data_broken_1.nii.gz',
         opath='img_mask_data_interpolated_1.nii.gz',direction=1,
         itype='full_simplified', offset=50):
    img_mask = nib.load(ipath)
    affine = img_mask.get_affine()
    img_mask_data = img_mask.get_fdata()#[:50,90:100,:50]
    del img_mask
    if itype == 'full_simplified':
        interp_mask = simple_interpolation(img_mask_data)
    elif itype == 'break_mask':
        interp_mask = break_mask(img_mask_data,
                                direction=direction)
    elif itype =='contour':
        j = {}
        for mask in [1,2]:
            j[mask] = {}
            j[mask]['down'], j[mask]['up'] = prawidlowa_interpolacja_konturow_2D(img_mask_data, direction=1,
                                        x_offset=offset, mask=mask)

        interp_mask = fill_interp_contours(img_mask_data, contours=j, offset=offset)
    fa = nib.Nifti1Image(interp_mask, affine)
    nib.save(fa, opath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolate hemisphere masks.')
    parser.add_argument('--ipath', help='input file with partial mask to be interpolated\
                     in .nii or .nii.gz format', type=str,
    default='img_mask_data_broken_1.nii.gz')
    parser.add_argument('--opath', help='output file to output interpolated mask into\
                      in .nii or .nii.gz format', type=str)
    parser.add_argument('--direction', help='0,1 or2 (depending on which axis to interpolate)',
                    type=int)
    parser.add_argument('--itype', help='type of interpolation to be performed: \
                    contour or full_simplified', type=str)
    parser.add_argument('--offset', help='x offset value', type=int, default=50)

    args = parser.parse_args()
    kwargs = args.__dict__    
    main(**kwargs)