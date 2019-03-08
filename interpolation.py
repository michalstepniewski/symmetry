#usage python3.6 interpolation
# options
#  --input file_with_input_mask.nii.gz file in nii.gz format \
#  --output file_to_output_interpolated_mask.nii
#  --type contour or full_simplified
#   I thought it best for the input to be
#   (when transformed to numpy array) to be 3D array with
#   None (or np. nan values in slices where )
#  --direction 0,1,2 (depending on which axis to interpolate)
#
import argparse
import os, numpy as np, nibabel as nib
from nibabel.testing import data_path
from scipy.interpolate import griddata
from tqdm import tqdm



parser = argparse.ArgumentParser(description='Interpolate hemisphere masks.')
parser.add_argument('--input', help='input file with partial mask to be interpolated\
                     in .nii or .nii.gz format', type=str)
parser.add_argument('--output', help='output file to output interpolated mask into\
                      in .nii or .nii.gz format', type=str)
parser.add_argument('--direction', help='0,1 or2 (depending on which axis to interpolate)',
                    type=int)
parser.add_argument('--type', help='type of interpolation to be performed: \
                    contour or full_simplified', type=str)

args = parser.parse_args()
#print(args.__dict__)

def main(input_path='img_mask_data_broken_1.nii.gz',
         output_path='img_mask_data_interpolated_1.nii.gz',direction=1):
    img_mask = nib.load(input_path)
    affine = img_mask.get_affine()
    img_mask_data = img_mask.get_fdata()#[:50,90:100,:50]
    del img_mask
    if direction ==0:
        filled = np.array([np.isfinite(img_mask_data[x,:,:]).any() \
         for x in range(img_mask_data.shape[direction])])        
    elif direction ==1:
        filled = np.array([np.isfinite(img_mask_data[:,x,:]).any() \
         for x in range(img_mask_data.shape[direction])])        
    elif direction ==2:
        filled = np.array([np.isfinite(img_mask_data[:,:,x]).any() \
         for x in range(img_mask_data.shape[direction])])        
        #x, y, z = np.where(np.isfinite(img_mask_data_broken_1))
    
    for x in range(img_mask_data.shape[direction]):
        if not filled[x]:
            #finc closest filled neighbours
            left_x = np.where(filled[:x])[0].max()
            right_x = np.where(filled[x:])[0].min()+x
            base_img1 = img_mask_data[:,right_x,:]
            base_img0 = img_mask_data[:,left_x,:]
            diff = (base_img1 - base_img0)
            diff_atomic = diff/float(right_x-left_x)
    return

if __name__ == '__main__':
    kwargs = args.__dict__    
    main(input_path=args.input_path)