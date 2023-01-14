import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy as cart

from xmitgcm import open_mdsdataset
import xmitgcm
import ecco_v4_py as ecco
import pandas as pd

from netCDF4 import Dataset

import seawater

from analysis_package import plotting_functions
from analysis_package import open_datasets

time_slice = np.arange(10,20)

data_dir = "./old_nctiles_monthly/"
uvvel_data_dir = "./nctiles_monthly/"

UVELMASS_var = "UVELMASS"
VVELMASS_var = "VVELMASS"


def calc_bottom_integrated_meridional_stf_in_zdepth(dataset):
	"""Calculate the bottom-integrated meridional stream function for a dataset
	containing ECCO data for UVELMASS and VVELMASS in native grid coordinates.


	Parameters
	----------
	path: string
		path to datafiles
	VAR: string 
		name of variable to extract, eg 'UVELMASS'
	grid_path: string
		path to grid file, 

	Returns
	_______
	variable_all_tiles: xarray dataset
		Xarray datset with tile files stacked along 'tile' dimension
	"""
	# test a slightly alternative depth integration method..
	transport_x = dataset["UVELMASS"]*dataset["drF"]*dataset["dyG"]
	transport_y = dataset["VVELMASS"]*dataset["drF"]*dataset["dxG"]

	cds = dataset.coords.to_dataset()
	grid = ecco.ecco_utils.get_llc_grid(cds)
	lat_transport = ecco.calc_meridional_trsp._initialize_trsp_data_array(cds, lat_vals)

	depth_integrated_trsp_x = transport_x
	depth_integrated_trsp_x.load()
	depth_integrated_trsp_y = transport_y
	depth_integrated_trsp_y.load()

	# perform the 
	depth_integrated_trsp_x = depth_integrated_trsp_x.isel(k=slice(None,None,-1))
	depth_integrated_trsp_x = depth_integrated_trsp_x.cumsum(dim='k')
	depth_integrated_trsp_x = -1*depth_integrated_trsp_x.isel(k=slice(None,None,-1))

	depth_integrated_trsp_y = depth_integrated_trsp_y.isel(k=slice(None,None,-1))
	depth_integrated_trsp_y = depth_integrated_trsp_y.cumsum(dim='k')
	depth_integrated_trsp_y = -1*depth_integrated_trsp_y.isel(k=slice(None,None,-1))
	    
	for lat in lat_vals:

	    # Compute mask for particular latitude band
	    lat_maskW, lat_maskS = ecco.vector_calc.get_latitude_masks(lat, cds['YC'], grid)
	    # Sum horizontally
	    lat_trsp_x = (depth_integrated_trsp_x * lat_maskW).sum(dim=['i_g','j','tile'])
	    lat_trsp_y = (depth_integrated_trsp_y * lat_maskS).sum(dim=['i','j_g','tile'])
	    lat_transport.loc[{'lat':lat}] = lat_trsp_x + lat_trsp_y

	return lat_transport


def stitch_tiles(dataarray):
	"""
	only works for c-type points

	"""

	# Western Atlantic Ocean Tiles
	tile10 = np.flip(dataarray.isel(tile=10).T,axis=0)
	tile11 = np.flip(dataarray.isel(tile=11).T,axis=0)
	tile12 = np.flip(dataarray.isel(tile=12).T,axis=0)

	# Eastern Atlantic Ocean Tiles
	tile0 = dataarray.isel(tile=0)
	tile1 = dataarray.isel(tile=1)
	tile2 = dataarray.isel(tile=2)

	# Indian Ocean Tiles (note these are ordered 5,4,3 North to South)
	tile3 = dataarray.isel(tile=3)
	tile4 = dataarray.isel(tile=4)
	tile5 = dataarray.isel(tile=5)

	# Pacific Ocean Tiles (note these are ordered 5,4,3 North to South)
	tile7 = np.flip(dataarray.isel(tile=7).T,axis=0)
	tile8 = np.flip(dataarray.isel(tile=8).T,axis=0)
	tile9 = np.flip(dataarray.isel(tile=9).T,axis=0)

	western_atlantic_tile = xr.concat((tile12,tile11,tile10),dim='i')
	eastern_atlantic_tile = xr.concat((tile0,tile1,tile2),dim='j')
	indian_tile = xr.concat((tile3,tile4,tile5),dim='j')
	mid_pacific_tile = xr.concat((tile9,tile8,tile7),dim='i')

	pacific_basin_tile = xr.concat((mid_pacific_tile,western_atlantic_tile),dim='j').rename({'i':'x','j':'y'}).drop('y').drop('x')
	indian_east_atlantic_tile = xr.concat((eastern_atlantic_tile,indian_tile),dim='i').rename({'i':'y','j':'x'}).drop('y').drop('x')

	world_tile = xr.concat((pacific_basin_tile,indian_east_atlantic_tile),dim='y')

	return world_tile

def plot_lat_transport(lat_transport):

	plt.figure(figsize=(12,6))
	plt.pcolor(lat_trsp_test2.lat,lat_trsp_test2.k,np.flip(lat_trsp_test2.mean(axis=0),axis=0))
	plt.xlabel("Latitude (deg north)")
	plt.ylabel("Depth (grid level)")
	plt.title("year-averaged GOC for 2010 (Preliminary Post-ECCO Results)")
	cbar = plt.colorbar()
	cbar.set_label("Depth-integrated stream function (Sv)")
	plt.show()

	return plt