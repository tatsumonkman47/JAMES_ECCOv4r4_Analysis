import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import dask.array as da
import xarray as xr
import os


from xmitgcm import open_mdsdataset
import xmitgcm
import ecco_v4_py as ecco


from netCDF4 import Dataset

import seawater

from analysis_package import plotting_functions
from analysis_package import open_datasets
from analysis_package import derive_potential_density_values_TEST
from analysis_package import ecco_masks

from importlib import reload

# reload modules for prototyping...
ecco_masks = reload(ecco_masks)
plotting_functions = reload(plotting_functions)
open_datasets = reload(open_datasets)
derive_potential_density_values_TEST = reload(derive_potential_density_values_TEST)



def integrate_over_lat_bands(trsp_x,trsp_y,time_slice,dens_minima,lat_vals,sub_surf=False):

	######################################################################################################################
	################################################### LOAD GRID ########################################################
	######################################################################################################################
	grid_path = "./ecco_grid/ECCOv4r3_grid.nc"
	grid = xr.open_dataset(grid_path)
	cds = grid.coords.to_dataset()
	grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)

	# set dimensions based on input dataset with modified vertical level spacing..
	pot_dens_dims = (len(time_slice),
	                 len(trsp_x.pot_rho),
	                 len(lat_vals))

	empty_pot_coords_data = np.zeros(pot_dens_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [time_slice, 
	              trsp_x.pot_rho, 
	              lat_vals]
	new_dims = ["time", 
	            "pot_rho", 
	            "lat"]

	depth_integrated_pdens_transport = xr.DataArray(data=empty_pot_coords_data,coords=new_coords,dims=new_dims)
	depth_integrated_pdens_transport.load()
	depth_integrated_pdens_transport_latx = depth_integrated_pdens_transport.copy(deep=True)
	depth_integrated_pdens_transport_latx.load()
	depth_integrated_pdens_transport_laty = depth_integrated_pdens_transport.copy(deep=True)
	depth_integrated_pdens_transport_laty.load()

	for lat in lat_vals:
		# Compute mask for particular latitude band
		print(str(lat)+' ',end='')
		# since transport values are in native grid coordaintes you need to combine the sum of the transports in 
		# the x and y direction and vis versa for tiles 0-5 and 7-12, respectively
		lat_maskX, lat_maskY = ecco.vector_calc.get_latitude_masks(lat, cds['YC'], grid_xmitgcm)

		# Subtract minimum density surface if need be
		if sub_surf == True:
			trsp_x_filtered = trsp_x.where(trsp_x.pot_rho > dens_minima.isel(lat=lat), other=0)
			trsp_y_filtered = trsp_y.where(trsp_y.pot_rho > dens_minima.isel(lat=lat), other=0)
			# Sum horizontally
			lat_trsp_x = (trsp_x_filtered * lat_maskX).sum(dim=['i_g','j','tile'],skipna=True).transpose("time","pot_rho")
			lat_trsp_y = (trsp_y_filtered * lat_maskY).sum(dim=['i','j_g','tile'],skipna=True).transpose("time","pot_rho")

		else:
			lat_trsp_x = (trsp_x * lat_maskX).sum(dim=['i_g','j','tile'],skipna=True).transpose("time","pot_rho")
			lat_trsp_y = (trsp_y * lat_maskY).sum(dim=['i','j_g','tile'],skipna=True).transpose("time","pot_rho")
		     
		depth_integrated_pdens_transport_latx.loc[{'lat':lat}] = lat_trsp_x
		depth_integrated_pdens_transport_laty.loc[{'lat':lat}] = lat_trsp_y

	return depth_integrated_pdens_transport_latx, depth_integrated_pdens_transport_laty



def analyze_overturning_output(transport_file_x,transport_file_y,dens_minima_file,data_dir,lat_vals=np.arange(-88,0),sub_surf=False,only_atl=False):

	######################################################################################################################
	############################################# CREATE DOMAIN MASKS ####################################################
	######################################################################################################################

	maskW = xr.open_dataarray("generic_masks/maskW.nc")
	maskS = xr.open_dataarray("generic_masks/maskS.nc")
	maskC = xr.open_dataarray("generic_masks/maskC.nc")

	southern_ocean_mask_W, southern_ocean_mask_S, so_atl_basin_mask_W, so_atl_basin_mask_S, so_indpac_basin_mask_W, so_indpac_basin_mask_S = ecco_masks.get_basin_masks(maskW, maskS, maskC)

	# Load minimum density array
	if sub_surf==True:
	    dens_minima = xr.open_dataarray(dens_minima_file)
	    print(dens_minima)
	else:
		dens_minima = None

	# Load transport arrays.
	trsp_x_0 = xr.open_dataarray(data_dir+transport_file_x)
	trsp_y_0 = xr.open_dataarray(data_dir+transport_file_y)
	print(trsp_x_0)
	print(trsp_y_0)
	trsp_x = trsp_x_0.transpose("j","i_g","pot_rho","tile","time").copy(deep=True)
	trsp_y = trsp_y_0.transpose("j_g","i","pot_rho","tile","time").copy(deep=True)

	# Get analysis time slice    
	time_slice = trsp_x.time.values

	# Get potential density levels:
	pot_dens_coord = trsp_x.pot_rho.values

	# Initialize output dataset
	pot_dens_dims = (len(time_slice),
	                 len(pot_dens_coord),
	                 len(lat_vals))

	empty_pot_coords_data = np.zeros(pot_dens_dims)

	new_coords = [time_slice, 
	              pot_dens_coord, 
	              lat_vals]
	new_dims = ["time", 
	            "pot_rho", 
	            "lat"]

	# Create three output fields
	depth_integrated_pdens_transport = xr.DataArray(data=empty_pot_coords_data,coords=new_coords,dims=new_dims)
	depth_integrated_pdens_transport.load()

	global_integrated_pdens_transport = depth_integrated_pdens_transport.copy(deep=True)
	global_integrated_pdens_transport.load()
	indpac_integrated_pdens_transport = depth_integrated_pdens_transport.copy(deep=True)
	indpac_integrated_pdens_transport.load()
	atl_integrated_pdens_transport = depth_integrated_pdens_transport.copy(deep=True)
	atl_integrated_pdens_transport.load()

	atl_integrated_trsp_x, atl_integrated_trsp_y = integrate_over_lat_bands(trsp_x*so_atl_basin_mask_W,
																			trsp_y*so_atl_basin_mask_S,
																			time_slice,
																			dens_minima,
																			lat_vals,
																			sub_surf=sub_surf)
	atlso_str_func = -1*atl_integrated_trsp_x -1*atl_integrated_trsp_y
	atl_integrated_pdens_transport = atl_integrated_pdens_transport + atlso_str_func

	if only_atl == False:
		indpac_integrated_trsp_x, indpac_integrated_trsp_y = integrate_over_lat_bands(trsp_x*so_indpac_basin_mask_W,
																					  trsp_y*so_indpac_basin_mask_S,
																					  time_slice,
																					  dens_minima,
																					  lat_vals,
																					  sub_surf=sub_surf)
		global_integrated_trsp_x, global_integrated_trsp_y = integrate_over_lat_bands(trsp_x,
																					  trsp_y,
																					  time_slice,
																					  dens_minima,
																					  lat_vals,
																					  sub_surf=sub_surf)
		indpacso_str_func = -1*indpac_integrated_trsp_x -1*indpac_integrated_trsp_y
		global_str_func = -1*global_integrated_trsp_x -1*global_integrated_trsp_y
		indpac_integrated_pdens_transport = indpac_integrated_pdens_transport + indpacso_str_func
		global_integrated_pdens_transport = global_integrated_pdens_transport + global_str_func

	return indpac_integrated_pdens_transport, atl_integrated_pdens_transport, global_integrated_pdens_transport


