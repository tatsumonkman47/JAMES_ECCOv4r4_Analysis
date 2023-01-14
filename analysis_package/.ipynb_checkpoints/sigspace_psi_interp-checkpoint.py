import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr


from xmitgcm import open_mdsdataset
import xmitgcm
import sys
sys.path.append('/Users/tatsumonkman/3rd_party_software/ECCOv4-py')
import ecco_v4_py as ecco


from netCDF4 import Dataset

import seawater

from analysis_package import plotting_functions
from analysis_package import open_datasets
from analysis_package import ecco_masks

from importlib import reload

data_dir = "./nctiles_monthly/"

UVELMASS_var = "UVELMASS"
VVELMASS_var = "VVELMASS"
BOLUS_UVEL_var = "BOLUS_UVEL"
BOLUS_VVEL_var = "BOLUS_VVEL"

grid_path = "./ECCO-GRID.nc"
grid = xr.open_dataset(grid_path)



def perform_potential_density_overturning_calculation(time_slice,PDENS_U_ds,PDENS_V_ds,UVELMASS_ds_raw,VVELMASS_ds_raw,
														UVELSTAR_ds_raw, VVELSTAR_ds_raw,pot_dens_coord,interpolation=True):
	""" 

	Parameters
	----------
	

	Returns
	_______
	
	"""
	def pdens_stencils(density,density_below,pot_dens_array):
	    potdens_stencil_0 = pot_dens_array > density
	    # this step is critical to remove low density anomalies in the deep ocean from stencil...
	    # not sure what to do about those, ignoring them for now
	    potdens_stencil = 1*(potdens_stencil_0.cumsum(dim="k") > 0)    
	    # set end-appended value equal to 1 for subtraction step..
	    potdens_stencil_shifted_up_one_cell = xr.concat((potdens_stencil.isel(k=slice(1,50)),potdens_stencil.isel(k=49)),dim="k").assign_coords(k=np.arange(0,50))
	    potdens_stencil_shifted_up_two_cell = xr.concat((potdens_stencil_shifted_up_one_cell.isel(k=slice(1,50)),potdens_stencil_shifted_up_one_cell.isel(k=49)),dim="k").assign_coords(k=np.arange(0,50))        
	    
	    potdens_stencil_shifted_down_one_cell = xr.concat((potdens_stencil.isel(k=0)*0,potdens_stencil.isel(k=slice(0,49))),dim="k").assign_coords(k=np.arange(0,50))
	    potdens_stencil_shifted_down_two_cell = xr.concat((potdens_stencil_shifted_down_one_cell.isel(k=0)*0,potdens_stencil_shifted_down_one_cell.isel(k=slice(0,49))),dim="k").assign_coords(k=np.arange(0,50))
	    potdens_stencil_shifted_down_three_cell = xr.concat((potdens_stencil_shifted_down_two_cell.isel(k=0)*0,potdens_stencil_shifted_down_two_cell.isel(k=slice(0,49))),dim="k").assign_coords(k=np.arange(0,50))

	    potdens_stencil_one_above_top_level = potdens_stencil_shifted_up_one_cell*1 - potdens_stencil*1
	    potdens_stencil_two_above_top_level = potdens_stencil_shifted_up_two_cell*1 - potdens_stencil_shifted_up_one_cell
	    # get rid of trailing negative values that occur at the ocean's bottom boundary..
	    potdens_stencil_one_above_top_level = potdens_stencil_one_above_top_level.where(potdens_stencil_one_above_top_level > 0, other=0)
	    potdens_stencil_two_above_top_level = potdens_stencil_two_above_top_level.where(potdens_stencil_two_above_top_level > 0, other=0)
	    
	    potdens_stencil_one_below_top_level = potdens_stencil_shifted_down_one_cell*1 - potdens_stencil_shifted_down_two_cell*1
	    potdens_stencil_two_below_top_level = potdens_stencil_shifted_down_two_cell*1 - potdens_stencil_shifted_down_three_cell*1
	    # get rid of trailing negative values that occur at the ocean's bottom boundary..
	    potdens_stencil_one_below_top_level = potdens_stencil_one_below_top_level.where(potdens_stencil_one_below_top_level > 0, other=0)
	    potdens_stencil_two_below_top_level = potdens_stencil_two_below_top_level.where(potdens_stencil_two_below_top_level > 0, other=0)
	    
	    potdens_stencil_top_level = potdens_stencil*1 - potdens_stencil_shifted_down_one_cell*1    
	    potdens_stencil_top_level = potdens_stencil_top_level.where(potdens_stencil_top_level > 0, other=np.nan)
	    
	    potdens_stencil_one_above_top_level = potdens_stencil_one_above_top_level.where(potdens_stencil_one_above_top_level > 0, other=np.nan)
	    potdens_stencil_one_below_top_level = potdens_stencil_one_below_top_level.where(potdens_stencil_one_below_top_level > 0, other=np.nan)

	    return potdens_stencil_top_level, potdens_stencil_one_above_top_level, potdens_stencil_shifted_down_one_cell


	cds = grid.coords.to_dataset()
	grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)

	transport_x = ((UVELMASS_ds_raw["UVELMASS"]
					+ UVELSTAR_ds_raw["UVELSTAR"]).fillna(0)*grid["drF"]*grid["dyG"] )
	transport_y = ((VVELMASS_ds_raw["VVELMASS"]
					+ VVELSTAR_ds_raw["VVELSTAR"]).fillna(0)*grid["drF"]*grid["dxG"] )
	print("got past addition UVELMASS_ds_raw[\"UVELMASS\"]+UVELSTAR_ds_raw[\"UVELSTAR\"]")
	# create infrastructure for integrating in depth space

	lat_vals = np.arange(-88,88)

	# create an empty array with a stretched depth dimension
	# Set the coordinates of the stretched depth dimension to potential density values..
	# add padding to either end of the pot. density coordinates
	# just trying with slightly coarser resolution 
	#(what pot density resolution is valid in this case?)

	# set dimensions based on input dataset with modified vertical level spacing..
	pot_dens_dims = (len(time_slice),
	                 len(pot_dens_coord),
	                 len(lat_vals))

	empty_pot_coords_data = np.zeros(pot_dens_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [time_slice, pot_dens_coord, lat_vals]
	new_dims = ["time", "sig", "lat"]

	# the potential density values have been interpolated to the edges of the grid cells
	pot_dens_array_x = PDENS_U_ds.copy(deep=True)
	pot_dens_array_y = PDENS_V_ds.copy(deep=True)
	pot_dens_array = pot_dens_array_x.rename({"i_g":"i"})


	depth_integrated_pdens_transport = xr.DataArray(data=empty_pot_coords_data,coords=new_coords,dims=new_dims)
	depth_integrated_pdens_transport.load()

	# Create an array for storing depth-integrated transport in native grid coordinates
	native_pot_dens_dims = (13,
						    len(time_slice),
							90,
							90,
							len(pot_dens_coord),
							)
	new_native_pot_dens_dims_x = ["tile","time","j","i_g","sig"]
	new_native_pot_dens_dims_y = ["tile","time","j_g","i","sig"]
	empty_native_pot_dens_data = np.zeros(native_pot_dens_dims)
	new_native_pot_dens_coords = (np.arange(0,13),
								  time_slice,
								  np.arange(0,90),
				  				  np.arange(0,90),
				  				  pot_dens_coord
				  				  )
	depth_integrated_native_pdens_U_transport = xr.DataArray(data=empty_native_pot_dens_data, 
														      	coords=new_native_pot_dens_coords, 
														   		dims=new_native_pot_dens_dims_x)
	depth_integrated_native_pdens_U_transport.load()
	depth_integrated_native_pdens_V_transport = depth_integrated_native_pdens_U_transport.copy(deep=True).rename({"i_g":"i","j":"j_g"})
	depth_integrated_native_pdens_V_transport.load()

	z_depth_hFacC = (grid.Zl.rename({"k_l":"k"}) - (grid.hFacC)*grid.drF/2)

	print("interpolation == ",interpolation)

	for density in pot_dens_coord:
	    print("Started " + str(density) + " surface for time slice " + str(time_slice))
	    pdens_stencil = pot_dens_array > density

	    if interpolation == True:
		    psten_top, psten_p1, pdens_stencil_down1 = pdens_stencils(density,None,pot_dens_array,)
		    print("got to checkpoint 0")
		    ############################################################################################################
		    ###########################################     START INTERPOLATION    #####################################
		    ############################################################################################################
		    # set end-appended value equal to 1 for subtraction step..	    
		    # multiply depth values by -1 to make them positive..
		    depth_top = (z_depth_hFacC*psten_top).min(dim="k")
		    depth_p1 = (z_depth_hFacC*psten_p1).min(dim="k")
		    pdens_top = (pot_dens_array*psten_top).max(dim="k")
		    pdens_p1 = (pot_dens_array*psten_p1).max(dim="k")
		    drF_top = (grid.drF*grid.hFacC*psten_top).max(dim="k")
		    drF_p1 = (grid.drF*grid.hFacC*psten_p1).max(dim="k")

		    depth_potdens_slope = (depth_p1 - depth_top)/(pdens_p1 - pdens_top)                                                
		    dpdens = density - pdens_top
		    depth_density = depth_potdens_slope*dpdens + depth_top
		    depth_Zl = (grid.Zl.rename({"k_l":"k"})*psten_top).min(dim="k")
		    depth_diff = depth_Zl - depth_density

		    # tells you how much of the cell above the top cell is filled
		    percent_p1 = -1*depth_diff.where(depth_diff<0,other=0)/drF_p1
		    # tells you how much of the top cell is filled
		    percent_top = 1 - depth_diff.where(depth_diff>=0,other=0)/drF_top
		    
		    print("got to checkpoint 2") 
		    # Perform vertical integration
		    transport_x_top_level = ((psten_top*percent_top).rename({"i":"i_g"})*transport_x).sum(dim="k")
		    transport_y_top_level = ((psten_top*percent_top).rename({"j":"j_g"})*transport_y).sum(dim="k")
		    transport_above_x_top_level = ((psten_p1*percent_p1).rename({"i":"i_g"})*transport_x).sum(dim="k")
		    transport_above_y_top_level = ((psten_p1*percent_p1).rename({"j":"j_g"})*transport_y).sum(dim="k")

		    # multiply the top and p1 level transports by the percentage of the cell that is filled
		    trsp_interpolated_x = transport_x_top_level.fillna(0) + transport_above_x_top_level.fillna(0)
		    trsp_interpolated_y = transport_y_top_level.fillna(0) + transport_above_y_top_level.fillna(0)
		    
		    # "transport_integral_x/y" is the vertical sum of the interpolated grid cell tranposrt
		    trsp_interpolated_x.load()
		    trsp_interpolated_y.load()
		    ############################################################################################################
		    ###########################################     END INTERPOLATION    #######################################
		    ############################################################################################################    
		    
		    pdens_stencil_down1_x = pdens_stencil_down1.rename({"i":"i_g"})
		    pdens_stencil_down1_y = pdens_stencil_down1.rename({"j":"j_g"})

		    # split the top cell in half since we are putting it into the interpolation,
		    # but only in cases where there actually is a cell above it.
		    depth_integrated_trsp_x_0 = transport_x*(pdens_stencil_down1_x.where(pdens_stencil_down1_x>0,other=np.nan))
		    depth_integrated_trsp_x_0.load()
		    depth_integrated_trsp_x = depth_integrated_trsp_x_0.sum(dim='k') + trsp_interpolated_x
		    depth_integrated_trsp_y_0 = transport_y*(pdens_stencil_down1_y.where(pdens_stencil_down1_y>0,other=np.nan))
		    depth_integrated_trsp_y_0.load()
		    depth_integrated_trsp_y = depth_integrated_trsp_y_0.sum(dim='k') + trsp_interpolated_y

	    elif interpolation == False:
		    print("Checkpoint 3")
		    potdens_stencil = 1*(pdens_stencil.cumsum(dim="k") > 0)   
		    potdens_stencil_x = potdens_stencil.rename({"i":"i_g"})
		    potdens_stencil_y = potdens_stencil.rename({"j":"j_g"})
		    depth_integrated_trsp_x_0 = transport_x*potdens_stencil_x
		    depth_integrated_trsp_x_0.load()
		    depth_integrated_trsp_x = depth_integrated_trsp_x_0.sum(dim='k',skipna=True)
		    depth_integrated_trsp_y_0 = transport_y*potdens_stencil_y
		    depth_integrated_trsp_y_0.load()
		    depth_integrated_trsp_y = depth_integrated_trsp_y_0.sum(dim='k',skipna=True)

	    depth_integrated_native_pdens_V_transport.loc[{"sig":density}] = depth_integrated_trsp_y.transpose("tile","time","j_g","i").values
	    depth_integrated_native_pdens_U_transport.loc[{"sig":density}] = depth_integrated_trsp_x.transpose("tile","time","j","i_g").values

	depth_integrated_native_pdens_U_transport.load()
	depth_integrated_native_pdens_V_transport.load()
	
	return_tuple = (depth_integrated_native_pdens_U_transport,
					depth_integrated_native_pdens_V_transport)

	return return_tuple







