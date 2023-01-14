import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import dask.array as da
import xarray as xr
import os

import ecco_v4_py as ecco


############################################################################################################
############################      FUNCTIONS TO DERIVE LATITUDINAL MASKS    #################################
############################################################################################################
def make_latC_mask(lat):
	# LOAD GRID
	grid_path = "ECCO-GRID.nc"
	grid = xr.open_dataset(grid_path)
	cds = grid.coords.to_dataset()
	grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)

	lat_maskX, lat_maskY =  ecco.vector_calc.get_latitude_masks(lat, cds['YC'], grid_xmitgcm)
	lat_maskC =  np.abs(lat_maskX.rename({"i_g":"i"})) + lat_maskY.rename({"j_g":"j"})

	# This step is to get rid of overlapping grid cells... 
	lat_maskC = lat_maskC.where(lat_maskC != 0, other=np.nan)*0+1

	return lat_maskC

def make_directional_lat_masks(lat):
	# LOAD GRID
	grid_path = "ECCO-GRID.nc"
	grid = xr.open_dataset(grid_path)
	cds = grid.coords.to_dataset()
	grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)

	lat_maskX, lat_maskY =  ecco.vector_calc.get_latitude_masks(lat, cds['YC'], grid_xmitgcm)
	lat_maskC = lat_maskX.rename({"i_g":"i"}) + lat_maskY.rename({"j_g":"j"})

	# This step is to get rid of overlapping grid cells... 
	lat_maskC = lat_maskC.where(lat_maskC != 0, other=np.nan)*0+1
	lat_maskX = lat_maskX.where(lat_maskX != 0, other=np.nan)*0+1
	lat_maskY = lat_maskY.where(lat_maskY != 0, other=np.nan)*0+1
	
	return lat_maskC, lat_maskX, lat_maskY


############################################################################################################
#############################     CREATE TIME-DEPTH-LATITUDE DATA ARRAY    #################################
############################################################################################################
def integrate_zonal_time(FIELD,time_slice=np.arange(0,288),lat_vals = np.arange(-88,88),
							depth = np.arange(0,50),make_directional=False):
	"""
	FIELD = xarray dataarray
	"""
	# set dimensions based on input dataset with modified vertical level spacing..
	tdl_array_dims = (len(time_slice),
	                 len(depth),
	                 len(lat_vals),
	                 )

	empty_data = np.zeros(tdl_array_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [time_slice,depth,lat_vals,]
	new_dims = ["time","depth","lat",]

	tdl_array = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
	tdl_array.load()

	for lat in lat_vals:
	    print(lat,end=" ")
	    if make_directional == False:
	    	lat_maskC = make_latC_mask(lat)
	    elif make_directional == True:
	    	lat_maskC, lat_maskX, lat_maskY = make_directional_lat_masks(lat)
	    a = (FIELD*lat_maskC).sum(dim="i").sum(dim="j").sum(dim="tile")
	    tdl_array.loc[{"lat":lat}] = a
	print("returned integrated averaged array")
	print(tdl_array)

	return tdl_array   



def integrate_zonal(FIELD,lat_vals = np.arange(-88,88),depth = np.arange(0,50),make_directional=False):
	"""
	FIELD = xarray dataarray
	"""
	# set dimensions based on input dataset with modified vertical level spacing..
	tdl_array_dims = (len(depth),
	                 len(lat_vals),
	                 )

	empty_data = np.zeros(tdl_array_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [depth,lat_vals,]
	new_dims = ["depth","lat",]

	tdl_array = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
	tdl_array.load()

	for lat in lat_vals:
	    print(lat,end=" ")
	    if make_directional:
	    	lat_maskC, lat_maskX, lat_maskY = make_directional_lat_masks(lat)
	    else:
	    	lat_maskC = make_latC_mask(lat)
	    a = (FIELD*lat_maskC).sum(dim="i").sum(dim="j").sum(dim="tile")
	    tdl_array.loc[{"lat":lat}] = a
	print("returned integrated array")
	print(tdl_array)

	return tdl_array   


def integrate_zonal_sigma(FIELD,pot_rho,lat_vals = np.arange(-88,88),make_directional=False):
	"""
	FIELD = xarray dataarray
	"""
	# set dimensions based on input dataset with modified vertical level spacing..
	tdl_array_dims = (len(pot_rho),
	                 len(lat_vals),
	                 )

	empty_data = np.zeros(tdl_array_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [pot_rho.values,lat_vals,]
	new_dims = ["pot_rho","lat",]

	tdl_array = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
	tdl_array.load()

	for lat in lat_vals:
	    print(lat,end=" ")
	    if make_directional == False:
	    	lat_maskC = make_latC_mask(lat)
	    elif make_directional == True:
	    	lat_maskC, lat_maskX, lat_maskY = make_directional_lat_masks(lat)
	    a = (FIELD*lat_maskC).sum(dim="i").sum(dim="j").sum(dim="tile")
	    tdl_array.loc[{"lat":lat}] = a
	print("returned integrated array")
	print(tdl_array)

	return tdl_array   


def average_zonal_time(FIELD,time_slice = np.arange(0,288),lat_vals = np.arange(-88,88),
						depth = np.arange(0,50),make_directional=False):
	"""
	FIELD = xarray dataarray
	"""
	# set dimensions based on input dataset with modified vertical level spacing..
	tdl_array_dims = (len(time_slice),
	                 len(depth),
	                 len(lat_vals),
	                 )

	empty_data = np.zeros(tdl_array_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [time_slice,depth,lat_vals,]
	new_dims = ["time","depth","lat",]

	tdl_array = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
	tdl_array.load()

	for lat in lat_vals:
	    print(lat,end=" ")
	    if make_directional == False:
	    	lat_maskC = make_latC_mask(lat)
	    elif make_directional == True:
	    	lat_maskC, lat_maskX, lat_maskY = make_directional_lat_masks(lat)
	    a = (FIELD*lat_maskC).mean(dim=["i","j","tile"])
	    tdl_array.loc[{"lat":lat}] = a
	print("returned zonally averaged array")
	print(tdl_array)

	return tdl_array   


def average_zonal(FIELD,lat_vals = np.arange(-88,88),depth = np.arange(0,50),make_directional=False):
	"""
	FIELD = xarray dataarray
	"""
	# set dimensions based on input dataset with modified vertical level spacing..
	tdl_array_dims = (len(depth),
	                 len(lat_vals),
	                 )

	empty_data = np.zeros(tdl_array_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [depth,lat_vals,]
	new_dims = ["k","lat",]

	tdl_array = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
	tdl_array.load()

	for lat in lat_vals:
	    print(lat,end=" ")
	    if make_directional == False:
	    	lat_maskC = make_latC_mask(lat)
	    elif make_directional == True:
	    	lat_maskC, lat_maskX, lat_maskY = make_directional_lat_masks(lat)
	    a = (FIELD*lat_maskC).mean(dim=["i","j","tile"])
	    tdl_array.loc[{"lat":lat}] = a
	print("returned zonally averaged array")
	#print(tdl_array)

	return tdl_array   


def average_zonal_sigma(FIELD,pot_rho,lat_vals = np.arange(-88,88),make_directional=False):
	"""
	FIELD = xarray dataarray
	"""
	# set dimensions based on input dataset with modified vertical level spacing..
	tdl_array_dims = (len(pot_rho),
	                 len(lat_vals),
	                 )

	empty_data = np.zeros(tdl_array_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [pot_rho.values,lat_vals,]
	new_dims = ["pot_rho","lat",]

	tdl_array = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
	tdl_array.load()

	for lat in lat_vals:
	    print(lat,end=" ")
	    if make_directional == False:
	    	lat_maskC = make_latC_mask(lat)
	    elif make_directional == True:
	    	lat_maskC, lat_maskX, lat_maskY = make_directional_lat_masks(lat)
	    a = (FIELD*lat_maskC).mean(dim=["i","j","tile"])
	    tdl_array.loc[{"lat":lat}] = a
	print("returned zonally averaged array")
	#print(tdl_array)

	return tdl_array   


def sigma2_zonal_surf_max(SIGMA2,maskC,monotonic=False,lat_vals=np.arange(-88,89)):
	"""
	SIGMA2: xarray dataarray, SIGMA2 values, can be max in time or full in k
	maskC: xarray datarray 
	monotonic: boolean, flag for whether or not to make the outcrops monotonic towards the poles
	lat_vals: ndarray, lat values over which to evaluate

	"""

	# set dimensions based on input dataset with modified vertical level spacing..
	
	if ("time" in SIGMA2) and ("k" in SIGMA2):
		SIGMA2_0tmax = SIGMA2.isel(k=0).max(dim="time")*maskC
	elif "k" in SIGMA2:
		SIGMA2_0tmax = SIGMA2.isel(k=0)*maskC
	elif "time" in SIGMA2:
		SIGMA2_0tmax = SIGMA2.max(dim="time")*maskC
	else:
		SIGMA2_0tmax = SIGMA2*maskC

	# Make the dataarray
	maxsig_array_dims = (len(lat_vals))
	empty_data = np.zeros(maxsig_array_dims)
	new_coords = [lat_vals,]
	new_dims = ["lat"]
	maxsig_darray = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
	maxsig_darray.load()	

	# Fill in the dataarray
	for lat in lat_vals:
		lat_maskC = make_latC_mask(lat)
		lat_maxsig = (SIGMA2_0tmax*lat_maskC).max(dim=["i","j","tile"])
		maxsig_darray.loc[{"lat":lat}] = lat_maxsig

	# If we don't want to make it monotonic, end here
	if monotonic == False:
		return maxsig_darray

	# Else we can make it monotonic towards the poles
	# This is a bit of a mess right now, but hopefully its reasonable efficient
	elif monotonic == True:
		maxsig_array = maxsig_darray.values.copy()
		i = 1
		# do the Southern Hemisphere values first
		while lat_vals[i] < 0:
			if maxsig_array[i] < maxsig_array[i+1]:
				maxsig_array[:i+1][np.argwhere(maxsig_array[:i+1]<maxsig_array[i+1])] = maxsig_array[i+1]
			i+=1
		# do the Northern Hemisphere values, if there are any
		while len(lat_vals[i:]) > 1:
			if maxsig_array[i] > maxsig_array[i+1]:
				maxsig_array[i+1:][np.argwhere(maxsig_array[i+1:] < maxsig_array[i])] = maxsig_array[i]
			i+=1

		maxsig_darray = maxsig_darray*0+maxsig_array

	return maxsig_darray

		



def plot_lat_field(tmp_plot,grid,levels=40,highest_depth=0,lowest_depth=50,lat_min=0,lat_max=170,
					w=30,h=15,colormap='viridis',contouf=True,pcolor=False,vmin=None,vmax=None,
					contours=True,title=None,cbarlabel=None,xlabel=None,ylabel=None):
    dep = -1*grid.Z
    
    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s

    plt.figure(figsize=(w,h))
    if contouf==True:
	    plt.contourf(tmp_plot.lat[lat_min:lat_max],
	                 dep[:lowest_depth],
	                 tmp_plot[:lowest_depth,lat_min:lat_max],
	                 levels=levels,
	                 vmin=vmin,vmax=vmax,
	                 cmap=colormap)
	    cbar = plt.colorbar()
	    cbar.set_label(cbarlabel,rotation=270,labelpad=30)

    if pcolor==True:
	    plt.pcolor(tmp_plot.lat[lat_min:lat_max],
	                 dep[:lowest_depth],
	                 tmp_plot[:lowest_depth,lat_min:lat_max],
	                 vmin=vmin,vmax=vmax,
	                 cmap=colormap)
	    cbar = plt.colorbar()
	    cbar.set_label(cbarlabel,rotation=270,labelpad=30)

    if contours==True:
	    CS = plt.contour(tmp_plot.lat[lat_min:lat_max],
	                     dep[:lowest_depth],
	                     tmp_plot[:lowest_depth,lat_min:lat_max],
	                     levels=levels,
	                     vmin=vmin,vmax=vmax,
	                     colors='k')
	    # Recast levels to new class
	    CS.levels = [nf(val) for val in CS.levels]
	    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.title(title,fontsize=24)
    plt.xticks(np.arange(tmp_plot.lat[lat_min],tmp_plot.lat[lat_max],5))
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    plt.grid()
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    
    

def plot_lat_field_diff(tmp_plot1,tmp_plot2,grid,time_slice1=np.arange(0,12),
						time_slice2=np.arange(276,288),levels=40,lowest_depth=37,
						lat_min=0,lat_max=170,w=30,h=10,colormap='viridis',vmin=None):
    dep = -1*grid.Z
    levels=levels

    tmp_plot1 = tmp_plot1.isel(time=time_slice1).mean(dim="time")
    tmp_plot2 = tmp_plot2.isel(time=time_slice2).mean(dim="time")
    
    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s

    plt.figure(figsize=(30,10))
    plt.contourf(tmp_plot1.lat[lat_min:lat_max],
                 dep[:lowest_depth],
                 (tmp_plot1-tmp_plot2)[:lowest_depth,lat_min:lat_max],
                 levels=levels,
                cmap=colormap)
    plt.colorbar()
    CS = plt.contour(tmp_plot1.lat[lat_min:lat_max],dep[:lowest_depth],
    					(tmp_plot1-tmp_plot2)[:lowest_depth,lat_min:lat_max],levels=levels,colors='k')
    # Recast levels to new class
    CS.levels = [nf(val) for val in CS.levels]
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.grid()
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()


