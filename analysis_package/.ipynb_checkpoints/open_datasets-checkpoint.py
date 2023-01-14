import xarray as xr 
import numpy as np
import os
import pandas as pd

uvvel_data_dir = "./nctiles_monthly/"

UVELMASS_var = "UVELMASS"
VVELMASS_var = "VVELMASS"
GM_PSIX_var = "GM_PsiX"
GM_PSIY_var = "GM_PsiY"


def open_combine_raw_ECCO_tile_files(path,VAR,time_slice,rename_indices=True,print_raw_file_meta=False,surface_field=False,decode_times_bool=True):
	""" Open and combine individual tile files for an xmitgcm dataset and return a complete dataset 
	will probably remove grid assignment to make this function more general..
	I am not adding in a grid for now since merging datasets is pretty computationally intensive.

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

	# add simple logic to read in climatology
	if "climatology" in path:
		time_slice = np.arange(0,12)
		decode_times_bool = False

	variable_dir = path+VAR+"/"
	variable_dict = {}
	variable_nc_dict = {}

	for i in range(1,14):
	    if i < 10:
	        variable_dict["tile_"+str(i)] = xr.open_dataset(variable_dir+VAR+".000"+str(i)+".nc",decode_times=decode_times_bool).load()
	        if print_raw_file_meta == True:
	        	print(variable_dict["tile_"+str(i)])
	    else:
	        variable_dict["tile_"+str(i)] = xr.open_dataset(variable_dir+VAR+".00"+str(i)+".nc",decode_times=decode_times_bool).load()
	        if print_raw_file_meta == True:
	        	print(variable_dict["tile_"+str(i)])

	# rename dimension indicies to match grid dims..
	# will need to change dimension index names if you load variables that aren't in the middle 
	# of each grid tile, eg UVELMASS is on the western face of each grid tile wherease 
	if "UVEL" in VAR and rename_indices==True and "monthly" in path:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"time", "i2":"k","i3":"j","i4":"i_g"})).isel(time=time_slice)
	elif "VVEL" in VAR and rename_indices==True and "monthly" in path:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"time", "i2":"k","i3":"j_g","i4":"i"})).isel(time=time_slice)
	elif "ECCOv4" in VAR and rename_indices==True:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"k","i2":"j","i3":"i"}))#.isel(time=time_slice)
		    variable_dict[tile].assign_coords(i=np.arange(0,90))
		    variable_dict[tile].assign_coords(j=np.arange(0,90))
	elif (surface_field==True) and rename_indices==True:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"time", "i2":"j","i3":"i"})).isel(time=time_slice)	
	elif rename_indices==True:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"time", "i2":"k","i3":"j","i4":"i"})).isel(time=time_slice)
	elif rename_indices==False and "diffkr" in VAR:
		for tile in variable_dict:
		    variable_dict[tile] = variable_dict[tile]
	elif rename_indices==False and "geothermalFlux" in VAR:
		for tile in variable_dict:
		    variable_dict[tile] = variable_dict[tile]
	elif rename_indices==False and "ggl" in VAR:
		for tile in variable_dict:
		    variable_dict[tile] = variable_dict[tile]
	elif rename_indices==False:
		for tile in variable_dict:
		    variable_dict[tile] = variable_dict[tile].isel(time=time_slice)
	else:
		for tile in variable_dict:
		    variable_dict[tile] = variable_dict[tile].isel(time=time_slice)
	# combine tiles along new dimension "tile", which is the last argument in this function 
	variable_all_tiles = xr.concat([variable_dict["tile_1"],variable_dict["tile_2"],variable_dict["tile_3"],
	                               variable_dict["tile_4"],variable_dict["tile_5"],variable_dict["tile_6"],
	                               variable_dict["tile_7"],variable_dict["tile_8"],variable_dict["tile_9"],
	                               variable_dict["tile_10"],variable_dict["tile_11"],variable_dict["tile_12"],
	                             variable_dict["tile_13"]],'tile')

	# assign tile coordinates..
	tile_coords = np.arange(0,13)

	#print(variable_all_tiles)

	variable_all_tiles.assign_coords(tile = tile_coords)

	# not merging with grid for now...
	print("Loaded " + VAR + " over time slice  \n")
	# turn into dask array for performance purposes!

	#dask_variable_all_tiles = variable_all_tiles.chunk((12, len(time_slice), 50, 90, 90))
	return variable_all_tiles



def open_ECCOv4r4_files(path,VAR,print_raw_file_meta=False,decode_times_bool=True):
	""" Open and combine individual tile files for an xmitgcm dataset and return a complete dataset 
	will probably remove grid assignment to make this function more general..
	I am not adding in a grid for now since merging datasets is pretty computationally intensive.

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
	years = np.arange(1992,2018)
	# add simple logic to read in climatology
	# NOT IMPLEMENTED YET
	if "climatology" in path:
		time_slice = np.arange(0,12)
		decode_times_bool = False

	
	variable_dict = {}
	variable_nc_dict = {}
	variable_year_list = []

	for year in years:
		variable_dir = path+VAR+"/"+str(year)+"/"
		monthly_list =[]
		years_index = np.asarray([])
		for i in range(1,13):
			if i < 10:
				if "snapshots" in variable_dir:
					file = f"{variable_dir}{VAR}_{str(year)}_0{str(i)}_01.nc"
				else:
					file = f"{variable_dir}{VAR}_{str(year)}_0{str(i)}.nc"
				if os.path.isfile(file):
					monthly_list.append(xr.open_dataset(file,decode_times=decode_times_bool).load())
					years_index = np.concatenate([years_index,[(year-1992)*12 + i]])
				else:
					print(f"{file} does not exist")
					pass
				if print_raw_file_meta == True:
					print(variable_dict["tile_"+str(i)])
			else:
				if "snapshots" in variable_dir:
					file = f"{variable_dir}{VAR}_{str(year)}_{str(i)}_01.nc"
				else:
					file = f"{variable_dir}{VAR}_{str(year)}_{str(i)}.nc"
				if os.path.isfile(file):
					monthly_list.append(xr.open_dataset(file,decode_times=decode_times_bool).load())
					years_index = np.concatenate([years_index,[(year-1992)*12 + i]])
				else:
					print(f"{file} does not exist")
					pass
				if print_raw_file_meta == True:
					print(variable_dict["tile_"+str(i)])
		variable_all_months = xr.concat(monthly_list,pd.Index(years_index, name='time'))

		variable_year_list.append(variable_all_months)
		print("Loaded " + VAR + " over "+str(year)+" \n")


	#dask_variable_all_tiles = variable_all_tiles.chunk((12, len(time_slice), 50, 90, 90))

	return variable_year_list
