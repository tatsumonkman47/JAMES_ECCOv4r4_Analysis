import numpy as np
import xarray as xr

from analysis_package import plotting_functions
from analysis_package import open_datasets
from analysis_package import ecco_masks
from analysis_package import sigspace_psi_interp


def overturning_time_loop(data_dir,name,PDENS_var,level_list,tail="_full.nc",time_split=np.arange(0,312),interpolation=True):
    grid_path = "ECCO-GRID.nc"
    grid = xr.open_dataset(grid_path)
    print("loaded grid")

    # A couple of hacky lines setting limits for the time loop
    if "climatology" in data_dir+tail:
        min_time = 0
        max_time = 1
    elif np.array_equal(time_split,np.arange(0,156)) == True:
        min_time = 0
        max_time = 13
    elif np.array_equal(time_split,np.arange(156,312)) == True:
        min_time = 13
        max_time = 26
    elif np.array_equal(time_split,np.arange(0,312)) == True:
        min_time = 0
        max_time = 26
    elif np.array_equal(time_split,np.arange(288,312)) == True:
        min_time = 24
        max_time = 26
    else:
        min_time = time_split[0]
        max_time = time_split[-1]
        
    print("min_time:", min_time)
    print("max_time:", max_time)

    time_slice = []
    for i in range(min_time,max_time):
        time_slice.append(np.arange(i*12,(i+1)*12))
    # make sure to use the "open_dataarray" command instead of open "open_dataset" 
    # or xarray will read in the file incorrectly
    if "yearly" in tail:
        time_slice = []
        time_slice.append(time_split)
        
    UVELMASS_ds = xr.open_dataset(data_dir+"UVELMASS"+tail).assign_coords(time=time_split).isel(time=time_split)
    VVELMASS_ds = xr.open_dataset(data_dir+"VVELMASS"+tail).assign_coords(time=time_split).isel(time=time_split)
    UVELSTAR_ds = xr.open_dataset(data_dir+"UVELSTAR"+tail).assign_coords(time=time_split).isel(time=time_split)
    VVELSTAR_ds = xr.open_dataset(data_dir+"VVELSTAR"+tail).assign_coords(time=time_split).isel(time=time_split)
    if "climatology" in tail:
        UVELMASS_ds = UVELMASS_ds.rename({"i":"i_g"}) 
        VVELMASS_ds = VVELMASS_ds.rename({"j":"j_g"}) 
        UVELSTAR_ds = UVELSTAR_ds.rename({"i":"i_g"}) 
        VVELSTAR_ds = VVELSTAR_ds.rename({"j":"j_g"})
    # set data file indecies starting from zero.
    UVELMASS_ds = UVELMASS_ds.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50),time=time_split)
    VVELMASS_ds = VVELMASS_ds.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50),time=time_split)
    UVELSTAR_ds = UVELSTAR_ds.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50),time=time_split)
    VVELSTAR_ds = VVELSTAR_ds.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50),time=time_split)

    if "yearly" in tail:  
        PDENS_ds = xr.open_dataset("SIGMA2_yearly.nc").assign_coords(time=time_split)
    else:
        PDENS_ds = xr.open_dataset("SIGMA2_full.nc").isel(time=time_split)
    # set data file indecies starting from zero.
    PDENS_ds = PDENS_ds.assign_coords(i=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50),time=time_split)

    for t_slice in time_slice:
        tiles = np.arange(0,13)
        # load data files from central directory
        UVELMASS_ds_raw = UVELMASS_ds.loc[{"time":t_slice}]
        VVELMASS_ds_raw = VVELMASS_ds.loc[{"time":t_slice}]
        UVELSTAR_ds_raw = UVELSTAR_ds.loc[{"time":t_slice}]
        VVELSTAR_ds_raw = VVELSTAR_ds.loc[{"time":t_slice}]
        PDENS_ds_raw = PDENS_ds.loc[{"time":t_slice}]
        PDENS_U_ds_raw = PDENS_ds_raw[PDENS_var].rename({"i":"i_g"})
        PDENS_V_ds_raw = PDENS_ds_raw[PDENS_var].rename({"j":"j_g"})
 
        returned_trsp = sigspace_psi_interp.perform_potential_density_overturning_calculation(t_slice,
                                                                                                PDENS_U_ds_raw,
                                                                                                PDENS_V_ds_raw,
                                                                                                UVELMASS_ds_raw,
                                                                                                VVELMASS_ds_raw, 
                                                                                                UVELSTAR_ds_raw, 
                                                                                                VVELSTAR_ds_raw,
                                                                                                level_list,
                                                                                                interpolation)
        returned_trsp[0].load()
        returned_trsp[1].load()

        returned_trsp[0].to_netcdf("./time_loop_output/"+name+"_trsp_x"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
        returned_trsp[1].to_netcdf("./time_loop_output/"+name+"_trsp_y"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")

        print("\ndone overturning_time_loop\n")


##############################################################################################################################
##############################################################################################################################
########################### RUN HERE ####################################
##############################################################################################################################
##############################################################################################################################


data_dir = "./"
sig2_highres_level_list = np.concatenate((np.arange(1026,1032,0.5),np.arange(1032,1035.5,0.25),np.arange(1035.5,1036.2,0.1), np.arange(1036.2,1036.7,0.05), np.arange(1036.7,1037.5,0.02), np.arange(1037.5,1038,0.1)))

overturning_time_loop(data_dir,"sig2_full","SIGMA2",sig2_highres_level_list,tail="_full.nc",time_split=np.arange(0,312),interpolation=True)

