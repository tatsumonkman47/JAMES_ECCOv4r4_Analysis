
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
import sys
sys.path.append('/Users/tatsumonkman/3rd_party_software/ECCOv4-py')
import ecco_v4_py as ecco 

from analysis_package import integrate_zonally
from analysis_package import ecco_masks

from netCDF4 import Dataset


######################################################################################################################
################################################### LOAD GRID ########################################################
######################################################################################################################
grid_path = "ECCO-GRID.nc"
grid = xr.open_dataset(grid_path)
cds = grid.coords.to_dataset()
grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)

######################################################################################################################
############################################# CREATE DOMAIN MASKS ####################################################
######################################################################################################################
maskW = xr.open_dataarray("generic_masks/maskW.nc")
maskS = xr.open_dataarray("generic_masks/maskS.nc")
maskC = xr.open_dataarray("generic_masks/maskC.nc")

southern_ocean_mask_W, southern_ocean_mask_S, southern_ocean_mask_C, so_atl_basin_mask_W, so_atl_basin_mask_S, so_atl_basin_mask_C, so_indpac_basin_mask_W, so_indpac_basin_mask_S, so_indpac_basin_mask_C = ecco_masks.get_basin_masks(maskW, maskS, maskC)

baffin_mask_C = ecco.get_basin_mask("baffin",maskC)
north_mask_C = ecco.get_basin_mask("north",maskC)
hudson_mask_C = ecco.get_basin_mask("hudson",maskC)
gin_mask_C = ecco.get_basin_mask("gin",maskC)
bering_mask_C = ecco.get_basin_mask("bering",maskC)
okhotsk_mask_C = ecco.get_basin_mask("okhotsk",maskC)
atl_mask_C = ecco.get_basin_mask("atl",maskC)

# masks with low depths filtered out
so_atl_maskC_low_depth_filtered = so_atl_basin_mask_C*maskC.where(grid.Depth > 100.)
so_indpac_maskC_low_depth_filtered = so_indpac_basin_mask_C*maskC.where(grid.Depth > 100.)

atl_midlat_basin_mask_C = so_atl_basin_mask_C.where(so_atl_basin_mask_C.lat<=50,other=np.nan).where(so_atl_basin_mask_C.lat>=-32,other=np.nan)




############################################################################################################
##################     A SIMPLE ISOPYCNAL DEPTH CALCULATION, WITH INTERPOLATION   ##########################
############################################################################################################


def pdens_stencils(density,pot_dens_array):
    """
    Generate 4d potential density stencils, with VAR=1 at stencil levels and VAR=nan elsewhere
    
    Attributes:
    ----------
    density: 
    density_below: 
    pot_dens_array:
    
    Dependencies:
    ------------

    """    
    
    potdens_stencil_0 = pot_dens_array > density
    
    # this step is critical to remove low density anomalies in the deep ocean from stencil...
    # not sure what to do about those, ignoring them for now
    potdens_stencil = 1*(potdens_stencil_0.cumsum(dim="k") > 0)   
    
    # set end-appended value equal to 1 for subtraction step..
    potdens_stencil_shifted_up_one_cell = potdens_stencil.shift(k=-1)
    potdens_stencil_shifted_down_one_cell = potdens_stencil.shift(k=1)
    potdens_stencil_shifted_down_two_cell = potdens_stencil.shift(k=2)
    potdens_stencil_one_above_top_level = potdens_stencil_shifted_up_one_cell*1 - potdens_stencil*1
    # get rid of trailing negative values that occur at the ocean's bottom boundary..
    potdens_stencil_one_above_top_level = potdens_stencil_one_above_top_level.where(potdens_stencil_one_above_top_level > 0, other=np.nan)
    potdens_stencil_one_below_top_level = potdens_stencil_shifted_down_one_cell*1 - potdens_stencil_shifted_down_two_cell*1
    # get rid of trailing negative values that occur at the ocean's bottom boundary..
    potdens_stencil_one_below_top_level = potdens_stencil_one_below_top_level.where(potdens_stencil_one_below_top_level > 0, other=np.nan)
    potdens_stencil_top_level = potdens_stencil*1 - potdens_stencil_shifted_down_one_cell*1       
    potdens_stencil_top_level = potdens_stencil_top_level.where(potdens_stencil_top_level > 0, other=np.nan)

    return potdens_stencil_top_level, potdens_stencil_one_above_top_level, potdens_stencil_shifted_down_one_cell




############################################################################################################
##################     A SIMPLE ISOPYCNAL VOLUME CALCULATION, WITH INTERPOLATION   ##########################
############################################################################################################

def sigma2_volume_simple_interp(pdens_level_list, pot_dens_array, grid, output_dataarray2, time=False,lessthan=False):
    """
    Interpolate sigma2 depth levels, get total volume below a certain density
    
    Attributes:
    ----------
    grid: mitgcm grid obect, passed to sigma_volume_simple_interp(...)
    sig_variable: string, prefix of sigma field files of which to load into memory
    
    Dependencies:
    ------------
    pdens_stencils()

    """
    z_hFacC_grid = (grid.Zl.rename({"k_l":"k"}) - grid.hFacC*grid.drF/2)
    #z_hFacC_grid = grid.Z
    stencil = pot_dens_array*0+1
    for density in pdens_level_list:
        print(density,end=" ")
        if lessthan:
            pdens_stencil = stencil.where(pot_dens_array>density,other=np.nan)
        else:
            pdens_stencil = stencil.where(pot_dens_array>=density,other=np.nan)            
        psten_top, psten_up1, pdens_sten_down1 = pdens_stencils(density,pot_dens_array,)
        # pdens is positive, get max
        pdens_slvl = (psten_top*pot_dens_array).max(dim="k",skipna=True)
        pdens_slvl_abv = (psten_up1*pot_dens_array).max(dim="k",skipna=True)
        # grid_Z_slvl is negative, get min
        grid_Z_slvl = (z_hFacC_grid*psten_top).min(dim="k",skipna=True) 
        grid_Z_slvl_abv = (z_hFacC_grid*psten_up1).min(dim="k",skipna=True)
        
        slope = (grid_Z_slvl_abv - grid_Z_slvl)/(pdens_slvl_abv - pdens_slvl)
        dpdens = density - pdens_slvl
        height_pdens_lvl = ((slope*dpdens).fillna(0) + grid_Z_slvl) + grid.Depth                                                                 
        volume_slvl = (grid.rA*height_pdens_lvl)
        volume_slvl.load()
        if time:
            output_dataarray2.loc[{"sig":density}] = volume_slvl.transpose("tile","j","i","time",).values
        else:
            output_dataarray2.loc[{"sig":density}] = volume_slvl.transpose("tile","j","i").values
        volume_slvl.close()
        
    return output_dataarray2




def sigma2_volume_nointerp(pdens_level_list, pot_dens_array, grid, output_dataarray2, time=False):
    """
    Interpolate sigma2 depth levels, get total volume below a certain density
    
    Attributes:
    ----------
    grid: mitgcm grid obect, passed to sigma_volume_simple_interp(...)
    sig_variable: string, prefix of sigma field files of which to load into memory
    
    Dependencies:
    ------------
    pdens_stencils()

    """
    print("hello world")
    z_hFacC_grid = (grid.Zl.rename({"k_l":"k"}) - grid.hFacC*grid.drF/2)
    z_hFacC_grid = grid.Z
    stencil = pot_dens_array*0+1
    for density in pdens_level_list:
        print(density,end=" ")
        pdens_stencil = stencil.where(pot_dens_array>density,other=np.nan)
        psten_top, psten_up1, pdens_sten_down1 = pdens_stencils(density,pot_dens_array,)
        # pdens is positive, get max 
        pdens_slvl = (psten_top*pot_dens_array).max(dim="k",skipna=True)
        pdens_slvl_abv = (psten_up1*pot_dens_array).max(dim="k",skipna=True)
        # grid_Z_slvl is negative, get min
        grid_Z_slvl = (z_hFacC_grid*psten_top).min(dim="k",skipna=True) 

        height_pdens_lvl = grid_Z_slvl + grid.Depth                                                                 
        volume_slvl = (grid.rA*height_pdens_lvl)
        volume_slvl.load()
        if time == True:
            output_dataarray2.loc[{"sig":density}] = volume_slvl.transpose("tile","j","i","time").values
        if time == False:
            output_dataarray2.loc[{"sig":density}] = volume_slvl.transpose("tile","j","i").values
        volume_slvl.close()
        
    return output_dataarray2





def calculate_isovol(grid,SIGMA,maskC,sig_levels,
                     time_slice=np.arange(0,288),subset_tseries=True,
                     save=True,label="", make_sigmax_arr = False,
                     interp=True,intime=True):
    """
    A wrapper function to set up and call the memory-intensive function sigma_volume_simple_interp(...)
    
    Attributes:
    ----------
    grid: mitgcm grid obect, passed to sigma_volume_simple_interp(...)
    sig_variable: string, prefix of sigma field files of which to load into memory
    
    Dependencies:
    ------------
    self.sigma2_volume_simple_interp()

    """
    print("hello")

    if make_sigmax_arr == True:
        SIGMA2_full = xr.open_dataset("SIGMA2_full.nc")
        SIGMA_tmax_k0 = (SIGMA2_full.SIGMA2*maskC).max(dim="time").isel(k=0)
        SIGMA_tmax_k0.to_netcdf("SIGMA2_tmax_k0.nc")
        # Make max sig arrays
        atl_maxsig_array = integrate_zonally.sigma2_zonal_surf_max(SIGMA_tmax_k0,
                                                                   so_atl_maskC_low_depth_filtered,
                                                                   monotonic=True)
        indpac_maxsig_array = integrate_zonally.sigma2_zonal_surf_max(SIGMA_tmax_k0,
                                                                      so_indpac_maskC_low_depth_filtered,
                                                                      monotonic=True)
    else:
        SIGMA_tmax_k0 = xr.open_dataarray("SIGMA2_tmax_k0.nc")
        atl_maxsig_array = xr.open_dataarray("atl_maxsig_array.nc")
        indpac_maxsig_array = xr.open_dataarray("indpac_maxsig_array.nc")

    # Values for "tiles", "i_vals", and "j_vals" are 
    tiles=np.arange(0,13)
    i_vals=np.arange(0,90)
    j_vals=np.arange(0,90)
    # set dimensions based on input dataset with sigma-space vertical levels..
    ntv_pdens_dims = (len(tiles),
                     len(sig_levels),
                     len(j_vals),
                     len(i_vals),
                     len(time_slice),
                     )
    new_coords = [tiles,sig_levels,i_vals,j_vals,time_slice]
    new_dims = ["tile","sig","j","i","time"]

    vol_sig_interp_out = xr.DataArray(data=np.zeros(ntv_pdens_dims),coords=new_coords,dims=new_dims)
    vol_sig_interp_out.assign_coords
    
    # Subset time series into yearlong chunks
    if subset_tseries==True:
        for t in time_slice:
            print(f"Started year {t}")
            if interp:
                vol_sig_interp_1yr = sigma2_volume_simple_interp(sig_levels,SIGMA.isel(time=int(t)),grid,vol_sig_interp_out.isel(time=t).copy(),time=False)
                vol_sig_interp_out.loc[{"time":int(t)}] = vol_sig_interp_1yr.values
            else: 
                vol_sig_interp_1yr = sigma2_volume_nointerp(sig_levels,SIGMA.isel(time=int(t)),grid,vol_sig_interp_out.isel(time=t).copy(),time=False)
                vol_sig_interp_out.loc[{"time":int(t)}] = vol_sig_interp_1yr.values
    else:
        if interp:
            vol_sig_interp_out = sigma2_volume_simple_interp(sig_levels,SIGMA.isel(time=time_slice),grid,vol_sig_interp_out.copy(),time=True)
        else: 
            vol_sig_interp_out = sigma2_volume_nointerp(sig_levels,SIGMA.isel(time=time_slice),grid,vol_sig_interp_out.copy(),time=True)
            

    if save==True:
        if interp:
            vol_sig_interp_out.to_netcdf("vol_sig"+label+"_interp_out.nc")
        else:
            vol_sig_interp_out.to_netcdf("vol_sig"+label+"_nointerp_out.nc")
        
    return vol_sig_interp_out

