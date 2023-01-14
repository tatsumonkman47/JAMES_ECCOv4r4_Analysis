from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import dask.array as da
import xarray as xr
import os

from seawater import eos80

from xmitgcm import open_mdsdataset
import xmitgcm
import sys
sys.path.append('/Users/tatsumonkman/3rd_party_software/ECCOv4-py')
import ecco_v4_py as ecco


from netCDF4 import Dataset


from analysis_package import plotting_functions
from analysis_package import open_datasets
from analysis_package import ecco_masks
from analysis_package import integrate_zonally
from analysis_package import lat_fields






from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))



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




def calculate_in_sig2space(FIELD,SIGMA2,sig_lvls,grid,interpolate=False):
    # Values for "tiles", "i_vals", and "j_vals" are 
    tiles=FIELD.tile.values
    i_vals=np.arange(0,90)
    j_vals=np.arange(0,90)
    # set dimensions based on input dataset with sigma-space vertical levels..
    if "time" in FIELD.dims:
        time_vals = FIELD.time.values

        ntv_pdens_dims = (len(time_vals),
                         len(tiles),
                         len(sig_lvls),
                         len(j_vals),
                         len(i_vals),
                         )
        new_coords = [time_vals,tiles,sig_lvls,i_vals,j_vals]
        new_dims = ["time","tile","sig","j","i"]
    else:
        ntv_pdens_dims = (
                         len(tiles),
                         len(sig_lvls),
                         len(j_vals),
                         len(i_vals),
                         )
        new_coords = [tiles,sig_lvls,i_vals,j_vals]
        new_dims = ["tile","sig","j","i"]
    output_da = xr.DataArray(data=np.zeros(ntv_pdens_dims),coords=new_coords,dims=new_dims)
    output_da.assign_coords    
    
    z_hFacC_grid = (grid.Zl.rename({"k_l":"k"}) - grid.hFacC*grid.drF/2)
    z_hFacC_grid = grid.Z
    stencil = SIGMA2*0+1
    if interpolate:
        print("interpolating...")
    
    for density in sig_lvls:
        print(density,end=" ")
        if interpolate:
            pdens_stencil = stencil.where(SIGMA2>density,other=np.nan)
            psten_top, psten_up1, pdens_sten_down1 = pdens_stencils(density,SIGMA2,)
            pdens_slvl = (psten_top*SIGMA2).max(dim="k",skipna=True)
            FIELD_slvl = (psten_top*FIELD).max(dim="k",skipna=True)
            pdens_slvl_abv = (psten_up1*SIGMA2).max(dim="k",skipna=True)
            FIELD_slvl_abv = (psten_up1*FIELD).max(dim="k",skipna=True)
            
            # grid_Z_slvl is negative, get min
            grid_Z_slvl = (z_hFacC_grid*psten_top).min(dim="k",skipna=True) 
            grid_Z_slvl_abv = (z_hFacC_grid*psten_up1).min(dim="k",skipna=True)  
            slope = (FIELD_slvl_abv - FIELD_slvl)/(pdens_slvl_abv - pdens_slvl)
            dpdens = density - pdens_slvl
            FIELD_lvl_interpolated = (slope*dpdens).fillna(0) + FIELD_slvl
            if "time" in FIELD.dims:
                output_da.loc[{"sig":density}] = FIELD_lvl_interpolated.transpose("time","tile","j","i").values + FIELD_slvl
            else:
                output_da.loc[{"sig":density}] = FIELD_lvl_interpolated.transpose("tile","j","i").values            
            
        else:
            potdens_stencil_top_level,_,__, = pdens_stencils(density,SIGMA2)
            FIELD_lvl = (FIELD*potdens_stencil_top_level).sum(dim="k",skipna=True)
            if "time" in FIELD.dims:
                output_da.loc[{"sig":density}] = FIELD_lvl.transpose("time","tile","j","i").values
            else:
                output_da.loc[{"sig":density}] = FIELD_lvl.transpose("tile","j","i").values
        
    return output_da



def integrate_vertically_in_sig2space(FIELD,SIGMA2,sig_lvls,grid,interpolate=False):
    # Values for "tiles", "i_vals", and "j_vals" are 
    tiles=FIELD.tile.values
    i_vals=np.arange(0,90)
    j_vals=np.arange(0,90)
    # set dimensions based on input dataset with sigma-space vertical levels..
    if "time" in FIELD.dims:
        time_vals = FIELD.time.values

        ntv_pdens_dims = (len(time_vals),
                         len(tiles),
                         len(sig_lvls),
                         len(j_vals),
                         len(i_vals),
                         )
        new_coords = [time_vals,tiles,sig_lvls,i_vals,j_vals]
        new_dims = ["time","tile","sig","j","i"]
    else:
        ntv_pdens_dims = (
                         len(tiles),
                         len(sig_lvls),
                         len(j_vals),
                         len(i_vals),
                         )
        new_coords = [tiles,sig_lvls,i_vals,j_vals]
        new_dims = ["tile","sig","j","i"]
    output_da = xr.DataArray(data=np.zeros(ntv_pdens_dims),coords=new_coords,dims=new_dims)
    output_da.assign_coords    
    
    z_depth_hFacC = (grid.Zl.rename({"k_l":"k"}) - (grid.hFacC)*grid.drF/2)
    
    if interpolate:
        print("interpolating...")
    
    for density in sig_lvls:
        print(density,end=" ")
        if interpolate:
            # this interpolation scheme treats tendencies as constant in grid boxes for conservation
            # purposes...
            psten_top, psten_p1, pdens_stencil_down1 = pdens_stencils(density,SIGMA2,)
            ############################################################################################################
            ###########################################     START INTERPOLATION    #####################################
            ############################################################################################################
            # set end-appended value equal to 1 for subtraction step..	    
            # multiply depth values by -1 to make them positive..
            depth_top = (z_depth_hFacC*psten_top).min(dim="k")
            depth_p1 = (z_depth_hFacC*psten_p1).min(dim="k")
            pdens_top = (SIGMA2*psten_top).max(dim="k")
            pdens_p1 = (SIGMA2*psten_p1).max(dim="k")
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

            # Perform vertical integration
            FIELD_top_level = ((psten_top*percent_top)*FIELD).sum(dim="k")
            FIELD_above_top_level = ((psten_p1*percent_p1)*FIELD).sum(dim="k")
                
            FIELD_interpolated = FIELD_top_level + FIELD_above_top_level
        
            depth_integrated_FIELD_0 = FIELD*(pdens_stencil_down1.where(pdens_stencil_down1>0,other=np.nan))
            depth_integrated_FIELD_0.load()
            FIELD_lvl = depth_integrated_FIELD_0.sum(dim='k') + FIELD_interpolated
            
        else:
            potdens_stencil_top_level = SIGMA2.where(SIGMA2>density,other=np.nan)*0+1
            FIELD_lvl = (FIELD*potdens_stencil_top_level).sum(dim="k",skipna=True)
        
        # Save to output file!
        if "time" in FIELD.dims:
            output_da.loc[{"sig":density}] = FIELD_lvl.transpose("time","tile","j","i").values
        else:
            output_da.loc[{"sig":density}] = FIELD_lvl.transpose("tile","j","i").values

    return output_da






def volume_weighted_mean_in_sig2space_interp(FIELD,SIGMA2,sig_lvls,grid,interpolate=False):
    # Values for "tiles", "i_vals", and "j_vals" are 
    tiles=FIELD.tile.values
    i_vals=np.arange(0,90)
    j_vals=np.arange(0,90)
    # set dimensions based on input dataset with sigma-space vertical levels..
    if "time" in FIELD.dims:
        time_vals = FIELD.time.values

        ntv_pdens_dims = (len(time_vals),
                         len(tiles),
                         len(sig_lvls),
                         len(j_vals),
                         len(i_vals),
                         )
        new_coords = [time_vals,tiles,sig_lvls,i_vals,j_vals]
        new_dims = ["time","tile","sig","j","i"]
    else:
        ntv_pdens_dims = (
                         len(tiles),
                         len(sig_lvls),
                         len(j_vals),
                         len(i_vals),
                         )
        new_coords = [tiles,sig_lvls,i_vals,j_vals]
        new_dims = ["tile","sig","j","i"]
    output_da = xr.DataArray(data=np.zeros(ntv_pdens_dims),coords=new_coords,dims=new_dims)
    output_da.assign_coords    
    
    z_depth_hFacC = (grid.Zl.rename({"k_l":"k"}) - (grid.hFacC)*grid.drF/2)
    volume = grid.drF*grid.rA*grid.hFacC
    volume = volume.where(volume>0,other=np.nan)
    
    if interpolate:
        print("interpolating...")
    
        for density in sig_lvls:
            print(density,end=" ")
            if interpolate:
                # this interpolation scheme treats tendencies as constant in grid boxes for conservation
                # purposes...
                psten_top, psten_p1, pdens_stencil_down1 = pdens_stencils(density,SIGMA2,)
                ############################################################################################################
                ###########################################     START INTERPOLATION    #####################################
                ############################################################################################################
                # set end-appended value equal to 1 for subtraction step..	    
                # multiply depth values by -1 to make them positive..
                depth_top = (z_depth_hFacC*psten_top).min(dim="k")
                depth_p1 = (z_depth_hFacC*psten_p1).min(dim="k")
                pdens_top = (SIGMA2*psten_top).max(dim="k")
                pdens_p1 = (SIGMA2*psten_p1).max(dim="k")
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

                # Perform vertical integration
                FIELD_top_level = ((psten_top*percent_top)*FIELD).sum(dim="k")
                FIELD_above_top_level = ((psten_p1*percent_p1)*FIELD).sum(dim="k")
                volume_top_level = ((psten_top*percent_top)*volume).sum(dim="k")
                volume_above_top_level = ((psten_p1*percent_p1)*volume).sum(dim="k")

                FIELD_interpolated = FIELD_top_level + FIELD_above_top_level
                volume_interpolated = volume_top_level + volume_above_top_level

                depth_integrated_FIELD_0 = FIELD*(pdens_stencil_down1.where(pdens_stencil_down1>0,other=np.nan))
                depth_integrated_FIELD_0.load()
                depth_integrated_volume_0 = volume*(pdens_stencil_down1.where(pdens_stencil_down1>0,other=np.nan))
                depth_integrated_volume_0.load()

                FIELD_timesvol_lvl = (depth_integrated_FIELD_0*depth_integrated_volume_0).sum(dim='k') + FIELD_interpolated*volume_interpolated
                vol_lvl = (depth_integrated_volume_0).sum(dim='k') + volume_interpolated
            
        # Save to output file!
        if "time" in FIELD.dims:
            output_da.loc[{"sig":density}] = (FIELD_timesvol_lvl/vol_lvl).transpose("time","tile","j","i").values
        else:
            output_da.loc[{"sig":density}] = (FIELD_timesvol_lvl/vol_lvl).transpose("tile","j","i").values


    else:
        print("not interpolating")
        for density in sig_lvls:
            print(density,end=" ")
            FIELD_timesvol_lvl=0
            potdens_volume_stencil_top_level = (SIGMA2.where(SIGMA2>density,other=np.nan)*0+1)*volume
            FIELD_lvl = (FIELD*potdens_volume_stencil_top_level).sum(dim="k",skipna=True)/potdens_volume_stencil_top_level.sum(dim="k")

            # Save to output file!
            if "time" in FIELD.dims:
                output_da.loc[{"sig":density}] = (FIELD_lvl).transpose("time","tile","j","i").values
            else:
                output_da.loc[{"sig":density}] = (FIELD_lvl).transpose("tile","j","i").values



            
            
    return output_da



