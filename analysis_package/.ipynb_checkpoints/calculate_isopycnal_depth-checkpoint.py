def calculate_isopycnal_depth(PDENS_U_data_array, PDENS_V_data_array, pot_dens_coord, grid, lat_vals=np.arange(-88,88)):
    """
    A function for calculating isopycnal depth fields given potential density datasets
    
    Parameters
    ----------
    PDENS_U_data_array: xarray dataarray
    PDENS_V_data_array: xarray dataarray
    pot_dens_coord: numpy array
    grid: mitgcm grid..
    lat_vals: numpy array
    
    Returns
    -------
    
    
    """
    ######################################################################################################################
    ############################################ CALCULATE ISOPYCNAL DEPTH ###############################################
    ######################################################################################################################

    # get grid
    cds = grid.coords.to_dataset()
    grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)

    # create an empty array with a stretched depth dimension
    # Set the coordinates of the stretched depth dimension to potential density values..
    # set dimensions based on input dataset with modified vertical level spacing..
    # Standard ECCO datasets are xarrays with dimensions (13,288,50,90,90) corresponding to 
    # coordinates (tile, time, k, j, i)
    if "tile" in PDENS_U_data_array.dims:
        pot_dens_dims = (len(PDENS_U_data_array.tile),
                         len(PDENS_U_data_array.time),
                         len(pot_dens_coord),
                         90,
                         90)

        empty_pot_coords_data = np.zeros(pot_dens_dims)
        # trying to make this as general as possible, but need to keep an eye on this..
        new_coords = [PDENS_U_data_array.tile.values, PDENS_U_data_array.time.values, pot_dens_coord, np.arange(0,90), np.arange(0,90)]
        new_dims = ["tile","time","pot_rho","j","i"]
    else:
        pot_dens_dims = (len(PDENS_U_data_array.time),
                         len(pot_dens_coord),
                         90,
                         90)

        empty_pot_coords_data = np.zeros(pot_dens_dims)
        # trying to make this as general as possible, but need to keep an eye on this..
        new_coords = [PDENS_U_data_array.time.values, pot_dens_coord, np.arange(0,90), np.arange(0,90)]
        new_dims = ["time","pot_rho","j","i"]            
    potential_density_level_data_array = xr.DataArray(data=empty_pot_coords_data, coords=new_coords, dims=new_dims)
    potential_density_level_data_array_x = potential_density_level_data_array.copy(deep=True).rename({"i":"i_g"})
    potential_density_level_data_array_y = potential_density_level_data_array.copy(deep=True).rename({"j":"j_g"})

    pot_dens_array_x = PDENS_U_data_array.copy(deep=True)
    pot_dens_array_y = PDENS_V_data_array.copy(deep=True)

    for density in pot_dens_coord:
        print("Started " + str(density) + " surface") 
        potdens_stencil_x_0 = pot_dens_array_x > density
        potdens_stencil_y_0 = pot_dens_array_y > density
        # this step is critical to remove low density anomalies in the deep ocean from the stencil...
        potdens_stencil_x = potdens_stencil_x_0.cumsum(dim="k") > 0
        potdens_stencil_y = potdens_stencil_y_0.cumsum(dim="k") > 0

        ##################################################################################################################
        ###########################################     START INTERPOLATION    ###########################################
        ##################################################################################################################

        # set end-appended value equal to 1 for subtraction step..
        potdens_stencil_x_shifted_up_one_cell = xr.concat((potdens_stencil_x.isel(k=slice(1,50)),
                                                           potdens_stencil_x.isel(k=49)),
                                                          dim="k").assign_coords(k=np.arange(0,50))
        potdens_stencil_y_shifted_up_one_cell = xr.concat((potdens_stencil_y.isel(k=slice(1,50)),
                                                           potdens_stencil_y.isel(k=49)),
                                                          dim="k").assign_coords(k=np.arange(0,50))
        potdens_stencil_x_shifted_down_one_cell = xr.concat((potdens_stencil_x.isel(k=0)*0,
                                                             potdens_stencil_x.isel(k=slice(0,49))),
                                                            dim="k").assign_coords(k=np.arange(0,50))
        potdens_stencil_y_shifted_down_one_cell = xr.concat((potdens_stencil_y.isel(k=0)*0,
                                                             potdens_stencil_y.isel(k=slice(0,49))),
                                                            dim="k").assign_coords(k=np.arange(0,50))

        potdens_stencil_x_one_above_top_level = potdens_stencil_x_shifted_up_one_cell*1 - potdens_stencil_x*1
        potdens_stencil_y_one_above_top_level = potdens_stencil_y_shifted_up_one_cell*1 - potdens_stencil_y*1
        # get rid of trailing negative values that occur at the ocean's bottom boundary..
        potdens_stencil_x_one_above_top_level = potdens_stencil_x_one_above_top_level.where(potdens_stencil_x_one_above_top_level > 0,
                                                                                            other=0)
        potdens_stencil_y_one_above_top_level = potdens_stencil_y_one_above_top_level.where(potdens_stencil_y_one_above_top_level > 0,
                                                                                            other=0)

        potdens_stencil_x_top_level = potdens_stencil_x*1 - potdens_stencil_x_shifted_down_one_cell*1
        potdens_stencil_y_top_level = potdens_stencil_y*1 - potdens_stencil_y_shifted_down_one_cell*1
        # turn zeros into nans..
        # NOTE SOMETIMES YOU GET PROTRUSIONS OF DENSITY ANOMALIES THAT SEEM TO CREATE TWO DENSITY SURFACES, LEADING TO A VALUE OF
        # 2 IN THE STENCIL.. I eliminated this using "potdens_stencil_x = potdens_stencil_x_0.cumsum(dim="k") > 0" a couple lines above.
        potdens_stencil_x_top_level = potdens_stencil_x_top_level.where(potdens_stencil_x_top_level > 0, other=np.nan)
        potdens_stencil_y_top_level = potdens_stencil_y_top_level.where(potdens_stencil_y_top_level > 0, other=np.nan)
        potdens_stencil_x_one_above_top_level = potdens_stencil_x_one_above_top_level.where(potdens_stencil_x_one_above_top_level > 0,
                                                                                            other=np.nan)
        potdens_stencil_y_one_above_top_level = potdens_stencil_y_one_above_top_level.where(potdens_stencil_y_one_above_top_level > 0,
                                                                                            other=np.nan)
        # multiply depth values by -1 to make them positive..
        depth_above_x_top_level_raw = (-1*potdens_stencil_x_one_above_top_level.fillna(0)*grid.Z).sum(dim="k")
        depth_x_top_level_raw = (-1*potdens_stencil_x_top_level.fillna(0)*grid.Z).sum(dim="k",skipna=True)
        depth_above_y_top_level_raw = (-1*potdens_stencil_y_one_above_top_level.fillna(0)*grid.Z).sum(dim="k",skipna=True)
        depth_y_top_level_raw = (-1*potdens_stencil_y_top_level.fillna(0)*grid.Z).sum(dim="k",skipna=True)
        # turn zeros into nans..
        depth_above_x_top_level = depth_above_x_top_level_raw.where(depth_above_x_top_level_raw > 0, other=np.nan)
        depth_x_top_level = depth_x_top_level_raw.where(depth_x_top_level_raw > 0, other=np.nan)
        depth_above_y_top_level = depth_above_y_top_level_raw.where(depth_above_y_top_level_raw > 0, other=np.nan)
        depth_y_top_level = depth_y_top_level_raw.where(depth_y_top_level_raw > 0, other=np.nan)

        potdens_above_x_top_level = (potdens_stencil_x_one_above_top_level.fillna(0)*pot_dens_array_x.fillna(0)).sum(dim="k")
        potdens_x_top_level = (potdens_stencil_x_top_level.fillna(0)*pot_dens_array_x.fillna(0)).sum(dim="k")
        potdens_above_y_top_level = (potdens_stencil_y_one_above_top_level.fillna(0)*pot_dens_array_y.fillna(0)).sum(dim="k")
        potdens_y_top_level = (potdens_stencil_y_top_level.fillna(0)*pot_dens_array_y.fillna(0)).sum(dim="k")
        # turn zeros into nans..
        potdens_above_x_top_level = potdens_above_x_top_level.where(potdens_above_x_top_level > 0, other=np.nan)
        potdens_x_top_level = potdens_x_top_level.where(potdens_x_top_level > 0, other=np.nan)
        potdens_above_y_top_level = potdens_above_y_top_level.where(potdens_above_y_top_level > 0, other=np.nan)
        potdens_y_top_level = potdens_y_top_level.where(potdens_y_top_level > 0, other=np.nan)


        # nan out non-existant top level cells (when density outcrops on the native-grid surface)
        depth_above_x_top_level = depth_above_x_top_level.where(depth_above_x_top_level != 0, other = np.nan)
        depth_x_top_level = depth_x_top_level.where(depth_x_top_level != 0, other=np.nan)
        depth_potdens_slope_x = (depth_above_x_top_level - depth_x_top_level)/(potdens_above_x_top_level - potdens_x_top_level)                                                
        depth_potdens_slope_y = (depth_above_y_top_level - depth_y_top_level)/(potdens_above_y_top_level - potdens_y_top_level)

        # this is an issue... need to account for those low desnity protrusions..
        h_array_x_0 = (density - potdens_x_top_level)*depth_potdens_slope_x
        h_array_x = -1*h_array_x_0.where(h_array_x_0 < 0, other=0)
        h_array_y_0 = (density - potdens_y_top_level)*depth_potdens_slope_y
        h_array_y = -1*h_array_y_0.where(h_array_y_0 < 0, other=0)
        
        depth_x_top_level = depth_x_top_level.where(potdens_above_x_top_level < density,other=0)
        depth_y_top_level = depth_y_top_level.where(potdens_above_y_top_level < density,other=0)
        
        depth_x_top_level.load()
        depth_y_top_level.load()
        h_array_x.load()
        h_array_y.load()
        
        # subtract heights since we are defining depth as a positive quantity
        total_depth_x = depth_x_top_level - h_array_x
        total_depth_x.load()
        total_depth_y = depth_y_top_level - h_array_y    
        total_depth_y.load()
        
        potential_density_level_data_array_x.loc[{"pot_rho":density}] = total_depth_x
        potential_density_level_data_array_y.loc[{"pot_rho":density}] = total_depth_y

        ##################################################################################################################
        ###########################################     END INTERPOLATION    #############################################
        ##################################################################################################################   
    
                                          
    return potential_density_level_data_array_x, potential_density_level_data_array_y
    


