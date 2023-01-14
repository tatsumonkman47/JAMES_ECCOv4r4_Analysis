import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import xarray as xr

import sys
sys.path.append('/Users/tatsumonkman/3rd_party_software/ECCOv4-py')
import ecco_v4_py as ecco


######################################################################################################################
################################################### LOAD GRID ########################################################
######################################################################################################################
grid_path = "ECCO-GRID.nc"
grid = xr.open_dataset(grid_path)
cds = grid.coords.to_dataset()
grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)



############################################################################################################
##########################     MAKE FUNCTIONS TO DERIVE LATITUDINAL MEAN ARRAYS    ###############################
############################################################################################################

def make_latC_mask(lat):
    lat_maskX, lat_maskY =  ecco.vector_calc.get_latitude_masks(lat, cds['YC'], grid_xmitgcm)
    lat_maskC =  -1*lat_maskX.rename({"i_g":"i"}) + lat_maskY.rename({"j_g":"j"})
    
    return lat_maskC


def make_directional_lat_masks(lat):
    lat_maskX, lat_maskY =  ecco.vector_calc.get_latitude_masks(lat, cds['YC'], grid_xmitgcm)

    return lat_maskX, lat_maskY


def make_time_depth_lat_dataarray(field,lat_vals=np.arange(-88,88)):
    # set dimensions based on input dataset with modified vertical level spacing..
    if "k" in field.dims:
        depth_dim = "k"
    elif "sig" in field.dims:
        depth_dim = "sig"

    if "time" in field.dims:
        pot_dens_dims = (len(field.time.values),len(field[depth_dim]),len(lat_vals),)
        empty_data = np.zeros(pot_dens_dims)
        new_coords = [field.time.values,field[depth_dim],lat_vals,]
        new_dims = ["time",depth_dim,"lat",]
        depth_lat_da = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
    else:
        pot_dens_dims = (len(field[depth_dim]),len(lat_vals),)
        empty_data = np.zeros(pot_dens_dims)
        new_coords = [field[depth_dim],lat_vals,]
        new_dims = [depth_dim,"lat",]
        depth_lat_da = xr.DataArray(data=empty_data,coords=new_coords,dims=new_dims)
        
    return depth_lat_da


def make_zonal_mean_dataarrays(field,basin_maskC,lat_vals=np.arange(-88,88),zonal="mean"):
    # field needs to be a C-grid ECCO dataarray
    depth_lat_da = make_time_depth_lat_dataarray(field)
    if "k" in field.dims:
        depth_dim = "k"
    elif "sig" in field.dims:
        depth_dim = "sig"
    
    for lat in lat_vals:
        print(lat,end=" ")
        lat_maskC = make_latC_mask(lat)
        lat_maskC = lat_maskC.where(lat_maskC > 0, other=np.nan)
        if zonal=="mean":
            a = (field*lat_maskC*basin_maskC).mean(dim=["i","j","tile"])
        if zonal=="sum":
            a = (field*lat_maskC*basin_maskC).sum(dim=["i","j","tile"])
        if "time" in field.dims:
            depth_lat_da.loc[{"lat":lat}] = a.transpose("time",depth_dim)
        else:
            depth_lat_da.loc[{"lat":lat}] = a

    return depth_lat_da


############################################################################################################
##################################     PLOT ZONAL MEAN DATASETS    #########################################
############################################################################################################


def plot_lat_field(tmp_plot,grid,levels=40,lowest_depth=37,lat_min=0,lat_max=170,w=30,h=10,colormap='viridis',vmin=None,title=""):
    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s

    dep = -1*grid.Z
    tmp_plot = tmp_plot

    plt.figure(figsize=(w,h))
    plt.contourf(tmp_plot.lat[lat_min:lat_max],
                 dep[:lowest_depth],
                 tmp_plot[:lowest_depth,lat_min:lat_max],
                 levels=levels,
                 vmin=vmin,
                 cmap=colormap)
    plt.colorbar()
    CS = plt.contour(tmp_plot.lat[lat_min:lat_max],
                     dep[:lowest_depth],
                     tmp_plot[:lowest_depth,lat_min:lat_max],
                     levels=levels,
                     vmin=vmin,
                     colors='k')
    # Recast levels to new class
    CS.levels = [nf(val) for val in CS.levels]
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)
    plt.xlabel("Latitude (\u00b0)",fontsize=24,fontname="Times New Roman")
    plt.ylabel("Depth (m)",fontsize=24,fontname="Times New Roman")
    plt.title(title,fontsize=36,fontname="Times New Roman")
    plt.grid()
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    
    
def plot_lat_field12(tmp_plot1,tmp_plot2,grid,levels=40,highest_depth=0,lowest_depth=37,lat_min=0,lat_max=170,w=30,h=10,colormap='viridis',vmin=None,vmax=None,title="",pcolormin=-50):
    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s

    dep = tmp_plot1.k

    plt.figure(figsize=(w,h))
    CS1 = plt.contour(tmp_plot1.lat[lat_min:lat_max],
                     grid.Z[highest_depth:lowest_depth],
                     tmp_plot1[highest_depth:lowest_depth,lat_min:lat_max],
                     levels=levels,
                     vmin=vmin,
                     vmax=vmax,
                     colors='k')
    CS2 = plt.contour(tmp_plot2.lat[lat_min:lat_max],
                     grid.Z[highest_depth:lowest_depth],
                     tmp_plot2[highest_depth:lowest_depth,lat_min:lat_max],
                     levels=levels,
                     vmin=vmin,
                     vmax=vmax,
                     colors='r')
    
    # Recast levels to new class
    CS1.levels = [nf(val) for val in CS1.levels]
    CS2.levels = [nf(val) for val in CS2.levels]
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=10)
    plt.clabel(CS2, CS2.levels, inline=True, fontsize=10)
    plt.xlabel("Latitude (\u00b0)",fontsize=24,fontname="Times New Roman")
    plt.ylabel("$\sigma_{2}$",fontsize=24,fontname="Times New Roman")
    plt.xticks(tmp_plot1.lat[::2])
    plt.yticks(grid.Z[highest_depth:lowest_depth])
    plt.title(title,fontsize=36,fontname="Times New Roman")
    plt.grid()
    plt.show()
    plt.close()

    

def plot_lat_field12_final(tmp_plot1, tmp_plot2, grid, levels=40, highest_depth=0,lowest_depth=37,lat_min=0,lat_max=170,w=30,h=10,colormap='viridis',vmin=None,vmax=None,title="",pcolormin=-50):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')    
    plt.figure(figsize=(w,h))

    
    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s
    
    plt.contourf(tmp_plot1.lat[lat_min:lat_max],
                     grid.Z[highest_depth:lowest_depth],
                     tmp_plot1[highest_depth:lowest_depth,lat_min:lat_max],
                     levels=levels,
                     vmin=vmin,
                     vmax=vmax+5,
                     cmap='coolwarm')
    
    CS1 = plt.contour(tmp_plot1.lat[lat_min:lat_max],
                     grid.Z[highest_depth:lowest_depth],
                     tmp_plot1[highest_depth:lowest_depth,lat_min:lat_max],
                     levels=levels,
                     vmin=vmin,
                     vmax=vmax,
                     colors='k')
    CS2 = plt.contour(tmp_plot2.lat[lat_min:lat_max],
                     grid.Z[highest_depth:lowest_depth],
                     tmp_plot2[highest_depth:lowest_depth,lat_min:lat_max],
                     levels=levels,
                     vmin=vmin,
                     vmax=vmax,
                     colors='r')
    #plt.vlines(-32,ymin=-5906,ymax=0,linewidth=3,linestyle="--")
    # Recast levels to new class
    CS1.levels = [nf(val) for val in CS1.levels]
    CS2.levels = [nf(val) for val in CS2.levels]
    plt.ylim(top=0,bottom=-5906)
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=15)
    plt.clabel(CS2, CS2.levels, inline=True, fontsize=15)
    plt.xlabel("Latitude (\u00b0)",fontsize=32,fontname="Times New Roman")
    plt.ylabel("Depth (m)",fontsize=32,fontname="Times New Roman")
    plt.xticks(np.arange(lat_min-85,lat_max-86,5),fontsize=20)
    tick_coords = [10,20,25,30,35,37,40,41,42,43,44,45,46,47,48,49]
    plt.yticks(grid.Z[tick_coords],fontsize=20)
    plt.title(title,fontsize=36,fontname="Times New Roman")
    plt.grid()
    plt.show()
    plt.close()



