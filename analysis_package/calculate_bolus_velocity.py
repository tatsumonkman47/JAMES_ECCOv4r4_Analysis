import warnings
warnings.filterwarnings('ignore')
import numpy as np
import xarray as xr
import cartopy as cart

from xmitgcm import open_mdsdataset
import xmitgcm
import ecco_v4_py as ecco

from netCDF4 import Dataset

import open_datasets

from importlib import reload

# reload modules during prototyping...
open_datasets = reload(open_datasets)



def calculate_bolus_velocity(grid_path="./ecco_grid/ECCOv4r3_grid.nc",tile_data_dir = "./nctiles_monthly/",time_slice=np.arange(0,288).):

	grid = xr.open_dataset(grid_path)
	GM_PSIX_var = "GM_PsiX"
	GM_PSIY_var = "GM_PsiY"

	GM_PSIX_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(tile_data_dir, 
	                                                  				GM_PSIX_var, 
	                                                  				time_slice)
	GM_PSIY_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(tile_data_dir, 
	                                                  				GM_PSIY_var, 
	                                                  				time_slice)

	GM_PSIX_ds_raw = GM_PSIX_ds_raw.drop("lon").drop("lat")
	GM_PSIY_ds_raw = GM_PSIY_ds_raw.drop("lon").drop("lat")

	# do some post-processing..
	GM_PSIX_ds_raw = GM_PSIX_ds_raw.assign_coords(k=np.arange(0,50),j=np.arange(0,90),i=np.arange(0,90))
	GM_PSIY_ds_raw = GM_PSIY_ds_raw.assign_coords(k=np.arange(0,50),j=np.arange(0,90),i=np.arange(0,90))

	# trim datasets if final nan padding value is present.. otherwise this won't change anything
	GM_PSIX_ds_raw = GM_PSIX_ds_raw.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	GM_PSIY_ds_raw = GM_PSIY_ds_raw.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	GM_PSIX_ds_raw.load()
	GM_PSIY_ds_raw.load()

	# add in tile coordinates
	tiles = np.arange(0,13)
	GM_PSIX_ds_raw["tile"] = tiles
	GM_PSIY_ds_raw["tile"] = tiles
	GM_PSIX_ds_raw = GM_PSIX_ds_raw.set_coords(["tile"])
	GM_PSIY_ds_raw = GM_PSIY_ds_raw.set_coords(["tile"])

	print(GM_PSIX_ds_raw)

	GM_PSIX_ds_raw["GM_PsiX"] = GM_PSIX_ds_raw.GM_PsiX.chunk((13,10,50,90,90))
	GM_PSIY_ds_raw["GM_PsiY"] = GM_PSIY_ds_raw.GM_PsiY.chunk((13,10,50,90,90))
	GM_PSIX_ds_plus_extra_k = xr.concat([GM_PSIX_ds_raw,GM_PSIX_ds_raw.isel(k=1)*0.],dim='k')
	GM_PSIY_ds_plus_extra_k = xr.concat([GM_PSIY_ds_raw,GM_PSIY_ds_raw.isel(k=1)*0.],dim='k')

	k_new_coords = np.arange(0,51)
	GM_PSIX_ds_plus_extra_k.coords.update({'k':k_new_coords})
	GM_PSIY_ds_plus_extra_k.coords.update({'k':k_new_coords})

	bolus_u = GM_PSIX_ds_plus_extra_k.copy(deep=True)
	bolus_v = GM_PSIY_ds_plus_extra_k.copy(deep=True)
	bolus_u = bolus_u.rename({'GM_PsiX':'bolus_uvel'})
	bolus_v = bolus_v.rename({'GM_PsiY':'bolus_vvel'})
	bolus_u.load()
	bolus_v.load()

	for k in range(0,50):
	    bolus_u.bolus_uvel[:,:,k,:,:] = (GM_PSIX_ds_plus_extra_k.GM_PsiX[:,:,k,:,:] - GM_PSIX_ds_plus_extra_k.GM_PsiX[:,:,k+1,:,:])/grid.drF[k]
	    bolus_v.bolus_vvel[:,:,k,:,:] = (GM_PSIY_ds_plus_extra_k.GM_PsiY[:,:,k,:,:] - GM_PSIY_ds_plus_extra_k.GM_PsiY[:,:,k+1,:,:])/grid.drF[k]


	bolus_u_final = bolus_u.isel(k=slice(0,50)).copy(deep=True).rename({'i':'i_g'})
	bolus_v_final = bolus_v.isel(k=slice(0,50)).copy(deep=True).rename({'j':'j_g'})

	bolus_u_final.to_netcdf("BOLUSUVELL_ds.nc")
	bolus_v_final.to_netcdf("BOLUSVVELL_ds.nc")






