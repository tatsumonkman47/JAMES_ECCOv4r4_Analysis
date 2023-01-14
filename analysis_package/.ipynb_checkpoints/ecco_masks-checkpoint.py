import sys
sys.path.append('/Users/tatsumonkman/3rd_party_software/ECCOv4-py')
import ecco_v4_py as ecco

import ecco_v4_py as ecco 
import numpy as np
import xarray as xr 

def get_basin_masks(maskW, maskS, maskC):
	
	mexico_mask_W, mexico_mask_S, mexico_mask_C = ecco.get_basin_mask("mexico",maskW,less_output=True), ecco.get_basin_mask("mexico",maskS,less_output=True), ecco.get_basin_mask("mexico",maskC,less_output=True)
	baffin_mask_W, baffin_mask_S, baffin_mask_C = ecco.get_basin_mask("baffin",maskW,less_output=True), ecco.get_basin_mask("baffin",maskS,less_output=True), ecco.get_basin_mask("baffin",maskC,less_output=True)
	north_mask_W, north_mask_S, north_mask_C = ecco.get_basin_mask("north",maskW,less_output=True), ecco.get_basin_mask("north",maskS,less_output=True), ecco.get_basin_mask("north",maskC,less_output=True)
	hudson_mask_W, hudson_mask_S, hudson_mask_C = ecco.get_basin_mask("hudson",maskW,less_output=True), ecco.get_basin_mask("hudson",maskS,less_output=True), ecco.get_basin_mask("hudson",maskC,less_output=True)
	gin_mask_W, gin_mask_S, gin_mask_C = ecco.get_basin_mask("gin",maskW,less_output=True), ecco.get_basin_mask("gin",maskS,less_output=True), ecco.get_basin_mask("gin",maskC,less_output=True)
	atl_mask_W, atl_mask_S, atl_mask_C = ecco.get_basin_mask("atl",maskW,less_output=True), ecco.get_basin_mask("atl",maskS,less_output=True), ecco.get_basin_mask("atl",maskC,less_output=True)

	full_atl_basin_mask_W = atl_mask_W + baffin_mask_W + north_mask_W + gin_mask_W + mexico_mask_W + hudson_mask_W
	full_atl_basin_mask_S = atl_mask_S + baffin_mask_S + north_mask_S + gin_mask_S + mexico_mask_S + hudson_mask_S
	full_atl_basin_mask_C = atl_mask_C + baffin_mask_C + north_mask_C + gin_mask_C + mexico_mask_C + hudson_mask_C

	ind_mask_W, ind_mask_S, ind_mask_C = ecco.get_basin_mask("ind",maskW,less_output=True), ecco.get_basin_mask("ind",maskS,less_output=True), ecco.get_basin_mask("ind",maskC,less_output=True)
	pac_mask_W, pac_mask_S, pac_mask_C = ecco.get_basin_mask("pac",maskW,less_output=True), ecco.get_basin_mask("pac",maskS,less_output=True), ecco.get_basin_mask("pac",maskC,less_output=True)
	southChina_mask_W, southChina_mask_S, southChina_mask_C = ecco.get_basin_mask("southChina",maskW,less_output=True), ecco.get_basin_mask("southChina",maskS,less_output=True), ecco.get_basin_mask("southChina",maskC,less_output=True)
	japan_W, japan_S, japan_C = ecco.get_basin_mask("japan",maskW,less_output=True), ecco.get_basin_mask("japan",maskS,less_output=True), ecco.get_basin_mask("japan",maskC,less_output=True)
	eastChina_W, eastChina_S, eastChina_C = ecco.get_basin_mask("eastChina",maskW,less_output=True), ecco.get_basin_mask("eastChina",maskS,less_output=True), ecco.get_basin_mask("eastChina",maskC,less_output=True)
	timor_W, timor_S, timor_C = ecco.get_basin_mask("timor",maskW,less_output=True), ecco.get_basin_mask("timor",maskS,less_output=True), ecco.get_basin_mask("timor",maskC,less_output=True)
	java_W, java_S, java_C = ecco.get_basin_mask("java",maskW,less_output=True), ecco.get_basin_mask("java",maskS,less_output=True), ecco.get_basin_mask("java",maskC,less_output=True)
	bering_mask_W, bering_mask_S, bering_mask_C = ecco.get_basin_mask("bering",maskW,less_output=True), ecco.get_basin_mask("bering",maskS,less_output=True), ecco.get_basin_mask("bering",maskC,less_output=True)
	okhotsk_mask_W, okhotsk_mask_S, okhotsk_mask_C = ecco.get_basin_mask("okhotsk",maskW,less_output=True), ecco.get_basin_mask("okhotsk",maskS,less_output=True), ecco.get_basin_mask("okhotsk",maskC,less_output=True)

	full_indpac_basin_mask_W = (ind_mask_W + pac_mask_W + southChina_mask_W + japan_W + eastChina_W 
	                            + timor_W + java_W + okhotsk_mask_W )
	full_indpac_basin_mask_S = (ind_mask_S + pac_mask_S + southChina_mask_S + japan_S + eastChina_S 
	                            + timor_S + java_S + okhotsk_mask_S + bering_mask_S)
	full_indpac_basin_mask_C = (ind_mask_C + pac_mask_C + southChina_mask_C + japan_C + eastChina_C 
	                            + timor_C + java_C + okhotsk_mask_C )

	j_transport_coords = np.arange(0,90)
	i_transport_coords = np.arange(0,90)
	# WATCH OUT FOR THIS INDEXING!
	tile_coords = np.arange(0,13)

	southern_ocean_mask_C = (maskC.where(maskC["lat"] < -33)*0 + 1.)
	southern_ocean_mask_C = southern_ocean_mask_C.assign_coords(j=j_transport_coords,
	                                                            i=i_transport_coords,
	                                                            tile=tile_coords)
	southern_ocean_mask_C.loc[{"tile":4}] = maskC.isel(tile=4).where(maskC.isel(tile=4)["lat"] < -32)*0+1
	southern_ocean_mask_C.loc[{"tile":1}] = maskC.isel(tile=1).where(maskC.isel(tile=4)["lat"] < -32)*0+1
	southern_ocean_mask_W = southern_ocean_mask_C.rename({"i":"i_g"}).drop("lon").drop("lat")
	southern_ocean_mask_S = southern_ocean_mask_C.rename({"j":"j_g"}).drop("lon").drop("lat")

	so_atl_basin_mask_W = full_atl_basin_mask_W + southern_ocean_mask_W.where(southern_ocean_mask_W==1, other=0) 
	so_atl_basin_mask_S = full_atl_basin_mask_S + southern_ocean_mask_S.where(southern_ocean_mask_S==1, other=0)
	so_atl_basin_mask_C = full_atl_basin_mask_C + southern_ocean_mask_C.where(southern_ocean_mask_C==1, other=0)
	so_atl_basin_mask_W = so_atl_basin_mask_W.where(so_atl_basin_mask_W > 0, other=np.nan)*0 + 1
	so_atl_basin_mask_S = so_atl_basin_mask_S.where(so_atl_basin_mask_S > 0, other=np.nan)*0 + 1
	so_atl_basin_mask_C = so_atl_basin_mask_C.where(so_atl_basin_mask_C > 0, other=np.nan)*0 + 1


	so_indpac_basin_mask_W = full_indpac_basin_mask_W.where(full_indpac_basin_mask_W==1,other=0) + southern_ocean_mask_W.where(southern_ocean_mask_W==1,other=0) 
	so_indpac_basin_mask_S = full_indpac_basin_mask_S.where(full_indpac_basin_mask_S==1,other=0) + southern_ocean_mask_S.where(southern_ocean_mask_S==1,other=0)
	so_indpac_basin_mask_C = full_indpac_basin_mask_C.where(full_indpac_basin_mask_C==1,other=0) + southern_ocean_mask_C.where(southern_ocean_mask_C==1,other=0)
	so_indpac_basin_mask_W = so_indpac_basin_mask_W.where(so_indpac_basin_mask_W != 0, other=np.nan)*0 + 1
	so_indpac_basin_mask_S = so_indpac_basin_mask_S.where(so_indpac_basin_mask_S != 0, other=np.nan)*0 + 1
	so_indpac_basin_mask_C = so_indpac_basin_mask_C.where(so_indpac_basin_mask_C != 0, other=np.nan)*0 + 1

	return southern_ocean_mask_W, southern_ocean_mask_S, southern_ocean_mask_C, so_atl_basin_mask_W, so_atl_basin_mask_S, so_atl_basin_mask_C, so_indpac_basin_mask_W, so_indpac_basin_mask_S, so_indpac_basin_mask_C

