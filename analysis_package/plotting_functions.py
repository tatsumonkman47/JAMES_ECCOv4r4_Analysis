import matplotlib.gridspec as gridspec
import xarray as xr 
import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np

def world_plot(dataset,label_tiles=True,ticks_on=False,cmin=None,cmax=None,cmap="viridis"):
	"""
	# A function to all plot ECCO grid tiles simulateously
	# data should be entered as an xarray dataarray, not a dataset
	# data should have dimensions (tile, i, j) = (13,i,j)
	# return a matplotlib plot object..

	# need to enter variable with timestep and k-level already selected like
	# etc: RHOAnoma_ds.RHOAnoma.isel(time=1,k=1)
	"""
	var_min = cmin
	var_max = cmax

	tmp_plt = dataset
	if cmin == cmax == None:
		# normalize colorscale across all grid 
		var_min = tmp_plt.min()
		var_max = tmp_plt.max()
	elif tmp_plt.min() == tmp_plt.max():
		var_min = None
		var_max = None

	# initialize figure and gridspec
	fig = plt.figure(figsize=(15,15))
	gs = gridspec.GridSpec(nrows=4,ncols=4,hspace=0.02,wspace=0.05)
	n_levels = 20

	# Eastern Atlantic Ocean tiles
	ax0 = fig.add_subplot(gs[3,1])
	ax1 = fig.add_subplot(gs[2,1])
	ax2 = fig.add_subplot(gs[1,1])

	# Indian Ocean tiles
	ax3 = fig.add_subplot(gs[3,2])
	ax4 = fig.add_subplot(gs[2,2])
	ax5 = fig.add_subplot(gs[1,2])

	# Arctic Ocean tile
	ax6 = fig.add_subplot(gs[0,2])

	# Pacific Ocean Tiles
	ax7 = fig.add_subplot(gs[1,3])
	ax8 = fig.add_subplot(gs[2,3])
	ax9 = fig.add_subplot(gs[3,3])

	# Western Atlantic Ocean tiles
	ax10 = fig.add_subplot(gs[1,0])
	ax11 = fig.add_subplot(gs[2,0])
	ax12 = fig.add_subplot(gs[3,0])



	im0 = ax0.contourf(tmp_plt.isel(tile=0),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)
	im1 = ax1.contourf(tmp_plt.isel(tile=1),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)
	im2 = ax2.contourf(tmp_plt.isel(tile=2),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)	
	im3 = ax3.contourf(tmp_plt.isel(tile=3),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)	
	im4 = ax4.contourf(tmp_plt.isel(tile=4),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)	
	im5 = ax5.contourf(tmp_plt.isel(tile=5),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)
	im6 = ax6.contourf(tmp_plt.isel(tile=6),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)	

	# reorient flipped grid tiles..	
	im7 = ax7.contourf(np.flip(tmp_plt.isel(tile=7).T,axis=0),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)	
	im8 = ax8.contourf(np.flip(tmp_plt.isel(tile=8).T,axis=0),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)
	im9 = ax9.contourf(np.flip(tmp_plt.isel(tile=9).T,axis=0),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)	
	im10 = ax10.contourf(np.flip(tmp_plt.isel(tile=10).T,axis=0),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)
	im11 = ax11.contourf(np.flip(tmp_plt.isel(tile=11).T,axis=0),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)
	im12 = ax12.contourf(np.flip(tmp_plt.isel(tile=12).T,axis=0),n_levels,vmin=var_min,vmax=var_max,cmap=cmap)

	if label_tiles == True:
		ax0.set_title("tile 0")
		ax1.set_title("tile 1")
		ax2.set_title("tile 2")
		ax3.set_title("tile 3")
		ax4.set_title("tile 4")
		ax5.set_title("tile 5")
		ax6.set_title("tile 6")
		ax7.set_title("tile 7")
		ax8.set_title("tile 8")
		ax9.set_title("tile 9")
		ax10.set_title("tile 10")
		ax11.set_title("tile 11")
		ax12.set_title("tile 12")

	if ticks_on == False:
		ax0.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax1.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax2.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax3.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax4.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax5.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax6.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax7.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax8.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax9.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax10.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)	
		ax11.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax12.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)

	fig.subplots_adjust(right=0.8)
	cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])

	# I chose tile 8 since I don't know how to manually adjust the colorbar...
	cbar = fig.colorbar(im1,cax=cb_ax)

	# return the plot object..
	return plt 

#
