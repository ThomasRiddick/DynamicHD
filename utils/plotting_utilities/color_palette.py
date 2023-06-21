'''
Created on Mar 8, 2017

@author: thomasriddick
'''

class ColorPalette:
    """Various set of colors to use in plots"""

    def __init__(self,palette_to_use='default'):
        """Class constructor. Initialize chosen palette"""
        palettes = {'default':self.initialize_default_palette,'gmd_paper':self.initialize_gmd_paper_palette}
        palettes[palette_to_use]()

    def initialize_default_palette(self):
        """Initialize the default color palette"""
        print("Using default colors")
        self.simple_catchment_and_flowmap_colors = ['blue','peru','white','black']
        self.create_colormap_colors = ['blue','peru','yellow','white','gray','red','black',
                                       'indigo','deepskyblue']
        self.create_colormap_alternative_colors = ['blue','peru','blueviolet','black','red','gray','green',
                                                   'yellow','deepskyblue']
        self.basic_flowmap_comparison_plot_colors = ['blue','peru','black','white','purple']
        self.flowmap_and_catchments_colors_single_color_flowmap = ['lightblue','peru','blue','red','grey',
                                                                   'darkgrey','lightgrey']
        self.flowmap_and_catchments_colors = ['lightblue','peru','black','blue','purple','red',
                                              'grey','darkgrey','lightgrey']
        self.flowmap_and_catchments_colors_with_glac = self.flowmap_and_catchments_colors + ['white']
        self.flowmap_and_catchments_colors_single_color_flowmap_with_glac =\
            self.flowmap_and_catchments_colors_single_color_flowmap + ['white']
        self.simple_thresholded_data_flowmap_and_catchments_colors = ['lightblue','black','peru',
                                                                      'white','blue','red',
                                                                      'green']
        self.extra_sea_color = ['cadetblue']
        self.extra_land_color = ['moccasin']
        self.common_truesink = ['white']
        self.ref_truesink = ['aqua']
        self.data_truesink = ['gold']

    def initialize_gmd_paper_palette(self):
        """Initialize the color palette for the GMD paper"""
        print("Using colors for GMD paper")
        self.initialize_default_palette()
        self.simple_catchment_and_flowmap_colors = ['lightblue','peru','grey','blue']
        self.create_colormap_colors = ['lightblue','peru','yellow','grey','white','red','black',
                                       'indigo','deepskyblue']
        self.create_colormap_alternative_colors = ['lightblue','peru','blueviolet','black','red','gray','green',
                                                   'yellow','deepskyblue']
        self.basic_flowmap_comparison_plot_colors = ['lightblue','peru','yellow','grey','white']
        self.flowmap_and_catchments_colors = ['lightblue','peru','hotpink','blue','green','red',
                                              'grey','darkgrey','lightgrey']
        self.flowmap_and_catchments_colors_with_glac = self.flowmap_and_catchments_colors + ['white']
        self.simple_thresholded_data_flowmap_and_catchments_colors = ['lightblue','peru','blue',
                                                                      'white','#984ea3','#e41a1c',
                                                                      '#4daf4a']
