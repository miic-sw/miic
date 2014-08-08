"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Sep 6, 2011
"""

# Main imports
import time
import numpy as np


# ETS imports
from chaco.api import \
    VPlotContainer, PlotAxis, LinePlot, ArrayDataSource, \
    DataRange1D, LinearMapper, add_default_grids, \
    HPlotContainer, ColorBar, jet, Plot, ArrayPlotData, PlotLabel, \
    OverlayPlotContainer, MultiLinePlot, MultiArrayDataSource, LabelAxis

from chaco.tools.api import PanTool, ZoomTool
from chaco.scales.api import CalendarScaleSystem
from chaco.scales_tick_generator import ScalesTickGenerator
from chaco.plot_graphics_context import PlotGraphicsContext
from chaco.example_support import COLOR_PALETTE

from enable.api import Component, ComponentEditor
from pyface.api import FileDialog, OK, error

try:
    from traits.api import HasTraits, Instance, Property, Array, \
        Float, Date, Enum, Str, List, Button, DelegatesTo, Range
    from traitsui.api import Item, View, VGroup, Tabbed
except ImportError:
    pass


def plot_corr(data, plot_type, p_title, x_lbl, y_lbl_type, y_lbl, y_labels, \
              scale_type, first_day):
    pass


class PlotCorr(HasTraits):

    data = Array()

    plot_type = Enum('CMap', 'Overlay', 'Vertical', 'Multiline')
    p_title = Str('Graph')
    x_lbl = Str('Time')
    y_lbl_type = Enum('Corr', 'Single', 'Custom')
    y_lbl = Str('Corr')
    y_labels = List(Str)
    scale_type = Enum('Time', 'default')
    first_day = Date()
    apply_btn = Button('Apply')
    save_btn = Button('Save')

    y_low = Float(100.0)
    y_high = Float(-1.0)

    multi_line_plot_renderer = Instance(MultiLinePlot)
    # Drives multi_line_plot_renderer.normalized_amplitude
    amplitude = Range(-20.0, 20.0, value=1)
    # Drives multi_line_plot_renderer.offset
    offset = Range(-15.0, 15.0, value=0)

    plot = Instance(Component)

    plot_vertical = Property(Instance(Component))
    plot_overlay = Property(Instance(Component))
    plot_cmap = Property(Instance(Component))
    plot_multiline = Property(Instance(Component))

    view = View(Tabbed(VGroup(Item('plot',
                                    editor=ComponentEditor(),
                                    width=600,
                                    height=400,
                                    show_label=False),
                              Item('amplitude', label='amp', visible_when="plot_type=='Multiline'"),
                              Item('offset', label='offset', visible_when="plot_type=='Multiline'"),
                              Item('save_btn', show_label=False, width=200, height=100),
                              label='Graph'),
                       VGroup(Item('plot_type', style='custom'),
                              Item('p_title', label='graph title'),
                              Item('x_lbl', label='x axis label'),
                              Item('y_lbl_type', style='custom'),
                              Item('y_lbl', label='y axis label', visible_when="y_lbl_type=='Single'"),
                              Item('y_labels', label='y axis labels', visible_when="y_lbl_type=='Custom'"),
                              Item('scale_type', style='custom'),
                              Item('first_day', enabled_when="scale_type=='Time'"),
                              Item('apply_btn', show_label=False, width=200, height=100),
                              label='Config',
                              springy=True)),
                resizable=True)

    ###########################################################################
    # Protected interface.
    ###########################################################################

    def _create_dates(self, numpoints, start=None, units="days"):
        """ Returns **numpoints** number of dates that evenly bracket the current
        date and time.  **units** should be one of "weeks", "days", "hours"
        "minutes", or "seconds".
        """
        units_map = { "weeks" : 7 * 24 * 3600,
                      "days" : 24 * 3600,
                      "hours" : 3600,
                      "minutes" : 60,
                      "seconds" : 1 }

        if start is None:
            start = time.time() # Now
        else:
            start = time.mktime(start.timetuple())

        dt = units_map[units]
        dates = np.linspace(start, start + numpoints * dt, numpoints)
        return dates

    def _apply_btn_fired(self):
        self._update_plot()

    def _save_btn_fired(self):

        filter = 'PNG file (*.png)|*.png|\nTIFF file (*.tiff)|*.tiff|'
        dialog = FileDialog(action='save as', wildcard=filter)

        if dialog.open() != OK:
            return

        filename = dialog.path

        width, height = self.plot.outer_bounds

        gc = PlotGraphicsContext((width, height), dpi=100)
        gc.render_component(self.plot)
        try:
            gc.save(filename)
        except KeyError, e:
            errmsg = ("The filename must have an extension that matches "
                      "a graphics format, such as '.png' or '.tiff'.")
            if str(e.message) != '':
                errmsg = ("Unknown filename extension: '%s'\n" %
                          str(e.message)) + errmsg

            error(None, errmsg, title="Invalid Filename Extension")

    def _data_changed(self):
        self._update_plot()

    def _amplitude_changed(self, amp):
        self.multi_line_plot_renderer.normalized_amplitude = amp

    def _offset_changed(self, off):
        self.multi_line_plot_renderer.offset = off
        # FIXME:  The change does not trigger a redraw.  Force a redraw by
        # faking an amplitude change.
        self.multi_line_plot_renderer._amplitude_changed()

    def _update_plot(self):

        if self.data.shape[0] > 100 and self.plot_type == 'Vertical':
            self.plot_type = 'CMap'

        if self.plot_type == 'Vertical':
            self.plot = self.plot_vertical
        elif self.plot_type == 'CMap':
            self.plot = self.plot_cmap
        elif self.plot_type == 'Overlay':
            self.plot = self.plot_overlay
        elif self.plot_type == 'Multiline':
            self.plot = self.plot_multiline

    #### Trait property getters/setters #######################################

    def _get_plot_vertical(self):

        if self.data is None or len(self.data.shape) == 1:
            return

        container = VPlotContainer(resizable="v", fill_padding=True, padding=30,
                                   stack_order="top_to_bottom", bgcolor="transparent",
                                   spacing=9)

        numpoints = self.data.shape[1]

        if self.scale_type == 'Time':
            index = self._create_dates(numpoints, start=self.first_day)
        else:
            index = range(numpoints)

        time_ds = ArrayDataSource(index)
        xmapper = LinearMapper(range=DataRange1D(time_ds))

        corr_mapper = None

        for (m, cDx) in enumerate(self.data):

            corr_ds = ArrayDataSource(cDx, sort_order="none")

            corr_mapper = LinearMapper(range=DataRange1D(corr_ds))

            if corr_mapper.range.low < self.y_low : self.y_low = corr_mapper.range.low
            if corr_mapper.range.high > self.y_high : self.y_high = corr_mapper.range.high

            corr_plot = LinePlot(index=time_ds, value=corr_ds,
                                index_mapper=xmapper,
                                value_mapper=corr_mapper,
                                edge_color="blue",
                                face_color="paleturquoise",
                                bgcolor="white",
                                border_visible=True,
                                padding_left=25)

            ###### Y axis #####################################################

            if self.y_lbl_type == 'Corr':
                vtitle = ("%d" % (m + 1)) + u"\u00B0" + " t_win"
            elif self.y_lbl_type == 'Single':
                vtitle = "" # One label for all the axis
            elif self.y_lbl_type == 'Custom' and \
                 len(self.y_labels) == self.data.shape[0] and \
                 self.y_labels[m] is not None:
                # a new value in the list defaults to None so raise an error before
                # the operator ends inputing it. 
                vtitle = self.y_labels[m]
            else:
                vtitle = ""

            left = PlotAxis(orientation='left',
                            title=vtitle,
                            title_font="modern 12",
                            #title_spacing=0,
                            tick_label_font="modern 8",
                            tick_visible=True,
                            small_axis_style=True,
                            axis_line_visible=False,
                            ensure_labels_bounded=True,
                            #tick_label_color="transparent",
                            mapper=corr_mapper,
                            component=corr_plot)

            corr_plot.overlays.append(left)

            ###### X axis #####################################################

            if m != (self.data.shape[0] - 1):
                if self.scale_type == 'Time':
                    # Set the plot's bottom axis to use the Scales ticking system
                    bottom_axis = PlotAxis(corr_plot, orientation="bottom",
                                           tick_label_color="transparent", # mapper=xmapper,
                                           tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem()))
                else:
                    bottom_axis = PlotAxis(orientation='bottom',
                                           title="",
                                           tick_label_color="transparent",
                                           tick_visible=True,
                                           small_axis_style=True,
                                           axis_line_visible=False,
                                           component=corr_plot)
            else:
                if self.scale_type == 'Time':
                    # Just the last axis shows tick_labels
                    bottom_axis = PlotAxis(corr_plot, orientation="bottom", title=self.x_lbl,
                                           tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem()))
                else:
                    bottom_axis = PlotAxis(orientation='bottom',
                                           title=self.x_lbl,
                                           title_font="modern 12",
                                           tick_visible=True,
                                           small_axis_style=True,
                                           axis_line_visible=False,
                                           component=corr_plot)


            corr_plot.overlays.append(bottom_axis)
            _, vgrid = add_default_grids(corr_plot)
            vgrid.tick_generator = bottom_axis.tick_generator

            corr_plot.tools.append(PanTool(corr_plot, constrain=True,
                                            constrain_direction="x"))
            corr_plot.overlays.append(ZoomTool(corr_plot, drag_button="right",
                                                  always_on=True,
                                                  tool_mode="box",
                                                  #axis="index",
                                                  max_zoom_out_factor=10.0,
                                                 ))

            container.add(corr_plot)

        for component in container.components:
            component.y_mapper.range.set_bounds(self.y_low, self.y_high)

        container.overlays.append(PlotLabel(self.p_title,
                                    component=container,
                                    overlay_position="outside top",
                                    font="modern 16"))

        if self.y_lbl_type == 'Single':
            container.overlays.append(PlotLabel(self.y_lbl,
                                                component=container,
                                                angle=90.0,
                                                overlay_position="outside left",
                                                font="modern 12"))

        container.padding_bottom = 50

        return container

    def _get_plot_overlay(self):

        if self.data is None or len(self.data.shape) == 1:
            return

        container = OverlayPlotContainer(resizable="v", fill_padding=True, padding=30,
                                         bgcolor="transparent", use_backbuffer=True)

        numpoints = self.data.shape[1]

        if self.scale_type == 'Time':
            index = self._create_dates(numpoints, start=self.first_day)
        else:
            index = range(numpoints)

        time_ds = ArrayDataSource(index)
        xmapper = LinearMapper(range=DataRange1D(time_ds))

        corr_mapper = None

        for (m, cDx) in enumerate(self.data):

            corr_ds = ArrayDataSource(cDx, sort_order="none")

            if corr_mapper is None:
                corr_mapper = LinearMapper(range=DataRange1D(corr_ds))

            corr_plot = LinePlot(index=time_ds, value=corr_ds,
                                index_mapper=xmapper,
                                value_mapper=corr_mapper,
                                color=tuple(COLOR_PALETTE[m % len(COLOR_PALETTE)]),
                                edge_color="blue",
                                face_color="paleturquoise",
                                #bgcolor="white",
                                border_visible=True,
                                padding_left=25)

            corr_mapper.range.add(corr_plot.value)


            if m == 0:
                ###### Y axis #####################################################

                left = PlotAxis(orientation='left',
                                title=self.y_lbl,
                                title_font="modern 12",
                                #title_spacing=0,
                                tick_label_font="modern 8",
                                tick_visible=True,
                                small_axis_style=True,
                                axis_line_visible=False,
                                ensure_labels_bounded=True,
                                #tick_label_color="transparent",
                                mapper=corr_mapper,
                                component=corr_plot)

                corr_plot.overlays.append(left)

                ###### X axis #####################################################

                if self.scale_type == 'Time':
                    # Just the last axis shows tick_labels
                    bottom_axis = PlotAxis(corr_plot, orientation="bottom", title=self.x_lbl,
                                           tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem()))
                else:
                    bottom_axis = PlotAxis(orientation='bottom',
                                           title=self.x_lbl,
                                           title_font="modern 12",
                                           tick_visible=True,
                                           small_axis_style=True,
                                           axis_line_visible=False,
                                           component=corr_plot)


                corr_plot.overlays.append(bottom_axis)

                ###### Grids #####################################################

                _, vgrid = add_default_grids(corr_plot)
                vgrid.tick_generator = bottom_axis.tick_generator

                ###### Tools #####################################################

                corr_plot.tools.append(PanTool(corr_plot, constrain=True,
                                                constrain_direction="x"))
                corr_plot.overlays.append(ZoomTool(corr_plot, drag_button="right",
                                                      always_on=True,
                                                      tool_mode="box",
                                                      #axis="index",
                                                      max_zoom_out_factor=10.0,
                                                     ))
            container.add(corr_plot)

        ###### Title #####################################################
        container.overlays.append(PlotLabel(self.p_title,
                                    component=container,
                                    overlay_position="outside top",
                                    font="modern 16"))

        container.padding_bottom = 50

        return container

    def _get_plot_cmap(self):

        if self.data is None or len(self.data.shape) == 1:
            return

        numpoints = self.data.shape[1]

        if self.scale_type == 'Time':
            index_x = self._create_dates(numpoints, start=self.first_day)
        else:
            index_x = np.arange(numpoints)

        x_bounds = (index_x[0], index_x[-1], len(index_x))
        y_bounds = (1, self.data.shape[0], self.data.shape[0])
        # Create a plot data obect and give it this data
        pd = ArrayPlotData()
        pd.set_data("imagedata", self.data)

        plot = Plot(pd)

        plot.img_plot("imagedata",
                      name="corr_plot",
                      origin="top left",
                      xbounds=x_bounds[:2],
                      ybounds=y_bounds[:2],
                      colormap=jet,
                      padding_left=25)

        corr_plot = plot.plots['corr_plot'][0]

        if self.scale_type == 'Time':
            # Just the last axis shows tick_labels
            bottom_axis = PlotAxis(corr_plot, orientation="bottom", title=self.x_lbl,
                                   tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem()))
        else:
            bottom_axis = PlotAxis(orientation='bottom',
                                   title=self.x_lbl,
                                   title_font="modern 12",
                                   tick_visible=True,
                                   small_axis_style=True,
                                   axis_line_visible=False,
                                   component=corr_plot)


        corr_plot.overlays.append(bottom_axis)

        corr_plot.tools.append(PanTool(corr_plot,
                                       constrain=True,
                                       constrain_direction="x",
                                       constrain_key="shift"))
        corr_plot.overlays.append(ZoomTool(corr_plot,
                                           drag_button="right",
                                           always_on=True,
                                           tool_mode="range",
                                           axis="index",
                                           max_zoom_out_factor=10.0,
                                           ))

        # Create the colorbar, handing in the appropriate range and colormap
        colorbar = ColorBar(index_mapper=LinearMapper(range=corr_plot.color_mapper.range),
                            color_mapper=corr_plot.color_mapper,
                            plot=corr_plot,
                            orientation='v',
                            resizable='v',
                            width=30,
                            padding_top=corr_plot.padding_top,
                            padding_bottom=corr_plot.padding_bottom,
                            padding_left=50,
                            padding_right=5)

        #colorbar.plot = corr_plot
        #colorbar.padding_top = corr_plot.padding_top
        #colorbar.padding_bottom = corr_plot.padding_bottom

        # Add pan and zoom tools to the colorbar
        pan_tool = PanTool(colorbar, constrain_direction="y", constrain=True)
        colorbar.tools.append(pan_tool)

        zoom_overlay = ZoomTool(colorbar, axis="index", tool_mode="range",
                                always_on=False, drag_button=None)
        colorbar.overlays.append(zoom_overlay)

        # Create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(corr_plot, colorbar, use_backbuffer=True, bgcolor="transparent",
                                   spacing=9, padding=30)

        container.overlays.append(PlotLabel(self.p_title,
                                    component=container,
                                    overlay_position="outside top",
                                    font="modern 16"))

        container.overlays.append(PlotLabel(self.y_lbl,
                                            component=container,
                                            angle=90.0,
                                            overlay_position="outside left",
                                            font="modern 12"))

        container.padding_bottom = 50

        return container

    def _get_plot_multiline(self):

        if self.data is None or len(self.data.shape) == 1:
            return

        numpoints = self.data.shape[1]

        if self.scale_type == 'Time':
            index_x = self._create_dates(numpoints, start=self.first_day)
        else:
            index_x = np.arange(numpoints)

        index_y = np.linspace(1, self.data.shape[0], self.data.shape[0])

        xs = ArrayDataSource(index_x)
        xrange = DataRange1D()
        xrange.add(xs)

        ys = ArrayDataSource(index_y)
        yrange = DataRange1D()
        yrange.add(ys)

        # The data source for the MultiLinePlot.
        ds = MultiArrayDataSource(data=self.data)

        corr_plot = \
            MultiLinePlot(
                index=xs,
                yindex=ys,
                index_mapper=LinearMapper(range=xrange),
                value_mapper=LinearMapper(range=yrange),
                value=ds,
                global_max=self.data.max(),
                global_min=self.data.min())

        corr_plot.value_mapper.range.low = 0
        corr_plot.value_mapper.range.high = self.data.shape[0] + 1

        self.multi_line_plot_renderer = corr_plot

        plot = Plot(title=self.p_title)

        if self.scale_type == 'Time':
            # Just the last axis shows tick_labels
            bottom_axis = PlotAxis(component=plot, orientation="bottom", title=self.x_lbl,
                                   tick_generator=ScalesTickGenerator(scale=CalendarScaleSystem()))
        else:
            bottom_axis = PlotAxis(orientation='bottom',
                                   title=self.x_lbl,
                                   title_font="modern 12",
                                   tick_visible=True,
                                   small_axis_style=True,
                                   axis_line_visible=False,
                                   component=plot)

        if self.y_lbl_type == 'Custom' and \
            len(self.y_labels) == self.data.shape[0]:
            # a new value in the list defaults to None so raise an error before
            # the operator ends inputing it. 

            left_axis = LabelAxis(component=plot,
                                  orientation='left',
                                  title=self.x_lbl,
                                  mapper=corr_plot.value_mapper,
                                  tick_interval=1.0,
                                  labels=self.y_labels,
                                  positions=index_y)
        else:
            left_axis = PlotAxis(component=plot,
                                 orientation='left',
                                 title=self.y_lbl,
                                 title_font="modern 12",
                                 #title_spacing=0,
                                 tick_label_font="modern 8",
                                 tick_visible=True,
                                 small_axis_style=True,
                                 axis_line_visible=False,
                                 ensure_labels_bounded=True,
                                 #tick_label_color="transparent",
                                 mapper=corr_plot.value_mapper)


        plot.overlays.extend([bottom_axis, left_axis])

        plot.add(corr_plot)

        plot.padding_bottom = 50

        return plot

class _plot_corr_view(HasTraits):

    engine = Instance(PlotCorr)

    data = DelegatesTo('engine')

    plot_type = DelegatesTo('engine')
    p_title = DelegatesTo('engine')
    x_lbl = DelegatesTo('engine')
    y_lbl_type = DelegatesTo('engine')
    y_lbl = DelegatesTo('engine')
    y_labels = DelegatesTo('engine')
    scale_type = DelegatesTo('engine')
    first_day = DelegatesTo('engine')
    apply_btn = Button('Apply')
    save_btn = Button('Save')

    # Drives multi_line_plot_renderer.normalized_amplitude
    amplitude = DelegatesTo('engine')
    # Drives multi_line_plot_renderer.offset
    offset = DelegatesTo('engine')

    plot = Instance(Component)

    traits_view = View(Tabbed(VGroup(Item('plot',
                                          editor=ComponentEditor(),
                                          width=600,
                                          height=400,
                                          show_label=False),
                                     Item('save_btn', show_label=False, width= -200, height= -100),
                                     Item('amplitude', label='amp', visible_when="plot_type=='Multiline'"),
                                     Item('offset', label='offset', visible_when="plot_type=='Multiline'"),
                                     label='Graph'),
                              VGroup(Item('plot_type', style='custom'),
                                     Item('p_title', label='graph title'),
                                     Item('x_lbl', label='x axis label'),
                                     Item('y_lbl_type', style='custom'),
                                     Item('y_lbl', label='y axis label', visible_when="y_lbl_type=='Single'"),
                                     Item('y_labels', label='y axis labels', visible_when="y_lbl_type=='Custom'"),
                                     Item('scale_type', style='custom'),
                                     Item('first_day', enabled_when="scale_type=='Time'"),
                                     Item('apply_btn', show_label=False, width= -200, height= -100),
                                     label='Config',
                                     springy=True)),
                       resizable=True)

    def _engine_default(self):
        return PlotCorr()

    def _apply_btn_fired(self):
        self._update_plot()

    def _save_btn_fired(self):

        filter = 'PNG file (*.png)|*.png|\nTIFF file (*.tiff)|*.tiff|'
        dialog = FileDialog(action='save as', wildcard=filter)

        if dialog.open() != OK:
            return

        filename = dialog.path

        width, height = self.plot.outer_bounds

        gc = PlotGraphicsContext((width, height), dpi=100)
        gc.render_component(self.plot)
        try:
            gc.save(filename)
        except KeyError, e:
            errmsg = ("The filename must have an extension that matches "
                      "a graphics format, such as '.png' or '.tiff'.")
            if str(e.message) != '':
                errmsg = ("Unknown filename extension: '%s'\n" %
                          str(e.message)) + errmsg

            error(None, errmsg, title="Invalid Filename Extension")

    def _data_changed(self):
        self._update_plot()

    def _amplitude_changed(self, amp):
        self.engine.multi_line_plot_renderer.normalized_amplitude = amp

    def _offset_changed(self, off):
        self.engine.multi_line_plot_renderer.offset = off
        # FIXME:  The change does not trigger a redraw.  Force a redraw by
        # faking an amplitude change.
        self.engine.multi_line_plot_renderer._amplitude_changed()

    def _update_plot(self):

        if self.plot_type == 'Vertical':
            self.plot = self.engine.plot_vertical
        elif self.plot_type == 'CMap':
            self.plot = self.engine.plot_cmap
        elif self.plot_type == 'Overlay':
            self.plot = self.engine.plot_overlay
        elif self.plot_type == 'Multiline':
            self.plot = self.engine.plot_multiline






