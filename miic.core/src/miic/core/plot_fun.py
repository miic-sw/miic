# -*- coding: utf-8 -*-
"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Nov 16, 2010
"""

# Main imports
import os
import numpy as np
import datetime
from numpy import arange
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from copy import copy

# ETS imports
try:
    BC_UI = True
    from traits.api import HasTraits, Instance, Array, Enum, Directory, \
        Str, Int, Bool, Float
    from traitsui.api import View, Item
except ImportError:
    BC_UI = False
    pass
    
# Chaco import
try:
    CHACO_PLOT = True
    from chaco.api \
        import Plot, ArrayPlotData, LinearMapper, \
        create_line_plot, OverlayPlotContainer, VPlotContainer, \
        add_default_axes, add_default_grids, PlotLabel, Legend, \
        ColorBar, HPlotContainer, jet

    from chaco.tools.api \
        import RangeSelection, RangeSelectionOverlay, \
        ZoomTool, LegendTool, PanTool

    from chaco.example_support import COLOR_PALETTE
except ImportError:
    CHACO_PLOT = False
    print "ImportError for chaco plotting tools. Some functions might not work."
    pass

# Enable import
try:
    ENABLE_API = True
    from enable.component_editor import ComponentEditor
    from enable.api import Component
except ImportError:
    ENABLE_API = False
    print "ImportError for enable module."
    pass

# Obspy import
from obspy.core import stream

from miic.core.miic_utils import from_str_to_datetime, flatten_recarray, \
    convert_time, dv_check, adv_check, spectrogram_check

from miic.core.stream import corr_trace_to_obspy

from miic.core.corr_mat_processing import corr_mat_normalize, corr_mat_trim


if BC_UI:
    def _create_plot_component_vertical(signals=Array,
                                        use_downsampling=False):
    
        # container = HPlotContainer(resizable = "hv", bgcolor="lightgray",
        #                            fill_padding=True, padding = 10)
        container = VPlotContainer(resizable="hv", bgcolor="lightgray",
                                   fill_padding=True, padding=50)
    
        nSignal, nSample = np.shape(signals)
        time = arange(nSample)
    
        value_range = None
        plots = {}
        for i in range(nSignal):
    
            plot = create_line_plot((time, signals[i]),
                            color=tuple(COLOR_PALETTE[i % len(COLOR_PALETTE)]),
                            width=1.0,
                            # orientation="v")
                            orientation="h")
            plot.origin_axis_visible = True
            # plot.origin = "top left"
            plot.padding_left = 10
            plot.padding_right = 10
            plot.border_visible = False
            plot.bgcolor = "white"
            if value_range is None:
                value_range = plot.value_mapper.range
            else:
                plot.value_range = value_range
                value_range.add(plot.value)
    
            container.add(plot)
            plots["Corr fun %d" % i] = plot
    
        # Add a legend in the upper right corner, and make it relocatable
        legend = Legend(component=plot, padding=10, align="ur")
        legend.tools.append(LegendTool(legend, drag_button="right"))
        plot.overlays.append(legend)
        legend.plots = plots
        # container.padding_top = 50
        container.overlays.append(PlotLabel("Correlation function",
                                            component=container,
                                            font="swiss 16",
                                            overlay_position="top"))
        # selection_overlay = RangeSelectionOverlay(component=plot)
        # plot.tools.append(RangeSelection(plot))
        zoom = ZoomTool(plot, tool_mode="box", always_on=False)
        # plot.overlays.append(selection_overlay)
        plot.overlays.append(zoom)
        return container
    

def _create_plot_component_overlay(signals, use_downsampling=False):

    container = OverlayPlotContainer(padding=40, bgcolor="lightgray",
                                     use_backbuffer=True,
                                     border_visible=True,
                                     fill_padding=True)

    nSignal, nSample = np.shape(signals)
    time = arange(nSample)

    value_mapper = None
    index_mapper = None
    plots = {}
    for i in range(nSignal):

        plot = create_line_plot((time, signals[i]),
                        color=tuple(COLOR_PALETTE[i % len(COLOR_PALETTE)]),
                        width=2.0)
        plot.use_downsampling = use_downsampling

        if value_mapper is None:
            index_mapper = plot.index_mapper
            value_mapper = plot.value_mapper
            add_default_grids(plot)
            add_default_axes(plot)
        else:
            plot.value_mapper = value_mapper
            value_mapper.range.add(plot.value)
            plot.index_mapper = index_mapper
            index_mapper.range.add(plot.index)
        if i % 2 == 1:
            plot.line_style = "dash"
        plot.bgcolor = "white"
        plots["Corr fun %d" % i] = plot
        container.add(plot)

    # Add a legend in the upper right corner, and make it relocatable
    #    legend = Legend(component=plot, padding=10, align="ur")
    #    legend.tools.append(LegendTool(legend, drag_button="right"))
    #    plot.overlays.append(legend)
    #    legend.plots = plots
    #    selection_overlay = RangeSelectionOverlay(component=plot)
    #    plot.tools.append(RangeSelection(plot))
    zoom = ZoomTool(plot, tool_mode="box", always_on=False)
    # plot.overlays.append(selection_overlay)
    plot.overlays.append(zoom)

    return container


def _create_plot_component_cmap(signals):

    nSignal, nSample = np.shape(signals)

    xbounds = (1, nSample, nSample)
    ybounds = (1, nSignal, nSignal)
    z = signals

    # Create a plot data obect and give it this data
    pd = ArrayPlotData()
    pd.set_data("imagedata", z)

    # Create the plot
    plot = Plot(pd)
    plot.img_plot("imagedata",
                  name="my_plot",
                  xbounds=xbounds[:2],
                  ybounds=ybounds[:2],
                  colormap=jet)

    # Tweak some of the plot properties
    plot.title = "Selectable Image Plot"
    # plot.padding = 50

    # Right now, some of the tools are a little invasive, and we need the
    # actual CMapImage object to give to them
    my_plot = plot.plots["my_plot"][0]

    # Attach some tools to the plot
    plot.tools.append(PanTool(plot))
    zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
    plot.overlays.append(zoom)

    # Create the colorbar, handing in the appropriate range and colormap
    colormap = my_plot.color_mapper
    colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                        color_mapper=colormap,
                        plot=my_plot,
                        orientation='v',
                        resizable='v',
                        width=30,
                        padding=20)
    colorbar.padding_top = plot.padding_top
    colorbar.padding_bottom = plot.padding_bottom

    # create a range selection for the colorbar
    range_selection = RangeSelection(component=colorbar)
    colorbar.tools.append(range_selection)
    colorbar.overlays.append(RangeSelectionOverlay(component=colorbar,
                                                   border_color="white",
                                                   alpha=0.8,
                                                   fill_color="lightgray"))

    # we also want to the range selection to inform the cmap plot of
    # the selection, so set that up as well
    range_selection.listeners.append(my_plot)

    # Create a container to position the plot and the colorbar side-by-side
    container = HPlotContainer(use_backbuffer=True)
    container.add(plot)
    container.add(colorbar)
    container.bgcolor = "lightgray"

    return container


def plot_nd(data):
    pass


def _plot_nd(data, p_type):
    if p_type == 'Vertical':
        plot = _create_plot_component_vertical(signals=data)
    elif p_type == 'Overlay':
        plot = _create_plot_component_overlay(signals=data)
    elif p_type == 'CMap':
        plot = _create_plot_component_cmap(signals=data)
    return plot


if BC_UI:
    class _plot_nd_view(HasTraits):
        data_view = Array
        p_type = Enum('Overlay', 'Vertical', 'CMap')
        plot = Instance(Component)
    
        traits_view = View(Item('p_type'),
                           Item('plot', editor=ComponentEditor(),
                                               show_label=False),
                           resizable=True,
                           id='miicapp.plot_fun.plotnd',
                           buttons=['OK', 'Help']
                           )

        def __init__(self, data=None, p_type='CMap'):
            if data == None:
                return
            self.data_view = data
    
            if self.data_view.ndim == 1:
                p_type = 'Vertical'
            else:
                p_type = 'CMap'

            self.plot = _plot_nd(self.data_view, p_type)
    
        def _data_changed(self):
            self.plot = _plot_nd(self.data_view, self.p_type)
    
        def _p_type_changed(self):
            self.plot = _plot_nd(self.data_view, self.p_type)
    

def plot_dv(dv,
            save_dir='.',
            figure_file_name=None,
            mark_time=None,
            normalize_simmat=False,
            sim_mat_Clim=[],
            figsize=(9, 11), dpi=72):
    """ Plot the "extended" dv dictionary

    This function is thought to plot the result of the velocity change estimate
    as output by :class:`~miic.core.stretch_mod.multi_ref_vchange_and_align`
    and successively "extended" to contain also the timing in the form
    {'time': time_vect} where `time_vect` is a :class:`~numpy.ndarray` of
    :class:`~datetime.datetime` objects.
    This addition can be done, for example, using the function
    :class:`~miic.core.miic_utils.add_var_to_dict`.
    The produced figure is saved in `save_dir` that, if necessary, it is
    created.
    It is also possible to pass a "special" time value `mark_time` that will be
    represented in the `dv/v` and `corr` plot as a vertical line; It can be
    a string (i.e. YYYY-MM-DD) or directly a :class:`~datetime.datetime`
    object.
    if the `dv` dictionary also contains a 'comb_mseedid' keyword, its `value`
    (i.e. MUST be a string) will be reported in the title.
    In case of the chosen filename exist in the `save_dir` directory, a prefix
    _<n> with n:0..+Inf, is added.
    The aspect of the plot may change depending on the matplotlib version. The
    recommended one is matplotlib 1.1.1

    :type dv: dict
    :param dv: velocity change estimate dictionary as output by
        :class:`~miic.core.stretch_mod.multi_ref_vchange_and_align` and
        successively "extended" to contain also the timing in the form
        {'time': time_vect} where `time_vect` is a :class:`~numpy.ndarray` of
        :class:`~datetime.datetime` objects.
    :type save_dir: string
    :param save_dir: Directory where to save the produced figure. It is created
        if it doesn't exist.
    :type figure_file_name: string
    :param figure_file_name: filename to use for saving the figure. If None
        the figure is displayed in interactive mode.
    :type mark_time: string or :class:`~datetime.datetime` object
    :param mark_time: It is a "special" time location that will be represented
        in the `dv/v` and `corr` plot as a vertical line.
    :type normalize_simmat: Bool
    :param normalize_simmat: if True the simmat will be normalized to a maximum
        of 1. Defaults to False
    :type sim_mat_Clim: 2 element array_like
    :param sim_mat_Clim: if non-empty it set the color scale limits of the
        similarity matrix image
    """

    check_state = dv_check(dv)

    # Check if the dv dictionary is "correct"
    if check_state['is_incomplete']:
        print "Error: Incomplete dv"
        print "Possible errors:"
        for key in check_state:
            if key is not 'is_incomplete':
                print "%s: %s" % (key, check_state[key])
        raise ValueError

    # For compatibility with TraitsUI
    if mark_time == '':
        mark_time = None

    if mark_time and type(mark_time) == str:
        mark_time = from_str_to_datetime(mark_time, datetimefmt=True)
    elif mark_time and type(mark_time) == datetime.datetime:
        pass
    elif mark_time:
        print "Error: wrong mark_time format!"
        mark_time = None

    if not os.path.isdir(save_dir):
        print "Warning: `save_dir` doesn't exist. Creating ..."
        os.mkdir(save_dir)
        print "Directory %s created" % save_dir

    # Create a unique filename if TraitsUI-default is given
    if figure_file_name == 'plot_default':
        fname = figure_file_name + '_change.png'
        exist = os.path.isfile(os.path.join(save_dir, fname))
        i = 0
        while exist:
            fname = "%s_%i" % (figure_file_name, i)
            exist = os.path.isfile(os.path.join(save_dir,
                                                fname + '_change.png'))
            i += 1
        figure_file_name = fname

    # Extract the data from the dictionary

    value_type = dv['value_type'][0]
    method = dv['method'][0]

    corr = np.squeeze(dv['corr'])
    dt = np.squeeze(dv['value'])
    sim_mat = dv['sim_mat']
    stretch_vect = np.squeeze(dv['second_axis'])

    rtime = convert_time(np.squeeze(dv['time']))

    # normalize simmat if requested
    if normalize_simmat:
        sim_mat = sim_mat/np.tile(np.max(sim_mat,axis=1),(sim_mat.shape[1],1)).T

#    if dv_type == 'single_ref':
#        dt = 1 - dt
#        stretch_vect = stretch_vect - 1

    n_stretching = stretch_vect.shape[0]

    stretching_amount = np.max(stretch_vect)

    # Adapt plot details in agreement with the type of dictionary that
    # has been passed

    if (value_type == 'stretch') and (method == 'single_ref'):

        tit = "Single reference dv/v"
        dv_tick_delta = 0.01
        dv_y_label = "dv/v"   # plotting velocity requires to flip the
                              #stretching axis

    elif (value_type == 'stretch') and (method == 'multi_ref'):

        tit = "Multi reference dv/v"
        dv_tick_delta = 0.01
        dv_y_label = "dv/v"   # plotting velocity requires to flip the
                              #stretching axis

    elif (value_type == 'shift') and (method == 'time_shift'):

        tit = "Time shift"
        dv_tick_delta = 5
        dv_y_label = "time shift (sample)"

    else:
        print "Error: Wrong dv type!"
        return

    f = plt.figure(figsize=figsize, dpi=dpi)

    if matplotlib.__version__ >= '1.1.1':
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    else:
        gs = [311, 312, 313]

    ax1 = f.add_subplot(gs[0])
    imh = plt.imshow(sim_mat.T, interpolation='none', aspect='auto')
    ###
    scale = stretch_vect[1] - stretch_vect[0]
    offset = stretch_vect[0]
    mod_ind = (np.round((dv['value']-offset) / scale).astype(int))
    mod_ind[np.isnan(dv['value'])] = 0
    ax1.plot(mod_ind,'b.')
    if 'model_value' in dv.keys():
        mod_ind = (np.round((dv['model_value']-offset) / scale).astype(int))
        mod_ind[np.isnan(dv['model_value'])] = 0
        ax1.plot(mod_ind,'g.')
    ax1.set_xlim(0,sim_mat.shape[0])
    ax1.set_ylim(0,sim_mat.shape[1])
    if value_type == 'stretch':
        ax1.invert_yaxis()
    ###
    if sim_mat_Clim:
        assert  len(sim_mat_Clim) == 2, "sim_mat_Clim must be a two element list"            
        imh.set_clim(sim_mat_Clim[0], sim_mat_Clim[1])
    plt.gca().get_xaxis().set_visible(False)
    ax1.set_yticks(np.floor(np.linspace(0, n_stretching - 1, 7)).astype('int'))

    if value_type == 'stretch':
        ax1.set_yticklabels(["%4.3f" % x for x in
            stretch_vect[np.floor(np.linspace(n_stretching - 1,0,
                                              7)).astype('int')[:-1]]])
    else:
        ax1.set_yticklabels(["%4.3f" % x for x in
            stretch_vect[np.floor(np.linspace(0,
                                              n_stretching - 1,
                                              7)).astype('int')[:-1]]])
    if 'stats' in dv:
        stats = flatten_recarray(dv['stats'])
        comb_mseedid = \
          stats['network'] + '.' + stats['station'] + \
          '.' + stats['location'] + '.' + stats['channel']

        tit = "%s estimate (%s)" % (tit, comb_mseedid)
    else:
        tit = "%s estimate" % tit

    ax1.set_title(tit)
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.label.set_rotation(270)
    ax1.set_xticklabels([])
    ax1.set_ylabel(dv_y_label)

    ax2 = f.add_subplot(gs[1])
    plt.plot(rtime, -dt, '.')
    if 'model_value' in dv.keys():
        plt.plot(rtime, -dv['model_value'],'g.')
    plt.xlim([rtime[0], rtime[-1]])
    plt.ylim((-stretching_amount, stretching_amount))
    if mark_time and not np.all(rtime < mark_time) \
        and not np.all(rtime > mark_time):
        plt.axvline(mark_time, lw=1, color='r')
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.label.set_rotation(270)
    ax2.set_ylabel(dv_y_label)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(dv_tick_delta))
    ax2.yaxis.grid(True, 'major', linewidth=1)
    ax2.xaxis.grid(True, 'major', linewidth=1)
    ax2.set_xticklabels([])

    ax3 = f.add_subplot(gs[2])
    plt.plot(rtime, corr, '.')
    if 'model_corr' in dv.keys():
        plt.plot(rtime, dv['model_corr'],'g.')
    plt.xlim([rtime[0], rtime[-1]])
    ax3.yaxis.set_ticks_position('right')
    ax3.set_ylabel("Correlation")
    plt.ylim((0, 1))
    if mark_time and not np.all(rtime < mark_time)\
        and not np.all(rtime > mark_time):
        plt.axvline(mark_time, lw=1, color='r')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax3.yaxis.grid(True, 'major', linewidth=1)
    ax3.xaxis.grid(True, 'major', linewidth=1)

    plt.subplots_adjust(hspace=0, wspace=0)

    if figure_file_name == None:
        plt.show()
    else:
        print 'saving to %s' % figure_file_name
        f.savefig(os.path.join(save_dir, figure_file_name + '_change.png'),
                  dpi=dpi)
        plt.close()


if BC_UI:
    class _plot_dv_view(HasTraits):
        save_dir = Directory('.')
        figure_file_name = Str('plot_default')
        mark_time = Str('2010-10-14')
    
        trait_view = View(Item('save_dir'),
                          Item('figure_file_name'),
                          Item('mark_time',
                               label='Mark Time (fmt. YYYY-MM-DD)'))


def plot_single_corr_matrix(corr_mat, seconds=0, filename=None,
                            normalize=True, normtype='absmax', norm_time_win=[None, None],
                            clim=[], cmap='viridis', figsize=(8, 6), dpi=72):
    """ Plot a single correlation matrix.

    A simple plot of the correlation matrix `corr_mat` is generated and
    saved under `filename` + '_corrMatrix.png'. If filename is an empty
    string (default) the figure is saved under
    './plot_default_corrMatrix.png'. An optional parameter
    `center_win_len` can be passed that restricts the length of the plotted
    traces to the respective number of samples in the center.

    :type corr_mat: dictionary of the type correlation matrix
    :param corr_mat: correlation matrix to be plotted
    :type seconds: int
    :param seconds: How many seconds will be taken from the central part
        of the correlation function (in a symmetric way respect to zero
        time lag)
    :type filename: string
    :param filename: complete path the file under which the figure is to be
        saved
    :type normalize: bool
    :param normalize: If True the matix will be normalized before plotting
    :type normtype: string
    :param normtype: one of the following 'energy', 'max', 'absmax', 'abssum'
        to decide about the way to calculate the normalization.
    :type norm_time_win: list
    :param norm_time_win: list contaning start- and endtime in seconds of time
        window used for normalization
    :type clim: list
    :param clim: lower and upper limit of the colorscale for plotting. Empty
        list uses default color range
    """

    zerotime = datetime.datetime(1971, 1, 1)

    if normalize:
        corr_mat = corr_mat_normalize(corr_mat, starttime=norm_time_win[0],
                                endtime=norm_time_win[1], normtype=normtype)

    if seconds != 0:
        corr_mat = corr_mat_trim(corr_mat, -seconds, +seconds)

    X = corr_mat['corr_data']

    if 'time' in corr_mat:
        time = convert_time(corr_mat['time'])
    else:
        time = np.linspace(0, X.shape[0], X.shape[0])

    tlag = np.linspace((convert_time([corr_mat['stats']
                ['starttime']])[0] - zerotime).total_seconds(),
                (convert_time([corr_mat['stats']['endtime']])[0] -
                zerotime).total_seconds(),
                corr_mat['stats']['npts'])

    plt.figure(figsize=figsize, dpi=dpi)
    plt.gcf().subplots_adjust(left=0.35)

    ax = plt.imshow(X,
                    aspect='auto',
                    origin='upper',
                    interpolation='nearest',
                    cmap=cmap,
                    extent=(tlag[0], tlag[-1], X.shape[0], 0))
    if clim:
        plt.clim(clim)

    plt.colorbar(format='%1.2g')

    ax.axes.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7,
                                                       integer=True,
                                                       symmetric=True))
    try:
        row, _ = X.shape
        # number of ylabel from 2 to 15
        ynbins = max(2, min(15, row // 15))
    except Exception:
        ynbins = 1

    ax.axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ynbins,
                                                       integer=True,
                                                       prune='upper'))

    ytickpos = np.array(ax.axes.yaxis.get_ticklocs()).astype('i')
    _ = ax.axes.yaxis.set_ticklabels(time[ytickpos])

    plt.ylabel('Days')
    plt.title('Correlation Matrix')
    plt.xlabel('Correlation time [s]')

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename + '_corrMatrix.png', dpi=dpi)
        plt.close()


if BC_UI:
    class _plot_single_corr_matrix_view(HasTraits):
        center_win_len = Int(0)
        filename = Str('./plot_default')
        normalize = Bool(True)
        normtype = Enum('absmax', 'energy', 'max', 'abssum')
    
        trait_view = View(Item('center_win_len'),
                          Item('filename'),
                          Item('normalize'),
                          Item('normtype'))


def plot_trace_distance_section(traces, scale=0, azim=0,
                                outfile=None, title=None, plot_type='wiggle',
                                annotate=False, moveout_vels=False, joint_norm=False,
                                figsize=(8, 6), dpi=72):
    """ Plot a time distance section of an obspy stream or list of traces.

    Take a list of correlation traces with proper geo information or an
    :class:`~obspy.core.stream.Stream` object and plots with distance section.
    The Orientation is controlled by the ``azim`` argument. Traces are
    plotted such that the difference of the interstation azimuth to ``azim`` is
    smaller than 90 degrees. If it is larger the stations are exchanged and the
    traces are flipped at the zero time. ``plot_type`` may be 'line', 'wiggle',
    'cwiggle' or 'plain_cwiggle'.
    If ``annotate`` is set to True, the station combination name will be
    printed on each curve (in the left side).

    :type traces: :class:`~obspy.core.stream.Stream` or list of traces
    :param traces: structure containing the traces to plot
    :type scale: float
    :param scale: maximum amplitude in kilometers (on the distance scale)
    :type azim: float
    :param azim: azimuth of the noise propagation direction that creates
        arrivals at positive lag time in cross-correlations. When the azimuth
        of the direction from the first to the second station differs by more
        than 90 degrees from this value the trace is flipped exchanging
        positive and negative lag times.
    :param outfile: file name to save the figure. If ``outfile`` is None a
        matplotlib figure is opened.
    :type title: string
    :param title: Set plot title
    :type plot_type: string
    :param plot_type: How to plot
    :type annotate: bool
    :param annotate: If True the station combination name will be printed on
        each curve (in the left side)
    :type joint_norm: bool
    :param joint_norm: joint normalization of all traces
    """

    zerotime = datetime.datetime(1971, 1, 1)

    if not isinstance(traces, stream.Stream):
        st = corr_trace_to_obspy(traces)
    else:
        st = traces

    plt.figure(figsize=figsize, dpi=dpi)

    jnorm = 0
    if joint_norm:
        for tr in st:
            if jnorm < np.max(np.abs(tr.data)):
                jnorm = np.max(np.abs(tr.data))

    maxdist=0
    for tr in st:
        start = tr.stats['starttime'].datetime
        end = tr.stats['endtime'].datetime
        dist = tr.stats['sac']['dist']
        maxdist=dist if (dist>maxdist) else maxdist
        if azim:
            azi = 2. * np.arctan(np.tan((tr.stats['sac']['az'] - azim) *
                                        np.pi / 360.))
        else:
            azi = 0
        if abs(azi) > (np.pi / 2):
            tim = arange(tr.stats['npts'])[-1::-1] / \
                tr.stats['sampling_rate'] + (zerotime - end).total_seconds()
        else:
            tim = arange(tr.stats['npts']) / \
                tr.stats['sampling_rate'] + (start - zerotime).total_seconds()

        if joint_norm:
            data = tr.data/jnorm
        else:
            data = tr.data / max(np.abs(tr.data))

        # fill wiggles
        if plot_type == 'wiggle':
            pdata = copy(data)
            pdata[0] = pdata[-1] = 0.
            pdata[pdata < 0] = 0.
            plt.fill(tim, pdata * scale + dist, 'k', linewidth=0)
        elif plot_type in ['cwiggle','plain_cwiggle']:
            pdata = copy(data)
            ndata = copy(data)
            pdata[0] = pdata[-1] = 0.
            pdata[pdata < 0] = 0.
            plt.fill(tim, pdata * scale + dist, 'r', linewidth=0)
            ndata[0] = ndata[-1] = 0.
            ndata[ndata > 0] = 0.
            plt.fill(tim, ndata / max(data) * scale + dist, 'b', linewidth=0)
        if plot_type != 'plain_cwiggle':
            plt.plot(tim, data * scale + dist, 'k', linewidth=0.5)
        if 'arrivals' in tr.stats['sac']:
            for phase in tr.stats['sac']['arrivals']:
                #print phase['time']-zerotime, phase['name']
                plt.plot((phase['time'].datetime-zerotime).total_seconds(),dist,marker=phase['style'],color=phase['color'])
        if annotate:
            plt.annotate(tr.stats['station'], xy=(max(tim), dist),
                         horizontalalignment='right',
                         verticalalignment='middle')

    if moveout_vels :
        plt.plot([0,0],[0,maxdist],'k',linewidth=0.3)
        plt.plot([0,maxdist/5.0],[0,maxdist],'r',linewidth=0.3)
        plt.plot([0,-maxdist/5.0],[0,maxdist],'r',linewidth=0.3)
        plt.plot([0,maxdist/4.0],[0,maxdist],'b',linewidth=0.3)
        plt.plot([0,-maxdist/4.0],[0,maxdist],'b',linewidth=0.3)
        plt.plot([0,maxdist/3.0],[0,maxdist],'g',linewidth=0.3)
        plt.plot([0,-maxdist/3.0],[0,maxdist],'g',linewidth=0.3)
        plt.plot([0,maxdist/2.0],[0,maxdist],'y',linewidth=0.3)
        plt.plot([0,-maxdist/2.0],[0,maxdist],'y',linewidth=0.3)
    if title:
        plt.title(title)

    plt.ylabel('Distance [km]')
    plt.xlabel('Correlation time [s]')
    plt.autoscale(axis='x',tight=True)
    if outfile == None:
        plt.show()
    else:
        plt.savefig(outfile + '_distance_section.png', dpi=dpi)
        plt.close()


if BC_UI:
    class _plot_trace_distance_section_view(HasTraits):
        scale = Float(0)
        azim = Float(0)
        outfile = Str('./plot_default')
        title = Str('Distance section')
        plot_type = Enum('wiggle', 'cwiggle', 'line')
    
        trait_view = View(Item('scale'),
                          Item('azim'),
                          Item('outfile'),
                          Item('title'),
                          Item('plot_type'))
                          

def plot_spectrogram(spectrogram, filename=None, freq_range=[],
                            clim=[], figsize=(8, 6), dpi=72):
    """ Plot a spectrogram.

    A simple plot of the spectrogram matrix `corr_mat` is generated and
    saved under `filename` + '_spectrogram.png'. If filename is an empty
    string (default) the figure is interactivly opened.

    :type spectrogram: dictionary of the type spectrogram
    :param spectrogram: spectrogram to be plotted
    :type filename: string
    :param filename: complete path the file under which the figure is to be
        saved
    :type freq_range: list like
    :param freq_range: limits of frequency range to plot
    :type clim: list
    :param clim: lower and upper limit of the colorscale for plotting. Empty
        list uses default color range
    """

    check = spectrogram_check(spectrogram)
    if not check['valid']:
        print 'spectrogram is not a valid spectrogram dictionary'
        return
    
    frequency = spectrogram['frequency'].flatten()
    if freq_range:
        start = np.argmax(frequency[frequency <= freq_range[0]])
        stop = np.argmax(frequency[frequency <= freq_range[1]])
        X = np.array(spectrogram['spec_mat'])[:,start:stop]
        frequency = frequency[start: stop]
    else:
        X = spectrogram['spec_mat']
        extent=[frequency[0], frequency[-1], X.shape[0], 0]

    if 'time' in spectrogram:
        time = convert_time(spectrogram['time'])
    else:
        time = np.linspace(0, X.shape[0], X.shape[0])

    plt.figure(figsize=figsize, dpi=dpi)
    plt.gcf().subplots_adjust(left=0.35)

    ax = plt.imshow(np.log(X),
                    aspect='auto',
                    origin='upper',
                    interpolation='none',
                    extent=[frequency[0], frequency[-1], X.shape[0], 0])
    if clim:
        plt.clim(clim)

    plt.colorbar(format='%1.2g')
    
    ax.axes.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7,
                                                       integer=True,
                                                       symmetric=True))
    try:
        row, _ = X.shape
        # number of ylabel from 2 to 15
        ynbins = max(2, min(15, row // 15))
    except Exception:
        ynbins = 1

    ax.axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ynbins,
                                                       integer=True,
                                                       prune='upper'))

    ytickpos = np.array(ax.axes.yaxis.get_ticklocs()).astype('i')

    _ = ax.axes.yaxis.set_ticklabels(time[ytickpos])

    plt.ylabel('Date')
    plt.title('Spectrogram')
    plt.xlabel('Frequency [Hz]')
    
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename + '_spectrogram.png', dpi=dpi)
        plt.close()


def apparent_dv_plot(adv_dict,
                     time_stamp,
                     inter_points=1000,
                     save_file_name=None,
                     ax=None,
                     use_gridspec=False,
                     shrink=1,
                     coord_sys='geo'):
    """ Apparent velocity change plot

    This function does plot the apparent velocity change that occurred at
    a certain time of interest.
    The dv values available at that time are interpolated on a regular grid
    taking into account their position as specified by ``stations_info``
    parameter.

    :type apparent_dv: :py:class:`~pandas.DataFrame`
    :param apparent_dv: Apparent dv DataFrame produced by, e.g.,
        :py:func:`~miic.core.macro.inversion_with_missed_data_macro`
    :type stations_info: :py:class:`~pandas.DataFrame`
    :param stations_info: Stations geo information. This DataFrame is supposed
        to have the stations names as index and, at least, two columns:
        'easting' and 'nothing'.
    :type time_stamp: :py:class:`~datetime.datetime`
    :param time_stamp: Day of interest
    :type inter_points: int
    :param inter_points: Interpolation points
    :type save_file_name: str
    :param save_file_name: Filename where to save the produced figure.
    """

    check_state = adv_check(adv_dict)

    # Check if the dv dictionary is "correct"
    if check_state['is_incomplete']:
        print "Error: Incomplete dv"
        print "Possible errors:"
        for key in check_state:
            if key is not 'is_incomplete':
                print "%s: %s" % (key, check_state[key])
        raise ValueError

    dv = adv_dict['apparent_dv'].ix[time_stamp]
    stations_info = adv_dict['points_geo_info']

    mask = dv.isnull()

    # Geo coordinates for the stations
    xpt = stations_info[~mask]['easting']
    ypt = stations_info[~mask]['northing']

    if coord_sys == 'geo':
        # Local projection that is valid for small areas
        xpt = np.array(xpt.tolist()) - xpt.mean()
        xpt *= np.pi / 180 * np.cos(np.mean(ypt) * np.pi / 180) * 6371
        ypt = np.array(ypt.tolist()) - ypt.mean()
        ypt *= np.pi / 180 * 6371

    elif coord_sys == 'utm':
        # Correct the coordinates to work with km
        xpt = xpt.astype('float64') / 1000
        ypt = ypt.astype('float64') / 1000

    else:
        raise ValueError

    # Differential apparent dv
    zpt = dv[~mask]

    # Construct the grid
    xg = np.linspace(np.nanmin(xpt), np.nanmax(xpt), inter_points)
    yg = np.linspace(np.nanmin(ypt), np.nanmax(ypt), inter_points)
    xgrid, ygrid = np.meshgrid(xg, yg)

    # Do the interpolation
    F = plt.mlab.griddata(xpt, ypt, zpt.values, xgrid, ygrid, interp='nn')

    # Create symmetric color scale
    vmax = F.max()
    vmin = F.min()
    mmax = max(abs(vmax), abs(vmin))

    if ax is None:
        f = plt.figure(figsize=(12, 6), dpi=300)
        ax = f.add_subplot(111)
    else:
        f = ax.get_figure()

    ax.axes.set_aspect('equal')

    plt.imshow(F, interpolation='none',
               extent=(xg[0], xg[-1], yg[0], yg[-1]),
               origin='lower', cmap='jet_r',
               vmax=mmax, vmin= -mmax, axes=ax)
    plt.hold('on')
    plt.scatter(xpt, ypt, c=zpt, cmap='jet_r',
                vmax=mmax, vmin= -mmax,
                marker='v', s=50, label='stations',
                edgecolors='k')

    plt.title('Apparent velocity change')
    plt.xlabel('easting (Km)')
    plt.ylabel('northing (Km)')
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')

    plt.colorbar(format='%-6.5f', ax=ax,
                 use_gridspec=use_gridspec, shrink=shrink)

    plt.legend(scatterpoints=1)

    if save_file_name is not None:
        f.savefig(save_file_name)


def diff_apparent_dv_plot(adv_dict,
                          period_b_start, period_b_stop,
                          period_a_start, period_a_stop,
                          inter_points=1000,
                          catalog=None,
                          save_file_name=None,
                          ax=None,
                          use_gridspec=False,
                          shrink=1,
                          coord_sys='geo'):
    """ Differential apparent velocity change plot

    This function does plot the differential apparent velocity change
    that occurred between two time periods of interest.
    The apparent velocity change is first averaged over the two specified
    time period. The differential values are computed for all the stations
    available at that time and they are interpolated on a regular grid
    taking into account their position.
    In case a catalog that reports geo information about important events is
    passed, all of them falling between ``period_b_stop`` and
    ``period_a_start`` are also marked on the plot.

    :type adv_dict: dictionary
    :param adv_dict: Dictionary that caonins the apparent dv DataFrame
        produced by, e.g.,
        :py:func:`~miic.core.macro.inversion_with_missed_data_macro`, and the
        points (stations) geographical infromation.
    :type period_b_start: :py:class:`~datetime.datetime`
    :param period_b_start: Stating day for the first period
    :type period_b_stop: :py:class:`~datetime.datetime`
    :param period_b_stop: Ending day of the first period
    :type period_a_start: :py:class:`~datetime.datetime`
    :param period_a_start: Stating day for the second period
    :type period_a_stop: :py:class:`~datetime.datetime`
    :param period_a_stop: Ending day of the second period
    :type inter_points: int
    :param inter_points: Interpolation points
    :type catalog: :py:class:`~pandas.DataFrame`
    :param catalog: This DataFrame is supposed to have a Multiindex where
        level=0 reports the day corresponding to each event and the level=1
        reports a generic marker (e.g. an increasing index) for the different
        locations interested by the event. Each location must be specified at
        least in four columns: 's_x' 's_y' -> coordinates of the starting point
                               'e_x' 'e_y' -> coordinates of the ending point
    :type save_file_name: str
    :param save_file_name: Filename where to save the produced figure.
    """

    check_state = adv_check(adv_dict)

    # Check if the dv dictionary is "correct"
    if check_state['is_incomplete']:
        print "Error: Incomplete dv"
        print "Possible errors:"
        for key in check_state:
            if key is not 'is_incomplete':
                print "%s: %s" % (key, check_state[key])
        raise ValueError

    apparent_dv = adv_dict['apparent_dv']
    stations_info = adv_dict['points_geo_info']

    dv_before = apparent_dv.ix[period_b_start:period_b_stop].mean(axis=0)

    dv_after = apparent_dv.ix[period_a_start:period_a_stop].mean(axis=0)

    mask = (np.isnan(dv_before) | np.isnan(dv_after))

    # Geo coordinates for the stations
    xpt = stations_info[~mask]['easting']
    ypt = stations_info[~mask]['northing']

    if coord_sys == 'geo':
        # Local projection that is valid for small areas
        xpt = np.array(xpt.tolist()) - xpt.mean()
        xpt *= np.pi / 180 * np.cos(np.mean(ypt) * np.pi / 180) * 6371
        ypt = np.array(ypt.tolist()) - ypt.mean()
        ypt *= np.pi / 180 * 6371

    elif coord_sys == 'utm':
        # Correct the coordinates to work with km
        xpt = xpt.astype('float64') / 1000
        ypt = ypt.astype('float64') / 1000

    else:
        raise ValueError

    # Differential apparent dv
    zpt = dv_after[~mask] - dv_before[~mask]

    # Construct the grid
    xg = np.linspace(np.nanmin(xpt), np.nanmax(xpt), inter_points)
    yg = np.linspace(np.nanmin(ypt), np.nanmax(ypt), inter_points)
    xgrid, ygrid = np.meshgrid(xg, yg)

    # Do the interpolation
    F = plt.mlab.griddata(xpt, ypt, zpt.values, xgrid, ygrid, interp='nn')

    # Create symmetric color scale
    vmax = F.max()
    vmin = F.min()
    mmax = max(abs(vmax), abs(vmin))

    # It a catalog is passed check if there are events in between the
    # two selected periods
    if catalog is not None:
        events = catalog.ix[period_b_stop:period_a_start]

    if ax is None:
        f = plt.figure(figsize=(12, 6), dpi=300)
        ax = f.add_subplot(111)
    else:
        f = ax.get_figure()

    ax.axes.set_aspect('equal')

    plt.imshow(F, interpolation='none',
               extent=(xg[0], xg[-1], yg[0], yg[-1]),
               origin='lower', cmap='jet_r',
               vmax=mmax, vmin= -mmax, axes=ax)
    plt.hold('on')
    plt.scatter(xpt, ypt, c=zpt, cmap='jet_r',
                vmax=mmax, vmin= -mmax,
                marker='v', s=50, label='stations',
                edgecolors='k')

    if catalog is not None:
        for (_, fs) in events.iterrows():
            plt.plot(np.array([fs['s_x'], fs['e_x']]).astype('float64') / 1000,
                     np.array([fs['s_y'], fs['e_y']]).astype('float64') / 1000,
                     lw=1, color='k', marker='o', markerfacecolor='w',
                     markeredgewidth=1, markeredgecolor='k',
                     markersize=5, label='fissure')

    plt.title('Differential apparent velocity change')
    plt.xlabel('easting (Km)')
    plt.ylabel('northing (Km)')
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')

    plt.colorbar(format='%-6.5f', ax=ax,
                 use_gridspec=use_gridspec, shrink=shrink)

    plt.legend(scatterpoints=1)

    if save_file_name is not None:
        f.savefig(save_file_name)
# EOF
