# -*- coding: utf-8 -*-
"""
@author:
Eraldo Pomponi

@copyright:
The MIIC Development Team (eraldo.pomponi@uni-leipzig.de)

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Dec 7, 2012
"""
# Main imports
import os
import collections

import psycopg2

import pandas.io.sql as psql

# ETS imports
try:
    BC_UI = True
    from traits.api import HasTraits, Str, Float, List, Password
    from traitsui.api import View, Item, HGroup
except ImportError:
    BC_UI = False
    pass

# Obspy import
from obspy.core.stream import Stream, _read
from obspy.core import UTCDateTime


def dbconn(host='localhost', dbname='seishub', user='seishub', password=''):
    """ Connects to a PostgreSQL database

    This function provides an interface to connect to a PostgreSQL database

    .. Note::

    Requires the module ``psycopg2``.

    :type host: str
    :param host: Hostname where the PostgreSQL DBMS is running
    :type dbname: str
    :pram dbname: Database name
    :type user: str
    :pram user: Username for connecting to the DB
    :type password: str
    :pram password: Password for connecting to the DB

    :rtype: obj
    :return: **conn**: Connection object
    """
    conn_string = """host='%s' dbname='%s' user='%s' password='%s'""" % \
        (host, dbname, user, password)
    # print the connection string we will use to connect
    print "Connecting to database\n    ->%s" % dbname

    try:
        # get a connection, if a connect cannot be made an exception will be
        # raised here
        conn = psycopg2.connect(conn_string)
    except Exception, e:
        print "Error %s" % e
        return None

    return conn


if BC_UI:
    class _dbconn_view(HasTraits):
    
        host = Str('localhost')
        dbname = Str('seishub')
        user = Str('seishub')
        password = Password()
    
        trait_view = View(Item('host'),
                          Item('dbname'),
                          Item('user'),
                          Item('password'))


def stream_dbread(starttime=None, time_interval=30,
                  networks=None, stations=None, locations=None,
                  channels=None, fformat=None, conn=None, nearest_sample=True,
                  DEBUG=False):
    """ Read data fetching their position in the archive from a DBMS

    This function reads the data corresponding to the passed parameters
    retrieving their position in the archive from a PostgreSQL DBMS that has
    been created with the ``obspy-indexer`` shell script available in
    :py:class:`~obspy.db`.
    To increase the flexibility of this function, the parameters `networks`,
    `stations`, `locations` and `channels` can be string or list of strings
    to read, at the same time, multiple traces.
    The resulting data are trimmed to the time span indicated by the
    ``starttime`` and ``time_interval`` parameters.

    .. Note::
    It must be notice that, in case of the ``starttime`` or the corresponding
    ``endtime = starttime + time_interval`` are corresponding exactly to the
    first/last point between two traces stored in different files, requesting
    just one second less could avoid reading two times the number of files.

    :type starttime: :py:class:`~obspy.core.UTCDateTime`
    :param starttime: Starting time
    :type time_interval: float
    :param time_interval: How many seconds are requested starting from the
        stating time (both ends included)
    :type networks: str or list of str
    :param networks: Network/s requested
    :type stations: str or list of str
    :param stations: Station/s requested
    :type locations: str or list of str
    :param locations: Location/s requested
    :type channels: str or list of str
    :param channels: Channel/s requested
    :type fformat: str
    :param fforma: One of the format supported by :py:class:`~obspy.core.read`
        function
    :type conn: object
    :param conn: Connection object as it is returned by
        :py:func:`~miic.core.stram.dbconn`

    :rtype: :py:class:`~obspy.core.stream.Stream`
    :return: **st**: Stream object that holds all the selected traces
             **n_trace**: Number of traces returned
    """

    # True if it is necessary to close the connection inside this function
    to_close = False
    if conn is None:
        conn_string = "host='localhost' dbname='seishub'\
            user='seishub' password='seishub'"
        # print the connection string we will use to connect
        if DEBUG:
            print "Connecting to database\n    ->%s" % (conn_string)

        try:
            # get a connection, if a connect cannot be made an exception
            # will be raised here
            conn = psycopg2.connect(conn_string)
            to_close = True
        except Exception, e:
            print "DB Connection Error %s" % e
            return None

    kwargs = {}

    if starttime is None:
        endtime = None
    else:
        if not isinstance(starttime, UTCDateTime):
            if isinstance(starttime, basestring):
                try:
                    starttime = UTCDateTime(starttime)
                    endtime = starttime + time_interval
                except ValueError:
                    starttime = endtime = None
            else:
                starttime = endtime = None
        else:
            endtime = starttime + time_interval

    kwargs['starttime'] = starttime
    kwargs['endtime'] = endtime
    kwargs['nearest_sample'] = True

    query_sql = """SELECT network, station, location, channel, file, path
    FROM default_waveform_channels
    INNER JOIN default_waveform_files ON
        (default_waveform_channels.file_id = default_waveform_files.id)
    INNER JOIN default_waveform_paths ON
        (default_waveform_files.path_id = default_waveform_paths.id)"""

    # Clauses to add to the query in case of proper passed parameters
    net_clause = ''
    station_clause = ''
    location_clause = ''
    channel_clause = ''
    tclause = ''
    fformat_clause = ''

    networks_ok = False
    if (networks is not None) and (isinstance(networks, \
                                              collections.Iterable)):
        networks_ok = True
        if isinstance(networks, basestring):
            networks = [networks]

        networks = [net.upper() for net in networks]
        net_clause = '(' + \
            (' or ').join(["network='%s'" % net.upper() \
                           for net in networks]) + ')'
    else:
        print "Wrong networks parameter"
        print "No network clause in generated sql query"

    stations_ok = False
    if (stations is not None) and (isinstance(stations, \
                                              collections.Iterable)):
        stations_ok = True
        if isinstance(stations, basestring):
            stations = [stations]

        stations = [station.upper() for station in stations]
        station_clause = '(' + \
            (' or ').join(["station='%s'" % station.upper() \
                           for station in stations]) + ')'
    else:
        print "Wrong stations parameter"
        print "No stations clause in generated sql query"

    locations_ok = False
    if (locations is not None) and (isinstance(locations, \
                                               collections.Iterable)):
        locations_ok = True
        if isinstance(locations, basestring):
            locations = [locations]

        locations = [location.upper() for location in locations]
        location_clause = '(' + \
            (' or ').join(["location='%s'" % location.upper() \
                           for location in locations]) + ')'
    else:
        print "Wrong locations parameter"
        print "No locations clause in generated sql query"

    channels_ok = False
    if (channels is not None) and (isinstance(channels, \
                                              collections.Iterable)):
        channels_ok = True
        if isinstance(channels, basestring):
            channels = [channels]

        channels = [channel.upper() for channel in channels]
        channel_clause = '(' + \
            (' or ').join(["channel='%s'" % channel.upper() \
                           for channel in channels]) + ')'
    else:
        print "Wrong channels parameter"
        print "No channels clause in generated sql query"

    # Traces that cross the starttime point
    stime_clause = ''
    # Traces that cross the endtime point
    etime_clause = ''
    # Traces that are contained in the interval [startime,endtime]
    stime_etime_clause = ''

    if isinstance(starttime, UTCDateTime):
        stime_clause = "(starttime <= '%s' and endtime > '%s')" % \
            (starttime.isoformat(), starttime.isoformat())
        etime_clause = "(starttime < '%s' and endtime >= '%s')" % \
            (endtime.isoformat(), endtime.isoformat())
        stime_etime_clause = "(starttime >= '%s' and endtime <= '%s')" % \
            (starttime.isoformat(), endtime.isoformat())

        # Create the complete time clause
        tclause = '(' + (' or ').join([tc for tc in \
                                               [stime_clause,
                                                etime_clause,
                                                stime_etime_clause]]) + ')'

    if (fformat is not None) and (isinstance(fformat, basestring)):
        fformat_clause = "(format='%s')" % fformat
    else:
        print "Wrong fformat parameter"
        print "No fformat clause in generated sql query"

    query_sql = query_sql + '\nWHERE \n' + \
                (' and\n').join([cl for cl in [net_clause,
                                               station_clause,
                                               location_clause,
                                               channel_clause,
                                               fformat_clause,
                                               tclause] if cl != ''])

    query_sql = query_sql + '\nORDER BY network,station,location,channel'

    if DEBUG:
        print "\n"
        print query_sql
        print "\n\n"

    # Execute the query
    try:
        df = psql.frame_query(query_sql, con=conn)

        # If the connection has been created inside the function, it is
        # correct to close the connection also inside the function
        if to_close:
            conn.close()

    except Exception, e:
        print "Problem querying the DB"
        print "sql = \n%s" % query_sql
        # If the connection has been created inside the function, it is
        # correct to close the connection also inside the function
        if to_close:
            conn.close()
        return None, None

    # Create the Stream obj and populate it
    st = Stream()
    for (_, rec) in df.iterrows():
        fname = os.path.join(rec['path'], rec['file'])
        for tr in _read(fname, fformat, headonly=False, **kwargs).traces:
            if (networks_ok) and (tr.stats.network not in networks):
                continue
            if (stations_ok) and (tr.stats.station not in stations):
                continue
            if (locations_ok) and (tr.stats.location not in locations):
                continue
            if (channels_ok) and (tr.stats.channel not in channels):
                continue
            st.append(tr)

    if st.count() > 0:
        # If the starttime is given then it trims the resulting traces
        if starttime:
            st.trim(starttime=starttime, endtime=endtime,
                    nearest_sample=nearest_sample)

        st.merge(method=1, fill_value=0, interpolation_samples=1)
    else:
        print "Empty stream"

    n_trace = st.count()
    return st, n_trace


if BC_UI:
    class _stream_dbread_view(HasTraits):
    
        time_interval = Float(30.0)
        networks = List(Str, value=['PF', 'YA'])
        stations = List(Str, value=['FOR', 'UV05'])
        locations = List(Str, value=['00', '10'])
        channels = List(Str, value=['HHZ', 'HLZ'])
        fformat = Str('MSEED')
    
        trait_view = View(HGroup(Item('time_interval'),
                                 HGroup(Item('networks'),
                                        Item('stations'),
                                        Item('locations'),
                                        Item('channels')),
                                 Item('fformat')))

