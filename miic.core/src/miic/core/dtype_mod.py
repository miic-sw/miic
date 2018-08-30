import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

SI_prefix = {'names':['E','P','T','G','M','k','h','da','','d','c','m','mu','n','p','f','a'],\
             'exp':[18, 15, 12, 9, 6, 3, 2, 1, 0, -1, -2, -3, -6, -9, -12, -15, -18]}

class Unit(object):
    def __init__(self, numerator=['1'], denominator=['1']):
        assert type(numerator) == list, 'numerator must be a list of strings'
        assert type(denominator) == list, 'denominator must be a list of strings'
        tdpf, num = self.set_units(numerator)
        self.dec_prefix = deepcopy(tdpf)
        tdpf,den = self.set_units(denominator)
        self.dec_prefix -= tdpf
        self.numerator = num
        self.denominator = den


    def set_units(self,units):
        baseunits = []
        dec_prefix = 0
        for unit in units:
            pf, bu = self.set_unit(unit)
            baseunits.append(bu)
            dec_prefix += pf
        return dec_prefix, baseunits
        
    def set_unit(self,unit):
        assert type(unit) == str, 'unit must be string: %s' % unit
        if unit[-1] == 'm':
            baseunit = 'm'
        elif unit[-1] == 'g':
            baseunit = 'g'
        elif unit[-1] == 's':
            baseunit = 's'
        elif unit[-1] == 'A':
            baseunit = 'A'
        elif str(unit[-3:]) == 'mol':
            baseunit = 'mol'
        elif unit[-2:] == 'cd':
            baseunit = 'cd'
        elif unit[-1] == '1':
            baseunit = '1'
        else:
            raise TypeError('unit not recognized: %s' % unit)
        prefix_str = unit[::-1].replace(baseunit,'',1)[::-1]
        try:
            index = SI_prefix['names'].index(prefix_str)
            dec_pref = SI_prefix['exp'][index]
        except ValueError:
            print 'prefix not recognized: %s' % unit
            raise
        return dec_pref, baseunit
    
    def copy(self):
        return deepcopy(self)
    
    def simplify(self):
        out = self.copy()
        units = {}
        for tnum in out.numerator:
            if tnum in units.keys():
                units[tnum] +=1
            else:
                units.update({tnum:1})
        for tden in out.denominator:
            if tden in units.keys():
                units[tden] -=1
            else:
                units.update({tden:-1})
        out.numerator = []
        out.denominator = []
        for unit in units.keys():
            if units[unit] < 0:
                for ind in range(-units[unit]):
                    out.denominator.append(unit)
            if units[unit] > 0:
                for ind in range(units[unit]):
                    out.numerator.append(unit)
        out.numerator = sorted(out.numerator)
        if not out.numerator:
            out.numerator = ['1']
        out.denominator = sorted(out.denominator)
        if not out.denominator:
            out.denominator = ['1']
        return out

    def __str__(self):
        num = ''
        for tu in set(self.numerator):
            cnt = self.numerator.count(tu)
            if tu != 1:
                if cnt != 1:
                    num += ' %s^%d' %(tu,cnt)
                else:
                    num += tu
        if num == '':
            num = '1'

        den = ''
        for tu in set(self.denominator):
            cnt = self.denominator.count(tu)
            if tu != '1':
                if cnt != 1:
                    den += ' %s^%d' %(tu,cnt)
                else:
                    den += ' %s' % tu
            
        if num != '1':
            try:
                index = SI_prefix['exp'].index(self.dec_prefix)
                prefix_str = SI_prefix['names'][index]
            except ValueError:
                prefix_str = '10^%d' % self.dec_prefix
            if den != '':
                out = '%s%s/%s' %(prefix_str, num.strip(), den.strip())
            else:
                out = '%s%s' %(prefix_str, num.strip())               
        else:
            if den != '':
                # put prefix in denominator
                try:
                    index = SI_prefix['exp'].index(-1 * self.dec_prefix)
                    prefix_str = SI_prefix['names'][index]
                    out = '1/%s%s' %(prefix_str, den.strip())
                except ValueError:
                    prefix_str = '10^%d' % self.dec_prefix
                    out = '%s*1/%s' %(prefix_str, num.strip(), den.strip())
            else:
                if self.dec_prefix == 0:
                    out = '1'
                else:
                    out = '10^%d' % self.dec_prefix
            
        out.strip()
        return out
        

    def __eq__(self,other):
        this = self.simplify()
        that = other.simplify()
        if (this.numerator == that.numerator) \
            and (this.denominator == that.denominator) \
            and (this.dec_prefix == that.dec_prefix):
            return True
        else:
            return False

                
    def __add__(self,other):
        assert isinstance(other,Unit), 'not a unit object %s' % other
        assert self == other, 'Units must be equal'
        return self.copy()
        
    
    def __mul__(self,other):
        out = self.copy()
        for tu in other.numerator:
            out.numerator.append(tu)
        for tu in other.denominator:
            out.denominator.append(tu)
        out.dec_prefix += other.dec_prefix
        return out
        

class Scalar(object):
    def __init__(self, data=0., s_type='', name='', unit=Unit()):
        assert type(s_type) == str, 's_type must be a string.'
        assert type(name) == str, 'name must be a string.'
        assert isinstance(unit,Unit), 'unit must be a Unit object.'
        try:
            float(data)
        except ValueError:
            print 'data must be a number.'
            return False
        self.data_type = 'scalar'
        self.type = s_type
        self.unit = unit
        self.name = name
        self.data = data
    
    def copy(self):
        return deepcopy(self)
        
    def get_data(self):
        return deepcopy(self.data)
        
    def __add__(self, other):
        out = self.copy()
        if type(other) == Scalar:
            assert out.unit == other.unit, 'Units must be equal'
            if out.type:
                if other.type:
                    assert out.type == other.type, 'Types must be equal'
            else:
                out.type = deepcopy(other.type)
            out.name += ' + %s' % other.name
            out.data += other.data
        else:
            out.data += other
        return out
                
    def __mul__(self,other):
        out = self.copy()
        if type(other) == Scalar:
            out.type += ' * %s' % other.type
            out.name += ' * %s' % other.name
            out.unit *= other.unit
            out.data *= other.data
        else:
            out.data *= other
        return out
    
    def __str__(self):
        out = '%e %s' % (self.data,self.unit)
        return out
            

class Spaced_values(object):
    def __init__(self,start=0, delta=0, length=0, s_type='', name='', unit=Unit()):
        assert type(s_type) == str, 's_type must be a string.'
        assert type(name) == str, 'name must be a string.'
        assert isinstance(unit,Unit), 'unit must be a Unit object.'
        assert type(length) == int, 'length must be an int.'
        self.data_type = 'Spaced_values'
        self.type = s_type
        self.unit = unit
        self.name = name
        self.start = start
        self.delta = delta
        self.length = length

    def copy(self):
        return deepcopy(self)
    
    def get_data(self):
        out = self.start + (self.delta * np.arange(self.length))
        return out
        
    def get_Scalar(self,ind):
        if ind >= self.length:
            raise IndexError
        if ind<0:
            ind += self.length
        data = self.start + (self.delta * ind)
        return Scalar(data=data, s_type=self.type, name=self.name, unit=self.unit)
    
    def __str__(self):
        out = ''
        if self.length < 10:
            for tind in range(self.length):
                out += str(self.get_Scalar(tind))+'\n'
        else:
            for tind in [0,1,2]:
                out += str(self.get_Scalar(tind))+'\n'
            out += '...\n'
            for tind in [-3,-2,-1]:
                out += str(self.get_Scalar(tind))+'\n'
        return out
        
    
class Series(object):
    def __init__(self, data=np.ndarray((0,)), s_type='', name='', unit=Unit()):
        assert type(s_type) == str, 's_type must be a string.'
        assert type(name) == str, 'name must be a string.'
        assert isinstance(unit,Unit), 'unit must be a Unit object.'
        assert type(data) == np.ndarray, 'data must be a numpy.ndarray'
        assert len(data.shape) == 1, 'data must be a single dimension numpy.ndarray'
            
        self.data_type = 'series'
        self.type = s_type
        self.unit = unit
        self.name = name
        self.data = deepcopy(data)
        self.length = len(data)
    
    def copy(self):
        return deepcopy(self)
    
    def get_Scalar(self,ind):
        if ind >= self.length:
            raise IndexError
        if ind<0:
            ind += self.length
        data = self.data[ind]
        return Scalar(data=data, s_type=self.type, name=self.name, unit=self.unit)
    
    def get_data(self):
        return deepcopy(self.data)


class Vector(object):
    def __init__(self, data=np.ndarray((0,)), v_type='', axis=[], name='', unit=Unit()):
        assert type(v_type) == str, 'v_type must be a string.'
        assert type(name) == str, 'name must be a string.'
        assert isinstance(unit,Unit), 'unit must be a Unit object.'
        assert type(axis) == list, 'axis must be a list.'
        for point in axis:
            assert type(point) == Scalar, 'All elements of axis must be dtype_mod.Scalar'
        assert type(data) == np.ndarray, 'data must be a numpy.ndarray'
        assert len(data.shape) == 1, 'data must be a single dimension numpy.ndarray'
        assert len(data) == len(axis), 'Length of data and axis must be equal'
        self.data_type = 'vector'
        self.type = v_type
        self.unit = unit
        self.name = name
        self.data = data
        self.axis = axis
    
    def copy(self):
        return deepcopy(self)
        
    def get_data(self):
        return deepcopy(self.data)


class Matrix(object):
    def __init__(self, data=np.ndarray((0,0)), m_type='', first_axis=Spaced_values(), second_axis=Spaced_values(), name='', unit=Unit()):
        assert type(m_type) == str, 'm_type must be a string.'
        assert type(name) == str, 'name must be a string.'
        assert isinstance(unit,Unit), 'unit must be a Unit object.'
        assert isinstance(first_axis,Spaced_values) or isinstance(first_axis,Series),\
                'first_axis must be a dtype_mod.Spaced_values or dtype_mod.Series'
        assert isinstance(second_axis,Spaced_values) or isinstance(second_axis,Series),\
                'second_axis must be a dtype_mod.Spaced_values a dtype_mod.Series'
        assert type(data) == np.ndarray, 'data must be a numpy.ndarray'
        assert len(data.shape) == 2, 'data must be a two dimension numpy.ndarray'
        self.data_type = 'matrix'
        self.type = m_type
        self.unit = unit
        self.name = name
        self.first_axis = first_axis
        self.second_axis = second_axis
        self.data = data
        self.shape = data.shape
        
    def copy(self):
        return deepcopy(self)
    
    def get_data(self):
        return self.data
        
    def plot(self):
        fig,ax = plt.subplots()
        im = ax.imshow(self.data,aspect='auto',extent=[self.first_axis.get_Scalar(0).data,self.first_axis.get_Scalar(-1).data,\
                       self.second_axis.get_Scalar(0).data,self.second_axis.get_Scalar(-1).data])
        xt = ax.get_xticklabels()
        ax.set_xlabel('%s [%s]' % (self.second_axis.name,self.second_axis.unit))
        ax.set_ylabel('%s [%s]' % (self.first_axis.name,self.first_axis.unit))
        plt.colorbar(im,ax=ax,label='%s [%s]' % (self.type,self.unit))
        plt.title(self.name)
        plt.show()
        


