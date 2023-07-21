import os
import numpy as np
import netCDF4 as nc
import h5py
import scipy as scio
import datetime
import pickle


# def GridData(data,lon,lat):
#     from scipy.interpolate import griddata
#     lon_grid,lat_grid = np.meshgrid(lon,lat)
#     avail_data = scio.loadmat("../mask_24.mat")["avail_data"].astype(bool).T
#     isnan = np.isnan(data)
#     a,b,values = lon_grid[~isnan],lat_grid[~isnan],data[~isnan]
#     points = np.array([(a[i],b[i]) for i in range(len(a))])

#     a,b = lon_grid[avail_data],lat_grid[avail_data]
#     grid_points = np.array([(a[i],b[i]) for i in range(len(a))])
#     data[avail_data] = griddata(points,values,grid_points)
#     return data


class DataObj():
    def __init__(self, date_name, param):
        self.date_name = date_name
        
        self.win_s   = param["window_s"]
        self.win_t   = param["window_t"]
        self.path    = param["path"]
        self.var     = param["var"]
        self.lon_var = param["lon_var"]
        self.lat_var = param["lat_var"]
        
        y,m,d = int(self.date_name[:4]),int(self.date_name[4:6]),int(self.date_name[6:])
        yd = (datetime.datetime(y,m,d)-datetime.datetime(y,1,1)).days+1
        name = "A{:0>4d}{:0>3d}.L3m_DAY".format(y,yd)
        for file in os.listdir(self.path):
            if name in file:
                self.date_name = file.split(".nc")[0]
        self.is_exist = os.path.exists(self.path+self.date_name+".nc")
        self.i         = []
        self.win_index = []
        self.lat       = []
        self.lon       = []
        self.date      = []
        
        
    def add_situ(self, i, win_index, lat, lon, date):
        self.i.append(i)
        self.win_index.append(win_index)
        self.lat.append(lat)
        self.lon.append(lon)
        self.date.append(date)
        
        
    def add_data_var(self):
        self.length = len(self.lat)
        self.data = np.nan*np.zeros([self.length, self.win_s, self.win_s])
        self.lat_center = np.zeros(self.length)
        self.lon_center = np.zeros(self.length)
        
    def add_data(self):
        if not self.is_exist: return

        data_file = self.path+self.date_name+".nc"
        with nc.Dataset(data_file) as f:
            lon_ext, lat_ext = f[self.lon_var][:], f[self.lat_var][:]
            var_ext = np.squeeze(f[self.var][:])
        var_ext.data[var_ext.mask == True] = np.nan
        var_ext = var_ext.data  
        # var_ext = GridData(var_ext,lon=lon_ext,lat=lat_ext)

        for i in range(self.length): 
            a, b, w = np.argmin(abs(lat_ext-self.lat[i])), np.argmin(abs(lon_ext-self.lon[i])), (self.win_s-1)//2
            c = np.arange(b-w,b+w+1) % len(lon_ext)

            self.lat_center[i] = lat_ext[a]
            self.lon_center[i] = lon_ext[b]

            extract = var_ext[a-w:a+w+1,c]
            if extract.shape==(self.win_s,self.win_s):
                self.data[i,:,:] =  extract
            else:
                d = extract.shape[0]
                if self.lat[i]>0: self.data[i,-d:,:] = extract
                else: self.data[i,:d,:] = extract




def main(param):
    # 读取HPLC数据
    with open(param["use_data"],'rb') as f:
        dates, lon, lat, _, _, _ = pickle.load(f)

    print("use length:{:d}".format(len(dates)))
    
    # 需要使用的天数据记录下来，这样可以只读取一次
    data_obj = {}
    for i in range(len(dates)):
        dates_win = [dates[i]+datetime.timedelta(days=t) for t in range(-(param["window_t"]-1)//2,(param["window_t"]-1)//2+1)]
        for win_index in range(len(dates_win)):
            date_name = "{:0>4d}{:0>2d}{:0>2d}".format(dates_win[win_index].year,
                                                       dates_win[win_index].month,
                                                       dates_win[win_index].day)
            
            if date_name not in data_obj:
                data_obj[date_name] = DataObj(date_name, param)
            data_obj[date_name].add_situ(i=i, 
                                         win_index=win_index, 
                                         lat=lat[i], 
                                         lon=lon[i],
                                         date=dates[i])
    print("all data file:{:d}".format(len(data_obj)))
    
    # 读取satellite数据
    m = len(data_obj)
    for i,obj in enumerate(data_obj):
        data_obj[obj].add_data_var()
        data_obj[obj].add_data() 
        print("read inforamtion, number:{:d}, percentage process:{:.4%}, all data file:{:d}".format(i+1, (i+1)/m, m), end="\r")
    print("\n read inforamtion done!")

    # 将读取的数据转化为对应的结果
    m, count = len(dates), 0
    data = np.nan*np.zeros([m,param["window_t"],
                            param["window_s"],
                            param["window_s"]])

    for j,obj in enumerate(data_obj):
        if data_obj[obj].is_exist:
            data[data_obj[obj].i,data_obj[obj].win_index,:,:] = data_obj[obj].data

        else:
            data[data_obj[obj].i,data_obj[obj].win_index,:,:] = np.nan*np.zeros([data_obj[obj].length,
                                                                                 data_obj[obj].win_s,
                                                                                 data_obj[obj].win_s])
        count = count + data_obj[obj].length
        print("write data inforamtion,  number: {:d},  percentage: {:.4%}".format(count, count/m/param["window_t"]), end="\r")
    print("\n write data inforamtion done!")
    
    return data