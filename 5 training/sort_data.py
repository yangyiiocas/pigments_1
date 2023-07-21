import scipy.io as scio
import pickle
import numpy as np
import os


def get_x(grid,lat,lon,chose_input):
    # 这里首先会计算各种参数，根据chose_inputs进行选择保存
    x = {}

    bbw = [0.00305149029009044,0.00223100068978965,0.00174739013891667,0.00147526187356561,0.00103128689806908,0.000909828173462302,0.000855824153404683,0.000455705419881269,0.000396181712858379,0.000370067748008296,0.000245936738792807,0.000138672316097654,0.000132197805214673,3.06449401250575e-05,9.77994477580069e-06,3.37679944095726e-06]
    aw = [0.00488336198031902,0.00637746788561344,0.0102276708930731,0.0137411449104548,0.0429367981851101,0.0505724288523197,0.0577158555388451,0.318534463644028,0.424950867891312,0.452210456132889,2.82907462120056,4.34513902664185,4.63710498809814,116.181182861328,642.374755859375,2557.63647460938]
    bands = [412,443,469,488,531,547,555,645,667,678]


    ### calculate bbp and aph
    bbp,adg,aph = {},{},{}
    for i,band in enumerate(bands):
        bbp["bbp_"+str(band)] = grid["bb_"+str(band)+"_giop"]-bbw[i]
        adg["adg_"+str(band)] = grid["adg_443_giop"]*np.exp(-grid["adg_s_giop"]*(band-443))
        aph["aph_"+str(band)] = grid["a_"+str(band)+"_giop"] - adg["adg_"+str(band)] - aw[i]
    
    ### Rrs
    Rrs = {}
    for band in bands:
        Rrs["Rrs_"+str(band)] = grid["Rrs_"+str(band)]
    
    ### plus bands information on Rrs
    Rrs_band = {}
    for i in range(len(bands)):
        for j in range(i+1,len(bands)):
            Rrs_band["Rrs_band_"+str(bands[i])+"_"+str(bands[j])] = (grid["Rrs_"+str(bands[i])]-grid["Rrs_"+str(bands[j])])/(bands[i]-bands[j])
    

    # 这里选择保存的参数
    # 注意，为了保存原始的数据，x["lon"], x["satellite_chla"]也被保存了，但是这并不是网络的输入数据，因此在下一步保存原始数据后会删掉
    ######## which is inter ########
    if chose_input["lat_lon"]:
        
        x["lon_sin"] = np.sin(lon/180.*np.pi)
        x["lat"] = lat
        x["lon"] = lon
    if chose_input["sst"]:
        x["sst"] = grid["sst"]
    if chose_input["par"]:
        x["par"] = grid["par"]
    if chose_input["Zeu"]:
        x["Zeu"] = grid["Zeu_lee"]
    if chose_input["chl_a"]:
        x["chlor_a"] = grid["chlor_a"]
    if chose_input["Rrs"]:
        for var in Rrs: x[var] = Rrs[var]
    if chose_input["bbp_aph"]:
        for var in bbp: x[var] = bbp[var]
        for var in aph: x[var] = aph[var]
    if chose_input["Rrs_band"]:
        for var in Rrs_band: x[var] = Rrs_band[var]
        
    x["satellite_chla"] = grid["chlor_a"]
    return x



def sort_train_test(pigments, pigments_use, x_dict, lat, lon, case1, save_sub_path, del_matchup_use=None):
    # x_bp作为匹配提取出的原始数据进行保存
    x_bp = x_dict.copy()
    # 这两个删除针对需要保存但是不输入模型的lon和chla原始数据
    del  x_dict["satellite_chla"]
    if "lon" in x_dict:
        del x_dict["lon"]
        
    # 剩下的数据都是提取输入的数据
    x_all = np.array(list(x_dict.values())).transpose(1,0)
    matchup_use = {}
    
    for var in pigments:

        # 只保存输入都存在的数据，如果不是1day的数据，还会删除和1day重复的数据
        if del_matchup_use is None:
            use = pigments_use[var]&(~np.isnan(x_all.sum(axis=1)))
        else:
            use = pigments_use[var]&(~np.isnan(x_all.sum(axis=1)))&(~del_matchup_use[var])
        # save raw data
        x_raw = {item:x_bp[item][use] for item in x_bp}
        y_raw = pigments[var][use] 

        # extract available data
        x = x_all[use]

        # 这里色素数据取对数
        pigment = pigments[var][use]
        pigment[pigment==0] = np.nan
        pigment = np.log10(pigment)
        y = pigment[:,np.newaxis]
        

        print(var,'sorted.')

        # save data 
        # 保存为以色素为文件名的输入x,输出y,xy的原始值(包括不进行输入时的lon chla),以及case1water.
        # 注意的是，在我们最后的模型中，我们对case1的限制已经做完了，实际上这里就的case1全为True.只是在我们all water的实验中会存在False
        with open("../0 save data/"+save_sub_path+"/"+var+".pkl",'wb') as f:
            pickle.dump([x,y,x_raw,y_raw,case1[use]], f)
    
        # 这里返回参数匹配的数据，为了不重复1day和7days匹配的
        matchup_use[var] = use
    return matchup_use



def get_data(chose_input):
    # 这里时获取matchups数据的部分，分别读取色素数据和两部分卫星数据
    # 输入的chose_inputs是选择作为输入的卫星数据
    with open("../0 save data/2 sorted_pigments.pkl",'rb') as f:
        dates, lon, lat, depth, pigments, pigments_use = pickle.load(f)
    with open("../0 save data/4 extract data.pkl",'rb') as f:
        grid,long_name,case1 = pickle.load(f)
    with open("../0 save data/4 extract data 7 day.pkl",'rb') as f:
        grid_7day,long_name,case1_7day = pickle.load(f)
    

    # 两部分卫星数据分别与色素数据匹配
    ## for 1 day dataset
    x_dict = get_x(grid,lat,lon,chose_input) # 根据chose_inputs提取输入的数据
    del_matchup_use = sort_train_test(pigments, pigments_use, x_dict, lat, lon, case1, "train data")


    ## for 7 days dataset
    # 在这里，需要删除1day的匹配数据，因此多传入了del_matchup_use
    x_dict = get_x(grid_7day,lat,lon,chose_input)
    sort_train_test(pigments, pigments_use, x_dict, lat, lon, case1, "train data 7 day",del_matchup_use)
    
    # 这里结果数据以储存文件的形式保存了，因此无返回值
    return 
