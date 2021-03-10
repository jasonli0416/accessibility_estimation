import pandas as pd
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_file(file):
    '''
    读取文件
    '''
    assert file[-3:] == 'shp' or file[-3:] == 'xls'

    if file[-3:] == 'xls':
        return pd.read_excel(file)
    if file[-3:] == 'shp':
        return gpd.read_file(file)

def to_file(gdf,file_dir):
    gdf.to_file(file_dir)

def value_classify(df, column_name, number=8):
    '''
    number: 分类数量
    '''
    mean_val = df[column_name].mean()
    std_val = df[column_name].std()
    std_val = np.abs(std_val)
    temp = df[column_name].copy(deep=True)
    if number==8 :
        temp[(df[column_name]-mean_val)>3*std_val] = 8
        temp[(2 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 3 * std_val)] = 7
        temp[(1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 2 * std_val)] = 6
        temp[(0 < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 1 * std_val)] = 5
        temp[(-1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 0)] = 4
        temp[(-2 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= -1 * std_val)] = 3
        temp[(-3 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= -2 * std_val)] = 2
        temp[(-3 * std_val >= (df[column_name] - mean_val))] = 1  # 相对可达性最低的区域
        print([sum(temp==x) for x in range(1,9)])

    if number==6:
        # 这种分类方式是专用于相对可达性计算的
        temp[(df[column_name]) > 2 * std_val] = 6
        temp[(1 * std_val < df[column_name]) & (df[column_name] <= 2 * std_val)] = 5
        temp[(0 < df[column_name]) & (df[column_name]<= 1 * std_val)] = 4
        temp[(-1 * std_val < df[column_name]) & (df[column_name]<= 0)] = 3
        temp[(-2 * std_val < df[column_name]) & (df[column_name]<= -1 * std_val)] = 2
        temp[(-2 * std_val >= df[column_name])] = 1  # 相对可达性最低的区域
        print([sum(temp==x) for x in range(1,7)])

    if number==-6:
        # 专用于对可达性计算结果的分类
        temp[(df[column_name]-mean_val) > 3 * std_val] = 6
        temp[(2 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 3 * std_val)] = 5
        temp[(1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 2 * std_val)] = 4
        temp[(0 < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 1 * std_val)] = 3
        temp[(-1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 0)] = 2
        temp[(-1 * std_val >= (df[column_name] - mean_val))] = 1

        print('平均值为：', mean_val, ' ', '标准差为：', mean_val)
        print([sum(temp == x) for x in range(1, 7)])
    if number==-7:
        # 专用于对可达性计算结果的分类
        temp[(df[column_name]-mean_val) > 4 * std_val] = 7
        temp[(3 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 4 * std_val)] = 6
        temp[(2 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 3 * std_val)] = 5
        temp[(1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 2 * std_val)] = 4
        temp[(0 < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 1 * std_val)] = 3
        temp[(-1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 0)] = 2
        temp[(-1 * std_val >= (df[column_name] - mean_val))] = 1
        print('平均值为：', mean_val, ' ', '标准差为：', mean_val)

        print([sum(temp == x) for x in range(1, 8)])
    if number==-5:
        #该分类专门用于熵值可达性结果的分类
        temp[(df[column_name]-mean_val) > 3 * std_val] = 5
        temp[(2 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 3 * std_val)] = 4
        temp[(1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 2 * std_val)] = 3
        temp[(0 < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 1 * std_val)] = 2
        temp[(-1 * std_val < (df[column_name] - mean_val)) & ((df[column_name] - mean_val) <= 0)] = 1
        print([sum(temp == x) for x in range(1, 6)])

    return temp


def iter_shpfile(dir_name, filter_type):

    '''
    从文件夹中遍历出文件
    '''
    for maindir, subdir, file_name_list in os.walk(dir_name):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容

            if ext in filter_type:
                # print(filename.split('.')[0])
                yield apath,filename.split('.')[0]

def generate_district_indexlist(dir_name = r'D:\multicities\data\深圳分区\分区去重结果\最终结果1', filter_type = ['.shp'], target_index=2):
    '''
    返回ArcGIS中已分类好的各区shp文件的对应索引
    target_index: 行政区的唯一ID

    '''

    for each,file_name in iter_shpfile(dir_name,filter_type):
        df_file = gpd.read_file(r'{}'.format(each)) #遍历得到shp文件
        # print(df_file)
        yield list(df_file.iloc[:, target_index]),file_name

def cal_4_sta_ind(df,target_index,panda_dic,name=None):
    '''
    输入df,然后对目标字段(target_index)进行基本统计值统计
    name 是统计区域的名称
    '''
    max_value = df[target_index].max()
    mean_value = df[target_index].mean()
    std_value = df[target_index].std()
    cv_value = std_value/mean_value
    panda_dic['最大值'].append(max_value)
    panda_dic['平均值'].append(mean_value)
    panda_dic['标准差'].append(std_value)
    panda_dic['变异系数'].append(cv_value)
    if name is not None:
        panda_dic['名称'].append(name)
    return panda_dic

def cal_5_sta_ind(df,target_index,panda_dic,name=None):
    '''
    输入df,然后对目标字段(target_index)进行基本统计值统计
    name 是统计区域的名称
    '''
    max_value = df[target_index].max()
    min_value = df[target_index].min()
    mean_value = df[target_index].mean()
    median_value = df[target_index].median()
    std_value = df[target_index].std()
    panda_dic['最大值'].append(max_value)
    panda_dic['最小值'].append(min_value)
    panda_dic['平均值'].append(mean_value)
    panda_dic['标准差'].append(std_value)
    panda_dic['中位数'].append(median_value)
    if name is not None:
        panda_dic['名称'].append(name)
    return panda_dic



if __name__ == '__main__':
    pass