import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn import preprocessing as pp
from accessibility_analyzing import utlis as ut
from accessibility_analyzing.accessibility_calculator import accessibility_calculator

"""
此模块下的代码没有使用
"""

def generate_access_index(dir_name=r'D:\multicities\data\深圳分区\可达性结算结果'):
    '''
    存放可达性计算结果的文件夹
    该文件夹下暂时只放置两个文件

    '''
    for each in ut.iter_shpfile(dir_name,['.shp']):
        t = gpd.read_file(r'{}'.format(each))
        yield t['access_ind']

def write_accind_2_file(file_dir):
    '''
    将可达性计算结果写入 file_dir 下的指定文件
    返回geo pandas 格式文件
    '''
    i = 1
    t = gpd.read_file(file_dir)
    for each in generate_access_index():
        name = 'access_ind_{}'.format(i)
        t[name] = each
        i+=1
    return t

def sklearn_pca_cal( n_component=3,
                 cal_df=None,
                 is_z_stand=False,
                 *args):
    '''
    使用PCA 前必须对数据进行标准化
     pca.fit_transform(X)# 返回降维后的数据
     pca.components_  #返回因子载荷，横向查看相应载荷，实际为协方差矩阵特征值对应的特征向量
     pca.explained_variance_ratio_ #返回方差贡献率
     pca.explained_variance_ #返回特征值

     注意： X*pca.components_ 结果等于 pca.fit_transform(X) 的输出结果

     cal_df: 进行pca计算的数据框
     is_z_stand: 这个参数不是简单的是否对输出结果进行z标准化，如果这个这个参数是False 此时计算的是直接对机会项进行PCA
     计算赋权(对原始非标准化结果直接赋权)，加和，百分化输出。 选择为True时 是对标准化后的可达性计算PCA， 赋权，加和，输出标准化后的
     原始三种可达性值 与 最终可达性值(赋权加和后结果)。
    '''
    column_name_list = []
    output_column_name_list = []

    for each in args:
        assert isinstance(each, str)
        column_name_list.append(each)
        output_column_name_list.append(each+'_sta')
    assert n_component <= len(column_name_list)

    acc_file = cal_df
    # X = acc_file[['access_ind','access_ind_1','access_ind_2']]#取出dataframe的三列
    X = acc_file[column_name_list]
    X = np.array(X)


    if is_z_stand==False:
        pca = PCA(n_components=n_component)
        NEWX = pca.fit_transform(X)  # 返回降维后的数据
        variance_ratio = pca.explained_variance_ratio_
        t = NEWX*variance_ratio
        t = t.sum(axis=1)
        t = t/t.sum(axis=0)# 计算百分值
        t = t.reshape(len(t), 1)
        results = np.concatenate((X,t),axis=1)# 最后结果，前三列为降维后的数据，最后一列为 加权相加的结果
        pca_index = 'pca_en_per'
        output_column_name_list.append(pca_index)
        results_df = pd.DataFrame(data=results, columns=output_column_name_list)
    else:
        scaler = pp.StandardScaler()
        X_scaler = scaler.fit_transform(X) #对数据进行标准化
        pca = PCA(n_components=n_component)
        NEWX = pca.fit_transform(X_scaler)  # 返回降维后的数据
        variance_ratio = pca.explained_variance_ratio_
        t = NEWX*variance_ratio #直接进行广播相乘
        t = t.sum(axis=1)
        t = t.reshape(len(t), 1)
        results = np.concatenate((X,t),axis=1)# 最后结果，前三列为降维后的数据，最后一列为 加权相加的结果
    # results_df = pd.DataFrame(data=results,columns=['acc_ind_sta','acc_ind_sta_1','acc_ind_sta_2','acc_ind_pca'])
        pca_index = 'pca_in_per'
        output_column_name_list.append(pca_index)
        results_df = pd.DataFrame(data=results,columns=output_column_name_list)

    del results_df[output_column_name_list[0]]
    del results_df[output_column_name_list[1]]
    del results_df[output_column_name_list[2]]
    final_re_df = pd.concat([acc_file, results_df], axis=1)
    final_re_df['e_0_pe_le'] = ut.value_classify(final_re_df, 'entr_0_per', number=-5)
    final_re_df['e_1_pe_le'] = ut.value_classify(final_re_df, 'entr_1_per', number=-5)
    final_re_df['e_2_pe_le'] = ut.value_classify(final_re_df, 'entr_2_1_p', number=-5)
    final_re_df['pca_en_le'] = ut.value_classify(final_re_df, pca_index, number=-5)

    return final_re_df

def entro_add(shp_dir,*args):

    df = ut.read_file(shp_dir)
    df['aggre_en']=0
    for each in args:
        assert isinstance(each, str)
        df['aggre_en']+=df[each]
    df['agg_en_per'] = df['aggre_en']/df['aggre_en'].sum()
    df['e_0_pe_le'] = ut.value_classify(df, 'entr_0_per', number=-5)
    df['e_1_pe_le'] = ut.value_classify(df, 'entr_1_per', number=-5)
    df['e_2_pe_le'] = ut.value_classify(df, 'entr_2_1_p', number=-5)
    df['agg_en_le'] = ut.value_classify(df, 'agg_en_per', number=-5)

    return df
if __name__ == '__main__':

    # df = sklearn_pca_cal(3,ut.read_file(r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp'),True,
    #             'entr_0_per','entr_1_per','entr_2_1_p')
    df = entro_add(r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
                   'entr_0','entr_1','entr_2_1')
    ut.to_file(df,r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')


