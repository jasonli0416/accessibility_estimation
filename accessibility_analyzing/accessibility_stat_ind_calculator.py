from accessibility_analyzing.utlis import cal_4_sta_ind,generate_district_indexlist
from accessibility_analyzing.accessibility_calculator import accessibility_calculator as AC
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt

'''
本文件中两个函数
dividing_region 函数用以计算分区可达性统计结果，最后以xls格式的文件输出在datarep文件夹下
plot_accessbility_stat 函数用以计算 对应城市 不同时间边界下的 可达性统计结果
'''
def dividing_region_plot():
    '''
    绘制可达性计算结果柱状图统计值
    '''
    pass

def dividing_region(access_index = 'access_index',opportunity_type='entr_0_per',target_index='index',
                    npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy",
                    research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
                    delete_shp_file=None,
                    deprived_boundary = 0.05,
                    time_boundary=3600,
                    panda_dic = None,
                    excel_save_path = r'./datarep/',
                    is_timefeecost_model=False,
                    divided_region_dir = r'D:\multicities\data\深圳分区\分区去重结果\最终结果1',
                    divided_rigion_index = 2,
                    reindex_index = None):
    '''
    对公交服务按区进行划分
    然后按照一定时间边界(3600s)输出分区可达性计算结果
    新增了参数，divided_region_index 这个是分区文件夹中，每个行政区shp文件的唯一识别码对应索引值
    '''
    ac = AC(target_index=target_index,npy_file=npy_file, research_area_file=research_area_file,
       delelte_shp_file=delete_shp_file, opportunity_index=opportunity_type, time_boundary=time_boundary,
       deprived_boundary=deprived_boundary,is_timefeecost_model=is_timefeecost_model)
    AC_result = ac.to_dataframe() # 可达性计算结果
    cal_4_sta_ind(AC_result,access_index,panda_dic,'全市')

    for each,file_name in generate_district_indexlist(dir_name=divided_region_dir,target_index=divided_rigion_index):
        t = AC_result[AC_result[target_index].isin(each)] #按行政区进行划分的最主要逻辑，
        cal_4_sta_ind(t, access_index, panda_dic, file_name)

    df = pd.DataFrame.from_dict(panda_dic)
    if reindex_index is not None:
        df = df.reindex(reindex_index) #对表中每个区的计算结果进行重新排序
    df.index = df['名称']
    del df['名称']
    df.to_excel(excel_save_path+r'/results_{0}_{1}.xls'.format(opportunity_type,time_boundary))

def plot_accessbility_stat_ind(access_index = 'access_index',
                           target_index='index',
                           npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_night_sz.npy",
                           research_area_file=r'D:/multicities/data/深圳分区/sz_10_acc_entro.shp',
                           deprived_boundary = 0.05,
                           mintime=1800,
                           maxtime=5400,
                           timegap=600,
                           color_list = ['cornflowerblue','orangered','lightgreen'],
                           fig_save_path=r'./datarep/',
                           is_timefeecost_model=False,
                           opportunity_list=['entr_0_per', 'entr_1_per', 'entr_2_1_p']):
    '''
    绘制不同时间边界下可达性统计结果

    '''
    assert isinstance(opportunity_list, list) #保证opportunity_list 是列表元素
    fig, ax = plt.subplots(len(opportunity_list), 1, sharex=True)
    for index,opportunity_type in enumerate(opportunity_list):

        panda_dic = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
        if len(opportunity_list) == 1:
            ax1 = ax
        else:
            ax1 = ax[index]
        ax2 = ax1.twinx()
        for each in range(mintime, maxtime + timegap, timegap):

            ac = AC(target_index=target_index, npy_file=npy_file, research_area_file=research_area_file,
                opportunity_index=opportunity_type, time_boundary=each,
                deprived_boundary=deprived_boundary, is_timefeecost_model=is_timefeecost_model)
            AC_result = ac.to_dataframe()  # 可达性计算结果
            cal_4_sta_ind(AC_result, access_index, panda_dic, each)

            df = pd.DataFrame.from_dict(panda_dic)
            df.index = df['名称']
            ax1.plot(df['名称'], df['平均值'], color=color_list[index], linewidth=1, label='        ')

            ax2.plot(df['名称'],df['变异系数'], color=color_list[index], linewidth=1,linestyle='--', label='        ')
            tick_list = [x for x in np.arange(mintime,(maxtime+timegap), timegap)]
            ax1.set_xticks(tick_list)
            ax1.set_xticklabels([str(int(x/60)) for x in tick_list])
            ax1.set_yticks([0,0.05,0.1,0.15,0.2,0.25,0.3])
            ax1.set_yticklabels(['0','','0.1','','0.2','','0.3'])
            ax1.axvline(x=3300, ls='-.', c='grey')
            ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
            ax2.set_yticklabels(['0', '', '1', '', '2', ''])
            ax1.grid(True)
        ax1.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        fig.subplots_adjust(right=0.8)
    plt.savefig(fig_save_path+'/{}'.format('test'))
    plt.show()

def plot_accessbility_stat_ind_comparasion(access_index = 'access_index',
                           target_index='index',
                           npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_night_sz.npy",
                           npy_file1= r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_night_sz_1108.npy",
                           research_area_file=r'D:/multicities/data/深圳分区/sz_10_acc_entro.shp',
                           deprived_boundary = 0.05,
                           mintime=1800,
                           maxtime=5400,
                           timegap=600,
                           color_list = ['cornflowerblue','orangered','lightgreen'],
                           color_list1 = ['mediumblue','darkred','forestgreen'],
                           fig_save_path=r'./datarep/',
                           is_timefeecost_model=False,
                           opportunity_list=['entr_0_per', 'entr_1_per', 'entr_2_1_p']):
    '''
    单独一个子图一个子图的输出结果
    '''


    for index,opportunity_type in enumerate(opportunity_list):
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        panda_dic = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
        panda_dic1 = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
        ax2 = ax1.twinx()
        for each in range(mintime, maxtime + timegap, timegap):

            ac = AC(target_index=target_index, npy_file=npy_file, research_area_file=research_area_file,
                opportunity_index=opportunity_type, time_boundary=each,
                deprived_boundary=deprived_boundary, is_timefeecost_model=is_timefeecost_model)

            ac1= AC(target_index=target_index, npy_file=npy_file1, research_area_file=research_area_file,
                    opportunity_index=opportunity_type, time_boundary=each,
                    deprived_boundary=deprived_boundary, is_timefeecost_model=is_timefeecost_model)

            AC_result = ac.to_dataframe()  # 可达性计算结果
            AC_result1 = ac1.to_dataframe()  # 可达性计算结果

            cal_4_sta_ind(AC_result, access_index, panda_dic, each)
            cal_4_sta_ind(AC_result1, access_index, panda_dic1, each)

            df = pd.DataFrame.from_dict(panda_dic)
            df1 = pd.DataFrame.from_dict(panda_dic1)

            df.index = df['名称']
            ax1.plot(df['名称'], df['平均值'], color=color_list[index], linewidth=1)
            ax1.plot(df1['名称'], df1['平均值'], color=color_list1[index], linewidth=1)

            ax2.plot(df['名称'],df['变异系数'], color=color_list[index], linewidth=1,linestyle='--')
            ax2.plot(df1['名称'],df1['变异系数'], color=color_list1[index], linewidth=1,linestyle='--')

            tick_list = [x for x in np.arange(mintime,(maxtime+timegap), timegap)]
            ax1.set_xticks(tick_list)
            ax1.set_xticklabels([str(int(x/60)) for x in tick_list])
            ax1.set_yticks([0,0.05,0.1,0.15,0.2,0.25,0.3])
            ax1.set_yticklabels(['0','0.05','0.1','0.15','0.2','0.25','0.3'])
            ax1.axvline(x=3300, ls='-.', c='grey')
            ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
            ax2.set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
            ax1.grid(True)

        plt.savefig(fig_save_path+'/{}'.format(opportunity_type))
        plt.show()

    pass




if __name__ == '__main__':

    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
    #     panda_dic = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
    #     dividing_region(opportunity_type=each, panda_dic=panda_dic)
    # plot_accessbility_stat_ind()

    #--------------------------论文中的深圳市可达性输出结果---------------------------------
    for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
        panda_dic = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
        dividing_region(opportunity_type=each,
                        time_boundary=3300,
                        panda_dic=panda_dic,
                        npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy",
                        is_timefeecost_model=True,
                        excel_save_path=r'./datarep/sz_access_dir/time_boundary_3300',
                        reindex_index=[0,2,7,8,6,5,10,9,3,1,4])
    # plot_accessbility_stat_ind(npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy",
    #                            is_timefeecost_model=True,
    #                            fig_save_path=r'./datarep/sz_access_dir/time_boundary_3300',
    #                            timegap=300)

    #--------------------------论文中的深圳市两期可达性对比分析---------------------------------


    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
    #     panda_dic = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
    #     dividing_region(opportunity_type=each,
    #                     time_boundary=3300,
    #                     panda_dic=panda_dic,
    #                     research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
    #                     npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz_1115.npy",
    #                     is_timefeecost_model=True,
    #                     excel_save_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300_1',
    #                     reindex_index=[0,2,7,8,6,5,10,9,3,1,4])

    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
    #     panda_dic = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
    #     dividing_region(opportunity_type=each,
    #                     time_boundary=3300,
    #                     panda_dic=panda_dic,
    #                     research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
    #                     npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz_1108.npy",
    #                     is_timefeecost_model=True,
    #                     excel_save_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300',
    #                     reindex_index=[0,2,7,8,6,5,10,9,3,1,4])

    # plot_accessbility_stat_ind(npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz_1108.npy",
    #                            is_timefeecost_model=True,
    #                            fig_save_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300',
    #                            timegap=300)

    # plot_accessbility_stat_ind_comparasion(
    #                            npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy",
    #                            npy_file1=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz_1108.npy",
    #                            is_timefeecost_model=True,
    #                            fig_save_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300',
    #                            timegap=300)

    #-------------------------以下为武汉市的相应计算结果------------------------------------
    # for each in [1800,2700,3600]:
    #     panda_dic = dict(最大值=[], 平均值=[], 标准差=[], 变异系数=[], 名称=[])
    #
    #     dividing_region(target_index='index1',
    #                     opportunity_type='emp_per',
    #                     panda_dic=panda_dic,
    #                     npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #                     is_timefeecost_model=True,
    #                     delete_shp_file=None,
    #                     deprived_boundary=None,
    #                     time_boundary=each,
    #                     excel_save_path=r'./datarep/wuhan_access_dir',
    #                     research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #                     divided_rigion_index = 4,
    #                     divided_region_dir=r'D:\multicities\data\wuhan\行政分区\ts_800_dsfz'
    #                     )

    # plot_accessbility_stat_ind(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #                            is_timefeecost_model=True,
    #                            target_index='index1',
    #                            research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #                            fig_save_path=r'./datarep/wuhan_access_dir',
    #                            opportunity_list=['emp_per'],
    #                            color_list=['orangered'],
    #                            deprived_boundary=None)