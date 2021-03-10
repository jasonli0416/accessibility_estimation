import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from accessibility_analyzing.accessibility_calculator import accessibility_calculator as AC #作为程序入口，暂时写成绝对引用
from accessibility_analyzing import utlis as ut


def deprived_changed(mintime, maxtime, timegap, deprived_boundary, entro_type, research_area_file,
                     npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_night_sz.npy",
                     delete_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                     is_timefeecost_model=False,
                     demography_index='Sum_PEO',
                     target_index='index'):
    """
    查询随着时间变化到底有多少人会被剥夺
    deprived_boundary: 被剥夺的可达性边界
    type: 选择何种机会
    """
    for each in range(mintime, maxtime+timegap,timegap): #TODO 这里可以解耦,单独再写一个模块,后面同样的逻辑会用到很多次
        ac = AC(target_index=target_index,npy_file=npy_file,research_area_file=research_area_file,demography_index=demography_index,
                delelte_shp_file=delete_shp_file,opportunity_index=entro_type,time_boundary=each,
                deprived_boundary=deprived_boundary,is_timefeecost_model=is_timefeecost_model)
        df_temp = ac.to_dataframe()
        temp = df_temp['deprived_pop'].sum() #查看有多少人被剥夺
        temp1 = df_temp[demography_index].sum()
        print(temp,temp1,temp/temp1)
        yield each, temp/temp1
        # print(temp)

def plot_comparison(mintime_range,maxtime_range,time_gap,deprived_boundary=0.05,
         is_timefeecost_model=False,delete_shp_file=None,
         filepath=r'C:\Users\43714\Desktop\temp.png',
         npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_night_sz.npy",
         npy_file1 = r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_night_sz.npy",
         oppor_index_list=['entr_0_per','entr_1_per','entr_2_1_p'],
         color_list=['cornflowerblue','orangered','lightgreen'],
         color_list1 = ['mediumblue','darkred','forestgreen'],
         research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
         demo_index="Sum_PEO",
         target_index='index'):
    """
    两期时间节点下的，剥夺人群对比绘图模块
    """

    fig = plt.figure()
    axe = plt.subplot()
    # color = color_list
    entroy_index = oppor_index_list
    for each, each1, each1_1 in zip(entroy_index, color_list, color_list1):
        x_label = []
        y_label = []
        for each2 in deprived_changed(mintime_range, maxtime_range, time_gap, deprived_boundary, each,
                                      delete_shp_file=delete_shp_file,
                                      is_timefeecost_model=is_timefeecost_model,
                                      npy_file=npy_file,
                                      research_area_file=research_area_file,
                                      demography_index=demo_index,
                                      target_index=target_index):
            x_label.append(each2[0] / 60)
            y_label.append(each2[1])
            l, = axe.plot(x_label, y_label, color=each1, linestyle=':',linewidth=1)
        l.set_label('                    ')
        x_label = []
        y_label = []
        for each2 in deprived_changed(mintime_range, maxtime_range, time_gap, deprived_boundary, each,
                                      delete_shp_file=delete_shp_file,
                                      is_timefeecost_model=is_timefeecost_model,
                                      npy_file=npy_file1,
                                      research_area_file=research_area_file,
                                      demography_index=demo_index,
                                      target_index=target_index):
            x_label.append(each2[0] / 60)
            y_label.append(each2[1])
            l, = axe.plot(x_label, y_label, color=each1_1,linewidth=1,)

        l.set_label('                    ')

    axe.axvline(x=55, ls='-.', c='grey')
    axe.grid(True)
    plt.legend()
    plt.yticks([x for x in np.arange(0, 1.2, 0.2)], ('0', '20%', '40%', '60%', '80%', '100%'))
    plt.xticks([x for x in np.arange(mintime_range / 60, (maxtime_range + time_gap) / 60,
                                     time_gap / 60)], ('30', '35', '40', '45', '50', '55', '60', '65', '70', '75'
                                                       , '80', '85', '90',))

    plt.savefig(filepath, dpi=300)
    plt.show()

def plot(mintime_range,maxtime_range,time_gap,deprived_boundary=0.05,
         is_timefeecost_model=False,delete_shp_file=None,
         filepath=r'C:\Users\43714\Desktop\temp.png',
         npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_night_sz.npy",
         oppor_index_list=['entr_0_per','entr_1_per','entr_2_1_p'],
         color_list=['cornflowerblue','orangered','lightgreen'],
         research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
         demo_index="Sum_PEO",
         target_index='index'):
    """
    绘图模块
    """
    fig = plt.figure()
    axe = plt.subplot()
    color = color_list
    entroy_index = oppor_index_list
    for each,each1 in zip(entroy_index,color):
        x_label = []
        y_label = []
        for each2 in deprived_changed(mintime_range,maxtime_range,time_gap,deprived_boundary,each,
                                      delete_shp_file=delete_shp_file,
                                      is_timefeecost_model=is_timefeecost_model,
                                      npy_file=npy_file,
                                      research_area_file=research_area_file,
                                      demography_index=demo_index,
                                      target_index=target_index):
            x_label.append(each2[0]/60)
            y_label.append(each2[1])
            l, =axe.plot(x_label,y_label,color = each1,)
        l.set_label('       ')

    axe.axvline(x=55,ls='-.',c='grey')
    axe.grid(True)
    plt.legend()
    plt.yticks([x for x in np.arange(0,1.2,0.2)],('0','20%','40%','60%','80%','100%'))
    plt.xticks([x for x in np.arange(mintime_range/60,(maxtime_range+time_gap)/60,
                                     time_gap/60)],('30','35','40','45','50','55','60','65','70','75'
                                                    ,'80','85','90',))

    plt.savefig(filepath,dpi=300)
    plt.show()

def deprived_stat_cal(shp_file_dir='./datarep/sz_access_dir',target_index = 'index',panda_dic=dict(),
                      deprived_index='deprived_p',demo_index="Sum_PEO",
                      excel_save_path=r'./datarep/',
                      divided_region_dir=r'D:\multicities\data\深圳分区\分区去重结果\最终结果1',
                      divided_rigion_index=2,
                      reindex_index = None,
                      excel_file_name = 'results_deprived'
                      ):
    '''
    shp_file_dir 为存放已经完成可达性计算的shp文件之路径名称
    '''

    for shp_file, _ in ut.iter_shpfile(shp_file_dir, ['.shp']):
        temp_str = '剥夺占比'+_
        temp_str1 = '名称'+_
        panda_dic[temp_str] = []
        panda_dic[temp_str1] = []
        AC_result = ut.read_file(shp_file)
        panda_dic[temp_str].append(AC_result[deprived_index].sum()/AC_result[demo_index].sum())
        panda_dic[temp_str1].append('全市')

        for each,file_name in ut.generate_district_indexlist(dir_name=divided_region_dir,target_index=divided_rigion_index):
            t = AC_result[AC_result[target_index].isin(each)] #按行政区进行划分的最主要逻辑，
            t = t[deprived_index].sum()/t[demo_index].sum()
            panda_dic[temp_str].append(t)
            panda_dic[temp_str1].append(file_name)

    df = pd.DataFrame.from_dict(panda_dic)
    if reindex_index is not None:
        df = df.reindex(reindex_index)
    df.to_excel(excel_save_path + r'/{}.xls'.format(excel_file_name))

if __name__ == "__main__":

# ----------------以下是论文中深圳第一期分析结果--------------
    # plot(1800, 5400, 300, 0.05,is_timefeecost_model=True,
    #      delete_shp_file=None,filepath=r'C:\Users\43714\Desktop\temp1.png',
    #      npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy")
    #
    # deprived_stat_cal(shp_file_dir=r'./datarep/sz_access_dir/time_boundary_3300',
    #                   excel_save_path=r'./datarep/sz_access_dir/time_boundary_3300',
    #                   reindex_index=[0,2,7,8,6,5,10,9,3,1,4])

# ----------------以下是论文中深圳第二期分析结果--------------

    # deprived_stat_cal(shp_file_dir=r'./datarep/sz_access_liangqi_dir/time_boundary_3300/NOV_results',
    #                   excel_save_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300',
                      # reindex_index=[0,2,7,8,6,5,10,9,3,1,4],
                      # )

    # plot_comparison(1800, 5400, 300, 0.05,is_timefeecost_model=True,
    #          delete_shp_file=None,filepath=r'C:\Users\43714\Desktop\temp1.png',
    #          npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy",
    #          npy_file1=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz_1108.npy")

    deprived_stat_cal(shp_file_dir=r'./datarep/sz_access_liangqi_dir/time_boundary_3300_1/NOV_result',
                      excel_save_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300_1',
                      reindex_index=[0,2,7,8,6,5,10,9,3,1,4])

#----------------以下是深圳是采用建筑面积计算的剥夺人群--------------
    # plot(1800, 5400, 300, 0.05, is_timefeecost_model=True,
    #      delete_shp_file=None, filepath=r'C:\Users\43714\Desktop\temp1.png',
    #      npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy",
    #      oppor_index_list = ['sc_ent_pe','sh_ent_pe','st_ent_pe'],
    #      research_area_file= r'D:\multicities\data\深圳分区\sz_10_jzmj_entro.shp'
    #      )
#----------------以下是武汉市的相应分析结果---------------
    # plot(1800, 7200, 600, 0.05, is_timefeecost_model=True,
    #      delete_shp_file=None,filepath=r'C:\Users\43714\Desktop\temp1.png',
    #      npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #      research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #      oppor_index_list=['emp_per'],
    #      color_list=['orangered'],
    #      demo_index='pop',
    #      target_index='index1'
    #      )
    #
    # deprived_stat_cal(shp_file_dir='./datarep/wuhan_access_dir/deprived_dir',
    #                   target_index='index1',
    #                   demo_index='pop',
    #                   divided_rigion_index=4,
    #                   divided_region_dir=r'D:\multicities\data\wuhan\行政分区\ts_800_dsfz',
    #                   excel_save_path=r'./datarep/wuhan_access_dir'
    #                   )
