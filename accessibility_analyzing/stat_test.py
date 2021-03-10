from accessibility_analyzing import utlis as ut
import numpy as np
import pandas as pd
import geopandas as gpd
from accessibility_analyzing.accessibility_calculator import AC_generator as AC_G
import scipy.stats as stats
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
from scipy.stats import levene
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def decile_calculator(input_file,is_resi,column_name, output_file=None, fws = 10):
    '''
    该模块用于基于某一字段，按照十分位，对分析单元进行分类。
    is_resi:是否有居民居住字段
    column_name 是待分类的字段名称
    '''

    df_origin = ut.read_file(input_file)
    df = df_origin[~df_origin[is_resi].isin([0])].copy(deep=True) #只选取有人居住的单元

    # for num, name in enumerate(column_name):

    # disv_name = '{}_std'.format(name)
    df['disv_std'] = (df[column_name] - np.mean(df[column_name])) / np.std(df[column_name])  # 首先进行z标准化

    #对标准化后的值求十分位数
    df['Decile_rank'] = fws - pd.qcut(df['disv_std'], fws, labels=False) # 按照十分位数，分类, 数值越小越严重


    df = df[['index','disv_std','Decile_rank']]
    df1 = df_origin.join(df.set_index('index'),on= 'index', lsuffix='_l', rsuffix='_r', how='left')
    df1['Decile_rank'] = df1['Decile_rank'].fillna(-1) #连接后有些空值填-1
    if output_file is not None:
        ut.to_file(df1,output_file)
        return df1
    else:
        return df1


def anova_oneway(df):
    '''
    单因素方差分析
    '''
    group1 = df[df['Decile_rank']==1]['access_index']#弱势人口最多的行政区
    group2 = df[df['Decile_rank']==2]['access_index']
    group3 = df[df['Decile_rank']==3]['access_index']
    group4 = df[df['Decile_rank']==4]['access_index']
    group5 = df[df['Decile_rank']==5]['access_index']
    group6 = df[df['Decile_rank']==6]['access_index']
    group7 = df[df['Decile_rank']==7]['access_index']
    group8 = df[df['Decile_rank']==8]['access_index']
    group9 = df[df['Decile_rank']==9]['access_index']
    group10 = df[df['Decile_rank']==10]['access_index']
    stat,p = levene(group1,group2,group3,group4,group5,group6,group7,group8,group9,group10) #最后验证的结果是不具有方差齐性
    F_statistic,pval = stats.f_oneway(group1,group2,group3,group4,group5,group6,
                                      group7,group8,group9,group10,)
    print('方差齐性检验结果',stat,p)
    print('方差分析结果',pval,F_statistic)


def multiple_comparsion(df):
    '''
    多重比较检验
    '''
    df = df[~df['Decile_rank'].isin([-1])] #将所有decile 为-1 的行删去
    mc = MultiComparison(df['access_index'], df['Decile_rank'])
    result = mc.tukeyhsd()

    print(result)


def multi_comparison_calculator(df_decile,mintime, maxtime, timegap, deprived_boundary, opportunity_index='agg_en_per',
                     npy_file=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy',
                     research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                     delete_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                     is_timefeecost_model=True):
    '''
    df1 含有decile 的人口分区数据
    df2 含有可达性计算计算结果的数据

    两份数据依靠index进行连接
    '''
    G = AC_G(mintime, maxtime, timegap, deprived_boundary, opportunity_index,
         npy_file=npy_file,
         research_area_file=research_area_file,
         delete_shp_file=delete_shp_file,
         is_timefeecost_model=is_timefeecost_model) #制造可达性生成器
    for each in G:
        ac_df = each.to_dataframe()
        ac_df = ac_df[['access_index','index']] #索引出我需要的两列
        temp_df = df_decile.join(ac_df.set_index('index'), on='index', how='left')
        # ut.to_file(temp_df, r'D:\multicities\data\深圳分区\temp.shp')
        anova_oneway(temp_df)
        multiple_comparsion(temp_df)


def is_equal_var(p_value, data1, data2):
    if p_value > 0.05:
        print('通过了方差齐性检验')
        s, p = stats.ttest_ind(data1, data2, equal_var=True)
    else:

        s, p = stats.ttest_ind(data1, data2, equal_var=False)
    return s, p
def plot_box(df_decile,mintime,maxtime,timegap,opportunity_index,fig_save_path='./datarep/sz_access_liangqi_dir/sz_access_stat/fig'):

    """
    绘制箱型图
    df: 时间节点1对应的df
    df1：时间节点2对应的df
    """
    AG = accessibility_generator(df_decile, mintime, maxtime, timegap,
                                 opportunity_index=opportunity_index,
                                 npy_file=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy',
                                 research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                                 delete_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                                 is_timefeecost_model=True,deprived_boundary=None)

    AG1 = accessibility_generator(df_decile, mintime, maxtime, timegap,
                                 opportunity_index=opportunity_index,
                                 npy_file=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz_1108.npy',
                                 research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                                 delete_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                                 is_timefeecost_model=True,deprived_boundary=None)
    try:
        while True:
            df,_,__=next(AG)
            df1,___,____=next(AG1)
            fig, ax = plt.subplots(1,10,figsize=(18,12))

            plt.title("time is {0}, opportunity is {1}".format(_,__))
            colors = mpl.cm.get_cmap('Spectral')
            col = colors(np.linspace(0,1,10))
            for each in range(10):
                group = df[df['Decile_rank'] == each+1]['access_index']
                group_ = df1[df1['Decile_rank'] == each+1]['access_index']
                sns.boxplot(ax=ax[each])
                flierprops=dict(marker='o',markerfacecolor='grey',markersize=3,linestyle='none',alpha=0.5)
                meanprops = dict(marker='D',markerfacecolor='red',markersize=8,linestyle='none',alpha=0.5)
                # medianprops = dict(color='white')
                # boxprops=dict(c=col[each])
                bp = ax[each].boxplot([group,group_],showmeans=True,widths=0.5,flierprops=flierprops,meanprops=meanprops,
                                      patch_artist=True,)
                bwith=0.3
                ax[each].spines['bottom'].set_linewidth(bwith)
                ax[each].spines['left'].set_linewidth(bwith)
                ax[each].spines['top'].set_linewidth(bwith)
                ax[each].spines['right'].set_linewidth(bwith)
                for patch in bp['boxes']:
                    patch.set(facecolor=col[each])
                if each != 0:
                    ax[each].set_yticks([])
                    ax[each].set_yticklabels([])
                # else:
                    # ax[each].set_yticks(fontsize=20)
            #
            plt.yticks(fontsize=10)
            plt.savefig(fig_save_path + '/{}_{}'.format(_, __))

            # plt.show()
    except StopIteration:
        return





def t_test(df,time=None,opportunity_type=None,excel_save_path='./datarep/sz_access_liangqi_dir/sz_access_stat'):
    '''
    由于实验数据不具有方差齐性，因此这里只能使用t检验，检验最差的两组，和最好的两组的实验结果。
    '''
    group1 = df[df['Decile_rank'] == 1]['access_index']  # 弱势人口最多的行政区
    group2 = df[df['Decile_rank'] == 2]['access_index']

    group6 = df[df['Decile_rank'] == 6]['access_index']
    group7 = df[df['Decile_rank'] == 7]['access_index']
    group8 = df[df['Decile_rank'] == 8]['access_index']
    group9 = df[df['Decile_rank'] == 9]['access_index']
    group10 = df[df['Decile_rank'] == 10]['access_index']


    stat, p1 = stats.levene(group1, group6)
    stat, p2 = stats.levene(group1, group7)
    stat, p3 = stats.levene(group1, group8)
    stat, p4 = stats.levene(group1, group9)
    stat, p5 = stats.levene(group1, group10)

    stat, p6 = stats.levene(group2, group6)
    stat, p7 = stats.levene(group2, group7)
    stat, p8 = stats.levene(group2, group8)
    stat, p9 = stats.levene(group2, group9)
    stat, p10 = stats.levene(group2, group10)

    panda_dic = dict(name1=[],name2=[],t_ind=[],p_val=[])
    print('关键平均值为：')
    print(group1.mean(),group2.mean(),group6.mean(),group7.mean(),group8.mean(),group9.mean(),group10.mean())

    s,p = is_equal_var(p1, group1, group6)
    print('decile1 与 decile6 的t检验结果：', 't统计量值：',s, '对应p值：',p)
    panda_dic['name1'].append('decile1')
    panda_dic['name2'].append('decile6')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s,p = is_equal_var(p2, group1, group7)
    print('decile1 与 decile7 的t检验结果：', 't统计量值：',s, '对应p值：',p)
    panda_dic['name1'].append('decile1')
    panda_dic['name2'].append('decile7')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s,p = is_equal_var(p3, group1, group8)
    print('decile1 与 decile8 的t检验结果：', 't统计量值：',s, '对应p值：',p)
    panda_dic['name1'].append('decile1')
    panda_dic['name2'].append('decile8')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s,p = is_equal_var(p4, group1, group9)
    print('decile1 与 decile9 的t检验结果：', 't统计量值：',s, '对应p值：',p)
    panda_dic['name1'].append('decile1')
    panda_dic['name2'].append('decile9')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s, p = is_equal_var(p5, group1, group10)
    print('decile1 与 decile10 的t检验结果：', 't统计量值：', s, '对应p值：', p)
    panda_dic['name1'].append('decile1')
    panda_dic['name2'].append('decil10')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s, p = is_equal_var(p6, group2, group6)
    print('decile2 与 decile6 的t检验结果：', 't统计量值：', s, '对应p值：', p)
    panda_dic['name1'].append('decile2')
    panda_dic['name2'].append('decile6')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s, p = is_equal_var(p7, group2, group7)
    print('decile2 与 decile7 的t检验结果：', 't统计量值：', s, '对应p值：', p)
    panda_dic['name1'].append('decile2')
    panda_dic['name2'].append('decile7')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s, p = is_equal_var(p8, group2, group8)
    print('decile2 与 decile8 的t检验结果：', 't统计量值：', s, '对应p值：', p)
    panda_dic['name1'].append('decile2')
    panda_dic['name2'].append('decile8')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s, p = is_equal_var(p9, group2, group9)
    print('decile2 与 decile9 的t检验结果：', 't统计量值：', s, '对应p值：', p)
    panda_dic['name1'].append('decile2')
    panda_dic['name2'].append('decile9')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))

    s, p = is_equal_var(p10, group2, group10)
    print('decile2 与 decile10 的t检验结果：', 't统计量值：', s, '对应p值：', p)
    panda_dic['name1'].append('decile2')
    panda_dic['name2'].append('decile10')
    panda_dic['t_ind'].append(str(s))
    panda_dic['p_val'].append(str(p))
    df_pandic = pd.DataFrame.from_dict(panda_dic)
    if time is not None:
        df_pandic.to_excel(excel_save_path+'/{}_{}.xls'.format(time,opportunity_type))


def accessibility_generator(
                     df_decile,mintime, maxtime, timegap, deprived_boundary, opportunity_index='agg_en_per',
                     npy_file=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy',
                     research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                     delete_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                     is_timefeecost_model=True):

    G = AC_G(mintime, maxtime, timegap, deprived_boundary, opportunity_index,
             npy_file=npy_file,
             research_area_file=research_area_file,
             delete_shp_file=delete_shp_file,
             is_timefeecost_model=is_timefeecost_model)  # 制造可达性生成器
    for each, each1, each2 in G:
        print('{}费效边界值下的可达性结果：'.format(each1))
        ac_df = each.to_dataframe()
        ac_df = ac_df[['access_index', 'index']]  # 索引出我需要的两列
        temp_df = df_decile.join(ac_df.set_index('index'), on='index', how='left')  # 将df_decile 与research_area_file
        yield temp_df, each1, each2 # each1 为时间，each2 为 机会类型

def t_test_calculator(df_decile,mintime, maxtime, timegap, deprived_boundary, opportunity_index='agg_en_per',
                     npy_file=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy',
                     research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                     delete_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                     is_timefeecost_model=True,
                     excel_save_path='./datarep/sz_access_liangqi_dir/sz_access_stat'):

    for temp_df, each1, each2 in accessibility_generator(df_decile,mintime, maxtime, timegap, deprived_boundary, opportunity_index=opportunity_index,
                     npy_file=npy_file,
                     research_area_file=research_area_file,
                     delete_shp_file=delete_shp_file,
                     is_timefeecost_model=is_timefeecost_model):
        t_test(temp_df,each1,each2,excel_save_path)

if __name__ == '__main__':
    # 以下代码生成十分位结果
    input_file = r'D:\multicities\data\深圳分区\shenzhen_pop.shp'
    is_resi_column = 'is_resi'
    # column = 'disv_sl'
    # column1 = 'disv_sh'
    # output_file = r'D:\multicities\data\深圳分区\pop_decile_sl.shp'
    # output_file1 = r'D:\multicities\data\深圳分区\pop_decile_sh.shp'
    #
    # column2 = 'dissl_p'
    # column3 = 'dissh_p'
    #
    # output_file2 = r'D:\multicities\data\深圳分区\pop_decile_sl_p.shp'
    # output_file3 = r'D:\multicities\data\深圳分区\pop_decile_sh_p.shp'
    #
    # df_sl = decile_calculator(input_file, is_resi_column, column2,output_file=output_file2)
    # df_sh = decile_calculator(input_file, is_resi_column, column3,output_file=output_file3)

    column4 = 'dist_p'
    output_file3 = r'D:\multicities\data\深圳分区\pop_decile_slsh_p.shp'

    # df_sh = decile_calculator(input_file, is_resi_column, column4,output_file=output_file3)
    df_slsh = decile_calculator(input_file, is_resi_column, column4)




    # print('生理性弱势群体分析')
    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
    #     print('   ')
    #     print('******************','机会类型: ',each,'********************')
    #     print('   ')
    #     multi_comparison_calculator(df_sl, mintime=1800, maxtime=3600, timegap=900,
    #                                 deprived_boundary=None, opportunity_index='entr_2_1_p',
    #                                 research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')
        #
        # t_test_calculator(df_sl, mintime=1800, maxtime=3600, timegap=900,
        #                             deprived_boundary=None, opportunity_index=each,
        #                             research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')

    # print('   ')
    # print('-------------------------------------------------------------------------',)
    # print('   ')
    #
    # print('社会性弱势群体分析')
    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
    #     print('   ')
    #     print('******************', '机会类型: ', each, '********************')
    #     print('   ')
    #     multi_comparison_calculator(df_sl, mintime=1800, maxtime=3600, timegap=900,
    #                                 deprived_boundary=None, opportunity_index='entr_2_1_p',
    #                                 research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')
        #
        # t_test_calculator(df_sh, mintime=1800, maxtime=3600, timegap=900,
        #                   deprived_boundary=None, opportunity_index=each,
        #                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')

    # print('   ')
    # print('-------------------------------------------------------------------------', )
    # print('   ')
    #
    # print('时间节点1弱势群体分析')
    # for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
    #     print('   ')
    #     print('******************', '机会类型: ', each, '********************')
    #     print('   ')
        # multi_comparison_calculator(df_slsh, mintime=1800, maxtime=3600, timegap=900,
        #                             deprived_boundary=None, opportunity_index=each,
        #                             research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')
        #
        # t_test_calculator(df_slsh, mintime=3300, maxtime=3400, timegap=100,
        #                   deprived_boundary=None, opportunity_index=each,
        #                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')

    print('   ')
    print('-------------------------------------------------------------------------', )
    print('   ')

    print('时间节点2弱势群体分析')
    for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
        print('   ')
        print('******************', '机会类型: ', each, '********************')
        print('   ')
    # multi_comparison_calculator(df_slsh, mintime=1800, maxtime=3600, timegap=900,
    #                             deprived_boundary=None, opportunity_index=each,
    #                             research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp')
    #
        t_test_calculator(df_slsh, mintime=3300, maxtime=3400, timegap=100,
                      deprived_boundary=None, opportunity_index=each,
                      research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                      excel_save_path='./datarep/sz_access_liangqi_dir/sz_access_stat/jiedian2',
                      npy_file=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz_1108.npy')
    #
    # print('   ')
    # print('-------------------------------------------------------------------------', )
    # print('   ')
    #
    # print('箱型图统计指标')
    # for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
    #     plot_box(df_slsh,3300,3400,100,each)


