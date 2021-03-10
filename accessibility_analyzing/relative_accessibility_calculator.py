from accessibility_analyzing.accessibility_calculator import accessibility_calculator as AC
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from accessibility_analyzing import utlis as ut
import matplotlib.pyplot as plt

'''


'''

def relative_calculator(ac,ac1,number=6):
    """
    计算相对可达性
    AC: 第一种交通工具的可达性,分析中以公交的可达性为基准
    AC1:第二种交通工具的可达性

    以ac为主，返回实例ac1
    """
    assert  isinstance(ac, AC) #必须是其实例
    assert  isinstance(ac1, AC)

    t = ac.to_dataframe() #以该实例为主实例
    t1 = ac1.to_dataframe()
    t['rel_acc'] = t['access_index']-t1['access_index']
    t['re_ac_cla'] = ut.value_classify(t,'rel_acc',number=number)


    return t

def calculator_generator(target_index='index',
                         npy_file_transit=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy',
                         npy_file_driving=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_driving_withfee_sz.npy',
                         research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
                         deprived_boundary=0.05,
                         output_file_path = './datarep/',
                         number=6,
                         column_name = ['entr_0_per', 'entr_1_per', 'entr_2_1_p','agg_en_per'],
                         time_boundary = 3600
                        ):
    '''
    根据机会类型序列，计算relative_calculator
    number: 分类数量
    '''
    for opportunity_type in column_name:

        ac = AC(target_index=target_index,npy_file=npy_file_transit, research_area_file=research_area_file,
                opportunity_index=opportunity_type, time_boundary=time_boundary,
                deprived_boundary=deprived_boundary,is_timefeecost_model=True,)

        ac1 = AC(target_index=target_index, npy_file=npy_file_driving, research_area_file=research_area_file,
                 opportunity_index=opportunity_type, time_boundary=time_boundary,
                 deprived_boundary=deprived_boundary, is_timefeecost_model=True)

        relative_calculator(ac,ac1,number=number).to_file(output_file_path+'/sz_rel_acc_{0}_{1}.shp'.format(opportunity_type,time_boundary))

def calculator_generator1(target_index='index',
                         npy_file_transit=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy',
                         npy_file_driving=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_driving_withfee_sz.npy',
                         research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                         deprived_boundary=0.05,
                         number=6,
                         opportunity_type = 'agg_en_per',
                         min_time_boundary=1800,
                         max_time_boundary=4800,
                         time_gap=600,
                         time_list=None
                        ):
    '''
    根据通勤耗费序列，计算relative_calculator
    number: 分类数量
    '''

    if time_list is not None:
        time_boundary_list = time_list
    else:
        time_boundary_list = [x for x in range(min_time_boundary, max_time_boundary+time_gap, time_gap)]

    for time_boundary in time_boundary_list:

        ac = AC(target_index=target_index,npy_file=npy_file_transit, research_area_file=research_area_file,
                opportunity_index=opportunity_type, time_boundary=time_boundary,
                deprived_boundary=deprived_boundary,is_timefeecost_model=True)

        ac1 = AC(target_index=target_index, npy_file=npy_file_driving, research_area_file=research_area_file,
                 opportunity_index=opportunity_type, time_boundary=time_boundary,
                 deprived_boundary=deprived_boundary, is_timefeecost_model=True)

        t = relative_calculator(ac,ac1,number=number)

        yield t, time_boundary #返回相对可达性计算结果迭代器

def seeking_value(df, cutoff_value, rel_acc_ind,pop_cum_ind,t_b):
    for each in cutoff_value:
        if each[0] * 60 == t_b:
            t = df[df[rel_acc_ind]>=each[1]]
            t1 = list(t[pop_cum_ind])[0] # 第一个元素既是所需要的截断值
            print('时间边界为',each[0],'分钟时，相对可达性结果大于',each[1],'的单元占： ',1-t1)
            print('此时的最小值为',min(df[rel_acc_ind]))

def plot_culma_pop_rel(  rel_acc_ind, pop_ind,
                         npy_file_transit=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_transit_withfee_sz.npy',
                         npy_file_driving=r'D:\pyprojectlbw\odtime_generate\datarep\2198_2197_driving_withfee_sz.npy',
                         research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
                         deprived_boundary=0.05,
                         output_file_path = r'C:\Users\43714\Desktop',
                         number=6,
                         opportunity_type = 'agg_en_per',
                         target_index='index',
                         min_time_boundary=1800,max_time_boundary=5400,time_gap=900,
                         time_list=None,
                         cutoff_value = None):
    """
    绘制人口，相对可达性占比图
    rel_acc_shp_dir: 存放相对可达性计算结果shp文件的路径
    rel_acc_ind: 存放相对可达性计算结果字段的名称
    pop_ind: 存放人口数量结果字段的名称
    cutoff_value: 截断值，初步另其为字典结构

    该函数绘制的图像 斜率越大的通勤时间越短
    """
    # df = ut.read_file(rel_acc_shp_dir)
    fig, ax = plt.subplots()
    for df,t_b in calculator_generator1(target_index=target_index,
                                        min_time_boundary=min_time_boundary ,max_time_boundary= max_time_boundary,time_gap= time_gap,
                                        npy_file_transit=npy_file_transit,
                                        npy_file_driving=npy_file_driving,
                                        research_area_file=research_area_file,
                                        deprived_boundary=deprived_boundary,
                                        number=number,
                                        opportunity_type=opportunity_type,
                                        time_list=time_list):

        df = df.sort_values(by=rel_acc_ind) #根据相对可达性结果 对整个dataframe 进行索引
        df['pop_per'] = df[pop_ind]/df[pop_ind].sum()
        df['pop_per_cum'] = df['pop_per'].cumsum() #计算人口数量的累积值
        seeking_value(df,cutoff_value,rel_acc_ind,'pop_per_cum',t_b)

        ax.plot(df[rel_acc_ind], df['pop_per_cum'],label = '       ')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1,])
    ax.set_yticklabels(['0', '20%', '40%', '60%', '80%', '100%',])
    ax.grid(True)
    plt.legend()
    plt.savefig(output_file_path+r'\temp1.png')

    plt.show()

def rel_acc_stat_cla(shp_file_dir,rel_acc_index,excel_save_path,target_index='index'):

    for shp_file, _ in ut.iter_shpfile(shp_file_dir, ['.shp']):
        panda_dic = dict(最大值=[], 最小值=[], 平均值=[], 标准差=[], 中位数=[], 名称=[])
        AC_result = ut.read_file(shp_file)
        ut.cal_5_sta_ind(AC_result, rel_acc_index, panda_dic, '全市')

        for each, file_name in ut.generate_district_indexlist():
            t = AC_result[AC_result[target_index].isin(each)]  # 按行政区进行划分的最主要逻辑，
            ut.cal_5_sta_ind(t, rel_acc_index, panda_dic, file_name)
        df = pd.DataFrame.from_dict(panda_dic)
        df.index = df['名称']
        del df['名称']

        df.to_excel(excel_save_path + r'/re_acc_sta_res_{}.xls'.format(_))






if __name__ == "__main__":
    '''
    相对可达性值越小，表示driving 的优势越小
    '''


    # 以下的代码仅仅为修改一下输出路径
    # calculator_generator(research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
    #                      output_file_path='./datarep/sz_rel_acc_dir/')

    # rel_acc_stat_cla(shp_file_dir=r'./datarep/sz_rel_acc_dir/', excel_save_path=r'./datarep',rel_acc_index='rel_acc')

    # 论文中所使用的最终代码
    plot_culma_pop_rel(
                       rel_acc_ind='rel_acc',#这个字段是固定的
                       pop_ind='Sum_PEO',
                       output_file_path='./datarep/sz_rel_acc_dir/sz_rel_acc_boundary_3300',
                       time_list=[30*60,45*60,55*60,65*60,75*60,85*60,90*60],
                       cutoff_value=[[30,0],[45,0],[55,0],[65,0],[75,0],[85,0]]
                        )

    # calculator_generator(research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
    #                      time_boundary=3300,output_file_path='./datarep/sz_rel_acc_dir/sz_rel_acc_boundary_3300')

    # 以下代码为改用建筑面积的尝试。

    # plot_culma_pop_rel(
    #     rel_acc_ind='rel_acc',
    #     pop_ind='Sum_PEO',
    #     output_file_path='./datarep/sz_access_build_s_dir/sz_rel_acc_dir',
    #     time_list=[45 * 60, 55 * 60, 65 * 60, 75 * 60, 85 * 60, 90 * 60],
    #     research_area_file = r'D:\multicities\data\深圳分区\sz_10_jzmj_entro.shp',
    #     opportunity_type='ag_ent_pe'
    #
    # )

    # 以下代码为武汉的分析场景
    # for each in [1800,3600,5400]:
    #     calculator_generator(target_index='index1',
    #                      npy_file_transit=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #                      npy_file_driving=r'D:\pyprojectlbw\odtime_generate\datarep\1797_1796_driving_withfee_wuhan.npy',
    #                      research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #                      deprived_boundary=None,
    #                      output_file_path = './datarep/wuhan_access_dir/rel_acc_dir/',
    #                      number=6,
    #                      column_name = ['emp_per'],
    #                      time_boundary = each)

    # plot_culma_pop_rel(
    #                  min_time_boundary=1800,
    #                  max_time_boundary=5400,
    #                  time_gap=1800,
    #                  rel_acc_ind='rel_acc',
    #                  pop_ind='pop',
    #                  npy_file_transit=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #                  npy_file_driving=r'D:\pyprojectlbw\odtime_generate\datarep\1797_1796_driving_withfee_wuhan.npy',
    #                  research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #                  deprived_boundary=None,
    #                  number=6,
    #                  opportunity_type = 'emp_per',
    #                  target_index='index1'
    # )
    #
    # plot_culma_pop_rel(
    #     min_time_boundary=1800,
    #     max_time_boundary=5400,
    #     time_gap=1800,
    #     rel_acc_ind='rel_acc',
    #     pop_ind='pop',
    #     npy_file_transit=r"D:/pyprojectlbw/odtime_generate/datarep/1541_1540_transit_withfee.npy",
    #     npy_file_driving=r'D:\pyprojectlbw\odtime_generate\datarep\1541_1540_driving_withfee.npy',
    #     research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_zc_withjobpop.shp',
    #     deprived_boundary=None,
    #     number=6,
    #     opportunity_type='emp_per',
    #     target_index='index1'
    # )