import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from accessibility_analyzing import utlis as ut

class accessibility_calculator:
    def __init__(self, npy_file, research_area_file,is_sysm_matrix=True, accessibility_type='culmu_type',
                 whether_count_opportunity=True,time_boundary=2700, demography_index='Sum_PEO',
                 opportunity_index=None,delelte_shp_file = None,
                 target_index='index',deprived_boundary=0.05,
                 output_file='./datarep/test.shp',is_timefeecost_model=False,yuan_per_sec = 0.016,
                 ):
        # 主要参数读取
        self.research_area_file = research_area_file
        self.npy_file = npy_file
        self.is_sysm_matrix = is_sysm_matrix
        self.accessibility_type = accessibility_type
        self.whether_count_opportunity = whether_count_opportunity
        self.time_boundary = time_boundary
        self.delete_shp_file = delelte_shp_file
        self.target_index = target_index # 按顺序编制的唯一编号。
        self.output_file = output_file
        self.opportunity_index = opportunity_index
        self.is_timefeecost_model = is_timefeecost_model# 选择计算时间费效模型，则输出的可达性计算文件中，会另外被添加三个字段，综合费效,时间费效，票价费效
        self.yuan_per_sec = yuan_per_sec
        self.cali_name = 'deleted_index'#被删除掉的元素名称
        # 读取文件模块
        self.od_matrix = self._read_matrix(self.npy_file, self.is_sysm_matrix)
        self.df_file = self._read_file(self.research_area_file)
        # end
        # if self.delelte_shp_file != None:
        #     self.od_matrix, self.delete_index = self._del_target_cell(self.od_matrix,self.is_sysm_matrix,self.delete_shp_file,self.target_index)

        self.df_file = self._calculate(self.opportunity_index, self.od_matrix, self.delete_shp_file,
                        self.target_index, self.df_file, self.yuan_per_sec, self.time_boundary, self.is_timefeecost_model,
                        self.whether_count_opportunity,self.accessibility_type)

        self.demo_index = demography_index  # 人口字段
        self.deprived_boundary = deprived_boundary
        if self.deprived_boundary is not None:
            self.df_file = self._calculate_deprived_people(self.deprived_boundary, self.df_file, self.cali_name,self.demo_index)
        self._classify(self.df_file, 'access_index', -6,)
    def _bacic_sta(self):
        pass
    def _read_matrix(self, file, is_sys_mat):
        od_matrix = np.load(file)
        if is_sys_mat == False:
            return od_matrix + od_matrix.T
        else:
            return od_matrix

    def _read_file(self, file):

        assert file[-3:] ==  'shp' or file[-3:] ==  'xls'

        if file[-3:] == 'xls':
            return pd.read_excel(file)
        if file[-3:] == 'shp':
            return gpd.read_file(file)

    def _classify(self,df,column_name, number):
        classify_column_name = column_name+'_le'
        df[classify_column_name]= ut.value_classify(df, column_name, number)

    def _del_target_cell(self,od_martix, target_cell_shapefile, target_index,):
        """
        target_cell_shapefile: 有些地段比如，有水体的地段需要将对应的分析单元的所有OD值均记为负数，以从可达性分析中剔除。
        target_index: 标记index

        注意函数输出的结果为对称矩阵
        """
        df = self._read_file(target_cell_shapefile)
        od = od_martix
        all_index = df[target_index]  # 所有需要被删除掉的index
        t = all_index.values
        od[t] = -10  # 将对应所有行变为负数
        od[:, t] = -10  # 将对应所有列变为负数
        # np.fill_diagonal(od, 0) #保持美观，将对角元素赋值为0 #更新，这个代码造成后期的bug
        return od,t

    def _calibration(self,df_file, cali_nam, del_list, target_index, ):
        """
        新建一列，用来表示哪些行属于被删除的网格，对应的行
        1 是被删除了的
        """
        df_file[cali_nam] = df_file.apply(lambda x: 1 if x[target_index] in del_list else 0, axis=1)
        return df_file
    def _cal_timefeecost_matrix(self,odmatrix,yuan_per_sec):
        '''
        计算时间费效综合可达性模型
        年平均收入 除以250天工作日
        '''

        t = np.modf(odmatrix)
        return t[0]*10000/yuan_per_sec+t[1], t[0]*10000,t[1]
    def _join_table(self, access, access_stat, df, target_index,):
        '''
        用来基于target_index将可达性计算结果与文件计算结果连接
        '''
        index = [x for x in range(len(df))]
        acc_ind = access
        df_dir = {'index_t': index, 'access_index': acc_ind,'ac_sta':access_stat}
        data = pd.DataFrame(df_dir)
        df1 = df.join(data.set_index('index_t'), on=target_index, how='left')
        # del df1['index_t']
        return df1
    def _join_table_timefeemodel(self, access, access_stat,time, fee, df, target_index,):
        '''
        用来基于target_index将可达性计算结果与文件计算结果连接
        '''
        index = [x for x in range(len(df))]
        acc_ind = access
        df_dir = {'index_t': index, 'access_index': acc_ind,'ac_sta':access_stat, "ac_t_sta": time, 'ac_f_sta' : fee}
        data = pd.DataFrame(df_dir)
        df1 = df.join(data.set_index('index_t'), on=target_index, how='left')
        # del df1['index_t']
        return df1

    def _calculate(self, opportunity_index, od_matrix, delete_shp_file,
                   target_index, df_file, yuan_per_sec, time_boundary, is_timefeecost_model,
                   whether_count_opportunity, accessibility_type):


        """
        解算模块
        """
        if self.delete_shp_file != None:

            od_matrix, delete_list = self._del_target_cell(od_matrix, delete_shp_file, target_index)
        else:
            #如果字段为空，自动将记录删除过字段的列表记录为空
            delete_list = [] #没有需要删除的元素，delete_list为空。
        df_file = self._calibration(df_file, cali_nam='deleted_index',del_list=delete_list, target_index=target_index)

        if is_timefeecost_model == True:#如果是考量时间与票价的综合模型，则使用一下模型
            od_matrix_final = self._cal_timefeecost_matrix(od_matrix, yuan_per_sec)
            od_matrix = od_matrix_final[0]
            fee_od_matrix =od_matrix_final[1]
            time_od_matrix = od_matrix_final[2]
            access_basic_stat_time = np.sum(time_od_matrix, axis=1)
            access_basic_stat_time = (access_basic_stat_time / len(od_matrix)) #转换为小时

            access_basic_stat_fee = np.sum(fee_od_matrix, axis=1)
            access_basic_stat_fee = access_basic_stat_fee / len(od_matrix)

        if accessibility_type == "culmu_type":
            temp_array = np.where(((od_matrix >= 0) & (od_matrix <= time_boundary)), 1,
                                  0)  # 自己可达自己的情况不用刨除
            access_basic_stat = np.sum(od_matrix, axis=1)
            access_basic_stat = (access_basic_stat/len(od_matrix)) # 计算每个单元的平均通勤时间，该模块计算完成后，需要参与表连接
            if whether_count_opportunity == False:  # 最简单的模式 仅仅只按照面积计算可达性
                access_area_num = np.sum(temp_array, axis=1)  # time_boundary 内可达的单元数量
                df_file['access_area_num'] = access_area_num
                # df_file['access_level'] = pd.DataFrame(classify(access_area_num))#得出每个分析单元再45分钟内可以到达其他分析单元的总数量,并将结果进行分类
            else:
                temp = temp_array * np.array([np.array(df_file[opportunity_index])
                                              for i in range(len(df_file))])  # 首先把df_file[population_index] 由纵向数组变化为横向数组
                # 然后进行单个元素级别相乘
                access_index = np.sum(temp, axis=1)
                if is_timefeecost_model == True:
                    df_file = self._join_table_timefeemodel(access_index, access_basic_stat,
                                                                 access_basic_stat_time, access_basic_stat_fee
                                                                 ,df_file, target_index)
                else:
                    df_file = self._join_table(access_index, access_basic_stat, df_file, target_index)
                # self.df_file['access_index'] = access_index

                # df_file['access_level'] = pd.DataFrame(classify(access_index))#结合字段值，得出每个分析单元再45分钟内可以到达其他分析单元的总数量,并将结果进行分类

        if accessibility_type == "gravity_type":
            pass

        return df_file

    def _calculate_deprived_people(self, deprived_boundary, df_file, cali_name, demo_index):
        """
        用来计算有多少人被剥夺
        被剥夺的人数将赋值为0
        deprived_boundary: 被剥夺的边界
        accind_index: 可达性指数的索引
        population_index: 人数的索引 用来计算有多少人会被排除在外
        """

        df_file['deprived_pop'] = np.where(((df_file['access_index'] <= deprived_boundary) & (df_file['access_index'] >= 0)
                                                 & (df_file[cali_name] == 0)),
                                      df_file[demo_index], 0)
        #
        df_file['no_deprived_pop'] = np.where(((df_file['access_index'] > deprived_boundary)
                                                    & (df_file[cali_name] == 0)), df_file[demo_index],0)
        df_file['deprived_access'] = np.where(df_file['access_index'] <= deprived_boundary, df_file['access_index'], 0)
        df_file['no_deprived_access'] = np.where(df_file['access_index'] <= deprived_boundary, 0, df_file['access_index'])

        return df_file

    def to_csv(self,file_path):
        return self.df_file.to_csv(file_path)

    def to_dataframe(self):
        return self.df_file

    def to_shp(self, file_path):
        t = self.df_file
        return t.to_file(file_path)

def AC_generator(mintime, maxtime, timegap, deprived_boundary, opportunity_index,
                     npy_file=r"D:\pyprojectlbw\odtime_generate\datarep\2198_2197_night_sz.npy",
                     research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
                     delete_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                     is_timefeecost_model=False):
    '''
    可达性生成器
    '''
    for each in range(mintime, maxtime + timegap, timegap):
        ac = accessibility_calculator(npy_file=npy_file, research_area_file=research_area_file,
                delelte_shp_file=delete_shp_file, opportunity_index=opportunity_index, time_boundary=each,
                deprived_boundary=deprived_boundary, is_timefeecost_model=is_timefeecost_model)
        yield ac,each,opportunity_index

if __name__ == "__main__":

    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:

        # ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_night_sz.npy",
        #                               research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
                                      # delelte_shp_file=r'D:\multicities\data\深圳分区\水库_Clip.shp',
                                      # opportunity_index=each,
                                      # time_boundary=3600,
                                      # deprived_boundary=0.05,
                                      # is_timefeecost_model=False)

        # ac.to_shp(file_path=r'D:\multicities\data\深圳分区\test\sz_acce_{0}_1h_nodel.shp'.format(each))

    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
    # 为了剥夺模块计算方便，重新计算一遍，将相应结果写入sz_access_dir 中。
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy",
    #                                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
    #                                   opportunity_index=each,
    #                                   time_boundary=3600,
    #                                   deprived_boundary=0.05,
    #                                   is_timefeecost_model=True)
    #
    #     ac.to_shp(file_path=r'./datarep/sz_access_dir/sz_access_{}_nodel_tfc.shp'.format(each))
    #

    #论文中最新的运行模块
    # for each in ['entr_0_per','entr_1_per','entr_2_1_p']:
    #
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy",
    #                                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
    #                                   opportunity_index=each,
    #                                   time_boundary=3300,
    #                                   deprived_boundary=0.05,
    #                                   is_timefeecost_model=True)
    #
    #     ac.to_shp(file_path=r'./datarep/sz_access_dir/time_boundary_3300/sz_access_{}_nodel_tfc.shp'.format(each))



    #------------------------以下为深圳的两期可达性计算结果----------------------------------
    # ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy",
    #                               research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
    #                               opportunity_index='agg_en_per',
    #                               time_boundary=3300,
    #                               deprived_boundary=0.05,
                                  # is_timefeecost_model=True)
    #
    # ac.to_shp(file_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300/sz_access_JUN_nodel_tfc.shp')
    #
    # ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz_1108.npy",
    #                               research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro_aggre.shp',
    #                               opportunity_index='agg_en_per',
    #                               time_boundary=3300,
    #                               deprived_boundary=0.05,
                                  # is_timefeecost_model=True)
    #
    # ac.to_shp(file_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300/sz_access_NOV_nodel_tfc.shp')

    #-------------------------以下为两期时间节点下深圳市三生空间的可达性变化评价-----------------------
    # for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy",
    #                                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
    #                                   opportunity_index=each,
    #                                   time_boundary=3300,
    #                                   deprived_boundary=0.05,
    #                                   is_timefeecost_model=True)
    #
    #     ac.to_shp(file_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300/JUN_results/sz_access_JUN_nodel_tfc_{}.shp'.format(each))


    # for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz_1108.npy",
    #                                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
    #                                   opportunity_index=each,
    #                                   time_boundary=3300,
    #                                   deprived_boundary=0.05,
    #                                   is_timefeecost_model=True)
    #
    #     ac.to_shp(file_path=r'./datarep//sz_access_liangqi_dir/time_boundary_3300/NOV_results/sz_access_NOV_nodel_tfc_{}.shp'.format(each))

    # 注：这个代码是用来测试1115的数据的，最后没有采用，还是使用的1108的那份数据。
    # for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz_1115.npy",
    #                                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
    #                                   opportunity_index=each,
    #                                   time_boundary=3300,
    #                                   deprived_boundary=0.05,
    #                                   is_timefeecost_model=True)
    #     ac.to_shp(file_path=r'./datarep//sz_access_liangqi_dir/time_boundary_3300_1/NOV_result/sz_access_NOV_nodel_tfc_{}.shp'.format(each))

    # 以下代码用来查看深圳市每个单元的到其他单元的平均出行时间
    # ac = accessibility_calculator(
    #     npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz.npy",
    #     research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
    #     opportunity_index='entr_0_per',
    #     time_boundary=3300,
    #     deprived_boundary=None,
    #     is_timefeecost_model=True)
    #
    # ac.to_shp(file_path=r'./datarep/sz_access_dir_basic_stat/sz_access_nodel_tfc_basic_stat.shp')

    # 以下代码用来测试最新版代码的效果
    for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
        ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2198_2197_transit_withfee_sz_1108.npy",
                                      research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
                                      opportunity_index=each,
                                      time_boundary=3300,
                                      deprived_boundary=0.05,
                                      is_timefeecost_model=True)
        ac.to_shp(file_path=r'./datarep//sz_access_liangqi_dir/test/sz_access_NOV_nodel_tfc_{}.shp'.format(each))

    # 以下代码用来测试深圳3.10收集的数据
    # for each in ['entr_0_per', 'entr_1_per', 'entr_2_1_p']:
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/2197_2198_transit_withfee_sz_210310.npy",
    #                                   research_area_file=r'D:\multicities\data\深圳分区\sz_10_acc_entro.shp',
    #                                   opportunity_index=each,
    #                                   time_boundary=3300,
    #                                   deprived_boundary=0.05,
    #                                   is_timefeecost_model=True,
    #                                   is_sysm_matrix=False)
    #
    #     ac.to_shp(file_path=r'./datarep/sz_access_liangqi_dir/time_boundary_3300_2_310/sz_access_310_nodel_tfc_{}.shp'.format(each))



    #------------------------以下为武汉市的计算结果----------------------------------
    # ac = accessibility_calculator(
    #     npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/1541_1540_transit_withfee.npy",
    #     research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_zc_withjob.shp',
    #     opportunity_index='SUM_grid_c',
    #     time_boundary=2700,
    #     deprived_boundary=None,
    #     is_timefeecost_model=True,
    #     target_index='index1', )
    # ac.to_shp(file_path=r'./datarep/wuhan_access_dir/wh_dsfz_access_job_nodel_tfc_2700.shp')

    # for time_range in [1800,2700,3600]:
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #                                   research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #                                   opportunity_index='emp_per',
    #                                   time_boundary=time_range,
    #                                   deprived_boundary=None,
    #                                   is_timefeecost_model=True,
    #                                   target_index='index1',)
    #     ac.to_shp(file_path=r'./datarep/wuhan_access_dir/wh_dsfz_access_job_nodel_tfc_{}.shp'.format(time_range))

    #以下代码为了仅仅只给每个文件添加一个deprived 字段
    # for time_range in [1800,3600,5400]:
    #     ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #                                   research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #                                   opportunity_index='emp_per',
    #                                   demography_index='pop',
    #                                   time_boundary=time_range,
    #                                   deprived_boundary=0.018,
    #                                   is_timefeecost_model=True,
    #                                   target_index='index1',
    #                                   )
    #     ac.to_shp(file_path=r'./datarep/wuhan_access_dir/deprived_dir1/wh_dsfz_access_job_nodel_tfc_{}.shp'.format(time_range))


    # ac = accessibility_calculator(npy_file=r"D:/pyprojectlbw/odtime_generate/datarep/1797_1796_wuhan_dsfz_transit_withfee.npy",
    #                               research_area_file=r'D:\multicities\data\wuhan\wuhan_ts_dsfz_withjobpop.shp',
    #                               opportunity_index='emp_per',
    #                               demography_index='pop',
    #                               time_boundary=1800,
    #                               deprived_boundary=None,
    #                               is_timefeecost_model=True,
    #                               target_index='index1',
    #                               )
    # ac.to_shp(file_path=r'./datarep/wuhan_access_dir/basic_stat/wh_dsfz_access_job_nodel_tfc_basic_stat.shp')