
训练阶段
	1. 训练文件：wmmse_deepunfolding_final.py
	2. 训练数据：6tx6rx4ants50disBS20-80dis_user0indoor_H_realizations_training.mat
	3. 运行环境：tensorflow 1.x版本以上


测试阶段
	1. 测试文件：Sparse_WMMSE_V.S._PDG_impact_I_Q_M_final.py  
	2. 测试数据：./test_data/...
	3. 主要函数：
		1）run_WMMSE：Full JT传输，使用WMMSE求解发射预编码，提供一个上届
		2）run_SWMMSE：Mingyi Hong 文章中的算法，得到一个stationary解
		3）PF_scheduling：PF-EZF的调度算法，两种模式：逐次加用户和逐次删用户
		4）Baseline_PF_EZF：单点非协作EZF算法
		5）Baseline_PF_RSRP_WMMSE：在PF_scheduling调度出的用户基础上，每用户选择RSRP最强的T的基站，然后使用WMMSE求解发射预编码
		6）Baseline_PF_RSRP_WMMSE_remove1：每用户选择RSRP最强的T的基站，WMMSE求解发射预编码过程中，每迭代2次删除速率较小的一部分用户
		7）Run_PDG：所提的深度展开方法，直接使用所学的梯度下降步长，可以修改参数xi调整平均每用户激活基站数
	
	4. 测试文件：Sparse_WMMSE_V.S._PDG_22.5.30_impact_of_iteration.py：探索迭代次数和固定步长的性能对比
	5. 测试文件：Sparse_WMMSE_V.S._PDG_22.5.30_impact_CSI_final：信道CSI的完整度信息怼性能的影响

2022.06.03：Sparse_WMMSE_V.S._PDG_impact_I_Q_M_final.py、Sparse_WMMSE_V.S._PDG_22.5.30_impact_of_iteration.py、Sparse_WMMSE_V.S._PDG_22.5.30_impact_CSI_final  SWMMSE、Unfolding的稀疏度 已经修改过来
