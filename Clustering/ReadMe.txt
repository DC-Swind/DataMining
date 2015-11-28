【运行说明】：
    本次实验首先在Java中开发，但是在实现spectral clustering的时候发现Java无法处理dataset 2，所以在matlab中重新实现了spectral。所以代码由Java和matlab两部分构成。
	
    如果提示找不到Jama相关的包，还请麻烦手动导入，Jama包在DM-Assignment3根目录下。	

    DM-Assignment3是在Eclipse下开发的JAVA工程。
    DM-Assignment3/src/Algorithm中是源代码文件。其中Setting.java为配置文件，对于每个dataset需要设置四个量，分别是12-14行以及19行。
    matlab是在matlab中开发的spectral clustering源代码。
    matlab代码的开头部分已写好两个dataset的配置，可以选择运行。

    Java、matlab的输入文件均在工程根目录下。对于dataset 1做过预处理，将类别为-1的改为0，以保持和dataset 2的一致。
    

【运行方式】：
    对于Java程序：
    在Eclipse中打开运行。
    入口函数main分别在算法对应的Java文件中。
 
    对于matlab程序：
    写好配置后，可以直接在matlab中运行。

【运行提示信息】：
    在程序中我写入了比较详细的提示信息，用来标注程序运行到哪一部分。所有程序最后输出purity 和 gini index后停止运行。

【运行时间】：
    在Java中，kmeans和NMF在dataset 1上的运行时间即使运行10次也较短，在dataset 2上由于kmeans和NMF需要重复10次，所以时间较长，k-means需要2分钟左右，NMF需要6分钟左右。
    在matlab中，处理dataset 1需要时间较短，处理dataset 2需要时间大约为5分钟。另外在matlab运行中，计算eigs的时候可能提示warning，经过研究此warning应该不会影响结果。

