import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2
import numpy as np

def drawhevc(hclass):
    prefix = 'HEVCresults'
    font = {'family': 'serif', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
    LineWidth = 2

    if hclass == 'B':
        bpp, psnr = [0.269053, 0.146596, 0.097074, 0.067416], [35.391955, 34.391431, 33.370759, 32.255203]
        FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')

        bpp, psnr = [0.271843, 0.146412, 0.090925, 0.056782], [35.104963, 34.104735, 33.066283, 31.911116]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
        
        bpp, psnr = [0.292566,0.161836, 0.106004, 0.072696], [35.490834,34.388072,33.31253, 32.153002]
        LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')

        bpp, psnr = [0.0688, 0.1093, 0.1883, 0.3536], [31.7031, 33.2380, 34.5965, 35.7882]
        RY, = plt.plot(bpp, psnr, "c-o", color="darkred", linewidth=LineWidth, label='RY_CVPR20')

        bpp, psnr = [0.2355, 0.1192, 0.0739, 0.0559], [35.5245, 34.416, 33.3334, 32.0969]
        Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, psnr =[0.040289277833824355, 0.061040674080699686, 0.09433236599713564, 0.15265098729928334], [31.616849353790283, 32.84782929992676, 33.94570527267456, 34.87294593811035] 
        DCVC, = plt.plot(bpp, psnr, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')

        # bpp = [0.0661, 0.09362000000000001, 0.14695999999999998, 0.2232]
        # psnr = [32.721524294756705, 33.61284158051482, 34.75174138791608, 35.718610694312616]
        # msssim = [0.959144, 0.964186, 0.97111, 0.9765840000000001]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC') #  some problem : this paper use resize the test sequence, which loses information and leads to higher results.

        bpp = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
        psnr = [31.82619106, 33.0220837, 34.10751308, 35.10115456]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.2824179, 0.15401706, 0.09166346, 0.06412788]
        psnr = [35.32781542, 34.2566437, 33.20279178, 32.05932868]
        DVCp, = plt.plot(bpp, psnr, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.569167464, 0.219282161, 0.110183846, 0.065360897]
        psnr = [35.79301018, 34.03927013, 32.59008599, 31.1074001]
        msssim = [0.977265628, 0.967603596, 0.957326897, 0.94278379]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.473727262, 0.185593416, 0.08897321, 0.048943254]
        psnr = [35.78820838, 34.14234791, 32.7354592, 31.24880809]
        msssim = [0.975606114, 0.966591679, 0.956803834, 0.942605916]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'C':
        bpp, psnr = [0.350834, 0.247527, 0.174569, 0.121681], [33.482413, 32.367214, 31.096299, 29.633528]
        FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')

        bpp, psnr = [0.345057, 0.244922, 0.16963, 0.107338], [32.58818, 31.563223, 30.362042, 28.849785]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        bpp, psnr = [0.44329, 0.2811825,  0.186065, 0.12845], [34.443885, 32.3978875, 30.7971525, 29.249145]
        LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')
        
        bpp, psnr = [0.1117, 0.1951, 0.3158, 0.5441], [28.4321, 30.4874, 32.2846, 34.2527]
        RY, = plt.plot(bpp, psnr, "c-o", color="darkred", linewidth=LineWidth, label='RY_CVPR20')

        bpp, psnr = [0.3777, 0.2371, 0.1446, 0.0982], [33.6704, 32.3317, 30.7426, 29.273]
        Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, psnr = [0.08087719868104853, 0.12708667685592465, 0.18865849206403185, 0.2718540310294746], [28.314245862960817, 29.971384420394898, 31.30502407550812, 32.284087176322934] 
        DCVC, = plt.plot(bpp, psnr, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')
        
        # bpp = [0.12157499999999999, 0.17965, 0.271025, 0.38745]
        # psnr = [28.63609903605147, 29.944387388321196, 31.54062511303266, 33.03500011196936]
        # msssim = [0.9578025, 0.9686675, 0.9776324999999999, 0.98348]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.133924025, 0.19340295, 0.272746925, 0.4122508326]
        psnr = [28.767909, 29.95094243, 31.25051302, 32.51022528]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.4111753326,  0.2770532, 0.185561475, 0.12688355]
        psnr = [33.89443968, 32.16821025, 30.68711723, 29.37904568]
        DVCp, = plt.plot(bpp, psnr, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.576088116, 0.328202857, 0.193107239, 0.116934184]
        psnr = [34.63515097, 32.32511581, 30.15303183, 28.10807277]
        msssim = [0.983930296, 0.975657708, 0.963020551, 0.943658353]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.525598708, 0.295866404, 0.169516387, 0.097295083]
        psnr = [34.62769455, 32.4512852, 30.35337589, 28.28509499]
        msssim = [0.983231578, 0.974819694, 0.962047889, 0.941757279]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'D':
        bpp, psnr = [0.382361, 0.275732, 0.195583, 0.136572], [33.687894, 32.427037, 31.009378, 29.455373]
        FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')

        bpp, psnr = [0.37856, 0.274477, 0.192981, 0.121238], [32.642264, 31.520746, 30.215077, 28.655284]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
        
        bpp, psnr = [0.4673675, 0.2995675, 0.19376,0.133075], [34.4069875, 32.2339375, 30.45044, 28.883315]
        LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')
        
        bpp, psnr = [0.1173, 0.1990, 0.3347, 0.5589], [28.6544, 30.8125, 32.7950, 34.7743]
        RY, = plt.plot(bpp, psnr, "c-o", color="darkred", linewidth=LineWidth, label='RY_CVPR20')

        bpp, psnr = [0.3785, 0.2395, 0.1496, 0.1036], [34.2622, 32.5959, 30.9119, 29.4029]
        Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, psnr = [0.08711765038211727, 0.13590804261362385, 0.19764378561327856, 0.28633494290419753], [28.154973974227904, 29.943532361984253, 31.531896958351137, 32.7050665140152] 
        DCVC, = plt.plot(bpp, psnr, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')

        
        # bpp = [0.11227499999999999, 0.17395000000000002, 0.264575, 0.38207500000000005]
        # psnr = [28.82520179437781, 30.23478970812226, 31.91116994008828, 33.53631860458036]
        # msssim = [0.96777, 0.9763375, 0.9830949999999999, 0.98793]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')
        
        bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
        psnr = [28.41229473, 29.72853673, 31.1727661, 32.53451213]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.4087568561,0.281171025, 0.199286125,0.13588435]
        psnr = [33.87303353,31.87365235, 30.47633143,29.02572843]
        DVCp, = plt.plot(bpp, psnr, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.62866645, 0.369079861, 0.216769748, 0.130388726]
        psnr = [34.8324324, 32.26764653, 29.94494703, 27.82689033]
        msssim = [0.987974466, 0.980878777, 0.969883378, 0.952591233]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.593745117, 0.347317166, 0.198283963, 0.113808322]
        psnr = [34.91076403, 32.50539009, 30.21889251, 28.07014606]
        msssim = [0.987591187, 0.98046689, 0.969045592, 0.951302461]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'E':
        bpp, psnr = [0.104291, 0.06196, 0.046346, 0.034603], [40.031664, 38.871796, 37.75919, 36.411769]
        FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')

        bpp, psnr = [0.102672, 0.059125, 0.041676, 0.028598], [39.933601, 38.773299, 37.673584, 36.296838]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
        
        bpp, psnr = [0.11462,0.069586667,0.050683333,0.037183333], [40.11296333,38.96264,37.73251333,36.31533667]
        LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')

        bpp, psnr = [0.063, 0.0292, 0.018, 0.0137], [39.4295, 38.3259, 37.2213, 35.8482]
        Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, psnr = [0.01683933672804216, 0.02703989761424335, 0.035054004512332156, 0.05369957348563228], [34.676219838460284, 36.17811138153076, 37.49404130299886, 38.48175212860107] 
        DCVC, = plt.plot(bpp, psnr, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')

        # bpp = [0.02223333333333333, 0.0281, 0.04416666666666667, 0.06459999999999999]
        # psnr = [36.119993400198794, 36.895272069830746, 37.95133358562974, 38.59336757723919]
        # msssim = [0.97991, 0.98177, 0.9831466666666667, 0.98547]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.03684806667, 0.04987243333, 0.07102866667, 0.1215183]
        psnr = [36.020002, 37.53390843, 38.70012267, 39.80463083]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.1071702, 0.06180936667, 0.04365763333, 0.03117206667]
        psnr = [39.96223793, 38.78827227, 37.68675003, 36.3244915]
        DVCp, = plt.plot(bpp, psnr, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.272495739, 0.116933239, 0.058484612, 0.035880386]
        psnr = [40.28897112, 38.7756175, 37.19223474, 35.48255472]
        msssim = [0.988984555, 0.986157917, 0.982596303, 0.977072748]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.216879054, 0.092386216, 0.046302764, 0.027135239]
        psnr = [40.50252558, 39.03431677, 37.54350387, 35.90135885]
        msssim = [0.988956497, 0.986107604, 0.98287309, 0.977855586]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    else:
        print('no such class : ', + hclass)
        exit(0)
    savepathpsnr = prefix + '/' + 'HEVCClass_' + hclass + '_psnr'# + '.eps'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)    
    if hclass != 'E':
        plt.legend(handles=[h264, h265, DVC, DVCp, RY, Liu, LU, RaFC, FVC, DCVC], loc=4)
    else:
        plt.legend(handles=[h264, h265, DVC, DVCp, Liu, LU, RaFC, FVC, DCVC], loc=4)

    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title('HEVC Class ' + hclass + ' dataset')
    # plt.savefig(savepathpsnr + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(savepathpsnr + '.png')
    plt.clf()

    # ----------------------------------------MSSSIM-------------------------------------------------
    if hclass == 'B':
        bpp, msssim = [0.443496, 0.261679, 0.142189, 0.08852], [0.98591, 0.98055, 0.972843, 0.964454]
        FVC, = plt.plot(bpp, msssim, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')
        
        bpp, msssim = [0.286055, 0.171946, 0.104357, 0.062172], [0.978676, 0.9724, 0.96476, 0.953966]
        RaFC, = plt.plot(bpp, msssim, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        bpp, msssim = [0.292566,0.161836, 0.106004, 0.072696], [0.974654,0.968922,0.961012,0.948494]
        LU, = plt.plot(bpp, msssim, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')
        
        bpp, msssim = [0.0861, 0.1324, 0.2307, 0.4242], [0.9596, 0.9669, 0.9763, 0.9824]
        RY, = plt.plot(bpp, msssim, "c-o", color="darkred", linewidth=LineWidth, label='RY_CVPR20')

        bpp, msssim = [0.5875, 0.3704, 0.1976, 0.1098], [0.98851, 0.98448, 0.97854, 0.97026]
        Liu, = plt.plot(bpp, msssim, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, msssim = [0.06475324752728144, 0.1195541512221098, 0.23559859975973765, 0.39073750951687497], [0.9598303587436676, 0.9699258782863617, 0.9789459744691849, 0.9851255168914795] 
        DCVC, = plt.plot(bpp, msssim, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')

        # bpp = [0.0661, 0.09362000000000001, 0.14695999999999998, 0.2232]
        # psnr = [32.721524294756705, 33.61284158051482, 34.75174138791608, 35.718610694312616]
        # msssim = [0.959144, 0.964186, 0.97111, 0.9765840000000001]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        msssim = [0.94666318, 0.95921518, 0.96812108, 0.97383252]
        bpp = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
        DVC, = plt.plot(bpp, msssim, "y-p", linewidth=LineWidth, label='DVC')

        bpp = [0.2824179, 0.15401706, 0.09166346, 0.06412788]
        msssim = [0.97436334, 0.96841286, 0.960519, 0.94919218]
        DVCp, = plt.plot(bpp, msssim, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.569167464, 0.219282161, 0.110183846, 0.065360897]
        psnr = [35.79301018, 34.03927013, 32.59008599, 31.1074001]
        msssim = [0.977265628, 0.967603596, 0.957326897, 0.94278379]
        h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.473727262, 0.185593416, 0.08897321, 0.048943254]
        psnr = [35.78820838, 34.14234791, 32.7354592, 31.24880809]
        msssim = [0.975606114, 0.966591679, 0.956803834, 0.942605916]
        h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')

    elif hclass == 'C':
        bpp, msssim = [0.375906, 0.248239, 0.162061, 0.112118], [0.987824, 0.982816, 0.975108, 0.964439]
        FVC, = plt.plot(bpp, msssim, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')

        bpp, msssim = [0.358622, 0.243691, 0.163198, 0.094394], [0.984432, 0.978034, 0.968136, 0.950041]
        RaFC, = plt.plot(bpp, msssim, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        bpp, msssim = [0.1263825, 0.1769925, 0.258575,  0.4006125], [0.95457, 0.9681725, 0.978235, 0.9829975]
        LU, = plt.plot(bpp, msssim, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')
        
        bpp, msssim = [0.1147, 0.1551, 0.2581, 0.4118], [0.9565, 0.9624, 0.9790, 0.9856]
        RY, = plt.plot(bpp, msssim, "c-o", color="darkred", linewidth=LineWidth, label='RY_CVPR20')

        bpp, msssim = [0.466, 0.2933, 0.178, 0.1147], [0.9905, 0.98659, 0.97995, 0.97085]
        Liu, = plt.plot(bpp, msssim, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, msssim = [0.08330556079921835, 0.13537671187913025, 0.218990166298124, 0.31722916534518475], [0.9553103259205818, 0.969731270223856, 0.9799207252264023, 0.9857787843048572] 
        DCVC, = plt.plot(bpp, msssim, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')


        # bpp = [0.12157499999999999, 0.17965, 0.271025, 0.38745]
        # psnr = [28.63609903605147, 29.944387388321196, 31.54062511303266, 33.03500011196936]
        # msssim = [0.9578025, 0.9686675, 0.9776324999999999, 0.98348]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')
        

        bpp = [0.133924025, 0.19340295, 0.272746925, 0.4122508326]
        msssim = [0.9523096, 0.96620995, 0.9763428823, 0.981115925]
        DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.4111753326,  0.2770532, 0.185561475, 0.12688355]
        msssim = [0.9837059, 0.97858455, 0.96894595, 0.955701925]
        DVCp, = plt.plot(bpp, msssim, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.576088116, 0.328202857, 0.193107239, 0.116934184]
        psnr = [34.63515097, 32.32511581, 30.15303183, 28.10807277]
        msssim = [0.983930296, 0.975657708, 0.963020551, 0.943658353]
        h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.525598708, 0.295866404, 0.169516387, 0.097295083]
        psnr = [34.62769455, 32.4512852, 30.35337589, 28.28509499]
        msssim = [0.983231578, 0.974819694, 0.962047889, 0.941757279]
        h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'D':
        bpp, msssim = [0.309253, 0.210086, 0.139261, 0.097161], [0.989619, 0.984708, 0.976946, 0.966873]
        FVC, = plt.plot(bpp, msssim, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')
        
        bpp, msssim = [0.365465, 0.255783, 0.176039, 0.102085], [0.987527, 0.981579, 0.972727, 0.958684]
        RaFC, = plt.plot(bpp, msssim, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        bpp, msssim = [0.1341975, 0.190485, 0.2775625,0.4304825], [0.9600475, 0.9738, 0.98289, 0.98739]
        LU, = plt.plot(bpp, msssim, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')
        
        bpp, msssim = [0.1072, 0.1496, 0.2376, 0.3794], [0.9633, 0.9693, 0.9833, 0.9891]
        RY, = plt.plot(bpp, msssim, "c-o", color="darkred", linewidth=LineWidth, label='RY_CVPR20')

        bpp, msssim = [0.3535, 0.2227, 0.1414, 0.0963], [0.99233, 0.98849, 0.98277, 0.97411]
        Liu, = plt.plot(bpp, msssim, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, msssim = [0.07223953101965082, 0.11525956880404717, 0.18456281938279667, 0.2637081038248208], [0.9587352877855301, 0.972116578668356, 0.9819355012476444, 0.987687586247921] 
        DCVC, = plt.plot(bpp, msssim, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')

        
        # bpp = [0.11227499999999999, 0.17395000000000002, 0.264575, 0.38207500000000005]
        # psnr = [28.82520179437781, 30.23478970812226, 31.91116994008828, 33.53631860458036]
        # msssim = [0.96777, 0.9763375, 0.9830949999999999, 0.98793]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
        msssim = [0.95808245, 0.971744025, 0.981360625, 0.9863982]
        DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.4087568561,0.281171025, 0.199286125,0.13588435]
        msssim = [0.98832555, 0.982999975, 0.974272125, 0.961972025]
        DVCp, = plt.plot(bpp, msssim, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.62866645, 0.369079861, 0.216769748, 0.130388726]
        psnr = [34.8324324, 32.26764653, 29.94494703, 27.82689033]
        msssim = [0.987974466, 0.980878777, 0.969883378, 0.952591233]
        h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.593745117, 0.347317166, 0.198283963, 0.113808322]
        psnr = [34.91076403, 32.50539009, 30.21889251, 28.07014606]
        msssim = [0.987591187, 0.98046689, 0.969045592, 0.951302461]
        h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'E':
        bpp, msssim = [0.175973, 0.093687, 0.052362, 0.035589], [0.992617, 0.989891, 0.986464, 0.982571]
        FVC, = plt.plot(bpp, msssim, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')
        
        bpp, msssim = [0.138196, 0.078092, 0.04724, 0.029643], [0.990446, 0.988069, 0.985217, 0.980766]
        RaFC, = plt.plot(bpp, msssim, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        bpp, msssim = [0.11462,0.069586667,0.050683333,0.037183333], [0.988766667, 0.9864, 0.982756667, 0.976416667]
        LU, = plt.plot(bpp, msssim, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')

        bpp, msssim = [0.2654, 0.1282, 0.0564, 0.0239], [0.99432, 0.99183, 0.9889, 0.98478]
        Liu, = plt.plot(bpp, msssim, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

        bpp, msssim = [0.0254131786643217, 0.0491818304258314, 0.10638169960451849, 0.21079487220626889], [0.9789208730061849, 0.9837300658226014, 0.9879869558413823, 0.9912863037983577] 
        DCVC, = plt.plot(bpp, msssim, "c-o", color="hotpink", linewidth=LineWidth, label='DCVC')

        # bpp = [0.02223333333333333, 0.0281, 0.04416666666666667, 0.06459999999999999]
        # psnr = [36.119993400198794, 36.895272069830746, 37.95133358562974, 38.59336757723919]
        # msssim = [0.97991, 0.98177, 0.9831466666666667, 0.98547]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.03684806667, 0.04987243333, 0.07102866667, 0.1215183]
        msssim = [0.9768797, 0.9829584333, 0.9865011667, 0.9887121667]
        DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.1071702, 0.06180936667, 0.04365763333, 0.03117206667]
        msssim = [0.9886938667, 0.9864564667, 0.9832413333, 0.9778337]
        DVCp, = plt.plot(bpp, msssim, "y-o", color="peru", linewidth=LineWidth, label='DVC++')

        bpp = [0.272495739, 0.116933239, 0.058484612, 0.035880386]
        psnr = [40.28897112, 38.7756175, 37.19223474, 35.48255472]
        msssim = [0.988984555, 0.986157917, 0.982596303, 0.977072748]
        h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.216879054, 0.092386216, 0.046302764, 0.027135239]
        psnr = [40.50252558, 39.03431677, 37.54350387, 35.90135885]
        msssim = [0.988956497, 0.986107604, 0.98287309, 0.977855586]
        h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')
    else:
        print('no such class : ', + hclass)
        exit(0)
        
    savepathmsssim = prefix + '/' + 'HEVCClass_' + hclass + '_msssim'# + '.eps'
    if hclass != 'E':
        plt.legend(handles=[h264, h265, DVC, DVCp, RY, Liu, LU, RaFC, FVC, DCVC], loc=4)
    else:
        plt.legend(handles=[h264, h265, DVC, DVCp, Liu, LU, RaFC, FVC, DCVC], loc=4)

    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('MS-SSIM')
    plt.title('HEVC Class ' + hclass + ' dataset')
    # plt.savefig(savepathmsssim + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(savepathmsssim + '.png')
    plt.clf()

    savepath = prefix + '/' + 'HEVCClass_' + hclass + '.png'
    img1 = cv2.imread(savepathpsnr + '.png')
    img2 = cv2.imread(savepathmsssim + '.png')

    image = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(savepath, image)

drawhevc('B')
drawhevc('C')
drawhevc('D')
drawhevc('E')