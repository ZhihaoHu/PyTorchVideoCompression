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

        
        bpp, psnr = [0.271843, 0.146412, 0.090925, 0.056782], [35.104963, 34.104735, 33.066283, 31.911116]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
        
        # bpp = [0.0661, 0.09362000000000001, 0.14695999999999998, 0.2232]
        # psnr = [32.721524294756705, 33.61284158051482, 34.75174138791608, 35.718610694312616]
        # msssim = [0.959144, 0.964186, 0.97111, 0.9765840000000001]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC') #  some problem : this paper use resize the test sequence, which loses information and leads to higher results.

        bpp = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
        psnr = [31.82619106, 33.0220837, 34.10751308, 35.10115456]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.569167464, 0.219282161, 0.110183846, 0.065360897]
        psnr = [35.79301018, 34.03927013, 32.59008599, 31.1074001]
        msssim = [0.977265628, 0.967603596, 0.957326897, 0.94278379]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.473727262, 0.185593416, 0.08897321, 0.048943254]
        psnr = [35.78820838, 34.14234791, 32.7354592, 31.24880809]
        msssim = [0.975606114, 0.966591679, 0.956803834, 0.942605916]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'C':
        bpp, psnr = [0.345057, 0.244922, 0.16963, 0.107338], [32.58818, 31.563223, 30.362042, 28.849785]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
        
        # bpp = [0.12157499999999999, 0.17965, 0.271025, 0.38745]
        # psnr = [28.63609903605147, 29.944387388321196, 31.54062511303266, 33.03500011196936]
        # msssim = [0.9578025, 0.9686675, 0.9776324999999999, 0.98348]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.133924025, 0.19340295, 0.272746925, 0.4122508326]
        psnr = [28.767909, 29.95094243, 31.25051302, 32.51022528]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.576088116, 0.328202857, 0.193107239, 0.116934184]
        psnr = [34.63515097, 32.32511581, 30.15303183, 28.10807277]
        msssim = [0.983930296, 0.975657708, 0.963020551, 0.943658353]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.525598708, 0.295866404, 0.169516387, 0.097295083]
        psnr = [34.62769455, 32.4512852, 30.35337589, 28.28509499]
        msssim = [0.983231578, 0.974819694, 0.962047889, 0.941757279]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'D':
        bpp, psnr = [0.37856, 0.274477, 0.192981, 0.121238], [32.642264, 31.520746, 30.215077, 28.655284]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        
        # bpp = [0.11227499999999999, 0.17395000000000002, 0.264575, 0.38207500000000005]
        # psnr = [28.82520179437781, 30.23478970812226, 31.91116994008828, 33.53631860458036]
        # msssim = [0.96777, 0.9763375, 0.9830949999999999, 0.98793]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')
        
        bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
        psnr = [28.41229473, 29.72853673, 31.1727661, 32.53451213]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.62866645, 0.369079861, 0.216769748, 0.130388726]
        psnr = [34.8324324, 32.26764653, 29.94494703, 27.82689033]
        msssim = [0.987974466, 0.980878777, 0.969883378, 0.952591233]
        h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.593745117, 0.347317166, 0.198283963, 0.113808322]
        psnr = [34.91076403, 32.50539009, 30.21889251, 28.07014606]
        msssim = [0.987591187, 0.98046689, 0.969045592, 0.951302461]
        h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'E':
        bpp, psnr = [0.102672, 0.059125, 0.041676, 0.028598], [39.933601, 38.773299, 37.673584, 36.296838]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        # bpp = [0.02223333333333333, 0.0281, 0.04416666666666667, 0.06459999999999999]
        # psnr = [36.119993400198794, 36.895272069830746, 37.95133358562974, 38.59336757723919]
        # msssim = [0.97991, 0.98177, 0.9831466666666667, 0.98547]
        # MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.03684806667, 0.04987243333, 0.07102866667, 0.1215183]
        psnr = [36.020002, 37.53390843, 38.70012267, 39.80463083]
        DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

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
    plt.legend(handles=[h264, h265, DVC, RaFC], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title('HEVC Class ' + hclass + ' dataset')
    # plt.savefig(savepathpsnr + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(savepathpsnr + '.png')
    plt.clf()

    # ----------------------------------------MSSSIM-------------------------------------------------
    if hclass == 'B':
        
        bpp, msssim = [0.286055, 0.171946, 0.104357, 0.062172], [0.978676, 0.9724, 0.96476, 0.953966]
        RaFC, = plt.plot(bpp, msssim, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        # bpp = [0.0661, 0.09362000000000001, 0.14695999999999998, 0.2232]
        # psnr = [32.721524294756705, 33.61284158051482, 34.75174138791608, 35.718610694312616]
        # msssim = [0.959144, 0.964186, 0.97111, 0.9765840000000001]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        msssim = [0.94666318, 0.95921518, 0.96812108, 0.97383252]
        bpp = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
        DVC, = plt.plot(bpp, msssim, "y-p", linewidth=LineWidth, label='DVC')

        bpp = [0.569167464, 0.219282161, 0.110183846, 0.065360897]
        psnr = [35.79301018, 34.03927013, 32.59008599, 31.1074001]
        msssim = [0.977265628, 0.967603596, 0.957326897, 0.94278379]
        h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.473727262, 0.185593416, 0.08897321, 0.048943254]
        psnr = [35.78820838, 34.14234791, 32.7354592, 31.24880809]
        msssim = [0.975606114, 0.966591679, 0.956803834, 0.942605916]
        h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')

    elif hclass == 'C':

        bpp, psnr = [0.358622, 0.243691, 0.163198, 0.094394], [0.984432, 0.978034, 0.968136, 0.950041]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

        # bpp = [0.12157499999999999, 0.17965, 0.271025, 0.38745]
        # psnr = [28.63609903605147, 29.944387388321196, 31.54062511303266, 33.03500011196936]
        # msssim = [0.9578025, 0.9686675, 0.9776324999999999, 0.98348]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')
        

        bpp = [0.133924025, 0.19340295, 0.272746925, 0.4122508326]
        msssim = [0.9523096, 0.96620995, 0.9763428823, 0.981115925]
        DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.576088116, 0.328202857, 0.193107239, 0.116934184]
        psnr = [34.63515097, 32.32511581, 30.15303183, 28.10807277]
        msssim = [0.983930296, 0.975657708, 0.963020551, 0.943658353]
        h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.525598708, 0.295866404, 0.169516387, 0.097295083]
        psnr = [34.62769455, 32.4512852, 30.35337589, 28.28509499]
        msssim = [0.983231578, 0.974819694, 0.962047889, 0.941757279]
        h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'D':
        
        bpp, psnr = [0.365465, 0.255783, 0.176039, 0.102085], [0.987527, 0.981579, 0.972727, 0.958684]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
        
        # bpp = [0.11227499999999999, 0.17395000000000002, 0.264575, 0.38207500000000005]
        # psnr = [28.82520179437781, 30.23478970812226, 31.91116994008828, 33.53631860458036]
        # msssim = [0.96777, 0.9763375, 0.9830949999999999, 0.98793]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
        msssim = [0.95808245, 0.971744025, 0.981360625, 0.9863982]
        DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

        bpp = [0.62866645, 0.369079861, 0.216769748, 0.130388726]
        psnr = [34.8324324, 32.26764653, 29.94494703, 27.82689033]
        msssim = [0.987974466, 0.980878777, 0.969883378, 0.952591233]
        h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

        bpp = [0.593745117, 0.347317166, 0.198283963, 0.113808322]
        psnr = [34.91076403, 32.50539009, 30.21889251, 28.07014606]
        msssim = [0.987591187, 0.98046689, 0.969045592, 0.951302461]
        h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')
    elif hclass == 'E':
        
        bpp, psnr = [0.138196, 0.078092, 0.04724, 0.029643], [0.990446, 0.988069, 0.985217, 0.980766]
        RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
        
        # bpp = [0.02223333333333333, 0.0281, 0.04416666666666667, 0.06459999999999999]
        # psnr = [36.119993400198794, 36.895272069830746, 37.95133358562974, 38.59336757723919]
        # msssim = [0.97991, 0.98177, 0.9831466666666667, 0.98547]
        # MLVC, = plt.plot(bpp, msssim, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')

        bpp = [0.03684806667, 0.04987243333, 0.07102866667, 0.1215183]
        msssim = [0.9768797, 0.9829584333, 0.9865011667, 0.9887121667]
        DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

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
    plt.legend(handles=[h264, h265, DVC, RaFC], loc=4)
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