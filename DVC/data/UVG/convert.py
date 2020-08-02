import os

num = 7
video_name = ['Beauty_1920x1024_120fps_420_8bit_YUV.yuv', 'HoneyBee_1920x1024_120fps_420_8bit_YUV.yuv', 'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV.yuv',  'YachtRide_1920x1024_120fps_420_8bit_YUV.yuv', 'Bosphorus_1920x1024_120fps_420_8bit_YUV.yuv',  'Jockey_1920x1024_120fps_420_8bit_YUV.yuv', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV.yuv']
short = ['Beauty', 'HoneyBee', 'ReadySteadyGo', 'YachtRide', 'Bosphorus', 'Jockey', 'ShakeNDry']

for i in range(num):
    saveroot = 'images/' + short[i]
    savepath = 'images/' + short[i] + '/im%03d.png'
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    print('ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i ' + 'videos_crop/' + video_name[i] +  ' ' + savepath)
    os.system('ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i ' + 'videos_crop/' + video_name[i] +  ' ' + savepath)
