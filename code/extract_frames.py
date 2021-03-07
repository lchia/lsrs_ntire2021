import os
from natsort import natsort
import glob      

def get_video_paths(dir_path, wildcard='*.mp4'):
    return natsort.natsorted(glob.glob(dir_path + '/' + wildcard))

def main():
    root = './'
    root_frames = os.path.join(root, 'SDR_4K_frames')
    if not os.path.exists(root_frames):
        os.makedirs(root_frames)

    SDR_dir_format = 'SDR_4K(Part%d)'
    for k in range(4):
        sdr_dir = SDR_dir_format%(k)
        print('sdr_dir: ', sdr_dir)

        files = get_video_paths(sdr_dir) 
        idx = 0
        for vf in files:
            idx += 1
            print('@SDR_4K(%d) >> %03d-%03d: %s'%(k, idx, len(files), vf))
            _, name = os.path.split(vf)
            name, _ = os.path.splitext(name)
            ffmpeg_cmd = 'ffmpeg -i %s -vf fps=20/8 %s/%s_'%(vf, root_frames, name)
            ffmpeg_cmd += '%03d.png'
            ffmpeg_cmd = ffmpeg_cmd.replace(')', '\)').replace('(', '\(')
            print('\t', ffmpeg_cmd)
            os.system(ffmpeg_cmd)
            


if __name__ == '__main__':
    main()