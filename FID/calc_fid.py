import os 
import pprint
import shlex
import subprocess

def fid_one_list(filePath, fileNAME, fid_dict={}):
    fid = 'python3 -m pytorch_fid'

    PATH_with_FILE = os.path.join(filePath, fileNAME)

    file_list = os.listdir(PATH_with_FILE)
    file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(PATH_with_FILE,fn)) if not os.path.isdir(os.path.join(PATH_with_FILE,fn)) else 0)
    max_value = int(file_list[-1]) + 1

    # start FID calc 
    for i in range(0, max_value, 100):

        real_path = ' ' + os.path.join(PATH_with_FILE, str(i)) + '/real_images'
        fake_path = ' ' + os.path.join(PATH_with_FILE, str(i)) + '/fake_images'

        command_line = fid + real_path + fake_path

        # split command line 
        args = shlex.split(command_line)

        res = subprocess.run(args, shell=False, stdout=subprocess.PIPE, text=True)

        fid_dict[i] = float(res.stdout[6:-1])

    with open('FID/' + fileNAME + '.log', 'w') as tf:

        print(PATH_with_FILE + '\n', file=tf)

        pprint.pprint(sorted(fid_dict.items(), key=lambda kv:kv[1]), stream=tf)

def fid_all_list(filePath, fid_dict={}):
    fid = 'python3 -m pytorch_fid'

    for i in os.listdir(filePath):
        PATH_with_FILE = os.path.join(filePath, i)
        FILE_NAME = i

        print('now file path:\t' + str(PATH_with_FILE) + '\n')

        file_list = os.listdir(PATH_with_FILE)
        file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(PATH_with_FILE,fn)) if not os.path.isdir(os.path.join(PATH_with_FILE,fn)) else 0)
        max_value = int(file_list[-1]) + 1

        # start FID calc
        for i in range(0, max_value, 100):

            real_path = ' ' + os.path.join(PATH_with_FILE, str(i)) + '/real_images'
            fake_path = ' ' + os.path.join(PATH_with_FILE, str(i)) + '/fake_images'

            command_line = fid + real_path + fake_path

            args = shlex.split(command_line)

            res = subprocess.run(args, shell=False, stdout=subprocess.PIPE, text=True)

            fid_dict[i] = float(res.stdout[6:-1])

        # write to the log
        with open('FID/' + FILE_NAME +'.log', "w") as tf:

            print(PATH_with_FILE + '\n', file=tf)

            pprint.pprint(sorted(fid_dict.items(), key=lambda kv:kv[1]), stream=tf)
            