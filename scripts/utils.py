import os
from glob import glob
from time import sleep


def use_gpu(func):
    def dummy():
        # Find out how many gpus are available
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
        try:
            cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            print('CUDA_VISIBLE_DEVICES is not set!')
            return 
        
        gpu_ids = [int(i.strip()) for i in  cuda_visible_devices.split(',')]
        occupied_gpus = [int(i.split('-')[-1]) for i in glob('.*_gpu-*')]
        free_gpus = [i for i in gpu_ids if i not in occupied_gpus]

        print(f'Installed GPUs: {gpu_ids}')
        print(f'Free GPUs: {free_gpus}')

        if len(free_gpus) > 0:
            booked_gpu = free_gpus[0]
        else:
            print('All GPUs are occupied! -> Return!')
            return

        print(f'--> Book GPU with ID {booked_gpu}')

        with open(f'.{os.getpid()}_gpu-{booked_gpu}', 'w'):
            pass
        
        sleep(2)
        # Control. If two processes want to use the same GPU the process with
        # the smaller pid wins
        occupied_gpus = [int(i.split('-')[-1]) for i in glob('.*_gpu-*')]
        if occupied_gpus.count(booked_gpu) > 1:
            print(f'--> Too many processes for GPU {booked_gpu}')
            if os.getpid() != min([int(i[1:].split('_')[0]) for i in glob('.*_gpu-*')]):
                os.remove(f'.{os.getpid()}_gpu-{booked_gpu}')
                booked_gpu = abs(booked_gpu*2 - 1)
                print(f'PID {os.getpid()} switches to GPU {booked_gpu}')
                with open(f'.{os.getpid()}_gpu-{booked_gpu}', 'w'):
                    pass
            else:
                print(f'PID {os.getpid()} stays on GPU {booked_gpu}')

        os.environ['CUDA_VISIBLE_DEVICES'] = str(booked_gpu)

        try:
            result = func()
        except:
            print(f'Error', end='') 
            raise
        finally:
            print(f'--> Released GPU {booked_gpu}')
            os.remove(f'.{os.getpid()}_gpu-{booked_gpu}')
        return result
    return dummy

if __name__ == '__main__':
    @use_gpu
    def test():
        print('this is a test')

    
