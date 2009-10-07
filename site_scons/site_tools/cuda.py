"""
SCons.Tool.cudacubin

CUDA Tool for SCons
generates .cubin files
"""

import os
import SCons.Tool
import SCons.Defaults

def generate_actions(source, target, env, for_signature):
    return 'nvcc -o %s -cubin %s %s' % (target[0], env['NVCCFLAGS'], source[0])

def generate(env):
        # default flags for the NVCC compiler
        env['NVCCFLAGS'] = ''

        # helpers
        home=os.environ.get('HOME', '')
        programfiles=os.environ.get('PROGRAMFILES', '')
        homedrive=os.environ.get('HOMEDRIVE', '')

        # find CUDA Toolkit path and set CUDA_TOOLKIT_PATH
        try:
                cudaToolkitPath = env['CUDA_TOOLKIT_PATH']
        except:
                paths=[home + '/NVIDIA_CUDA_TOOLKIT',
                       home + '/Apps/NVIDIA_CUDA_TOOLKIT',
                           home + '/Apps/NVIDIA_CUDA_TOOLKIT',
                           home + '/Apps/CudaToolkit',
                           home + '/Apps/CudaTK',
                           '/usr/local/NVIDIA_CUDA_TOOLKIT',
                           '/usr/local/CUDA_TOOLKIT',
                           '/usr/local/cuda_toolkit',
                           '/usr/local/CUDA',
                           '/usr/local/cuda',
                           '/Developer/NVIDIA CUDA TOOLKIT',
                           '/Developer/CUDA TOOLKIT',
                           '/Developer/CUDA',
                           programfiles + 'NVIDIA Corporation/NVIDIA CUDA TOOLKIT',
                           programfiles + 'NVIDIA Corporation/NVIDIA CUDA',
                           programfiles + 'NVIDIA Corporation/CUDA TOOLKIT',
                           programfiles + 'NVIDIA Corporation/CUDA',
                           programfiles + 'NVIDIA/NVIDIA CUDA TOOLKIT',
                           programfiles + 'NVIDIA/NVIDIA CUDA',
                           programfiles + 'NVIDIA/CUDA TOOLKIT',
                           programfiles + 'NVIDIA/CUDA',
                           programfiles + 'CUDA TOOLKIT',
                           programfiles + 'CUDA',
                           homedrive + '/CUDA TOOLKIT',
                           homedrive + '/CUDA']
                for path in paths:
                        if os.path.isdir(path):
                                print 'scons: CUDA Toolkit found in ' + path
                                cudaToolkitPath = path
                                break
                if cudaToolkitPath == None:
                        sys.exit("Cannot find the CUDA Toolkit path. Please modify your SConscript or add the path in cudaenv.py")
        env['CUDA_TOOLKIT_PATH'] = cudaToolkitPath

        # add nvcc to PATH
        env.PrependENVPath('PATH', cudaToolkitPath + '/bin')

        # add required libraries
        env.Append(CPPPATH=[cudaToolkitPath + '/include'])
        env.Append(LIBPATH=[cudaToolkitPath + '/lib'])
        env.Append(LIBS=['cudart'])

        bldr = env.Builder(generator = generate_actions,
                       suffix = '.cubin',
                       src_suffix = '.cu')
        
        env.Append(BUILDERS = {'Cubin' : bldr})

def exists(env):
        return env.Detect('nvcc')

