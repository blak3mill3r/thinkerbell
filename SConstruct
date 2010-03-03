import os, glob

SetOption( 'num_jobs', 4 ) # Set this to the number of processors you have.  TODO: Automate this.

libthinkerbell_sources           = glob.glob( 'src/*.cc' ) + glob.glob( 'src/**/*.cc' )
libthinkerbell_cuda_sources      = glob.glob( 'src/*.cu' ) + glob.glob( 'src/**/*.cu' )
libthinkerbell_headers           = glob.glob( 'thinkerbell/*.h' )  + glob.glob( 'thinkerbell/**/*.h' )

env = Environment(tools=('default', 'cuda'))

env['ENV'] = {'PATH':os.environ['PATH'], 'TERM':os.environ['TERM'], 'HOME':os.environ['HOME']} # Environment variables required by colorgcc.
env['LIBPATH'] = [ './', '/usr/local/lib', '/usr/local/cuda/lib/' ]
env['CCFLAGS'] = [ '-Wno-deprecated' ] #'-Wall', '-W', '-Wshadow', '-Wpointer-arith', '-Wcast-qual', '-Wwrite-strings', '-Wconversion', '-Winline', '-Wredundant-decls', '-Wno-unused', '-Wno-deprecated' ]
env['CPPPATH'] = [ './src', './', '/home/blake/w/cudamm/', '/usr/include/cuda/' ]
env['LIBS'] = [ 'cudamm', 'cuda', 'boost_serialization-mt' ]

#if ARGUMENTS.get('debug', 0):
env['CCFLAGS'] += ['-g' ]
#else:
#    env['CCFLAGS'] += ['-O3' ]

env.SharedLibrary( source = libthinkerbell_sources, target = 'lib/thinkerbell' )
env.Cubin( 'cudasrc/rbm_kernels', NVCCPATH = env['CPPPATH'] )
env.Cubin( 'cudasrc/mersenne_twister_kernels', NVCCPATH = env['CPPPATH'] )
env.Command( 'tags', libthinkerbell_sources + libthinkerbell_headers, 'ctags -o $TARGET $SOURCES' )

### Testing ###
Export('env')

testEnv = env.Clone()
testEnv.Tool('unittest',
         toolpath=['tools'],
         UTEST_MAIN_SRC=File('tools/boost_auto_test_main.cc'),
         LIBS=['boost_unit_test_framework-mt']
 )

Export('testEnv')

# grab stuff from sub-directories.
env.SConscript(dirs = ['test'])

