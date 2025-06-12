set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.8.93")
set(CMAKE_CUDA_DEVICE_LINKER "/usr/local/cuda/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/usr/local/cuda/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "12.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.8.93")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "90-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/lib/stubs;/usr/local/cuda/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/mvapich2-2.3.7-1-qv3gjagtbx5e3rlbdy6iy2sfczryftyt/include;/opt/slurm/current/include;/opt/pmix/4.2.6/include;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/libpciaccess-0.17-pjfe4ct4gfm5k26s36hmewhbz4k232dl/include;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/zlib-1.2.13-3qhjpij2pji47kfanmlflvwk5ljcn5lh/include;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/mpc-1.3.1-3zvixith5i243hwiill4exyu3lf2q6ad/include;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/mpfr-4.2.0-tox4bdsc25zr763rnq5gzlukfbvw7uj7/include;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/gmp-6.2.1-n7dzsse5e3f6w5z6q6cuqursydg6yypo/include;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/include/c++/12.3.0;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/include/c++/12.3.0/x86_64-pc-linux-gnu;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/include/c++/12.3.0/backward;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include-fixed;/usr/local/include;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda/targets/x86_64-linux/lib/stubs;/usr/local/cuda/targets/x86_64-linux/lib;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib/gcc/x86_64-pc-linux-gnu/12.3.0;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib/gcc;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib64;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib64;/lib64;/usr/lib64;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/mvapich2-2.3.7-1-qv3gjagtbx5e3rlbdy6iy2sfczryftyt/lib;/opt/slurm/current/lib;/opt/pmix/4.2.6/lib;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/libpciaccess-0.17-pjfe4ct4gfm5k26s36hmewhbz4k232dl/lib;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/zlib-1.2.13-3qhjpij2pji47kfanmlflvwk5ljcn5lh/lib;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/mpc-1.3.1-3zvixith5i243hwiill4exyu3lf2q6ad/lib;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/mpfr-4.2.0-tox4bdsc25zr763rnq5gzlukfbvw7uj7/lib;/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/gmp-6.2.1-n7dzsse5e3f6w5z6q6cuqursydg6yypo/lib;/storage/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
