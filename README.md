# GDWE
Source code and datasets for the paper "Graph-based Dynamic Word Embeddings" accepted by IJCAI 2022

## Installation

> **Environment:**
>
> gcc 4.4.7 or higher is required.
>
> automake 1.1.16

```shell
autoscan # generate `autoscan.log` and `configure.scan`
mv configure.scan configure.ac
```

change the content of `configure.ac` as follows:

```
AC_PREREQ([2.69])
AC_INIT([FULL-PACKAGE-NAME], [VERSION], [BUG-REPORT-ADDRESS])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_RANLIB

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h sys/time.h])

AC_CHECK_LIB([thread/pthread/pthread_create])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AM_INIT_AUTOMAKE
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_STRTOD
AC_CHECK_FUNCS([floor gettimeofday pow sqrt strtol])

CXXFLAGS="-std=c++0x -O3 -funroll-loops -pthread"

AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT
```

```shell
aclocal # generate `aclocal.m4`
autoconf # generate `configure`
autoheader # generate `config.h` and `config.h.in`
automake --add-missing # generate `Makefile.in` etc.
bash ./configure
make
```



## Basic Usage

### Data Preparation

`corpus/` provides dataset NYT for experiments in the GDWE paper.



### Training

The following command trains the GDWE model on the NYT corpus:

```   sh
% bash scripts/train_NYT_graph.sh
```

For more model settings, you can use `-h` to show the arguments

```
% ./src/yskip -h
```



### Cross-time Alignment Evaluation

The following command uses pre-trained word embeddings to proceed space alignment evaluation test:

```   sh
% cd eval
% python eval_align_online2_test.py --embdir [emb_path] --result [result_file_name] 
```

Example:

``` sh
% cd eval
% python eval_align_online2_test.py --embdir ../model/e0.2_c5_N4_i1_b1000_E0.2_J0.05_A1.25/ --result gdwe-nyt-test
```



### Text Stream Classification

The following command uses pre-trained word embeddings to proceed text stream classification:

``` 
% bash scripts/run_classification.sh
```



