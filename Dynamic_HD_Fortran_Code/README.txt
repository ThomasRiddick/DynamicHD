Important Note
This code can ONLY be compiled with versions of gfortran that include the bug fix in 
version 5.x for a bug in the compilation of polymorphic variables. 
The bug can be recognised by error of the format 'undefined reference to __vtab_REAL_4' 
Gfortran from gcc-6.2.0 is known to work
There also appears to be problems with gcc-7.2.0 with overloaded constructors