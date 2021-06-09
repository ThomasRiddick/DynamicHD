#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

string home = getenv("HOME");
#if DARWIN
string datadir =  home + "/Documents/data";
#else
string datadir = home;
#endif

#endif
