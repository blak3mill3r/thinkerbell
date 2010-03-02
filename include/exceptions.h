#ifndef __THINKERBELL_EXCEPTIONS_H__
#define __THINKERBELL_EXCEPTIONS_H__

#include <exception>

namespace thinkerbell {

// exceptions:
class c_exception : public std::exception
{
  virtual const char* what() const throw()
  { return "Thinkerbell exception"; }
};

class c_memory_exception : public c_exception
{
  virtual const char* what() const throw()
  { return "Thinkerbell memory exception"; }
};

extern c_exception exception;
extern c_memory_exception memory_exception;

}

#endif


