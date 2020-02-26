#ifndef UTIL_H
#define UTIL_H

#include <boost/math/tools/roots.hpp>
#include <unordered_map>


double get_upperlimit(double eps, double R){
  double guess = 1;
  double min = 0;
  double max = 1e6;
  int digits = 16;
  double result = boost::math::tools::newton_raphson_iterate([eps](const double x){ return std::make_tuple(x * exp(x) - 1.0/eps,  exp(x) + x*exp(x));}, guess, min, max, digits);
  
  double ulimit = R * result;
  return ulimit;
}

#endif // end of UTIL_H
