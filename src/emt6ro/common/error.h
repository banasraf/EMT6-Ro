#ifndef EMT6RO_COMMON_ERROR_H_
#define EMT6RO_COMMON_ERROR_H_

#include <iostream>
#include <string>
#include <sstream>

namespace emt6ro {

template <typename ...Args>
std::string make_string(const Args & ...args);

template <typename Arg, typename ...Args>
std::string make_string(const Arg &arg, const Args & ... args) {
  std::ostringstream ss;
  ss << arg << make_string(args...);
  return ss.str();
}

template <>
inline std::string make_string() {
  return "";
}

#define ENFORCE(COND, ...) \
do {if (!COND) throw std::runtime_error(std::string("Check failed: " #COND "; ") + make_string(__VA_ARGS__));} while(0);
  
} // namespace emt6ro


#endif  // EMT6RO_COMMON_ERROR_H_