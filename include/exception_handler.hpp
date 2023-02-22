#ifndef __EXCEPTIONHANDLER_HPP__
#define __EXCEPTIONHANDLER_HPP__
#include <CL/sycl.hpp>
#include <exception>
#include <iostream>

namespace fpga_tools {

void exception_handler(sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
}

}  // namespace fpga_tools

#endif  //__EXCEPTIONHANDLER_HPP__