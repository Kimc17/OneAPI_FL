
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime> 
#include "QLearner.hpp"

using namespace cl::sycl;

int main() {

        FrozenLake* env = new FrozenLake();
        QLearner* agent = new QLearner();

        unsigned t0, t1;
        double time;

        // create device selector for the device of your interest
        // FPGA_EMULATOR defined in makefile-fpga/Makefile
        #if defined(FPGA_EMULATOR)
        // DPC++ extension: FPGA emulator selector on systems without FPGA card
        ext::intel::fpga_emulator_selector device_selector;
        #else
        // DPC++ extension: FPGA selector on systems with FPGA card
        ext::intel::fpga_selector device_selector;
        #endif

    
    try {
        queue q(device_selector, dpc_common::exception_handler);
        
        std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
        

        t0=clock();
        agent -> learn(env, 1000000, q);
        t1 = clock();
        time = (double(t1-t0)/CLOCKS_PER_SEC);
        std::cout << "Learning time: " << time << std::endl;

        agent -> print();

        /*t0=clock();
        agent -> play(env, 10000);
        t1 = clock();
        time = (double(t1-t0)/CLOCKS_PER_SEC);
        std::cout << "Playing time: " << time << " for 10000 cases" << std::endl;*/
        
    }
    catch (sycl::exception const &e) {
    
        // Catches exceptions in the host code
        std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

        // Most likely the runtime couldn't find FPGA hardware!
        if (e.code().value() == CL_DEVICE_NOT_FOUND) {
          std::cerr << "If you are targeting an FPGA, please ensure that your "
                       "system has a correctly configured FPGA board.\n";
          std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
          std::cerr << "If you are targeting the FPGA emulator, compile with "
                       "-DFPGA_EMULATOR.\n";
        }
        std::terminate();
      }

         //delete [] (agent -> q_table);
         //std::std::cout << "Learning time: " << time << std::std::endl;

        // agent -> print();

  return (EXIT_SUCCESS);
}
