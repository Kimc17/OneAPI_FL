
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

#include <iostream>
#include <math.h> 
#include <algorithm>
#include <vector>
#include <ctime> 
#include <limits>

#include <cstdlib>

#define N 4
#define LEFT  0
#define DOWN  1
#define RIGHT  2
#define UP  3

extern SYCL_EXTERNAL int rand(void);

constexpr int M = 16;
constexpr int O =  4;

using namespace cl::sycl;

// declare the kernel name as a global to reduce name mangling
class Kernel;


int to_s(int row, int col)
{
    return row * N + col;
}
class FrozenLake
{
public:
    struct Result
    {
        double p;
        int new_state;
        double reward;
        bool done;
    };
    //std::vector<Result>(*P)[4] = new std::vector<Result>[16][4];
    std::vector<Result> P[16][4];

private:
    char m_board[16] = {
        'S', 'F', 'F', 'F',
        'F', 'H', 'F', 'H',
        'F', 'F', 'F', 'H',
        'H', 'F', 'F', 'G'};
    int actualState;
    //std::vector<Result> P[N * N][4];
public:
    FrozenLake() // FrozenLakeEnv
    {
        actualState = 0;
        calculate_p();
    }

public:

    SYCL_EXTERNAL void reset() 
    {
        actualState = 0;
    }

    int getActualState() 
    {
        return actualState;
    }

    void setActualState(int state) 
    {
        actualState = state;
    }


private:
    SYCL_EXTERNAL int inc(int row, int col, int action)
    {
        if (action == LEFT)
        {
            col = max(col - 1, 0);
        }
        else if (action == DOWN)
        {
            row = min(row + 1, 3);
        }
        else if (action == RIGHT)
        {
            col = min(col + 1, 3);
        }
        if (action == UP)
        {
            row = max(row - 1, 0);
        }
        return to_s(row, col);
    }
    SYCL_EXTERNAL Result update_probability_matrix(int row, int col, int action)
    {
        Result result;
        result.new_state = inc(row, col, action);
        result.reward = 0.0;
        result.done = false;

        if (m_board[result.new_state] == 'H')
        {
            result.done = true;
        }
        else if (m_board[result.new_state] == 'G')
        {
            result.reward = 1.0;
            result.done = true;
        }
        return result;
    }

    void calculate_p()
    {
        for (int row = 0; row < N; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                int s = to_s(row, col);
                for (int a = 0; a < 4; ++a)
                {
                    char letter = m_board[s];
                    Result r;
                    int b;

                    if (letter == 'G' | letter == 'H')
                    {
                        r = {1.0, s, 0.0, true};
                        P[s][a].push_back(r);
                    }
                    else
                    {
                        b = (((a - 1) % 4) + 4) % 4;
                        r = update_probability_matrix(row, col, b);
                        r.p = 1.0 / 3.0;
                        P[s][a].push_back(r);

                        r = update_probability_matrix(row, col, a);
                        r.p = 1.0 / 3.0;
                        P[s][a].push_back(r);

                        b = (((a + 1) % 4) + 4) % 4;
                        r = update_probability_matrix(row, col, b);
                        r.p = 1.0 / 3.0;
                        P[s][a].push_back(r);
                    }
                }
            }
        }            
    }
};

class QLearner
{
public:
    QLearner()
        : q_table{
              0,
          }
    {
    }

//private:
     double q_table[N * N][4];
     //double( *q_table)[4] = new double[16][4];
public:
    void learn(FrozenLake* env,int totalEpisodes, queue &q)
    {
        
        srand(time(0));
        
        // Intialize c_back
       for (int s = 0; s < 16; s++){
           for (int a = 0; a < 4; a++){
               q_table[s][a] = 0.0;       
           }      
       }
           
        buffer a_buf(reinterpret_cast<double *>(q_table), range(4, 16));
        buffer b_buf(reinterpret_cast<std::vector<FrozenLake::Result> *>(env -> P), range(4, 16));

        // Submit command group to queue 
        q.submit([&](handler &h) {
        
        // Read from a and c, write to b
        accessor a(a_buf, h, read_only);
        accessor b(a_buf, h, write_only, no_init);
        accessor c(b_buf, h, read_only);

            
        double learning_rate = 0.1;
        double discount_rate = 0.99;
        double min_exploration_rate = 0.01;
        double max_exploration_rate = 1;
        double exploration_rate_decay = 0.001;
        
        // Execute kernel.
        //h.single_task<Kernel>([=]() [[intel::kernel_args_restrict]]{
        h.parallel_for(totalEpisodes, [=](auto episode) {
           //for (int episode = 1; episode <= totalEpisodes; ++episode) {
           int state = 0;
           //int rewards_current_episode = 0;
           double exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * exp(-exploration_rate_decay * episode);
           for (int step = 1; step <= 100; ++step)
           {
                double exploration_rate_threshold = (double) rand() / RAND_MAX;
                int actionIndex = 0;
                std::vector<FrozenLake::Result> prob_n =   c[state][actionIndex];
               // Send the action chosen to the environment and get result with reward and state                

                double random_n = (double) rand() / RAND_MAX;  
                double csprob_n = 0;
                //Calculate cumsum       
               int max_i = 0;

               for (int i = 0; i < prob_n.size(); ++i)
                {
                    csprob_n += prob_n.data()[i].p;
                    if (csprob_n > random_n)
                    {
                        max_i = i;
                    }
                    else 
                    {
                        max_i = 0;
                    }
                }

               FrozenLake::Result result = prob_n.data()[max_i];
               if(state == 15){
                   state = 0;
               }else {
                   state += 1;
               }
               
               //state = prob_n.data()[ej].new_state; //AYUDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA!!!!!!!!!!!
 
               int max =  a[state][0];
               for(int i = 0; i < 4; ++i) {
                  if(max < a[state][i])
                    max = a[state][i];
               }

               //Update Q-table for Q(s,a)  using the Q-Learning formula
               b[state][actionIndex] = a[state][actionIndex] * (1 - learning_rate) + learning_rate * (result.reward + discount_rate * max);
               
               // Move to the next state and add the reward gotten);
               //state = result.new_state;
               //rewards_current_episode += result.reward;
               // Break if fail or goal were reached
               if (result.done)
                  break;

              } 

        });

      });
    }
        
        
    void print()
    {
        for (int row = 0; row < N; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
               int s = to_s(row, col);
                 for (int a = 0; a < N; ++a)
                {
                     std::cout << q_table[s][a] << "                |          ";

                }

                std::cout << std::endl;
            }
        }
    }
};


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
        
         //FrozenLake env;
         //QLearner agent;
        


         //t0=clock();
         agent -> learn(env, 1000000, q);
         //t1 = clock();
         //time = (double(t1-t0)/CLOCKS_PER_SEC);
         
         //delete [] (env -> P);

         /*t0=clock();
         agent -> play(env, 10000);
         t1 = clock();
         time = (double(t1-t0)/CLOCKS_PER_SEC);
         std::cout << "Playing time: " << time << " for 10000 cases" << std::endl;*/
        
    }
    catch (exception const &e) {
    
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
         std::cout << "Learning time: " << time << std::endl;

         agent -> print();


    
  

  return (EXIT_SUCCESS);
}
