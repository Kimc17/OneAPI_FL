
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime> 
#include "FrozenLake.hpp"



class QLearner
{
public:
    QLearner()
        : q_table{
              0,
          }
    {
    }

private:
    double q_table[N * N][4];

public:
    void learn( FrozenLake *env, int totalEpisodes, queue &q)
    {        
        srand(time(0));
        
       // Intialize q_table
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
            h.parallel_for(totalEpisodes, [=](auto episode) {
                
            int state = 0;
            int rewards_current_episode = 0;
            // Exploration rate decay
            double exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * exp(-exploration_rate_decay * episode);

            for (int step = 1; step <= 100; ++step)
            {

                double exploration_rate_threshold = (double)rand() / RAND_MAX;
                int actionIndex = 0;
                if (exploration_rate < exploration_rate_threshold)
                {
                    // Take a probably smart action
                                        // Take a probably smart action
                    for(int i = 0; i < 4 ; ++i)
                    {
                        if(a[state][actionIndex] < a[state][i])
                        {
                           actionIndex = i;
                        }
                    }
                }
                else
                {
                    // Take a random action to explore
                    actionIndex = rand() % 4;
                }

                // Send the action chosen to the environment and get result with reward and state
                // Try to move with that action
                double csprob_n = 0;
                int ind = 0;
                double random_n = (double) rand() / RAND_MAX;
                std::vector<FrozenLake::Result> prob_n = c[state][actionIndex];

                //Calculate cumsum
                for (int i = 0; i < prob_n.size(); i++)
                {   
                    csprob_n += prob_n.data()[i].p;
                    if (csprob_n > random_n)
                    {
                        ind = i;
                        break;
                    }

                }
                FrozenLake::Result result =  c[state][actionIndex].data()[ind];

                double max = a[result.new_state][0];
                for (int i = 0; i < 4; ++ i)
                {
                    if(max < a[result.new_state][i])
                    {
                        max = a[result.new_state][i];
                    }
                }

                //  Update Q-table for Q(s,a) using the Q-Learning formula
                b[state][actionIndex] = a[state][actionIndex] * (1 - learning_rate) +  learning_rate * (result.reward + discount_rate * max);
                
                // Move to the next state and add the reward gotten
                //const int state =  c[state][actionIndex].data()[ind].new_state;
                
                if(state == 15)
                {
                    state = 0;
                }
                else 
                {
                    //const int state = result.new_state;
                    state += 1;
                }
     
                // Break if fail or goal were reached
                if (result.done)
                    break;
            }
            });
        });
    }

    /*void play(FrozenLake *env, int episodes) const
    {
        srand(time(0));
        float misses = 0;
        for (int i = 0; i < episodes; ++i)
        {
            int steps = 0;
            int  agentState = 0;
            while (true)
            {
                // Select an action index
                int actionIndex = 0;
                for(int i = 0; i < 4 ; ++i)
                {
                    if(a[agentState][actionIndex] < a[agentState][i])
                    {
                       actionIndex = i;
                    }
                }
                
                // Try to move with that action
                double csprob_n = 0;
                int ind = 0;
                double random_n = (double) rand() / RAND_MAX;
                std::vector<FrozenLake::Result> prob_n = c[agentState][actionIndex];

                //Calculate cumsum
                for (int i = 0; i < prob_n.size(); i++)
                {   
                    csprob_n += prob_n[i].p;
                    if (csprob_n > random_n)
                    {
                        ind = i;
                        break;
                    }
   
                }
                FrozenLake::Result result = c[agentState][actionIndex][ind];

                agentState = result.new_state;

                // Verify if the game is done
                if (result.done)
                {
                    if (result.reward == 1)
                    {
                        //std::cout << "You have get the goal after: " << steps << std::endl;
                        break;
                    }
                    else
                    {
                        //std::cout << "You fell in a hole!" << std::endl;
                        misses += 1;
                        break;
                    }
                }
                steps++;
            }
        }
        double overage = (float)(misses / episodes) * 100;
        std::cout << "You fell in the hole: " << overage<< " % of the times" << std::endl;
    }*/

    void print()
    {
        for (int row = 0; row < N; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                int s = to_s(row, col);
                for (int a = 0; a < N; ++a)
                {
                    if (q_table[s][a] == 0)
                    {
                        std::cout << q_table[s][a] << "                |          ";
                    }
                    else
                    {
                        std::cout << q_table[s][a] << "         |         ";
                    }
                }

                std::cout << std::endl;
            }
        }
    }
};
