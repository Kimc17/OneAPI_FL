
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime> 
#include "FrozenLake.hpp"

using namespace std;

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
    void learn(FrozenLake *env, int totalEpisodes)
    {
        double learning_rate = 0.1;
        double discount_rate = 0.99;
        double min_exploration_rate = 0.01;
        double max_exploration_rate = 1;
        double exploration_rate_decay = 0.001;
        srand(time(0));

        int rewards_all_episodes[totalEpisodes];

        for (int episode = 1; episode <= totalEpisodes; ++episode)
        {
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
                    for(int i = 0; i < 4 ; ++i)
                    {
                        if(q_table[state][actionIndex] < q_table[state][i])
                        {
                           actionIndex = i;
                        }
                    }
                    //actionIndex = std::distance(q_table[state], std::max_element(q_table[state], q_table[state] + 4));
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
                std::vector<FrozenLake::Result> prob_n = env -> P[state][actionIndex];

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
                FrozenLake::Result result = env -> P[state][actionIndex][ind];

                //  Update Q-table for Q(s,a) using the Q-Learning formula
                q_table[state][actionIndex] = q_table[state][actionIndex] * (1 - learning_rate) +
                                              learning_rate * (result.reward + discount_rate *
                                                                                   *(std::max_element(q_table[result.new_state], q_table[result.new_state] + 4)));
                // Move to the next state and add the reward gotten
                state = result.new_state;
                
                rewards_current_episode += result.reward;
                // Break if fail or goal were reached
                if (result.done)
                    break;
            }


            // Save total reward from episode
            rewards_all_episodes[episode] = rewards_current_episode;
        }
        int count = 0;
        int turn = 1;
        int sum = 0;
        double average = 0;
    }

    void play(FrozenLake *env, int episodes) const
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
                                    // Take a probably smart action
                    int actionIndex = 0;
                    for(int i = 0; i < 4 ; ++i)
                    {
                        if(q_table[agentState][actionIndex] < q_table[agentState][i])
                        {
                           actionIndex = i;
                        }
                    }
                
                // Try to move with that action
                double csprob_n = 0;
                int ind = 0;
                double random_n = (double) rand() / RAND_MAX;
                std::vector<FrozenLake::Result> prob_n = env -> P[agentState][actionIndex];

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
                FrozenLake::Result result = env -> P[agentState][actionIndex][ind];

                agentState = result.new_state;

                // Verify if the game is done
                if (result.done)
                {
                    if (result.reward == 1)
                    {
                        //cout << "You have get the goal after: " << steps << endl;
                        break;
                    }
                    else
                    {
                        //cout << "You fell in a hole!" << endl;
                        misses += 1;
                        break;
                    }
                }
                steps++;
            }
        }
        double overage = (float)(misses / episodes) * 100;
        cout << "You fell in the hole: " << overage<< " % of the times" << endl;
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
                    if (q_table[s][a] == 0)
                    {
                        cout << q_table[s][a] << "                |          ";
                    }
                    else
                    {
                        cout << q_table[s][a] << "         |         ";
                    }
                }

                cout << endl;
            }
        }
    }
};
