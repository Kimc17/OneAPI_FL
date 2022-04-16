#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime> 

using namespace std;

#define N 4
#define LEFT 0
#define DOWN 1
#define RIGHT 2
#define UP 3

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

private:
    char m_board[16] = {
        'S', 'F', 'F', 'F',
        'F', 'H', 'F', 'H',
        'F', 'F', 'F', 'H',
        'H', 'F', 'F', 'G'};
    int actualState;
    //std::vector<Result> P[N * N][4];
    std::vector<Result> P[N * N][4];
    
public:
    FrozenLake() // FrozenLakeEnv
    {
        actualState = 0;
        calculate_p();
    }

public:
    Result step(int a)
    {
        int i = categorical_sample(P[actualState][a]);
        Result result = P[actualState][a][i];
        actualState = result.new_state;
        return result;
    }
    void reset()
    {
        actualState = 0;
    }

    int getActualState() const
    {
        return actualState;
    }

    void setActualState(int state)
    {
        actualState = state;
    }

private:
    int inc(int row, int col, int action)
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
    Result update_probability_matrix(int row, int col, int action)
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

    int categorical_sample(std::vector<Result> prob_n)
    {
        double csprob_n[prob_n.size()] = {0}; 
        // initialize an accumulator variable
        double acc = 0;
        //Calculate cumsum
        for (int i = 0; i < prob_n.size(); i++)
        {
            acc += prob_n[i].p;
            csprob_n[i] = acc;
        }

        double random_n = (double) rand() / RAND_MAX;

        for (int i = 0; i < prob_n.size(); ++i)
        {
            if (csprob_n[i] > random_n)
            {
                return i;
            }
        }
        return 0;
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

private:
    double q_table[N * N][4];

public:
    void learn(FrozenLake &env, int totalEpisodes)
    {
        double learning_rate = 0.1;
        double discount_rate = 0.99;
        double exploration_rate = 1;
        double min_exploration_rate = 0.01;
        double max_exploration_rate = 1;
        double exploration_rate_decay = 0.001;
        srand(time(0));

        int rewards_all_episodes[totalEpisodes];

        for (int episode = 1; episode <= totalEpisodes; ++episode)
        {
            env.reset();
            int rewards_current_episode = 0;
            // Exploration rate decay
            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * exp(-exploration_rate_decay * episode);

            for (int step = 1; step <= 100; ++step)
            {
                double exploration_rate_threshold = (double)rand() / RAND_MAX;
                int state = env.getActualState();
                int actionIndex = 0;
                if (exploration_rate < exploration_rate_threshold)
                {
                    // Take a probably smart action
                    actionIndex = std::distance(q_table[state], std::max_element(q_table[state], q_table[state] + 4));
                }
                else
                {
                    // Take a random action to explore
                    actionIndex = rand() % 4;
                }

                // Send the action chosen to the environment and get result with reward and state
                auto result = env.step(actionIndex);
                //  Update Q-table for Q(s,a) using the Q-Learning formula
                q_table[state][actionIndex] = q_table[state][actionIndex] * (1 - learning_rate) +
                                              learning_rate * (result.reward + discount_rate *
                                                                                   *(std::max_element(q_table[result.new_state], q_table[result.new_state] + 4)));

                // Move to the next state and add the reward gotten
                env.setActualState(result.new_state);
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

        /**for (int i = 0; i < totalEpisodes; ++i)
        {
            if (count < 1000)
            {
                sum += rewards_all_episodes[i];
            }
            else
            {
                average = (double)sum / 1000;
                cout << count * turn << " : " << average << endl;
                count = 0;
                turn += 1;
                sum = 0;
            }
            count += 1;
        }*/
    }

    void play(FrozenLake &env, int episodes) const
    {
        srand(time(0));
        float misses = 0;
        for (int i = 0; i < episodes; ++i)
        {
            int steps = 0;
            env.reset();
            while (true)
            {
                // Get the actual state
                int agentState = env.getActualState();
                // Select an action index
                int actionIndex = std::distance(q_table[agentState],
                                                std::max_element(q_table[agentState], q_table[agentState] +4));
                // Try to move with that action
                auto result = env.step(actionIndex);

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

int main()
{
    FrozenLake env;
    QLearner agent;

    unsigned t0, t1;
    double time;

    t0=clock();
    agent.learn(env, 1000000);
    t1 = clock();
    time = (double(t1-t0)/CLOCKS_PER_SEC);
    cout << "Learning time: " << time << endl;

    agent.print();

    t0=clock();
    agent.play(env, 10000);
    t1 = clock();
    time = (double(t1-t0)/CLOCKS_PER_SEC);
    cout << "Playing time: " << time << " for 10000 cases" << endl;

    return 0;
}
