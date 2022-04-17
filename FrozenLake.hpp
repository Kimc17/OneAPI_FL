
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
    typedef struct 
    {
        double p;
        int new_state;
        double reward;
        bool done;
    } Result;

    std::vector<Result> P[N * N][4];

private:
    char m_board[16] = {
        'S', 'F', 'F', 'F',
        'F', 'H', 'F', 'H',
        'F', 'F', 'F', 'H',
        'H', 'F', 'F', 'G'};
    int actualState;
    
public:
    FrozenLake() // FrozenLakeEnv
    {
        actualState = 0;
        calculate_p();
    }

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

private:
    Result update_probability_matrix(int row, int col, int action)
    {
        Result* result = new Result;
        result -> new_state = inc(row, col, action);
        result -> reward = 0.0;
        result -> done = false;

        if (m_board[result -> new_state] == 'H')
        {
            result -> done = true;
        }
        else if (m_board[result -> new_state] == 'G')
        {
            result -> reward = 1.0;
            result -> done = true;
        }
        return *result;
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
                    Result* r = new Result;
                    int b;

                    if (letter == 'G' | letter == 'H')
                    {
                        *r = {1.0, s, 0.0, true};
                        P[s][a].push_back(*r);
                    }
                    else
                    {
                        b = (((a - 1) % 4) + 4) % 4;
                        *r = update_probability_matrix(row, col, b);
                        r -> p = 1.0 / 3.0;
                        P[s][a].push_back(*r);

                        *r = update_probability_matrix(row, col, a);
                        r -> p = 1.0 / 3.0;
                        P[s][a].push_back(*r);

                        b = (((a + 1) % 4) + 4) % 4;
                        *r = update_probability_matrix(row, col, b);
                        r -> p = 1.0 / 3.0;
                        P[s][a].push_back(*r);
                    }
                }
            }
        }
    }
};
