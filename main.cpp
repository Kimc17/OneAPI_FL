#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime> 
#include "QLearner.hpp"

using namespace std;


int main()
{
    FrozenLake* env = new FrozenLake();
    QLearner* agent = new QLearner();

    unsigned t0, t1;
    double time;

    t0=clock();
    agent -> learn(env, 1000000);
    t1 = clock();
    time = (double(t1-t0)/CLOCKS_PER_SEC);
    cout << "Learning time: " << time << endl;

    agent -> print();

    t0=clock();
    agent -> play(env, 10000);
    t1 = clock();
    time = (double(t1-t0)/CLOCKS_PER_SEC);
    cout << "Playing time: " << time << " for 10000 cases" << endl;

    return 0;
}
