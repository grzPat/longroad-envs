#include <iostream>
#include <chrono> //debug
#include <thread> //debug
#include <Eigen/Dense>
#include <vector>
#include <numeric> //iota
#include <Eigen/Sparse>
#include <random>
#include <iostream> //openmp



//define fehlt noch

namespace longroad{
class Lane_i
{
private:
    /* data */
    int color;
    Eigen::VectorXi cars;
    Eigen::VectorXi hold;
    Eigen::VectorXi move;
    Eigen::VectorXi move_pre;
    Eigen::VectorXi move_last;

    Eigen::VectorXi ind_down;  //maybe put in some vector property class
    Eigen::VectorXi ind_up;
public:
    Lane_i(int size,int color);

    void move_cars(Eigen::VectorXi actions);
    void move_cars(Eigen::VectorXi actions,Eigen::VectorXi yellow);

    void circ_shift_down(Eigen::VectorXi &vector_to_shift);
    void circ_shift_up(Eigen::VectorXi &vector_to_shift);
    Eigen::VectorXi get_circ_shift_down(Eigen::VectorXi &vector_to_shift);
    Eigen::VectorXi get_circ_shift_up(Eigen::VectorXi &vector_to_shift);
    //~Lane_i();

    double calc_speedup_reward();  

    Eigen::VectorXi get_hold();
    Eigen::MatrixXi get_states();
    void set_states(Eigen::VectorXi);
};




class World_i
{
private:
    /* data */
    Lane_i *Lane1;
    Lane_i *Lane2;
    Lane_i *Lane3;
    Lane_i *Lane4;

    u_int seed;
    std::mt19937 generator;
    
    double steptime; 
    int steps;// max steps for measuring time!! 2,147,483,647

    Eigen::VectorXi signal_state;
    Eigen::VectorXi last_actions;

    Eigen::MatrixXi last_states;
    Eigen::VectorXd last_rewards;

    //env parameters
    const int size;
    const bool measure_time;
    const bool yellow;
    const double global_cost; //action ressource cost
    const double global_reward; //collaboration reward

    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end;



    //reward matrix for local rewards


public:
    World_i(int size, bool measure_time,  bool yellow, double global_cost, double global_reward);
    void step(Eigen::VectorXi  actions);
    void setSeed(uint seed);
    void reset();

    //Getters for pyenv
    const Eigen::MatrixXi &lastStates() { return last_states; }
    const Eigen::VectorXd &lastRewards() { return last_rewards; }
    double avgTime() {return steptime;}

    void move_lanes(Eigen::VectorXi  actions);
    void move_lanes_yellow(Eigen::VectorXi  actions);
    
    //~World_i();
};






}
