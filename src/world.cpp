#include "world.hpp"

namespace longroad{

Eigen::VectorXi create_random_vec(int size,float prob,std::mt19937 &rd_generator){
    int t = size*prob;
    std::vector<int> shuffle_vec(size,0);
    std::fill_n(shuffle_vec.begin(),t,1);
    std::shuffle(std::begin(shuffle_vec), std::end(shuffle_vec), rd_generator);
    Eigen::Map<Eigen::VectorXi> vec(shuffle_vec.data(), shuffle_vec.size());
    Eigen::VectorXi result = vec;
    return result;
}    

void Lane_i::circ_shift_down(Eigen::VectorXi &vector_to_shift){
    vector_to_shift = this->ind_down.unaryExpr(vector_to_shift);
}
void Lane_i::circ_shift_up(Eigen::VectorXi &vector_to_shift){
    vector_to_shift = this->ind_up.unaryExpr(vector_to_shift);
}
Eigen::VectorXi Lane_i::get_circ_shift_down(Eigen::VectorXi &vector_to_shift){
    Eigen::VectorXi vector_shifted = this->ind_down.unaryExpr(vector_to_shift);
    return vector_shifted;
}
Eigen::VectorXi Lane_i::get_circ_shift_up(Eigen::VectorXi &vector_to_shift){
    Eigen::VectorXi vector_shifted = this->ind_up.unaryExpr(vector_to_shift);
    return vector_shifted;
}




Lane_i::Lane_i(int size, int color) : color(color){
    
    
    cars = Eigen::VectorXi::Zero(size);//create_random_vec(size,0.2,world_generator); // TODO:Lane prob fixed at compile time !! Change for global reward
    hold = Eigen::VectorXi::Zero(size);//create_random_vec(size,0.5,world_generator); 
    move = Eigen::VectorXi::Zero(size);
    move_pre = Eigen::VectorXi::Zero(size);
    move_last = Eigen::VectorXi::Zero(size);
    std::vector<int> ind_d(size); 
    std::iota(std::begin(ind_d)+1, std::end(ind_d), 0);
    ind_d[0]=size-1;
    Eigen::Map<Eigen::VectorXi> v_d(ind_d.data(),ind_d.size());
    ind_down = v_d;

    std::vector<int> ind_u(size); 
    std::iota(std::begin(ind_u), std::end(ind_u)-1, 1);
    ind_u[size-1]=0;
    Eigen::Map<Eigen::VectorXi> v_u(ind_u.data(),ind_u.size());
    ind_up = v_u;

}


void Lane_i::move_cars(Eigen::VectorXi  actions){
    int z;
    z = (this->color<=1) ? 0 : 1;
    Eigen::VectorXi move;
    this->hold = this->cars.cwiseProduct(actions - z*(2*actions - Eigen::VectorXi::Ones(cars.size())));
    this->move = this->cars.cwiseProduct(actions - (1-z)*(2*actions - Eigen::VectorXi::Ones(cars.size())));
    this->move_pre = this->move;
    if(this->color%2==1){
        this->circ_shift_up(this->move);
    }
    else{
        this->circ_shift_down(this->move);
    }
    this->cars = this->hold + this->move;
}
void Lane_i::move_cars(Eigen::VectorXi  actions, Eigen::VectorXi yellow){
    int z;
    z = (this->color<=1) ? 0 : 1;
    
    this->hold = this->cars.cwiseProduct(actions - z*(2*actions - Eigen::VectorXi::Ones(cars.size())));
    this->move = this->cars.cwiseProduct(actions - (1-z)*(2*actions - Eigen::VectorXi::Ones(cars.size())));
    this->hold = this->hold + this->move.cwiseProduct(yellow);
    this->move = this->move.cwiseProduct((Eigen::VectorXi::Ones(yellow.size()) - yellow));
    this->move_pre = this->move;
    if(this->color%2==1){
        this->circ_shift_up(move);
    }
    else{
        this->circ_shift_down(move);
    }
    this->cars = this->hold + this->move;
}

double Lane_i::calc_speedup_reward(){
    double speeders = 0.0;
    speeders = (this->move_pre.dot(this->move_last))/((double) this->move.size()); 
    this-> move_last = this->move;
    return speeders;
}

Eigen::VectorXi Lane_i::get_hold(){
    return this->hold;
}

Eigen::MatrixXi Lane_i::get_states(){ //Adds MA observations
    Eigen::MatrixXi state(this->cars.size(),3);
    state << this->cars,get_circ_shift_down(this->cars),get_circ_shift_up(this->cars);
    return state;
}

void Lane_i::set_states(Eigen::VectorXi state){ //Adds MA observations
    this->cars<<state;
}


World_i::World_i(int size, bool measure_time,  bool yellow, double global_cost, double global_reward):
                size(size),measure_time(measure_time), yellow(yellow), global_cost(global_cost), global_reward(global_reward){
    
    // init values for avg time measurement
    steptime = 0.0;
    steps = 1;
    t_start = std::chrono::high_resolution_clock::now();
    t_end = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    this->seed = rd();
    this->generator = std::mt19937(this->seed);
    Lane1 = new Lane_i(size,0); Lane1->set_states(create_random_vec(size,0.2,this->generator));
    Lane2 = new Lane_i(size,1); Lane2->set_states(create_random_vec(size,0.2,this->generator));
    Lane3 = new Lane_i(size,2); Lane3->set_states(create_random_vec(size,0.2,this->generator));
    Lane4 = new Lane_i(size,3); Lane4->set_states(create_random_vec(size,0.2,this->generator));
    last_actions = create_random_vec(size,0.5,this->generator); 
    last_states = Eigen::MatrixXi::Zero(size,15); //3 states per lane for all 4 lanes + 3 action or phase states 
    last_states << this->Lane1->get_states(),this->Lane2->get_states(),this->Lane3->get_states(),this->Lane4->get_states(),Eigen::VectorXi::Zero(size),Eigen::VectorXi::Zero(size),Eigen::VectorXi::Zero(size);
    last_rewards = Eigen::VectorXd::Zero(size);
    signal_state = Eigen::VectorXi::Zero(size);
}

void World_i::reset(){
    last_actions = create_random_vec(size,0.5,this->generator); 
    Lane1->set_states(create_random_vec(size,0.2,this->generator));
    Lane2->set_states(create_random_vec(size,0.2,this->generator));
    Lane3->set_states(create_random_vec(size,0.2,this->generator));
    Lane4->set_states(create_random_vec(size,0.2,this->generator));
    last_states = Eigen::MatrixXi::Zero(size,15);
    last_states << this->Lane1->get_states(),this->Lane2->get_states(),this->Lane3->get_states(),this->Lane4->get_states(),Eigen::VectorXi::Zero(size),Eigen::VectorXi::Zero(size),Eigen::VectorXi::Zero(size);
    last_rewards = Eigen::VectorXd::Zero(size);
}

void World_i::setSeed(uint seed){
    this->seed = seed;
    this->generator = std::mt19937(this->seed);
}

void World_i::move_lanes(Eigen::VectorXi actions){
    #pragma omp parallel sections 
    {
        #pragma omp section
        {this->Lane1->move_cars(actions);}
        #pragma omp section 
        {this->Lane2->move_cars(actions);}
        #pragma omp section 
        {this->Lane3->move_cars(actions);}
        #pragma omp section 
        {this->Lane4->move_cars(actions);}
    }

}
void World_i::move_lanes_yellow(Eigen::VectorXi actions){
    Eigen::VectorXi yellow;
    this->signal_state = this->signal_state  + 2*actions - Eigen::VectorXi::Ones(actions.size());
    this->signal_state = this->signal_state.cwiseMin(2);
    this->signal_state = this->signal_state.cwiseMax(-2);
    yellow = this->signal_state.unaryExpr([](int x) {return (abs(x) < 2) ? 1 : 0;});

    #pragma omp parallel sections // parallelization faster about size 10^5
    {
        #pragma omp section
        {this->Lane1->move_cars(actions,yellow);}
        #pragma omp section 
        {this->Lane2->move_cars(actions,yellow);}
        #pragma omp section 
        {this->Lane3->move_cars(actions,yellow);}
        #pragma omp section 
        {this->Lane4->move_cars(actions,yellow);}
    }
}

void World_i::step(Eigen::VectorXi  actions){
    
    if(measure_time){
        t_start = std::chrono::high_resolution_clock::now();
    }
    Eigen::VectorXi reward_i;
    Eigen::VectorXd reward_d;
    if(yellow){
        this->move_lanes_yellow(actions);
    }
    else{
        this->move_lanes(actions);
    }
    
    reward_i = this->Lane1->get_hold() + this->Lane2->get_hold() 
             + this->Lane3->get_hold() + this->Lane4->get_hold();
    reward_i *= -1;
    reward_d = reward_i.cast<double>();
    if(this->global_cost>=0.001){
        // Minimum 0.001 factor 
        double re_1 = (double) (actions-last_actions).cwiseAbs().sum()/actions.size(); //TODO: maybe times 4
        reward_d = reward_d.array() - re_1*this->global_cost;
    }
    if(this->global_reward>=0.001){
        // Minimum 0.001 factor 
        double re_2 = this->Lane1->calc_speedup_reward() + this->Lane2->calc_speedup_reward() 
                  + this->Lane3->calc_speedup_reward() + this->Lane4->calc_speedup_reward();
        reward_d = reward_d.array() +re_2*this->global_reward;
    }
    this->last_rewards = reward_d;
    if(yellow){// States of the for lanes and neighbours + phase state of agent and neighbors
        this->last_states << this->Lane1->get_states(),this->Lane2->get_states(),this->Lane3->get_states(),this->Lane4->get_states(),this->signal_state,Lane1->get_circ_shift_down(this->signal_state),Lane1->get_circ_shift_up(this->signal_state);
    }
    else{//States of the for lanes and neighbours + last action of agent and neighbours
        this->last_states << this->Lane1->get_states(),this->Lane2->get_states(),this->Lane3->get_states(),this->Lane4->get_states(),actions,Lane1->get_circ_shift_down(actions),Lane1->get_circ_shift_up(actions);
    }
    this->last_actions=actions;

    if(measure_time){
       t_end = std::chrono::high_resolution_clock::now();
       double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
       steptime += (elapsed_time_ms-steptime)/steps;
    }
   
}



}  //namespace longroad


int main(int, char**) {
    // int size = 10;
    // longroad::World_i world(size);
}