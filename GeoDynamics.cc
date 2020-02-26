/*
 * GeoDynamics Model undirected version 
 */
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h> 
#include <unistd.h>

#include <iostream> 
#include <fstream>
#include <iterator> 
#include <random>
#include <algorithm>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics.h>

#include <boost/tokenizer.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/functional/hash.hpp>

#include <zlib.h>
#include "util.hpp"
#include <omp.h>



// type defs
using JsonTree = boost::property_tree::ptree;
typedef uint32_t time_type;

// globals
std::string filebasename; // base filename
const gsl_rng_type * GSL_RNG_T;
gsl_rng * gsl_rand;

// system variables 
gsl_vector * resource;
gsl_vector * strength;
struct Edge{
  unsigned int i;
  unsigned int j;
  double weight;
  double distance;
};
Edge * edgelist;
gsl_vector * capacity;

void initialize_state(JsonTree &opt);



namespace Params{
  time_type total_time, checkpoint_interval, transient;
  std::vector<time_type> checkpoints;
  double d, d0, eps, R, R0;
  uint32_t M, N; // num of edges, nodes

  void load_edges(std::string filename);
  void set_params(JsonTree &opt) {
    N       = opt.get<int>("dynamics.nodes" , 0);
    assert (N >0);
    d       = opt.get<double>("dynamics.d" , 0.2);  
    eps     = opt.get<double>("dynamics.eps" , 0.01);  
    R       = opt.get<double>("dynamics.R" , 1.0);
    R0      = opt.get<double>("dynamics.R0" , R); // in km
    total_time  = opt.get<int>("dynamics.total_time" , 1e4);  
    checkpoint_interval  = opt.get<int>("dynamics.checkpoint_interval" , total_time/20);  
    transient  = opt.get<int>("dynamics.transient" , 0.3 * total_time);  

    load_edges(opt.get<std::string>("edgefile", "test.edge"));

    capacity = gsl_vector_calloc(N);
    for(size_t i =0; i< capacity->size;++i){capacity->data[i] = 1;}

    // print parameters and settings
    std::cerr << "## Model parameters \n"
      << "d     = " << d << std::endl
      << "eps   = " << eps << std::endl
      << "R     = " << R << std::endl
      << "R0    = " << R0 << std::endl
      << "N     = " << N << std::endl
      << "M     = " << M << std::endl
      << "## System parameters \n"
      << "total_time  = " << total_time << "\n"
      << "checkpoint interval  = " << checkpoint_interval << "\n"
      << std::endl;
  }
} // end of namespace [Params]


// functions
double cost_func(double distance, double scale) { double r = distance/scale; return r*exp(r); }
double current_R(double time) {
  double T1 = Params::transient;
  if (time > T1){
    return Params::R;
  } else {
    return (time)/(T1) * (Params::R - Params::R0) + Params::R0;
  }
}


void save_pagerank(const std::string filename);
void save_traffic(const std::string filename, double percentage);
void resource_fwrite(std::string filename);
void edgelist_fwrite(std::string filename);

void main_routine(){
  using namespace Params;

  // tmp variables
  gsl_vector * diffusion     = gsl_vector_calloc(N);
  gsl_vector * strength_next = gsl_vector_calloc(N);

  // for checkpoint jobs
  // the rate of change of variable
  FILE * fp_rate = fopen( (filebasename + ".rate").c_str(), "w" );
  gsl_vector * prev_resource = gsl_vector_calloc(N);
  gsl_vector_memcpy(prev_resource, resource);
  fprintf(fp_rate, "Time,Rate,SumX\n");

  // Run
  std::cerr << "==  RUN ==" << std::endl;
  uint32_t currentstep = 0;
  uint32_t maxstep     = int(Params::total_time);
  double tmp_R = current_R(currentstep);

#pragma omp parallel
  {
    // thread-local variables
    std::vector<double> diffusion_local(N, 0.0);
    std::vector<double> s_next_local(N, 0.0);

    // time step loop
    while(currentstep < maxstep){
      // reset thread-local variables
      for(uint32_t i=0; i<N;++i){
        diffusion_local[i] = 0; s_next_local[i] = 0;
      }

      // zero fill on  temp. variables
#pragma omp for nowait
      for(uint32_t  i = 0; i < N; ++i){
        diffusion->data[i]     = 0;
        strength_next->data[i] = 0;
      }
      // barrier until all pre-processes are finished.
#pragma omp barrier

      // loop for edges (update w_ij(shared), diff(local), strength_next(local)
#pragma omp for schedule(static) nowait
      for(unsigned int m=0; m < M ; ++m){
        int i     = edgelist[m].i;
        int j     = edgelist[m].j;
        double w  = edgelist[m].weight;
        double cij  = cost_func(edgelist[m].distance, tmp_R);
        double xi = resource->data[i];
        double xj = resource->data[j];

        if(strength->data[j] > 0) { diffusion_local[i] += w * xj/ strength->data[j]; }
        if(strength->data[i] > 0) { diffusion_local[j] += w * xi/ strength->data[i]; }

        double wnew = 0;
        // skip large cost edges
        if (cij < (1.0/eps) ){ wnew = (1.0 - eps*cij)*w  + eps * xi*xj; }	

        s_next_local[i] += wnew;
        s_next_local[j] += wnew;
        edgelist[m].weight = wnew; // update weight
      } // end of edge loop [nowait]

      // summarlize thread-local variables (x,s) to the shared variables
      for(uint32_t i=0; i<N; ++i) {
        if(diffusion_local[i] >0){
#pragma omp atomic update
          diffusion->data[i]     += diffusion_local[i];
        }
        if(s_next_local[i] >0){
#pragma omp atomic update
          strength_next->data[i] += s_next_local[i];
        }
      }

      // barrier for saveing the shared variables
#pragma omp barrier

      // update resource,x and strength
#pragma omp for schedule(static) 
      for(uint32_t i = 0; i < N; ++i){
        if (strength->data[i] > 0){ // connected
          resource->data[i] = (1-d)*capacity->data[i] + d * diffusion->data[i]; // x(t+1) = (1-D)*C_i + D sum_j T_ij x_j
        } else{ // disconnected
          resource->data[i] = (1-d)*capacity->data[i] + d * resource->data[i]; //  x(t+1) -x_i(t) = (1-D)(C_i - x_i)
        }
        strength->data[i] = strength_next->data[i];
      }

      // prepare for next step
#pragma omp single
      {
        currentstep +=1;
        tmp_R = current_R(currentstep);
      }
#pragma omp barrier

      // checkpoint job if ( (currentstep % checkpoint_interval == 0) or (currentstep < 10))
      if ( currentstep % checkpoint_interval == 0){ 
#pragma omp single nowait
        {
          fprintf(stderr, "Checkpoing [%d / %d] \n", currentstep, total_time );
        }

        // the rate of change of variable
#pragma omp single nowait
        {
          // change rate of resource
          double norm = 0;
          // resource sum
          double sum = 0;
          for(size_t i = 0; i < resource->size; ++i){
            norm += fabs( resource->data[i] - prev_resource->data[i]);
            sum  += resource->data[i];
          }
          norm /= checkpoint_interval * resource->size;
          fprintf(fp_rate, "%d,%.10f,%f\n", currentstep, norm, sum/resource->size);
          gsl_vector_memcpy(prev_resource, resource);
        }
      }
#pragma omp barrier
    } // end of step-loop
  } // end of parallels

  fclose(fp_rate);
  gsl_vector_free(diffusion);
  gsl_vector_free(strength_next);
  gsl_vector_free(prev_resource);
} // end of mainroutine



int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Error! Usage :  <program> [setting json file] \n";
    exit(1);
  }

  // initialize.
  JsonTree option_parser;
  boost::property_tree::read_json(argv[1], option_parser);
  filebasename = option_parser.get<std::string>("base", "test");

  // setup GSL random
  {
    gsl_rng_env_setup();
    const gsl_rng_type * GSL_RNG_T = gsl_rng_default;
    gsl_rand = gsl_rng_alloc(GSL_RNG_T);

    long int  seed = option_parser.get<double>("seed",-1);
    if (seed < 0){ // get seed from hardware random device
      std::random_device rd; 
      seed = rd();
    }
    gsl_rng_set(gsl_rand, seed);
    std::cerr << "seed = " << seed << "\n";
  }

  // set parameters
  std::cerr << "==  Setup  ==" << std::endl;
  Params::set_params(option_parser);

  // initial conditions
  std::cerr << "==  Initial condition  ==" << std::endl;
  initialize_state(option_parser);


  // Dump init
  {
    std::cerr << "==  Dump init ==" << std::endl;
    save_pagerank(filebasename + ".init.x");
  }

  // main routine
  main_routine();


  // Finish
  {
    save_pagerank(filebasename + ".last.x");
  }

  std::cerr << "== FINISH  ==" << std::endl;
  return 0;
}

namespace Params{
  void load_edges(std::string filename){
    std::cerr << "== Reading weighted edge list file : " << filename << "  ==" << std::endl;

    const int maxlen = 256;
    char buf[maxlen];

    double read_file_epsilon = 1e-6;
    double upperlimit = get_upperlimit(eps,R); // in km
    std::cerr << " reading edges ( <= upperlimit " << upperlimit << " (km)" << std::endl;

    // temp memory
    std::vector<int> tmp_edgelist;
    std::vector<double> tmp_distance;

    { // read file
      uint32_t m = 0;

      gzFile stream = gzopen(filename.c_str(), "r");
      if (stream == NULL) { std::cerr << "Unable to open edge file: " << filename << std::endl; exit(2) ;}

      int v1,v2;
      double w12,w21;
      while(gzgets(stream,buf, maxlen - 1) != NULL){
        sscanf(buf, "%d %d %lf %lf",&v1,&v2,&w12, &w21);

        // skip distance > upper limit
        if( std::max(w12,w21) > upperlimit){continue;}

        tmp_edgelist.push_back(v1);
        tmp_edgelist.push_back(v2);
        if (std::abs(w12 - w21) > read_file_epsilon ){
          std::cerr << boost::format("Edge file error : w12 and w21 should be equal :V1=%d, V2=%d, w12=%.15f, w21=%.15f") % v1 %v2 % w12 %w21 << std::endl; exit(2);
        }
        tmp_distance.push_back( 0.5*(w12+w21) );
        m += 1;
      }
      gzclose(stream);
    }
    { // store
      Params::M = tmp_distance.size();
      edgelist = (Edge*) malloc(M*sizeof(Edge));

      for(uint32_t m =0; m < M; ++m){
        edgelist[m].i = tmp_edgelist[2*m];
        edgelist[m].j = tmp_edgelist[2*m+1];
        edgelist[m].distance = tmp_distance[m];
      }
    }
  }
} // end of namespace [Params]


void initialize_state(JsonTree &option_parser){
  using namespace Params;

  // resource Init
  std::cerr << "initalize x" << std::endl;
  resource = gsl_vector_calloc(N);

  std::string init_type = option_parser.get<std::string>("dynamics.init_type", "uniform");
  std::cerr << "inital condition type = " << init_type << std::endl;
  double variance  = option_parser.get<double>("dynamics.init_random_strength", 0.01);
  {
    double min = 1.0 - variance;
    double max = 1.0 + variance;
    fprintf(stderr, "inital condition of x : uniform in (%f,%f)\n", min,max);
    for(unsigned int i =0; i < resource->size ; ++i){
      double val = min + (max  - min) * gsl_rng_uniform_pos(gsl_rand);
      gsl_vector_set(resource, i , val);
    }
  }

  // normalize of x (to be N)
  {
    double sum_of_x = 0;
    for(uint32_t  i = 0; i < N; ++i){sum_of_x += resource->data[i];}
    double mod = double(N) / sum_of_x;
    for(uint32_t i = 0; i < N; ++i){ resource->data[i] = resource->data[i] * mod;}
  }

  // Weight Init 
  std::cerr << "initalize weight" << std::endl;
  {
    double init_upper_limit = get_upperlimit(eps,R0);
    fprintf(stderr, "inital condition of weight uniform in (0,%f) if distance < upper limit (%f) by R0=%f\n", variance, init_upper_limit, R0);

    for(unsigned int m =0; m < M; ++m){
      double cij  = cost_func(edgelist[m].distance, R0);
      if (cij < (1.0/eps) ){ 
        double val = variance * gsl_rng_uniform_pos(gsl_rand);
        edgelist[m].weight = val;
      } else{
        edgelist[m].weight = 0;
      }
    }
  }

  // Strength Init
  std::cerr << "initalize strength" << std::endl;
  strength = gsl_vector_calloc(N);
  gsl_vector_set_zero(strength);
  for(unsigned int m=0; m < M ; ++m){
    strength->data[edgelist[m].i] += edgelist[m].weight;
    strength->data[edgelist[m].j] += edgelist[m].weight;
  }

  std::cerr << "initalize finished" << std::endl;
}

void save_pagerank(const std::string filename){
  std::cerr << "save pagerank" << std::endl;
  FILE *fp;
  if ((fp = fopen(filename.c_str(), "w")) == NULL) { std::cerr << "Open file error : " << filename << std::endl; exit(1); }
  fprintf(fp,"ID,PageRank\n");
  for(size_t i = 0; i < resource->size ; ++i){ fprintf(fp, "%lu,%e\n", i, resource->data[i]); }
  fclose(fp);
  std::cerr << "save pagerank end" << std::endl;
}

