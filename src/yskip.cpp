#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include "util.h"
#include "timer.h"
#include "skipgram.h"


using namespace yskip;


struct Configuration {
  
  int  graph_train_method;
  bool use_graph;
  int  iter_num;
  int  thread_num;
  int  mini_batch_size;
  int  random_seed;
  bool binary_mode;
  bool verbose;
  int start_year;
  int end_year;
  const char* train_file;
  const char* model_file;
  const char* initial_model_file;
  const char* initial_graph_file;
  Configuration();
};


Configuration::Configuration() {

  graph_train_method = 0;
  use_graph          = true;
  thread_num         = 10;
  mini_batch_size    = 1000;
  iter_num           = 10;
  random_seed        = time(NULL);
  binary_mode        = false;
  verbose            = true;
  start_year         = 1990;
  end_year           = 2016;
  train_file         = NULL;
  model_file         = NULL;
  initial_model_file = NULL;
  initial_graph_file = NULL;
}


void print_help() {

  std::cerr << "yskip [option] <train> <model>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "GDWE model paramters:" << std::endl;
  std::cerr << " -d, --dimensionality=INT           Dimensionality of word embeddings (default: 100)" << std::endl;
  std::cerr << " -w, --window-size=INT              Window size (default: 5)" << std::endl;
  std::cerr << " -n, --negative-sample=INT          Number of negative samples (default: 5)" << std::endl;
  std::cerr << " -a, --alpha=FLOAT                  Distortion parameter (default: 0.75)" << std::endl;
  std::cerr << " -s, --subsampling-threshold=FLOAT  Subsampling threshold (default: 1.0e-5)" << std::endl;
  std::cerr << " -u, --unigram-table-size=INT       Unigram table size used for negative sampling (default: 1e8)" << std::endl;
  std::cerr << " -m, --max-vocabulary-size=INT      Maximum vocabulary size (default: 1e6)" << std::endl;
  std::cerr << " -e, --eta=FLOAT                    Initial learning rate of AdaGrad (default: 0.1)" << std::endl;
  std::cerr << " -b, --mini-batch-size=INT          Mini-batch size (default: 10000)" << std::endl;
  std::cerr << " -B, --binary-mode                  Read/write models in a binary format" << std::endl;
  std::cerr << " -i, --iteration-numbedr            Iteration number in batch learning (default: 5)" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Misc.:" << std::endl;
  std::cerr << " -T, --thread-num=INT               Number of threads (default: 10)" << std::endl;
  std::cerr << " -I, --initial-model=FILE           Initial model (default: NULL)" << std::endl;
  std::cerr << " -G, --initial-graph=FILE           Initial graph (default: NULL)" << std::endl;
  std::cerr << " -r, --random-seed=INT              Random seed (default: current Unix time)" << std::endl;
  std::cerr << " -g, --graph-training-method=INT    Training graph method" << std::endl;
  std::cerr << "                                    0: direct extract (default)" << std::endl;
  //std::cerr << "                                    1: matrix factorization" << std::endl;
  //std::cerr << "                                    2: random walk" << std::endl;
  std::cerr << "                                    3: not use graph" << std::endl;
  std::cerr << " -A, --alpha-loss=FLOAT             Weight between L1/L2 (default: 0.75)" << std::endl;
  std::cerr << " -k, --knowledge-method=INT         Using knowledge graph method" << std::endl;
  std::cerr << "                                    0: use both long-term and short-term knowledge (default)" << std::endl;
  std::cerr << "                                    1: only use long-term knowledge" << std::endl;
  std::cerr << "                                    2: only use short-term knowledge" << std::endl;
  std::cerr << " -K, --long-kg-num=INT              Number of long-term WKGs (default: 5)" << std::endl;
  std::cerr << " -E, --edge-thresh=FLOAT            threshold of edge weights (default: 0.1)" << std::endl;
  std::cerr << " -N, --neb-thresh=INT               threshold of neighborhood number (default: 4)" << std::endl;
  std::cerr << " -J, --jaccard-thresh=FLOAT         threshold of jaccard coefficient (default: 0.1)" << std::endl;
  std::cerr << " -c, --context-num=INT              Direct Extract Context Number (default: 5)" << std::endl;
  std::cerr << " -y, --start-year=INT               Start Year (default: 1990)" << std::endl;
  std::cerr << " -Y, --end-year=INT                 End Year (default: 2016)" << std::endl;
  std::cerr << " -q, --quiet                        Do not show progress messages" << std::endl;
  std::cerr << " -h, --help                         Show this message" << std::endl;
}


int parse_arg(int argc, char* argv[], Skipgram::Option &option, Configuration& config) {

  int opt;
  char* endptr;
  struct option longopts[] = {
    {"dimensionality",        required_argument, NULL, 'd'},
    {"wind-size",             required_argument, NULL, 'w'},
    {"negative-sample-num",   required_argument, NULL, 'n'},
    {"alpha",                 required_argument, NULL, 'a'},
    {"subsampling-threshold", required_argument, NULL, 's'},
    {"unigram-table-size",    required_argument, NULL, 'u'},
    {"max-vocabulary-size",   required_argument, NULL, 'm'},
    {"eta",                   required_argument, NULL, 'e'},
    {"mini-batch-size",       required_argument, NULL, 'b'},
    {"binary-mode",           required_argument, NULL, 'B'},    
    {"iteration-number",      required_argument, NULL, 'i'},
    {"initial-model",         required_argument, NULL, 'I'},
    {"initial-graph",         required_argument, NULL, 'G'},
    {"thread-num",            required_argument, NULL, 'T'},
    {"random-seed",           required_argument, NULL, 'r'},
    {"graph-training-method", required_argument, NULL, 'g'},
    {"alpha-loss",            required_argument, NULL, 'A'},
    {"knowledge-method",      required_argument, NULL, 'k'},
    {"long-kg-num",           required_argument, NULL, 'K'},
    {"edge-thresh",           required_argument, NULL, 'E'},
    {"neb-thresh",            required_argument, NULL, 'N'},
    {"jaccard-thresh",        required_argument, NULL, 'J'},
    {"context-num",           required_argument, NULL, 'c'},
    {"start-year",            required_argument, NULL, 'y'},
    {"end-year",              required_argument, NULL, 'Y'},
    {"quiet",                 no_argument,       NULL, 'q'},
    {"help",                  no_argument,       NULL, 'h'},
    {0,                       0,                 0,    0  },
  };
  int knowledge_method;
  while((opt=getopt_long(argc, argv, "d:w:e:u:m:b:Bl:i:n:a:s:T:r:g:A:k:K:E:N:J:I:G:c:y:Y:hq", longopts, NULL)) != -1){
    switch(opt){   
    case 'd':
      option.vec_size = strtol(optarg, &endptr, 10);
      assert(0 < option.vec_size);
      break;
    case 'e':
      option.eta = atof(optarg);
      assert(0.0 < option.eta);
      break;
    case 'w':
      option.window_size = strtol(optarg, &endptr, 10);
      assert(0 < option.window_size);
      break;
    case 'n':
      option.neg_sample_num = strtol(optarg, &endptr, 10);
      assert(0 < option.neg_sample_num);
      break;
    case 'u':
      option.unigram_table_size = strtol(optarg, &endptr, 10);
      assert(100 <= option.unigram_table_size);
      break;
    case 'm':
      option.max_vocab_size = strtol(optarg, &endptr, 10);
      assert(100 <= option.max_vocab_size);
      break;
    case 'b':
      config.mini_batch_size = strtol(optarg, &endptr, 10);
      option.bn = config.mini_batch_size;
      assert(0 < config.mini_batch_size);
      break;
    case 'B':
      config.binary_mode = true;
      break;
    case 'i':
      config.iter_num = strtol(optarg, &endptr, 10);
      break;
    case 'a':
      option.alpha = atof(optarg);
      assert(0.0 < option.alpha);
      assert(option.alpha <= 1.0);
      break;
    case 's':
      option.subsampling_threshold = atof(optarg);
      assert(0.0 < option.subsampling_threshold);
      break;
    case 'T':
      config.thread_num = strtol(optarg, &endptr, 10);
      assert(0 < config.thread_num);
      break;
    case 'r':
      config.random_seed = strtol(optarg, &endptr, 10);
      break;
    case 'g':
      config.graph_train_method = strtol(optarg, &endptr, 10);
      //assert(config.graph_train_method == 0 || config.graph_train_method == 1 || config.graph_train_method == 2 || config.graph_train_method == 3);
      assert(config.graph_train_method == 0 || config.graph_train_method == 3);
      config.use_graph = (config.graph_train_method < 3);
      break;
    case 'A':
      option.alpha_loss = atof(optarg);
      assert(0.0 < option.alpha_loss);
      break;
    case 'k':
      knowledge_method = strtol(optarg, &endptr, 10);
      assert(knowledge_method == 0 || knowledge_method == 1 || knowledge_method == 2);
      if (knowledge_method == 0 || knowledge_method == 1) option.use_long = true;
      else option.use_long = false;
      if (knowledge_method == 0 || knowledge_method == 2) option.use_short = true;
      else option.use_short = false;
      break;
    case 'K':
      option.k = strtol(optarg, &endptr, 10);
      assert(0 < option.k);
      break;
    case 'E':
      option.e_thresh = atof(optarg);
      assert(0.0 < option.e_thresh);
      assert(option.e_thresh <= 1.0);
      break;
    case 'N':
      option.n_thresh = strtol(optarg, &endptr, 10);
      assert(0 < option.n_thresh);
      break;
    case 'J':
      option.pm_thresh = atof(optarg);
      assert(0.0 <= option.pm_thresh);
      assert(option.pm_thresh <= 1.0);
      break;
    case 'q':
      config.verbose = false;
      break;
    case 'c':
      option.cn = strtol(optarg, &endptr, 10);
      break;
    case 'y':
      config.start_year = strtol(optarg, &endptr, 10);
      assert(0 < config.start_year);
      break;
    case 'Y':
      config.end_year = strtol(optarg, &endptr, 10);
      assert(0 < config.end_year);
      break;
    case 'I':
      config.initial_model_file = optarg;
      break;
    case 'G':
      config.initial_graph_file = optarg;
      break;
    case 'h':
      print_help();
      return FAILURE;
    }
  }
  if (optind + 2 != argc) {
    print_help();
    return FAILURE;
  }
  config.train_file = argv[optind];
  config.model_file = argv[optind+1];
  return SUCCESS;
}


inline void print_progress(const count_t sent_num) {

  if (sent_num%100000 == 0) {
    std::fprintf(stderr, "*");
  }else if (sent_num%10000 == 0) {
    std::fprintf(stderr, ".");
  }
  if (sent_num%1000000 == 0) {
    std::fprintf(stderr, " %ldm\n", sent_num/1000000);
  }
}


inline void print_speed(const timeval start_time, const uint64_t progress, const char* unit) {

  timeval current_time;
  gettimeofday(&current_time, NULL);
  double elapsed = interval(start_time, current_time);
  std::fprintf(stderr, "\rspeed: %.2fk (%d %ss/%.2f sec)", static_cast<real_t>(progress)/1000.0/elapsed, progress, unit, elapsed);
}


inline void train_graph(Skipgram& skipgram, const Configuration& config, Random& random, int sent_num) {

  time_t use_graph_start_time = time(NULL);

  real_t* grad;
  posix_memalign((void**)&grad, 128, sizeof(real_t)*skipgram.vec_size());
  int target_cnt = skipgram.train_with_graph(config.graph_train_method, grad, random);
  free(grad);

  time_t use_graph_time = time(NULL) - use_graph_start_time;
  if (config.verbose && sent_num % (skipgram.mini_batch_size()*10) == 0) {
    std::fprintf(stderr, "(%d) use graph done (%d words/%ld sec)\n", sent_num/skipgram.mini_batch_size(), target_cnt, use_graph_time);
    std::fprintf(stderr, "Graph(nodes,edges) Gi(%d,%d) Ga(%d,%d)\n", skipgram.Gi().nodes_size(), skipgram.Gi().edges_size(), skipgram.Ga().nodes_size(), skipgram.Ga().edges_size());
  }

  //skipgram.initialize_Gi();
}


inline void update_graph(Skipgram& skipgram, const Configuration& config, Random& random, int sent_num) {

  time_t update_graph_start_time = time(NULL);

  skipgram.update_graph(config.graph_train_method, random);

  time_t update_graph_time = time(NULL) - update_graph_start_time;
  if (config.verbose && sent_num % (skipgram.mini_batch_size()*1) == 0){
    std::fprintf(stderr, "(%d) update graph done (%ld sec) Ga(%d, %d)\n\n", sent_num/skipgram.mini_batch_size(), update_graph_time, skipgram.Ga().nodes_size(), skipgram.Ga().edges_size());
  }
} 


inline void asyc_sgd2(Skipgram& skipgram, const int start, const int end, const std::vector<std::vector<std::string>>& mini_batch, Random& random) {

  real_t* grad;
  posix_memalign((void**)&grad, 128, sizeof(real_t)*skipgram.vec_size());
  for (int i = start; i < end; ++i) {
    skipgram.train(mini_batch[i], grad, random);
  }
  free(grad);
}


inline void asyc_sgd(Skipgram& skipgram, const Configuration& config, const std::vector<std::vector<std::string>> mini_batch, Random& random) {

  int n = mini_batch.size();
  std::vector<std::thread> threads; 
  for (int i = 0; i < config.thread_num; ++i) {
    threads.push_back(std::thread(&asyc_sgd2, std::ref(skipgram), i*n/config.thread_num, std::min<int>((i+1)*n/config.thread_num, n), std::ref(mini_batch), std::ref(random)));
  }
  for (int i = 0; i < config.thread_num; ++i) {
    threads[i].join();
  }
}


inline std::string int2str(int n) {

  std::string str = "";
  if (n == 0) {
    str = "0";
    return str;
  }
  
  while (n > 0) {
    char c = '0' + (n%10);
    str = c + str;
    n /= 10;
  }
  return str;
}


inline int train(Skipgram& skipgram, const Configuration& config, const Skipgram::Option& option, Random& random) {

  std::string train_dir = config.train_file;
  std::string postfix = ".txt";

  std::string model_dir = config.model_file;
  std::string w2v_postfix = ".w2v";
  std::string g_postfix = ".graph";

  if (config.verbose) {
    if (!config.use_graph) {
      std::fprintf(stderr, "Training GDWE w/o WKG ...\n");
    }
    else {
      std::fprintf(stderr, "Training GDWE w WKG");
      if (option.use_long) std::fprintf(stderr, "[G_L]");
      if (option.use_short) std::fprintf(stderr, "[G_s]");
      std::fprintf(stderr, " ...\n");
    }
  }

  for (int i = config.start_year; i <= config.end_year; i ++) {
    skipgram.initialize_Gi();
    skipgram.initialize_Ga();

    std::string train_file = train_dir + int2str(i) + postfix;

    FILE* is = fopen(train_file.c_str(), "r");
    if (is == NULL) {
      std::fprintf(stderr, "failed to open %s\n", config.train_file);
      return FAILURE;
    }
    setvbuf(is, NULL, _IOFBF, BUFF_SIZE);

    char line[BUFF_SIZE];
    
    if (config.verbose) {
      //std::fprintf(stderr, " done (vocab size=%ld)\n", skipgram.vocab().size());
      std::fprintf(stderr, "Training year:%d\n", i);
    }

    /*****************************************************
     *  SGD
     *****************************************************/
    time_t start_time = time(NULL);
    count_t sent_num = 0;
    std::vector<std::vector<std::string>> mini_batch;

    int mb_idx = 0;

    rewind(is);
    while (fgets(line, BUFF_SIZE, is) != NULL) {
      line[strlen(line)-1] = '\0';
        
      mini_batch.push_back(tokenize(line));

      skipgram.update_unigram_table(mini_batch.back(), random);

      ++sent_num;
      if (mini_batch.size() == config.mini_batch_size || feof(is) != 0) {
        if (config.use_graph) {
          for (int i = 0; i < mini_batch.size(); i ++) {
            for (int j = 0; j < mini_batch[i].size(); j ++) {
              skipgram.update_Gi(mini_batch[i], j);
            }
          }
          update_graph(skipgram, config, random, sent_num);

          // 2001 event track
          if (i == 2001) {
            std::string graph_path = model_dir + "/" + int2str(i) + "/" + int2str(mb_idx) + g_postfix;
            if(skipgram.Ga().save_E(graph_path, 0.0) == FAILURE) {
              return FAILURE;
            }
          }
        }

        skipgram.rebuild_unigram_table(random); // make sure that the unigram table is calculated without approximation

        for (int iter = 0; iter < config.iter_num; ++iter) {
          asyc_sgd(skipgram, config, mini_batch, random);
          if (config.use_graph) {
            train_graph(skipgram, config, random, sent_num);
          }
        }

        mini_batch.clear();
        skipgram.initialize_Gi();
        
        if (config.verbose) {
          print_progress(sent_num);
        }

        ++ mb_idx;
      }
    }
    fclose(is);
    
    //
    time_t elapsed_time = time(NULL) - start_time;;
    if (config.verbose) {
      std::fprintf(stderr, " done (%lf=%ld/%ld sent/sec)\n", static_cast<double>(sent_num)/static_cast<double>(elapsed_time), sent_num, elapsed_time);
    }

    // update Gp
    if (config.use_graph) {
      time_t Gp_time = time(NULL);

      skipgram.update_Gp(random);

      if (config.verbose) {
        std::fprintf(stderr, "update Gp_ done (%ld sec)\n", time(NULL) - Gp_time);
      }
    }

    // save
    /*
    if (skipgram.save(config.model_file, config.binary_mode) == FAILURE) {
      return FAILURE;
    }
    */

    // save word embedding
    std::string w2v_filename = model_dir + int2str(i) + w2v_postfix;
    if (skipgram.save_word2vec_emb(w2v_filename.c_str()) == SUCCESS) {
      std::fprintf(stderr, "save word2vec %s\n", w2v_filename.c_str());
    }
    else {
      std::fprintf(stderr, "fail saving word2vec %s\n", w2v_filename.c_str());
      return FAILURE;
    }

    // save graph
    if (config.use_graph) {
      std::string graph_path = model_dir + int2str(i) + g_postfix;
      if(skipgram.Ga().save_E(graph_path, 0.0) == FAILURE) {
        return FAILURE;
      }

      int p_kid = skipgram.kid()-skipgram.k()-1;
      if (p_kid >= 0) {
        std::string p_graph_path = model_dir + int2str(i) + "_p" + g_postfix;
        if(skipgram.Gp(p_kid).save_E(p_graph_path, 0.0) == FAILURE) {
          return FAILURE;
        }
      }
    }
  }
  
  return SUCCESS;
}

int main(int argc, char **argv) {

  /*
   * parse arguments
   */
  Configuration config;
  Skipgram::Option option;
  if (parse_arg(argc, argv, option, config) == FAILURE) {
    return FAILURE;
  }
  
  /*
   * initialize model
   */
  if (config.verbose) {
    std::fprintf(stderr, "Initializing model...");
  }
  Random random(config.random_seed);
  Skipgram skipgram(option, random);
  if (config.initial_model_file != NULL) {
    // configuration specified by the option is overwritten
    if (skipgram.load(config.initial_model_file, config.binary_mode) == FAILURE) {
      return FAILURE;
    }
  }

  if (config.initial_graph_file != NULL) {
    // configuration specified by the option is overwritten
    if (skipgram.load_graph(config.initial_graph_file) == FAILURE) {
      return FAILURE;
    }
  }


  if (config.verbose) {
    std::fprintf(stderr, " done\n");
  }
  
  /*
   * train model
   */
  if (train(skipgram, config, option, random) == FAILURE) {
    return FAILURE;
  }
  return SUCCESS;
}

