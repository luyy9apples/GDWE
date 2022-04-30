#pragma once
#include <stdio.h>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <utility>
#include <algorithm>
#include <math.h>

#include "util.h"
#include "random.h"

namespace yskip {

struct Edge {
	std::string node[2];
	real_t weight;

	Edge() {
		weight = 0;
	}

	Edge(const std::string& n1, const std::string& n2, real_t w) {
		node[0] = n1;
		node[1] = n2;
		weight = w;
	}
};

class Graph {
public:
	std::map<std::string, std::set<int> > V; // V={(word, edges)}
	std::vector<Edge> E; // E=[edge]
	std::set<int> trash;

	real_t g_total_weight;
	real_t g_max_weight;

	std::map<std::string, std::set<std::string> > NE;

	Graph();
	~Graph() {};
	
	// basic operation
	int add_edge(const Edge& e);
	// update algorithm
	int proximity_select(real_t e_thresh, int n_thresh, const std::vector<real_t>& log_table, Random& random);
	void build_ne(std::map<std::string, std::set<std::string> >& NE, const std::string& word);
	int union_graph(const Graph& g);

	int save(const std::string& filename);
	int load(const std::string& filename);

	int save_E(const std::string& filename);
	int save_E(const std::string& filename, real_t thresh);
	int load_E(const std::string& filename);

	// get attribute
	int edges_size();
	int nodes_size();

	// for knowledge extraction
	std::vector<std::string> direct_extract(const std::string& t, const int cn, Random& random);
	std::vector<std::string> direct_extract(const std::string& t, Graph& Gt, const int cn, const int n_thresh, const real_t pm_thresh, Random& random);
	
private:
	int add_E(const Edge& e);
	bool del_V(const std::string& n, int eid);

	real_t total_weight(const std::string& word);
	real_t graph_total_weight();
};


/*
 * private methods
 */

inline int Graph::add_E(const Edge& e) {

	int e_idx = -1;
	if (!trash.empty()) {
		e_idx = (*(trash.begin()));
		E[e_idx] = e;
		trash.erase(trash.begin());
	}
	else {
		e_idx = E.size();
		E.push_back(e);
	}

	return e_idx;
}


inline bool Graph::del_V(const std::string& n, int eid) {

	std::map<std::string, std::set<int> >::iterator iter = V.find(n);
	if (iter == V.end()) return false;
	std::string n1 = E[eid].node[0];
	std::string n2 = E[eid].node[1];
	if (!(iter->second).erase(eid)) return false;
	NE[n1].erase(n2);
	NE[n2].erase(n1);
	if ((iter->second).empty()) V.erase(iter);
	NE.erase(n);

	g_total_weight -= E[eid].weight;

	return true;
}


inline real_t Graph::total_weight(const std::string& word) {

	real_t tw = 0.0;
	for(std::set<int>::iterator iter = V[word].begin(); iter != V[word].end(); iter ++) {
		Edge e = E[*iter];
		if(e.weight > 0) tw += e.weight;
	}
	return tw;
}


inline real_t Graph::graph_total_weight() {

	real_t g_total_w = 0.;
	for (std::vector<Edge>::iterator iter = E.begin(); iter != E.end(); iter ++) {
		g_total_w += (iter->weight);
	}
	return g_total_w;
}


/*
 * public methods
 */

inline Graph::Graph() {

	g_total_weight = 0.;
	g_max_weight = 0.;
}


inline int Graph::add_edge(const Edge& e) {

	g_total_weight += e.weight;

	std::map<std::string, std::set<int> >::iterator iter_0 = V.find(e.node[0]);
	std::map<std::string, std::set<int> >::iterator iter_1 = V.find(e.node[1]);
	if (iter_0 == V.end() || iter_1 == V.end()) {
		int e_idx = this->add_E(e);
		
		std::set<int> new_s;
		new_s.insert(e_idx);

		std::set<std::string> init_set;

		if (iter_0 == V.end()) {
			V.insert(std::make_pair(e.node[0], new_s));
			NE[e.node[0]] = init_set;
		}
		else {
			(iter_0->second).insert(e_idx);
		}

		if (iter_1 == V.end()) {
			V.insert(std::make_pair(e.node[1], new_s));
			NE[e.node[1]] = init_set;
		}
		else {
			(iter_1->second).insert(e_idx);
		}

		NE[e.node[0]].insert(e.node[1]);
		NE[e.node[1]].insert(e.node[0]);

		if (e.weight > g_max_weight) g_max_weight = e.weight;

		return e_idx;
	}
	else {
		int exist_e_idx = -1;
		std::set<int>::const_iterator e_iter_0 = (iter_0->second).begin();
		std::set<int>::const_iterator e_iter_1 = (iter_1->second).begin();
		while (e_iter_0 != (iter_0->second).end() && e_iter_1 != (iter_1->second).end()) {
			if ((*e_iter_0) == (*e_iter_1)) {
				exist_e_idx = (*e_iter_0);
				break;
			}
			else if ((*e_iter_0) < (*e_iter_1)) {
				e_iter_0 ++;
			}
			else {
				e_iter_1 ++;
			}
		}

		if (exist_e_idx == -1) {
			int e_idx = this->add_E(e);

			(iter_0->second).insert(e_idx);
			(iter_1->second).insert(e_idx);
			NE[e.node[0]].insert(e.node[1]);
			NE[e.node[1]].insert(e.node[0]);

			if (e.weight > g_max_weight) g_max_weight = e.weight;

			return e_idx;
		}
		else {
			E[exist_e_idx].weight += e.weight;

			if (E[exist_e_idx].weight > g_max_weight) g_max_weight = E[exist_e_idx].weight;

			return exist_e_idx;
		}
	}
}


inline void Graph::build_ne(std::map<std::string, std::set<std::string> >& NE, const std::string& word) {

	std::set<std::string> ne;
	for (std::set<int>::iterator iter = V[word].begin(); iter != V[word].end(); iter ++) {
		Edge e_t = E[(*iter)];
		if (e_t.node[0] == e_t.node[1]) continue;
		if (e_t.node[0] == word) ne.insert(e_t.node[1]);
		else ne.insert(e_t.node[0]); 
	}
	NE[word] = ne;
}


inline int Graph::proximity_select(real_t e_thresh, int n_thresh, const std::vector<real_t>& log_table, Random& random) {

	//real_t smoothed_g_max_weight = std::pow(g_max_weight, 0.75);
	real_t smoothed_g_max_weight;
	if ((int)g_max_weight < log_table.size()) smoothed_g_max_weight = log_table[(int)g_max_weight];
	//else smoothed_g_max_weight = std::pow(g_max_weight, 0.75);
	else smoothed_g_max_weight = std::log(g_max_weight);

	std::vector<int> del_e_idx;

	int max_ne_size = 0;

	for (int i = 0; i < E.size(); i ++) {
		if (trash.find(i) != trash.end()) continue;

		Edge e = E[i];
		//real_t smoothed_e_weight = std::pow(e.weight, 0.75);
		real_t smoothed_e_weight;
		if ((int)e.weight < log_table.size()) smoothed_e_weight = log_table[(int)e.weight];
		//else smoothed_e_weight = std::pow(e.weight, 0.75);
		else smoothed_e_weight = std::log(e.weight);
		real_t s_e = 0.;

		std::set<std::string> NE_0, NE_1;
		if (NE.find(e.node[0]) == NE.end()) build_ne(NE, e.node[0]);
		if (NE.find(e.node[1]) == NE.end()) build_ne(NE, e.node[1]);
		NE_0 = NE[e.node[0]];
		NE_1 = NE[e.node[1]];

		if (NE_0.size() < n_thresh && NE_1.size() < n_thresh) {
			del_e_idx.push_back(i);
			continue;
		}

		real_t s2 = 0.;
		// edge weight only
		s2 = smoothed_e_weight / smoothed_g_max_weight;

		s_e += s2;
		//if (i % 1000 == 0) std::fprintf(stderr, "%d\t\ts_e = %.4f (%d %.4f / %d %.4f)\n", i, s_e, (int)e.weight, smoothed_e_weight, (int)g_max_weight, smoothed_g_max_weight);
		if (s_e > e_thresh) continue;
		// else delete Edge e
		del_e_idx.push_back(i);
	}

	for (int i = 0; i < del_e_idx.size(); i ++) {
		int idx = del_e_idx[i];
		Edge e = E[idx];
		trash.insert(idx);
		this->del_V(e.node[0], idx);
		this->del_V(e.node[1], idx);
	}

	return this->edges_size();
}


inline int Graph::union_graph(const Graph& g) {

	for (int i = 0; i < g.E.size(); i ++) {
		if (g.trash.find(i) != g.trash.end()) continue;
		this->add_edge(g.E[i]);
	}
	return this->edges_size();
}


inline int Graph::save(const std::string& filename) {

	FILE* os = NULL;
	os = fopen(filename.c_str(), "w");
	if (os == NULL) {
		std::fprintf(stderr, "cannot open %s\n", filename.c_str());
		return FAILURE;
	}
	setvbuf(os, NULL, _IOFBF, BUFF_SIZE);

	// header
	std::fprintf(os, "%d\t%d\t%d\n", this->nodes_size(), this->edges_size(), E.size());

	// V
	for (std::map<std::string, std::set<int> >::iterator iter = V.begin(); iter != V.end(); iter ++) {
		std::fprintf(os, "%s\t", (iter->first).c_str());
		std::fprintf(os, "%d\t", (iter->second).size());
		for (std::set<int>::iterator e_iter = (iter->second).begin(); e_iter != (iter->second).end(); e_iter ++) {
			if (e_iter == (iter->second).begin()) {
				std::fprintf(os, "%d", (*e_iter));
			}
			else {
				std::fprintf(os, " %d", (*e_iter));
			}
		}
		std::fprintf(os, "\n");
	}

	// E
	for (int i = 0; i < E.size(); i ++) {
		if (E[i].weight <= 0) continue;
		std::fprintf(os, "%d", i);
		std::fprintf(os, "\t%s", E[i].node[0].c_str());
		std::fprintf(os, "\t%s", E[i].node[1].c_str());
		std::fprintf(os, "\t%f\n", E[i].weight);
	}

	fclose(os);

	return SUCCESS;
}


inline int Graph::load(const std::string& filename) {

	FILE* is = NULL;
	is = fopen(filename.c_str(), "r");

	setvbuf(is, NULL, _IOFBF, BUFF_SIZE);
	if(is == NULL) {
		std::fprintf(stderr, "fialed to open %s\n", filename.c_str());
		return FAILURE;
	}

	// read header
	char line[BUFF_SIZE];
	if (fgets(line, BUFF_SIZE, is) == NULL) {
		std::fprintf(stderr, "%s is empty\n", filename.c_str());
		return FAILURE;
	}
	line[strlen(line)-1] = '\0';
	int nodes_size_, edges_size_, max_edges_size_;
	if (sscanf(line, "%d\t%d\t%d\n", &nodes_size_, &edges_size_, &max_edges_size_) != 3) {
		std::fprintf(stderr, HERE "invalid format (%s): %s\n", filename.c_str(), line);
		return FAILURE;
	}

	V.clear();
	E = std::vector<Edge>(max_edges_size_);
	trash.clear();

	char key_word[BUFF_SIZE], s1[BUFF_SIZE], n1[BUFF_SIZE], n2[BUFF_SIZE];
	int edges_cnt, eid;
	real_t ew;
	int line_cnt = 0;
	while (fgets(line, BUFF_SIZE, is) != NULL) {
		line[strlen(line)-1] = '\0';
		// load V
		if (line_cnt < nodes_size_) {
			if(sscanf(line, "%s\t%d %[^\t]", key_word, &edges_cnt, s1) != 3) {
				std::fprintf(stderr, HERE "invalid format (%s): %s\n", filename.c_str(), line);
				return FAILURE;
			}

			int* data;
			posix_memalign((void**)data, 32, sizeof(int)*edges_cnt);
			if (fread(data, sizeof(int), static_cast<size_t>(edges_cnt), is) != static_cast<size_t>(edges_cnt)) {
				std::fprintf(stderr, HERE "invalid format (%s): %s\n", filename.c_str(), line);
				return FAILURE;
			}

			std::set<int> es_t;
			for (int i = 0; i < edges_cnt; i ++) {
				es_t.insert(data[i]);
			}
			V.insert(std::make_pair(key_word, es_t));

			free(data);
		}
		// load E
		else {
			if(sscanf(line, "%d\t%s\t%s\t%f\n", &eid, n1, n2, &ew) != 4) {
				std::fprintf(stderr, HERE "invalid format (%s): %s\n", filename.c_str(), line);
				return FAILURE;
			}
			E[eid].node[0] = n1;
			E[eid].node[1] = n2;
			E[eid].weight = ew;
		}

		line_cnt ++;
	}


	for (int i = 0; i < E.size(); i ++) {
		if (E[i].weight <= 0) {
			trash.insert(i);
		}
	}

	fclose(is);

	return SUCCESS;
}


inline int Graph::save_E(const std::string& filename) {

	FILE* os = NULL;
	os = fopen(filename.c_str(), "w");
	if (os == NULL) {
		std::fprintf(stderr, "cannot open %s\n", filename.c_str());
		return FAILURE;
	}
	setvbuf(os, NULL, _IOFBF, BUFF_SIZE);

	// E
	for (int i = 0; i < E.size(); i ++) {
		if (E[i].weight <= 0) continue;
		std::fprintf(os, "%s", E[i].node[0].c_str());
		std::fprintf(os, "\t%s", E[i].node[1].c_str());
		std::fprintf(os, "\t%f\n", E[i].weight);
	}

	fclose(os);

	return SUCCESS;
}


inline int Graph::save_E(const std::string& filename, real_t thresh) {

	FILE* os = NULL;
	os = fopen(filename.c_str(), "w");
	if (os == NULL) {
		std::fprintf(stderr, "cannot open %s\n", filename.c_str());
		return FAILURE;
	}
	setvbuf(os, NULL, _IOFBF, BUFF_SIZE);

	// E
	for (int i = 0; i < E.size(); i ++) {
		if (trash.find(i) != trash.end()) continue;
		if (E[i].weight < thresh) continue;
		std::fprintf(os, "%s", E[i].node[0].c_str());
		std::fprintf(os, "\t%s", E[i].node[1].c_str());
		std::fprintf(os, "\t%f\n", E[i].weight);
	}

	fclose(os);

	return SUCCESS;
}


inline int Graph::load_E(const std::string& filename) {

	FILE* is = NULL;
	is = fopen(filename.c_str(), "r");

	setvbuf(is, NULL, _IOFBF, BUFF_SIZE);
	if(is == NULL) {
		std::fprintf(stderr, "fialed to open %s\n", filename.c_str());
		return FAILURE;
	}

	V.clear();
	E.clear();
	trash.clear();

	char line[BUFF_SIZE];
	char n1[BUFF_SIZE], n2[BUFF_SIZE];
	real_t ew;
	while (fgets(line, BUFF_SIZE, is) != NULL) {
		line[strlen(line)-1] = '\0';
		// load E
		if(sscanf(line, "%s\t%s\t%f\n", n1, n2, &ew) != 3) {
			std::fprintf(stderr, HERE "invalid format (%s): %s\n", filename.c_str(), line);
			return FAILURE;
		}
		Edge e(n1, n2, ew);
		add_edge(e);
	}

	fclose(is);

	return SUCCESS;
}


inline int Graph::edges_size() {

	return (E.size() - trash.size());
}


inline int Graph::nodes_size() {

	return V.size();
}


inline std::vector<std::string> Graph::direct_extract(const std::string& t, const int cn, Random& random) {

	std::vector<std::string> v;

	std::map<std::string, std::set<int> >::iterator iter = V.find(t);
	if (iter == V.end() || (iter->second).empty()) return v;

	// randomly sample
	std::vector<Edge> es;
	real_t total_weight = 0;
	for (std::set<int>::iterator e_iter = (iter->second).begin(); e_iter != (iter->second).end(); e_iter ++) {
		Edge e = E[(*e_iter)];

		if (E[(*e_iter)].node[0] == t) {
			e.weight /= V[E[(*e_iter)].node[1]].size();
		}
		else {
			e.weight /= V[E[(*e_iter)].node[0]].size();
		}
		total_weight += e.weight;
		es.push_back(e);
	}

	std::vector<real_t> samp_prod;
	real_t acc_prod = 0.0;
	for (int i = 0; i < es.size()-1; i ++) {
		acc_prod += (es[i].weight / total_weight);
		samp_prod.push_back(acc_prod);
	}
	samp_prod.push_back(1.01);

	for (int i = 0; i < cn; i ++) {
		real_t sp = random.uniform(0.0, 1.0);
		int s_eid = lower_bound(samp_prod.begin(), samp_prod.end(), sp) - samp_prod.begin();
		Edge se = es[s_eid];
		if (se.node[0] == t) v.push_back(se.node[1]);
		else v.push_back(se.node[0]);
	}
	return v;
}


inline std::vector<std::string> Graph::direct_extract(const std::string& t, Graph& Gt, const int cn, const int n_thresh, const real_t pm_thresh, Random& random) {

	std::vector<std::string> v;

	std::map<std::string, std::set<int> >::iterator iter = V.find(t);
	if (iter == V.end() || (iter->second).empty()) return v;

	// compare nebs of t in G(this) and Gt
	if (NE.find(t) == NE.end()) build_ne(NE, t);
	std::set<std::string>& NE_0 = NE[t];

	if (Gt.NE.find(t) == Gt.NE.end()) Gt.build_ne(Gt.NE, t);
	std::set<std::string>& NE_1 = Gt.NE[t];

	if (NE_0.size() < n_thresh || NE_1.size() < n_thresh) return v;

	std::set<std::string> NE_I;
	std::set_intersection(NE_0.begin(), NE_0.end(), NE_1.begin(), NE_1.end(), inserter(NE_I, NE_I.begin()));
	std::set<std::string> NE_U;
	std::set_union(NE_0.begin(), NE_0.end(), NE_1.begin(), NE_1.end(), inserter(NE_U, NE_U.begin()));
	real_t pm_score = (NE_I.size()*1.0)/NE_U.size();

	real_t r = random.uniform(0.0, 1.0);
	if (r < 0.0001) {
		std::fprintf(stderr, "%s score=%.4f\n", t.c_str(), pm_score);
		std::set<std::string>::iterator it = NE_0.begin();
		for (int i = 0; i < 5; ++ i) {
			if (it == NE_0.end()) break;
			std::fprintf(stderr, "%s ", it->c_str());
			it ++;
		}
		std::fprintf(stderr, "\n");

		it = NE_1.begin();
		for (int i = 0; i < 5; ++ i) {
			if (it == NE_1.end()) break;
			std::fprintf(stderr, "%s ", it->c_str());
			it ++;
		}
		std::fprintf(stderr, "\n\n");
	}

	if (pm_score < pm_thresh) return v;

	// randomly sample
	std::vector<Edge> es;
	real_t total_weight = 0;
	for (std::set<int>::iterator e_iter = (iter->second).begin(); e_iter != (iter->second).end(); e_iter ++) {
		Edge e = E[(*e_iter)];

		if (E[(*e_iter)].node[0] == t) {
			e.weight /= V[e.node[1]].size();
		}
		else {
			e.weight /= V[e.node[0]].size();
		}
		total_weight += e.weight;
		es.push_back(e);
	}

	std::vector<real_t> samp_prod;
	real_t acc_prod = 0.0;
	for (int i = 0; i < es.size()-1; i ++) {
		acc_prod += (es[i].weight / total_weight);
		samp_prod.push_back(acc_prod);
	}
	samp_prod.push_back(1.01);

	for (int i = 0; i < cn; i ++) {
		real_t sp = random.uniform(0.0, 1.0);
		int s_eid = lower_bound(samp_prod.begin(), samp_prod.end(), sp) - samp_prod.begin();
		Edge se = es[s_eid];
		if (se.node[0] == t) v.push_back(se.node[1]);
		else v.push_back(se.node[0]);
	}

	if (r < 0.0001) {
		for (int i = 0; i < v.size(); i ++) {
			std::fprintf(stderr, "%s ", v[i].c_str());
		}
		std::fprintf(stderr, "\n\n");
	}

	return v;
}

}
