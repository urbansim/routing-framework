#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Algorithms/TrafficAssignment/Adapters/CCHAdapter.h"
#include "Algorithms/TrafficAssignment/Adapters/DijkstraAdapter.h"
#include "Algorithms/TrafficAssignment/AllOrNothingAssignment.h"
#include "DataStructures/Graph/Attributes/LatLngAttribute.h"
#include "DataStructures/Graph/Attributes/TraversalCostAttribute.h"
#include "DataStructures/Graph/Graph.h"

namespace {

struct Arc {
  int tail;
  int head;
  int cost;
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: BenchmarkWeightedCCH nodes.csv arcs.csv demand.csv\n";
    return 2;
  }
  using Graph = StaticGraph<VertexAttrs<LatLngAttribute>, EdgeAttrs<TraversalCostAttribute>>;

  std::vector<LatLng> coordinates;
  std::ifstream nodes(argv[1]);
  std::string line;
  std::getline(nodes, line);
  while (std::getline(nodes, line)) {
    const auto f = split(line);
    if (f.size() != 3 || std::stoi(f[0]) != coordinates.size())
      throw std::runtime_error("nodes must be dense and zero based");
    coordinates.emplace_back(std::stod(f[1]), std::stod(f[2]));
  }

  std::vector<std::vector<Arc>> outgoing(coordinates.size());
  std::ifstream arcs(argv[2]);
  std::getline(arcs, line);
  while (std::getline(arcs, line)) {
    const auto f = split(line);
    Arc arc{std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2])};
    outgoing.at(arc.tail).push_back(arc);
  }

  Graph graph;
  for (int tail = 0; tail < coordinates.size(); ++tail) {
    graph.appendVertex(coordinates[tail]);
    for (const auto& arc : outgoing[tail]) graph.appendEdge(arc.head, arc.cost);
  }

  std::vector<ClusteredOriginDestination> demand;
  std::ifstream demands(argv[3]);
  std::getline(demands, line);
  while (std::getline(demands, line)) {
    const auto f = split(line);
    demand.emplace_back(std::stoi(f[0]), std::stoi(f[1]), 0, 0, std::stod(f[2]));
  }

#ifdef BENCHMARK_DIJKSTRA
  using Adapter = trafficassignment::DijkstraAdapter<Graph, TraversalCostAttribute>;
#else
  using Adapter = trafficassignment::CCHAdapter<Graph, TraversalCostAttribute>;
#endif
  const auto started = std::chrono::steady_clock::now();
  AllOrNothingAssignment<Adapter> assignment(graph, demand, false);
  const auto preprocessed = std::chrono::steady_clock::now();
  assignment.run();
  const auto finished = std::chrono::steady_clock::now();

  double demand_volume = 0;
  double shortest_path_objective = 0;
  for (int i = 0; i < demand.size(); ++i) {
    demand_volume += demand[i].volume;
    shortest_path_objective += demand[i].volume * assignment.stats.lastDistances[i];
  }
  double total_flow = 0;
  double objective = 0;
  FORALL_EDGES(graph, edge) {
    total_flow += assignment.trafficFlowOn(edge);
    objective += assignment.trafficFlowOn(edge) * graph.traversalCost(edge);
  }
  const auto seconds = [](const auto begin, const auto end) {
    return std::chrono::duration<double>(end - begin).count();
  };
  std::cout << std::setprecision(17)
            << "{\"nodes\":" << graph.numVertices()
            << ",\"arcs\":" << graph.numEdges()
            << ",\"od_pairs\":" << demand.size()
            << ",\"preprocess_seconds\":" << seconds(started, preprocessed)
            << ",\"assignment_seconds\":" << seconds(preprocessed, finished)
            << ",\"total_seconds\":" << seconds(started, finished)
            << ",\"demand_volume\":" << demand_volume
            << ",\"shortest_path_objective_integer_cost\":" << shortest_path_objective
            << ",\"total_edge_flow\":" << total_flow
            << ",\"objective_integer_cost\":" << objective << "}\n";
}
