#define TA_LOG_K 0

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Algorithms/TrafficAssignment/Adapters/CCHAdapter.h"
#include "DataStructures/Graph/Attributes/LatLngAttribute.h"
#include "DataStructures/Graph/Attributes/TraversalCostAttribute.h"
#include "DataStructures/Graph/Graph.h"
#include "Tools/Constants.h"

int main() {
  using Graph = StaticGraph<
      VertexAttrs<LatLngAttribute>, EdgeAttrs<TraversalCostAttribute>>;
  using Adapter = trafficassignment::CCHAdapter<Graph, TraversalCostAttribute>;

  const std::vector<std::vector<std::pair<int, int>>> outgoing = {
      {{1, 2}, {2, 9}},
      {{0, 2}, {2, 3}, {3, 8}},
      {{0, 9}, {1, 3}, {3, 1}, {4, 7}},
      {{1, 8}, {2, 1}, {4, 2}, {5, 9}},
      {{2, 7}, {3, 2}, {5, 3}},
      {{3, 9}, {4, 3}},
  };
  Graph graph;
  for (int tail = 0; tail < static_cast<int>(outgoing.size()); ++tail) {
    graph.appendVertex(LatLng(0.01 * tail, 0.02 * tail));
    for (const auto& arc : outgoing[tail]) graph.appendEdge(arc.first, arc.second);
  }

  std::vector<std::vector<int>> expected(6, std::vector<int>(6, INFTY));
  for (int u = 0; u < 6; ++u) {
    expected[u][u] = 0;
    for (const auto& arc : outgoing[u])
      expected[u][arc.first] = std::min(expected[u][arc.first], arc.second);
  }
  for (int k = 0; k < 6; ++k)
    for (int u = 0; u < 6; ++u)
      for (int v = 0; v < 6; ++v)
        if (expected[u][k] != INFTY && expected[k][v] != INFTY)
          expected[u][v] = std::min(expected[u][v], expected[u][k] + expected[k][v]);

  Adapter adapter(graph);
  adapter.preprocess();
  adapter.customize();
  auto query = adapter.getQueryAlgoInstance();
  std::array<int, Adapter::K> sources;
  std::array<int, Adapter::K> targets;
  std::array<double, Adapter::K> volumes;
  volumes.fill(0.0);
  for (int pass = 0; pass < 5; ++pass) {
    for (int source = 5; source >= 0; --source) {
      for (int target = 0; target < 6; ++target) {
        sources.fill(source);
        targets.fill(target);
        query.run(sources, targets, volumes, 1);
        if (query.getDistance(target, 0) != expected[source][target]) {
          std::cerr << "query reuse mismatch for " << source << " -> " << target
                    << ": " << query.getDistance(target, 0) << " != "
                    << expected[source][target] << '\n';
          return 1;
        }
      }
    }
  }
  std::cout << "scalar CCH query reuse OK\n";
  return 0;
}
