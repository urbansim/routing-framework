#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Algorithms/TrafficAssignment/Adapters/CCHAdapter.h"
#include "Algorithms/TrafficAssignment/AllOrNothingAssignment.h"
#include "DataStructures/Graph/Attributes/LatLngAttribute.h"
#include "DataStructures/Graph/Attributes/TraversalCostAttribute.h"
#include "DataStructures/Graph/Graph.h"

namespace {

constexpr double COST_SCALE = 100000.0;

struct Arc {
  int tail;
  int head;
  double freeFlow;
  double alpha;
  double beta;
  double capacity;
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double result = 0;
#pragma omp parallel for reduction(+ : result) schedule(static)
  for (int i = 0; i < static_cast<int>(a.size()); ++i) result += a[i] * b[i];
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: BenchmarkWeightedCCHEquilibrium nodes.csv arcs.csv demand.csv\n";
    return 2;
  }
  using Graph = StaticGraph<VertexAttrs<LatLngAttribute>, EdgeAttrs<TraversalCostAttribute>>;

  std::vector<LatLng> coordinates;
  std::ifstream nodes(argv[1]);
  std::string line;
  std::getline(nodes, line);
  while (std::getline(nodes, line)) {
    const auto f = split(line);
    if (f.size() != 3 || std::stoi(f[0]) != static_cast<int>(coordinates.size()))
      throw std::runtime_error("nodes must be dense and zero based");
    coordinates.emplace_back(std::stod(f[1]), std::stod(f[2]));
  }

  std::vector<std::vector<Arc>> outgoing(coordinates.size());
  std::ifstream arcs(argv[2]);
  std::getline(arcs, line);
  while (std::getline(arcs, line)) {
    const auto f = split(line);
    if (f.size() != 6) throw std::runtime_error("invalid equilibrium arc row");
    Arc arc{std::stoi(f[0]), std::stoi(f[1]), std::stod(f[2]), std::stod(f[3]),
            std::stod(f[4]), std::stod(f[5])};
    outgoing.at(arc.tail).push_back(arc);
  }

  Graph graph;
  std::vector<double> freeFlow, alpha, beta, capacity;
  for (int tail = 0; tail < static_cast<int>(coordinates.size()); ++tail) {
    graph.appendVertex(coordinates[tail]);
    for (const auto& arc : outgoing[tail]) {
      graph.appendEdge(arc.head, static_cast<int>(std::llround(COST_SCALE * arc.freeFlow)));
      freeFlow.push_back(arc.freeFlow);
      alpha.push_back(arc.alpha);
      beta.push_back(arc.beta);
      capacity.push_back(arc.capacity);
    }
  }

  std::vector<ClusteredOriginDestination> demand;
  std::ifstream demands(argv[3]);
  std::getline(demands, line);
  while (std::getline(demands, line)) {
    const auto f = split(line);
    demand.emplace_back(std::stoi(f[0]), std::stoi(f[1]), 0, 0, std::stod(f[2]));
  }

  using Adapter = trafficassignment::CCHAdapter<Graph, TraversalCostAttribute>;
  const auto started = std::chrono::steady_clock::now();
  AllOrNothingAssignment<Adapter> assignment(graph, demand, false);
  const auto preprocessed = std::chrono::steady_clock::now();

  const int numEdges = graph.numEdges();
  std::vector<double> flows(numEdges), aon(numEdges), sd(numEdges), psd(numEdges);
  std::vector<double> costs(numEdges), hessian(numEdges), fwDirection(numEdges);
  std::vector<double> d1(numEdges), d2(numEdges), xValue(numEdges), wValue(numEdges);
  std::vector<double> target(numEdges), direction(numEdges);

  assignment.run();
  for (int e = 0; e < numEdges; ++e) flows[e] = assignment.trafficFlowOn(e);
  sd = flows;
  psd = flows;
  double previousLambda = 1.0;
  double gap = INFINITY;
  int iterations = 0;
  double aonSeconds = assignment.stats.lastRoutingTime / 1000.0;
  double customizationSeconds = assignment.stats.lastCustomizationTime / 1000.0;
#ifdef TA_CCH_PROFILE
  double searchCpuSeconds = assignment.lastSearchTime;
  double loadingCpuSeconds = assignment.lastLoadingTime;
  double mergeSeconds = assignment.lastMergeTime;
  double propagationSeconds = assignment.lastPropagationTime;
#endif

  constexpr int MAX_ITERATIONS = 1000;
  constexpr double TOLERANCE = 1e-4;
  for (int k = 0; k < MAX_ITERATIONS; ++k) {
#pragma omp parallel for schedule(static)
    for (int e = 0; e < numEdges; ++e) {
      const double vc = std::max(0.0, flows[e] / std::max(1.0, capacity[e]));
      costs[e] = freeFlow[e] * (1.0 + alpha[e] * std::pow(vc, beta[e]));
      graph.traversalCost(e) = static_cast<int>(std::llround(COST_SCALE * costs[e]));
    }

    assignment.run();
    aonSeconds += assignment.stats.lastRoutingTime / 1000.0;
    customizationSeconds += assignment.stats.lastCustomizationTime / 1000.0;
#ifdef TA_CCH_PROFILE
    searchCpuSeconds += assignment.lastSearchTime;
    loadingCpuSeconds += assignment.lastLoadingTime;
    mergeSeconds += assignment.lastMergeTime;
    propagationSeconds += assignment.lastPropagationTime;
#endif
#pragma omp parallel for schedule(static)
    for (int e = 0; e < numEdges; ++e) {
      aon[e] = assignment.trafficFlowOn(e);
      fwDirection[e] = aon[e] - flows[e];
    }
    const double tx = dot(costs, flows);
    const double ty = dot(costs, aon);
    gap = (tx - ty) / std::max(tx, 1.0);
    iterations = k + 1;
    if (gap < TOLERANCE) break;

    target = aon;
    direction = fwDirection;
    if (k >= 1) {
#pragma omp parallel for schedule(static)
      for (int e = 0; e < numEdges; ++e) {
        const double cap = std::max(1.0, capacity[e]);
        const double vc = std::max(0.0, flows[e] / cap);
        hessian[e] = freeFlow[e] * alpha[e] * beta[e] *
                     std::pow(vc, beta[e] - 1.0) / cap;
        d1[e] = sd[e] - flows[e];
      }
      if (k == 1 || previousLambda >= 1.0 - 1e-8) {
        double q = 0, r = 0;
#pragma omp parallel for reduction(+ : q, r) schedule(static)
        for (int e = 0; e < numEdges; ++e) {
          q += hessian[e] * d1[e] * fwDirection[e];
          r += hessian[e] * d1[e] * d1[e];
        }
        const double den = q - r;
        const double blend = std::abs(den) < 1e-12 ? 0.0 :
            std::max(0.0, std::min(0.99999, q / den));
#pragma omp parallel for schedule(static)
        for (int e = 0; e < numEdges; ++e)
          target[e] = (1.0 - blend) * aon[e] + blend * sd[e];
      } else {
        double muNum = 0, muDen = 0, nuNum = 0, nuDen = 0;
#pragma omp parallel for reduction(+ : muNum, muDen, nuNum, nuDen) schedule(static)
        for (int e = 0; e < numEdges; ++e) {
          d2[e] = psd[e] - flows[e];
          xValue[e] = previousLambda * d1[e] + (1.0 - previousLambda) * d2[e];
          wValue[e] = d2[e] - d1[e];
          muNum += hessian[e] * xValue[e] * fwDirection[e];
          muDen += hessian[e] * xValue[e] * wValue[e];
          nuNum += hessian[e] * d1[e] * fwDirection[e];
          nuDen += hessian[e] * d1[e] * d1[e];
        }
        const double mu = std::abs(muDen) > 1e-12 ? std::max(0.0, -muNum / muDen) : 0.0;
        const double nuRaw = std::abs(nuDen) > 1e-12 ?
            -nuNum / nuDen + mu * previousLambda / (1.0 - previousLambda) : 0.0;
        const double nu = std::max(0.0, nuRaw);
        const double b0 = 1.0 / (1.0 + nu + mu);
#pragma omp parallel for schedule(static)
        for (int e = 0; e < numEdges; ++e)
          target[e] = b0 * aon[e] + nu * b0 * sd[e] + mu * b0 * psd[e];
      }
#pragma omp parallel for schedule(static)
      for (int e = 0; e < numEdges; ++e) direction[e] = target[e] - flows[e];
      if (dot(costs, direction) >= 0.0) {
        target = aon;
        direction = fwDirection;
      }
    }

    auto derivative = [&](const double lambda) {
      double result = 0;
#pragma omp parallel for reduction(+ : result) schedule(static)
      for (int e = 0; e < numEdges; ++e) {
        const double x = std::max(0.0, flows[e] + lambda * direction[e]);
        const double vc = x / std::max(1.0, capacity[e]);
        result += freeFlow[e] * (1.0 + alpha[e] * std::pow(vc, beta[e])) * direction[e];
      }
      return result;
    };
    double lambda = 0.0;
    if (derivative(0.0) < 0.0) {
      if (derivative(1.0) <= 0.0) {
        lambda = 1.0;
      } else {
        double lo = 0.0, hi = 1.0;
        const double xTolerance = std::max(std::min(1e-6, gap * 1e-5), 1e-12);
        while (hi - lo > xTolerance) {
          const double mid = (lo + hi) / 2.0;
          if (derivative(mid) <= 0.0) lo = mid; else hi = mid;
        }
        lambda = (lo + hi) / 2.0;
      }
    }
    psd = sd;
    sd = target;
    previousLambda = lambda;
#pragma omp parallel for schedule(static)
    for (int e = 0; e < numEdges; ++e)
      flows[e] = std::max(0.0, flows[e] + lambda * direction[e]);
  }

  double objective = 0;
#pragma omp parallel for reduction(+ : objective) schedule(static)
  for (int e = 0; e < numEdges; ++e) {
    const double cap = std::max(1.0, capacity[e]);
    objective += freeFlow[e] * (flows[e] + alpha[e] * std::pow(flows[e], beta[e] + 1.0) /
        ((beta[e] + 1.0) * std::pow(cap, beta[e])));
  }
  const auto finished = std::chrono::steady_clock::now();
  const auto seconds = [](const auto begin, const auto end) {
    return std::chrono::duration<double>(end - begin).count();
  };
  std::cout << std::setprecision(17)
            << "{\"nodes\":" << graph.numVertices()
            << ",\"arcs\":" << graph.numEdges()
            << ",\"od_pairs\":" << demand.size()
            << ",\"iterations\":" << iterations
            << ",\"relative_gap\":" << gap
            << ",\"objective\":" << objective
            << ",\"preprocess_seconds\":" << seconds(started, preprocessed)
            << ",\"routing_seconds\":" << aonSeconds
            << ",\"customization_seconds\":" << customizationSeconds
#ifdef TA_CCH_PROFILE
            << ",\"search_cpu_seconds\":" << searchCpuSeconds
            << ",\"loading_cpu_seconds\":" << loadingCpuSeconds
            << ",\"merge_wall_seconds\":" << mergeSeconds
            << ",\"propagation_wall_seconds\":" << propagationSeconds
#endif
            << ",\"solve_seconds\":" << seconds(preprocessed, finished)
            << ",\"total_seconds\":" << seconds(started, finished) << "}\n";
}
