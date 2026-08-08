#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <vector>

#include "DataStructures/Utilities/OriginDestination.h"
#include "Stats/TrafficAssignment/AllOrNothingAssignmentStats.h"
#include "Tools/CommandLine/ProgressBar.h"
#include "Tools/Simd/AlignedVector.h"
#include "Tools/Timer.h"

// Implementation of an iterative all-or-nothing traffic assignment. Each OD pair is processed in
// turn and the corresponding OD flow (in our case always a single flow unit) is assigned to each
// edge on the shortest path between O and D. Other O-D paths are not assigned any flow. The
// procedure can be used with different shortest-path algorithms.
template <typename ShortestPathAlgoT>
class AllOrNothingAssignment {
 private:
  using InputGraph = typename ShortestPathAlgoT::InputGraph;

 public:
  // Constructs an all-or-nothing assignment instance.
  AllOrNothingAssignment(const InputGraph& graph,
                         const std::vector<ClusteredOriginDestination>& odPairs,
                         const bool verbose = true)
      : stats(odPairs.size()),
        shortestPathAlgo(graph),
        inputGraph(graph),
        odPairs(odPairs),
        verbose(verbose) {
    Timer timer;
    shortestPathAlgo.preprocess();
    stats.totalPreprocessingTime = timer.elapsed();
    stats.lastRoutingTime = stats.totalPreprocessingTime;
    stats.totalRoutingTime = stats.totalPreprocessingTime;
    if (verbose) std::cout << "  Prepro: " << stats.totalPreprocessingTime << "ms" << std::endl;
  }

  // Assigns all OD flows to their currently shortest paths.
  void run(const int skipInterval = 1) {
    Timer timer;
    ++stats.numIterations;
    if (verbose) std::cout << "Iteration " << stats.numIterations << ": " << std::flush;
    shortestPathAlgo.customize();
    stats.lastCustomizationTime = timer.elapsed();

    timer.restart();
    ProgressBar bar(std::ceil(1.0 * odPairs.size() / (K * skipInterval)), verbose);
    trafficFlows.assign(inputGraph.numEdges(), 0);
    stats.startIteration();
#ifdef TA_CCH_PROFILE
    lastSearchTime = 0;
    lastLoadingTime = 0;
    lastMergeTime = 0;
    lastPropagationTime = 0;
#endif
    auto totalNumPairsSampledBefore = 0;
    #pragma omp parallel
    {
      auto queryAlgo = shortestPathAlgo.getQueryAlgoInstance();
      auto checksum = int64_t{0};
      auto prevMinPathCost = int64_t{0};
      auto avgChange = 0.0;
      auto maxChange = 0.0;
      auto numPairsSampledBefore = 0;

      #pragma omp for schedule(dynamic, 64) nowait
      for (auto i = 0; i < odPairs.size(); i += K * skipInterval) {
        // Run multiple shortest-path computations simultaneously.
        std::array<int, K> sources;
        std::array<int, K> targets;
        std::array<double, K> volumes;
        sources.fill(odPairs[i].origin);
        targets.fill(odPairs[i].destination);
        volumes.fill(odPairs[i].volume * skipInterval);
        auto k = 1;
        for (; k < K && i + k * skipInterval < odPairs.size(); ++k) {
          sources[k] = odPairs[i + k * skipInterval].origin;
          targets[k] = odPairs[i + k * skipInterval].destination;
          volumes[k] = odPairs[i + k * skipInterval].volume * skipInterval;
        }
        queryAlgo.run(sources, targets, volumes, k);

        for (auto j = 0; j < k; ++j) {
          // Maintain the avg and max change in the OD distances between the last two iterations.
          const auto dst = odPairs[i + j * skipInterval].destination;
          const auto dist = queryAlgo.getDistance(dst, j);
          const auto prevDist = stats.lastDistances[i + j * skipInterval];
          const auto change = 1.0 * std::abs(dist - prevDist) / prevDist;
          numPairsSampledBefore += prevDist != -1;
          checksum += dist;
          prevMinPathCost += prevDist != -1 ? dist : 0;
          stats.lastDistances[i + j * skipInterval] = dist;
          avgChange += std::max(0.0, change);
          maxChange = std::max(maxChange, change);
        }
        ++bar;
      }

      #pragma omp critical (combineResults)
      {
#ifdef TA_CCH_PROFILE
        lastSearchTime += queryAlgo.getSearchTime();
        lastLoadingTime += queryAlgo.getLoadingTime();
        const auto mergeStarted = std::chrono::steady_clock::now();
#endif
        queryAlgo.addLocalToGlobalFlows();
#ifdef TA_CCH_PROFILE
        lastMergeTime += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mergeStarted).count();
#endif
        stats.lastChecksum += checksum;
        stats.prevMinPathCost += prevMinPathCost;
        stats.avgChangeInDistances += avgChange;
        stats.maxChangeInDistances = std::max(stats.maxChangeInDistances, maxChange);
        totalNumPairsSampledBefore += numPairsSampledBefore;
      }
    }
    bar.finish();

#ifdef TA_CCH_PROFILE
    const auto propagationStarted = std::chrono::steady_clock::now();
#endif
    shortestPathAlgo.propagateFlowsToInputEdges(trafficFlows);
#ifdef TA_CCH_PROFILE
    lastPropagationTime = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - propagationStarted).count();
#endif
    stats.lastQueryTime = timer.elapsed();
    stats.avgChangeInDistances /= totalNumPairsSampledBefore;
    stats.finishIteration();

    if (verbose) {
      std::cout << " done.\n";
      std::cout << "  Checksum: " << stats.lastChecksum;
      std::cout << "  Custom: " << stats.lastCustomizationTime << "ms";
      std::cout << "  Queries: " << stats.lastQueryTime << "ms";
      std::cout << "  Routing: " << stats.lastRoutingTime << "ms\n";
      std::cout << std::flush;
    }
  }

  // Returns the traffic flow on edge e.
  const double& trafficFlowOn(const int e) const {
    assert(e >= 0); assert(e < inputGraph.numEdges());
    return trafficFlows[e];
  }

  AllOrNothingAssignmentStats stats; // Statistics about the execution.
#ifdef TA_CCH_PROFILE
  double lastSearchTime = 0;
  double lastLoadingTime = 0;
  double lastMergeTime = 0;
  double lastPropagationTime = 0;
#endif

 private:
  // The maximum number of simultaneous shortest-path computations.
  static constexpr int K = ShortestPathAlgoT::K;

  using ODPairs = std::vector<ClusteredOriginDestination>;

  ShortestPathAlgoT shortestPathAlgo; // Algorithm computing shortest paths between OD pairs.
  const InputGraph& inputGraph;       // The input graph.
  const ODPairs& odPairs;             // The OD pairs to be assigned onto the graph.
  AlignedVector<double> trafficFlows;    // The traffic flows on the edges.
  const bool verbose;                 // Should informative messages be displayed?
};
