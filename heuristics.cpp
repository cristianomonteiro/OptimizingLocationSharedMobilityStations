#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <iomanip>

std::unordered_map<int, std::unordered_set<int>>& processFile(const std::string& filename) {
    static std::unordered_map<int, std::unordered_set<int>> mapEdgesNearby;
    // Clears the map because it is static and so values of earlier runs may still exist
    mapEdgesNearby.clear();

    // Open the file
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return mapEdgesNearby;
    }

    std::string line;
    int lineCount = 0;

    // Count the number of lines in the file
    while (std::getline(inputFile, line)) {
        lineCount++;
    }

    // Reset the file position to the beginning
    inputFile.clear();
    inputFile.seekg(0, std::ios::beg);

    // Reserve space in the mapEdgesNearby to avoid rehashing
    mapEdgesNearby.reserve(lineCount);

    // Read each line of the file
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int vertexU;

        if (iss >> vertexU) {
            // Create an unordered_set to store the integers in the line
            std::unordered_set<int> intSet;

            // Add each integer to the set
            int num;
            while (iss >> num) {
                intSet.insert(num);
            }

            // Add the first integer and its corresponding set to the map
            mapEdgesNearby[vertexU] = intSet;
        }
    }

    // Close the file
    inputFile.close();

    return mapEdgesNearby;
}

std::vector<std::tuple<int, int, int, float, float>>& getEdgesOrdger(const std::string& filename) {
    static std::vector<std::tuple<int, int, int, float, float>> edgesOrder;
    // Clears the map because it is static and so values of earlier runs may still exist
    edgesOrder.clear();

    // Open the file
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return edgesOrder;
    }

    std::string line;
    // Read each line of the file
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int vertexU, vertexV, edgeKey;
        float edgeUtility, edgeCost;

        if (iss >> vertexU >> vertexV >> edgeKey >> edgeUtility >> edgeCost) {
            edgesOrder.push_back(std::make_tuple(vertexU, vertexV, edgeKey, edgeUtility, edgeCost));
        }
    }

    // Close the file
    inputFile.close();

    return edgesOrder;
}

float mergeSets(std::unordered_map<int, std::unordered_set<int>>& mapEdgesNearby, std::vector<std::tuple<int, int, int, float, float>>& edgesOrder, float budget) {
    std::unordered_set<int> forbiddenEdges;
    float objectiveFunction = 0.0, totalCost = 0.0;

    int vertexU, vertexV, edgeKey;
    float edgeUtility, edgeCost;
    for (int i = 0; i < edgesOrder.size(); i++) {
        std::tie(vertexU, vertexV, edgeKey, edgeUtility, edgeCost) = edgesOrder[i];
        if (edgeCost + totalCost > budget) {
            continue;
        }
        // Get the set from mapEdgesNearby based on the edge's key
        auto it = mapEdgesNearby.find(edgeKey);
        if (it != mapEdgesNearby.end()) {
            std::unordered_set<int>& edgesNearby = it->second;

            // Check if there is an intersection between edgesNearby and forbiddenEdges
            bool hasIntersection = false;
            for (const auto& value : edgesNearby) {
                if (forbiddenEdges.count(value)) {
                    hasIntersection = true;
                    break;
                }
            }

            if (!hasIntersection) {
                // Insert values from edgesNearby into forbiddenEdges
                forbiddenEdges.insert(edgesNearby.begin(), edgesNearby.end());

                // Sum the float values from the utylity value and edge cost
                objectiveFunction += edgeUtility;
                totalCost += edgeCost;
            }
        }
    }

    //std::cout << "Objective Function: " << objectiveFunction << std::endl;    
    //std::cout << "Rate of edges/forbidden edges: " << (float)forbiddenEdges.size() / (float)mapEdgesNearby.size() << std::endl;
    return objectiveFunction;
}

void runWriteMetrics(std::unordered_map<int, std::unordered_set<int>>& mapEdgesNearby, const float splitMultiplier,
                    const std::string& filenameOrder, const std::string& filenameWrite, float budget) {
    std::string rootFolder = "heuristics_results/";
    std::string folderObjectiveValues = rootFolder + "objective_values/";
    std::string folderRunTimes = rootFolder + "run_times/";

    // Open the files
    std::ofstream outputFileObjectiveValue(folderObjectiveValues + filenameWrite, std::ios_base::app);
    std::ofstream outputFileRunTimes(folderRunTimes + filenameWrite, std::ios_base::app);
    if (!outputFileObjectiveValue.is_open() || !outputFileRunTimes.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
    }
    
    std::vector<std::tuple<int, int, int, float, float>>& edgesOrder = getEdgesOrdger(filenameOrder);

    auto startTime = std::chrono::high_resolution_clock::now();
    float objectiveValue = mergeSets(mapEdgesNearby, edgesOrder, budget);
    auto stopTime = std::chrono::high_resolution_clock::now();
    auto runTime = std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime);

    outputFileObjectiveValue << splitMultiplier << " " << std::fixed << objectiveValue << std::endl;
    outputFileRunTimes << splitMultiplier << " " << std::fixed << runTime.count() << std::endl;

    // Close the files
    outputFileObjectiveValue.close();
    outputFileRunTimes.close();
}

int main() {
    // unsync the I/O
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // Defining constants
    const int spaceBetweenStations = 200;
    const float infinity = std::numeric_limits<float>::max();
    const int basePower = 2;

    // Running for each edges split multiplier
    for (float splitMultiplier = 0; splitMultiplier <= 9; splitMultiplier += 0.5) {
        std::string strSplitMultiplier;
        // Checking if splitMultiplier has an integer value
        if (floor(splitMultiplier) == splitMultiplier) {
            strSplitMultiplier = std::to_string((int)floor(splitMultiplier));
        } else {
            std::stringstream strStream;
            strStream << std::fixed << std::setprecision(1) << splitMultiplier;
            strSplitMultiplier = strStream.str();
        }

        const std::string suffixFileInput = "_" + strSplitMultiplier + "_" + std::to_string(spaceBetweenStations);
        const std::string filenameMaxHeuristicOrder = "max_utility_order" + suffixFileInput;
        const std::string filenameDijkstraHeuristicOrder = "dijkstra_order" + suffixFileInput;
        const std::string costsSuffix = "_costs";

        //std::cout << "Edges multiplier: " << splitMultiplier << std::endl;
        
        // Populate mapEdgesNearby with some initial values
        std::string filenameEdgesNearby = "edges_nearby" + suffixFileInput + ".txt";
        std::unordered_map<int, std::unordered_set<int>>& mapEdgesNearby = processFile(filenameEdgesNearby);

        // Running many times to measure time spent
        for (int k = 1; k <= 40; k++) {
            runWriteMetrics(mapEdgesNearby, splitMultiplier, filenameMaxHeuristicOrder + ".txt", filenameMaxHeuristicOrder + ".txt", infinity);
            runWriteMetrics(mapEdgesNearby, splitMultiplier, filenameDijkstraHeuristicOrder + ".txt", filenameDijkstraHeuristicOrder + ".txt", infinity);

            // Running considering costs
            for (int power = 0; power <= 29; power++) {
                float budget = pow(basePower, power);
                //std::cout << "Budget: " << budget << ", power: " << power << std::endl;

                std::string powerString = "_" + std::to_string(power);
                runWriteMetrics(mapEdgesNearby, splitMultiplier, filenameMaxHeuristicOrder + ".txt", filenameMaxHeuristicOrder + powerString + costsSuffix + ".txt", budget);
                runWriteMetrics(mapEdgesNearby, splitMultiplier, filenameDijkstraHeuristicOrder + ".txt", filenameDijkstraHeuristicOrder + powerString + costsSuffix + ".txt", budget);
            }
        
            //std::cout << std::endl << std::endl;
        }
    }
    return 0;
}
