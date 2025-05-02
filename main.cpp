#include <mpi.h>
#include "header/MPIHandler.h"
#include "header/SIRModel.h"
#include "header/CSVParser.h"
#include "header/GridSimulation.h"
#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <string>
#include <unordered_map> // Keep for neighbor maps
#include <unordered_set> // Needed for buildBlockNeighborMap

// --- Restored Function Implementations ---

// Builds neighbor map for a simple 2D grid ( toroidal or not depending on logic )
std::unordered_map<int, std::vector<int>> build2DGridNeighborMap(
    int rows, int cols,
    const std::unordered_map<int, int>& cellToBlock,
    std::unordered_map<int, std::vector<int>>& ghostNeighbors
) {
    std::unordered_map<int, std::vector<int>> neighbors;

    int totalCells = rows * cols;
    for (int i = 0; i < totalCells; ++i) {
        if (cellToBlock.find(i) == cellToBlock.end()) continue;

        int row = i / cols;
        int col = i % cols;

        std::vector<int> neighborList;
        std::vector<int> directions = {
            (row > 0)        ? (i - cols) : -1,
            (row < rows - 1) ? (i + cols) : -1,
            (col > 0)        ? (i - 1)    : -1,
            (col < cols - 1) ? (i + 1)    : -1
        };

        for (int neighborId : directions) {
            if (neighborId != -1 && cellToBlock.find(neighborId) != cellToBlock.end()) {
                if (cellToBlock.at(i) == cellToBlock.at(neighborId)) {
                    neighborList.push_back(neighborId);
                } else {
                    ghostNeighbors[i].push_back(neighborId);
                }
            }
        }

        neighbors[i] = neighborList;
    }

    return neighbors;
}

// Builds a map of adjacent blocks based on cell adjacency
std::unordered_map<int, std::vector<int>> buildBlockNeighborMap(
    const std::map<int, std::list<int>> &allBlocks, // Needs blocks map
    const std::unordered_map<int, std::vector<int>> &cellNeighborMap)
{ // Needs cell neighbor map

    std::unordered_map<int, std::vector<int>> blockNeighborMap;
    if (allBlocks.empty() || cellNeighborMap.empty())
    {
        return blockNeighborMap; // Return empty if input is empty
    }

    // Efficiently map cell ID -> block ID
    std::unordered_map<int, int> cellToBlockMap;
    for (const auto &[blockId, cellList] : allBlocks)
    {
        for (int cellId : cellList)
        {
            cellToBlockMap[cellId] = blockId;
        }
    }

    // Determine block neighbors
    for (const auto &[blockId, cellList] : allBlocks)
    {
        std::unordered_set<int> neighborBlockIdsSet; // Use set to automatically handle duplicates
        for (int cellId : cellList)
        {
            // Check neighbors of this cell
            auto it_neighbors = cellNeighborMap.find(cellId);
            if (it_neighbors != cellNeighborMap.end())
            {
                for (int neighborCellId : it_neighbors->second)
                {
                    // Find which block the neighbor cell belongs to
                    auto it_block = cellToBlockMap.find(neighborCellId);
                    if (it_block != cellToBlockMap.end())
                    {
                        int neighborBlockId = it_block->second;
                        // Add if it's a different block
                        if (neighborBlockId != blockId)
                        {
                            neighborBlockIdsSet.insert(neighborBlockId);
                        }
                    }
                }
            }
        }
        // Convert set to vector for the final map
        if (!neighborBlockIdsSet.empty())
        {
            blockNeighborMap[blockId] = std::vector<int>(neighborBlockIdsSet.begin(), neighborBlockIdsSet.end());
        }
    }
    return blockNeighborMap; // Make sure to return the map
}

// --- End Restored Functions ---

int main(int argc, char *argv[])
{
    const int blockSize = 4;
    MPIHandler mpi(argc, argv);

    if (argc < 1) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create SIR model
    SIRModel model(0.3, 0.1, 0.2, 100);

    // Load data and distribute it
    std::vector<std::vector<double>> fullData;
    std::map<int, std::list<int>> allBlocks;
    std::map<std::string, int> cells;
    std::unordered_map<int, std::vector<int>> blockNeighborMap;
    std::unordered_map<int, int> cellToBlock;

    if (mpi.getRank() == 0)
    {
        fullData = CSVParser::loadUSStateData("./data/sorted_initial_conditions.csv");
    }

    std::vector<SIRCell> localGrid = mpi.distributeData(fullData, [](const std::vector<double>& rowData) {
        return CSVParser::mapToSIR(rowData); // Call static method from CSVParser
    });
        // Debug print block neighbors
        for (const auto &[blockId, neighborBlocks] : blockNeighborMap)
        {
            std::cout << "Block " << blockId << " neighbors: ";
            for (int neighborBlock : neighborBlocks)
            {
                std::cout << neighborBlock << " ";
            }
            std::cout << "\n";
        }
    }

    // Create and configure simulation
    GridSimulation simulation(model, mpi.getRank(), mpi.getSize());
    simulation.initialize(localGrid, mpi.getSize());

    // Run the simulation
    // Create and broadcast blockToRankMap
    std::unordered_map<int, int> blockToRankMap;
    if (mpi.getRank() == 0)
    {
        int totalBlocks = allBlocks.size(); // Use size determined by rank 0
        int blockCounter = 0;
        for (const auto &[blockId, _] : allBlocks)
        { // Iterate in consistent order if needed
            int targetRank = -1;
            // Recalculate distribution logic to find owner rank for each block
            int tempBlocksPerProc = totalBlocks / mpi.getSize();
            int tempExtraBlocks = totalBlocks % mpi.getSize();
            for (int r = 0; r < mpi.getSize(); ++r)
            {
                int rNumBlocks = (r < tempExtraBlocks) ? (tempBlocksPerProc + 1) : tempBlocksPerProc;
                int rankStartBlockIndex = (r < tempExtraBlocks) ? (r * (tempBlocksPerProc + 1)) : (r * tempBlocksPerProc + tempExtraBlocks);
                if (blockCounter >= rankStartBlockIndex && blockCounter < rankStartBlockIndex + rNumBlocks)
                {
                    targetRank = r;
                    break;
                }
            }
            if (targetRank != -1)
            {
                blockToRankMap[blockId] = targetRank;
            }
            else
            {
                std::cerr << "Rank 0 Error: Could not determine owner rank for block " << blockId << " (index " << blockCounter << ")" << std::endl;
            }
            blockCounter++;
        }
    }


    int map_size = (mpi.getRank() == 0) ? blockToRankMap.size() : 0;
    MPI_Bcast(&map_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> map_data(map_size * 2);
    if (mpi.getRank() == 0) {
        int i = 0;
        for (const auto& pair : blockToRankMap) {
            map_data[i++] = pair.first;
            map_data[i++] = pair.second;
        }
    }
    MPI_Bcast(map_data.data(), map_size * 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi.getRank() != 0) {
        for (int i = 0; i < map_size * 2; i += 2) {
            blockToRankMap[map_data[i]] = map_data[i + 1];
        }
    }
    simulation.setBlockToRankMap(blockToRankMap);

    simulation.setGridFromLocalData(localBlocks, localCellData);
    simulation.setBlockInfo(localBlocks, blockNeighborMap);

    std::vector<std::vector<double>> localResults = simulation.runSimulation();
    std::vector<double> globalResults = mpi.gatherResults(localResults);
    int numLocalSteps = localResults.size();
    mpi.writeResults(globalResults, numLocalSteps);

    // Gather and write results
    std::vector<double> globalFlatResults = mpi.gatherResults(localResults);
    if (mpi.getRank() == 0) {
        mpi.writeResults(globalFlatResults, localResults.size());
    }

    return 0;
}
