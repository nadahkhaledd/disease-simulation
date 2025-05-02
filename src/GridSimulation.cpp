#include "../header/GridSimulation.h"
#include "../header/SIRCell.h"
#include "../header/SIRModel.h"
#include "../header/CSVParser.h"

#include <mpi.h>
#include <vector>
#include <map>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <array>
#include <cstdio>
#include <memory>

std::string getCurrentDirectory() {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("pwd", "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result;
}

GridSimulation::GridSimulation(const SIRModel& m, int mpiRank, int mpiSize)
    : model(m), rank(mpiRank), size(mpiSize) {}

std::vector<SIRCell>& GridSimulation::getGrid() {
    return grid;
}

int GridSimulation::getLocalSize() const {
    return static_cast<int>(grid.size());
}

void GridSimulation::setGhostNeighborMap(const std::unordered_map<int, std::vector<int>>& map) {
    ghostNeighborMap = map;
}

void GridSimulation::setGrid(const std::vector<SIRCell>& initialGrid) {
    grid = initialGrid;
    std::cerr << "Rank " << rank << " Warning: setGrid called directly. MPI mappings might be invalid." << std::endl;
    globalToLocalCellIndex.clear();
        // Use model to compute update using neighbors
        newGrid[i] = model.rk4StepWithNeighbors(grid[i], neighbors);
    }

    // Normalize the grid to ensure S + I + R = 1 for each cell
    for (auto& cell : newGrid) {
        double sum = cell.getS() + cell.getI() + cell.getR();
        if (sum > 0) {
            cell.setS(cell.getS() / sum);
            cell.setI(cell.getI() / sum);
            cell.setR(cell.getR() / sum);
        } else {
            cell.setS(1.0);
            cell.setI(0.0);
            cell.setR(0.0);
        }
    }

    grid = newGrid;
}

std::map<std::string, int> GridSimulation::createCellsMap() {
    std::map<std::string, int> cells;
    std::string filePath;
    try {
        std::string currentDir = getCurrentDirectory();
        filePath = currentDir + "/data/sorted_initial_conditions.csv";
    } catch (const std::runtime_error& e) {
        std::cerr << "Error getting current directory: " << e.what() << ". Cannot create cells map." << std::endl;
        return {};
    }
    std::ifstream infile(filePath);
    if (!infile) {
        std::cerr << "Error: Could not open file for createCellsMap: " << filePath << "\n";
        return {};
    }
    std::string line;
    int cellId = 0;
    bool headerFound = false;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        if (!headerFound && line.find("Province_State") != std::string::npos) {
            headerFound = true;
            continue;
        }
        if (!headerFound) continue;
        std::istringstream ss(line);
        std::string stateOrIdentifier;
        if (std::getline(ss, stateOrIdentifier, ',')) {
            stateOrIdentifier.erase(0, stateOrIdentifier.find_first_not_of(" \t\n\r\f\v"));
            stateOrIdentifier.erase(stateOrIdentifier.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!stateOrIdentifier.empty()) {
                cells[stateOrIdentifier] = cellId++;
            }
        }
    }
    infile.close();
    return cells;
}

std::map<int, std::list<int>> GridSimulation::divideIntoBlocks(
    const std::map<std::string, int>& cells, int numBlocks) {
    std::map<int, std::list<int>> blocks;
    if (blockSize <= 0 || cells.empty()) return blocks;
    std::vector<int> sortedCellIds;
    for (const auto& pair : cells) sortedCellIds.push_back(pair.second);
    std::sort(sortedCellIds.begin(), sortedCellIds.end());

    // Calculate block size dynamically
    int blockSize = sortedCellIds.size() / numBlocks;
    int extra = sortedCellIds.size() % numBlocks;

    // Assign sorted cell IDs to blocks
    int startIndex = 0;
    for (int blockId = 0; blockId < numBlocks; ++blockId) {
        int currentBlockSize = blockSize + (blockId < extra ? 1 : 0); // Distribute remainder evenly
        for (int i = 0; i < currentBlockSize; ++i) {
            blocks[blockId].push_back(sortedCellIds[startIndex++]);
        }
    }

    return blocks;
}

void GridSimulation::setGridFromLocalData(
    const std::map<int, std::list<int>>& localBlocks,
    const std::map<int, std::vector<double>>& localCellData) {

    grid.clear();
    globalToLocalCellIndex.clear();

    std::vector<int> sortedBlockIds;
    for (const auto& pair : localBlocks) sortedBlockIds.push_back(pair.first);
    std::sort(sortedBlockIds.begin(), sortedBlockIds.end());

    std::vector<int> orderedCellIds;
    for (int blockId : sortedBlockIds) {
        const auto& cellList = localBlocks.at(blockId);
        orderedCellIds.insert(orderedCellIds.end(), cellList.begin(), cellList.end());
    }

    grid.reserve(orderedCellIds.size());
    int localIndex = 0;
    for (int cellId : orderedCellIds) {
        auto it = localCellData.find(cellId);
        if (it != localCellData.end()) {
            grid.push_back(CSVParser::mapToSIR(it->second));
            globalToLocalCellIndex[cellId] = localIndex++;
        }
    }
}

void GridSimulation::setBlockInfo(
    const std::map<int, std::list<int>>& localBlocks,
    const std::unordered_map<int, std::vector<int>>& blockNeighbors) {

    localBlockMap = localBlocks;
    blockNeighborMap = blockNeighbors;

    if (!grid.empty()) {
        std::unordered_map<int, int> tempIndexMap;
        std::vector<int> orderedCellIds;
        std::vector<int> sortedBlockIds;
        for (const auto& pair : localBlocks) sortedBlockIds.push_back(pair.first);
        std::sort(sortedBlockIds.begin(), sortedBlockIds.end());
        for (int blockId : sortedBlockIds) {
            const auto& cellList = localBlocks.at(blockId);
            orderedCellIds.insert(orderedCellIds.end(), cellList.begin(), cellList.end());
        }
        for (size_t i = 0; i < orderedCellIds.size(); ++i) {
            tempIndexMap[orderedCellIds[i]] = static_cast<int>(i);
        }
        globalToLocalCellIndex = std::move(tempIndexMap);
    } else {
        globalToLocalCellIndex.clear();
    }
}

void GridSimulation::setCellNeighborMap(const std::unordered_map<int, std::vector<int>>& map) {
    cellNeighborMap = map;
}

void GridSimulation::setBlockToRankMap(const std::unordered_map<int, int>& map) {
    blockToRankMap = map;
}

void GridSimulation::updateGrid() {
    if (grid.empty()) return;
    std::vector<SIRCell> newGrid = grid;
    for (size_t i = 0; i < grid.size(); ++i) {
        newGrid[i] = model.rk4Step(grid[i]);
    }
    grid = std::move(newGrid);
}

void GridSimulation::updateGridNew() {
    if (grid.empty()) return;
    std::vector<SIRCell> newGrid = grid;
    std::vector<int> localIdxToGlobalId(grid.size(), -1);
    for (const auto& pair : globalToLocalCellIndex) {
        if (pair.second >= 0 && static_cast<size_t>(pair.second) < localIdxToGlobalId.size()) {
            localIdxToGlobalId[pair.second] = pair.first;
        }
    }
    for (size_t i = 0; i < grid.size(); ++i) {
        std::vector<SIRCell> neighbors;
        int globalId = localIdxToGlobalId[i];
        if (globalId != -1 && cellNeighborMap.count(globalId)) {
            for (int neighborGlobalID : cellNeighborMap.at(globalId)) {
                if (globalToLocalCellIndex.count(neighborGlobalID)) {
                    neighbors.push_back(grid[globalToLocalCellIndex.at(neighborGlobalID)]);
                }
            }
        }
        newGrid[i] = model.rk4StepWithNeighbors(grid[i], neighbors);
    }
    grid = std::move(newGrid);
}

std::map<int, std::list<int>> GridSimulation::divideIntoOptimalBlocks(
    const std::map<std::string, int>& cells, int numProcesses) {
    int totalCells = static_cast<int>(cells.size());

    std::cout << "Finding optimal block distribution for " << totalCells << " cells...\n";

    // Find all divisors of totalCells
    std::vector<int> divisors;
    for (int i = 1; i <= totalCells / 2; ++i) {
        if (totalCells % i == 0) {
            divisors.push_back(i);
        }
    }
    divisors.push_back(totalCells); // Add totalCells itself as a divisor

    // Define a target range for cells per block (not too few, not too many)
    // For 50 cells, a good range might be 5-10 cells per block
    const int minCellsPerBlock = 2;
    const int maxCellsPerBlock = 10;

    // Find divisors that give us cells per block within our target range
    std::vector<std::pair<int, int>> validConfigurations; // (blocks, cellsPerBlock)
    for (int blocks : divisors) {
        int cellsPerBlock = totalCells / blocks;
        if (cellsPerBlock >= minCellsPerBlock && cellsPerBlock <= maxCellsPerBlock) {
            validConfigurations.push_back({blocks, cellsPerBlock});
        }
    }

    // If no valid configurations found, relax constraints
    if (validConfigurations.empty()) {
        for (int blocks : divisors) {
            int cellsPerBlock = totalCells / blocks;
            validConfigurations.push_back({blocks, cellsPerBlock});
        }
    }

    // Score each valid configuration
    int bestNumBlocks = 1;
    double bestScore = -1.0;

    for (const auto& [blocks, cellsPerBlock] : validConfigurations) {
        // Calculate how "square" the grid of blocks would be
        int blockRows = static_cast<int>(std::sqrt(blocks));
        while (blocks % blockRows != 0) {
            blockRows--;
        }
        int blockCols = blocks / blockRows;

        // Calculate how "square" each block would be
        int cellRows = static_cast<int>(std::sqrt(cellsPerBlock));
        while (cellsPerBlock % cellRows != 0) {
            cellRows--;
        }
        int cellCols = cellsPerBlock / cellRows;

        // Calculate final grid dimensions
        int totalRows = blockRows * cellRows;
        int totalCols = blockCols * cellCols;

        // Score based on:
        // 1. How close the grid is to being square (ratio of 1.0 is perfect)
        // 2. How close each block is to being square
        // 3. Preference for more blocks (but not too many)
        double gridRatio = static_cast<double>(totalRows) / totalCols;
        if (gridRatio > 1.0) gridRatio = 1.0 / gridRatio; // Ensure ratio is <= 1.0

        double blockRatio = static_cast<double>(blockRows) / blockCols;
        if (blockRatio > 1.0) blockRatio = 1.0 / blockRatio;

        double cellRatio = static_cast<double>(cellRows) / cellCols;
        if (cellRatio > 1.0) cellRatio = 1.0 / cellRatio;

        // Combine factors into a single score
        double score = gridRatio * 0.5 + blockRatio * 0.3 + cellRatio * 0.2;

        // Bonus for having more blocks (but not too many)
        double blockBonus = 0.0;
        if (blocks >= 2 && blocks <= 20) {
            blockBonus = static_cast<double>(blocks) / 20.0; // Max bonus for 20 blocks
        } else if (blocks > 20) {
            blockBonus = 1.0; // Full bonus for > 20 blocks
        }

        score += blockBonus * 0.2; // Add block bonus with 20% weight

        std::cout << "  Option: " << blocks << " blocks with " << cellsPerBlock
                  << " cells each. Grid: " << totalRows << "x" << totalCols
                  << " Score: " << score << "\n";

        // Select the configuration with the highest score
        if (score > bestScore) {
            bestScore = score;
            bestNumBlocks = blocks;
        }
    }

    int cellsPerBlock = totalCells / bestNumBlocks;
    std::cout << "Optimal distribution: " << bestNumBlocks << " blocks with "
              << cellsPerBlock << " cells each.\n";

    // Divide cells into the best number of blocks
    return divideIntoBlocks(cells, bestNumBlocks);
}

std::vector<std::vector<double>> GridSimulation::runSimulation() {
    // ... Full implementation from previous step should be here ...
    // Ensure all std::cout, std::cerr, std::endl uses are valid
    // Ensure the placeholder for findRankOwningCell is handled or replaced
    std::vector<std::vector<double>> localResults;
    if (grid.empty()) {
        std::cout << "Rank " << rank << " has no cells. Entering synchronization loop." << std::endl;
        if (size > 1) {
             for (int step = 0; step < model.getNumSteps(); ++step) {
                 MPI_Barrier(MPI_COMM_WORLD);
             }
        }
        return localResults;
    }
    std::vector<int> localIndexToGlobalId(grid.size());
    for(const auto& [globalId, localIndex] : globalToLocalCellIndex) {
        if (localIndex >= 0 && static_cast<size_t>(localIndex) < grid.size()) {
            localIndexToGlobalId[localIndex] = globalId;
        } else {
             std::cerr << "FATAL Error: Rank " << rank << " Invalid local index " << localIndex << " for global ID " << globalId << std::endl; MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const int DOUBLES_PER_CELL = 3;
    const int MPI_TAG_COUNT = 100;
    const int MPI_TAG_DATA = 200;
    std::cout << "Rank " << rank << " starting simulation loop for " << model.getNumSteps() << " steps." << std::endl;
    for (int step = 0; step < model.getNumSteps(); ++step) {
        std::map<int, std::set<int>> localIndicesToSendToRank;
        std::map<int, std::set<int>> globalIdsToReceiveFromRank;
        for (size_t localIndex = 0; localIndex < grid.size(); ++localIndex) {
            int globalId = localIndexToGlobalId[localIndex];
            if (!cellNeighborMap.count(globalId)) continue;
            const std::vector<int>& neighborGlobalIds = cellNeighborMap.at(globalId);
            for (int neighborGlobalId : neighborGlobalIds) {
                if (globalToLocalCellIndex.find(neighborGlobalId) == globalToLocalCellIndex.end()) {
                    int owningRank = -1; bool foundOwner = false;
                    // --- REPLACE THIS WITH EFFICIENT LOOKUP ---
                     for(const auto& [b_id, b_rank] : blockToRankMap) { if (b_rank == rank) continue; /* if (isCellInBlock(...)) { owningRank=b_rank; foundOwner=true; break; } */ }
                    // --- END REPLACE ---
                    if (foundOwner && owningRank != -1) {
                        localIndicesToSendToRank[owningRank].insert(static_cast<int>(localIndex));
                        globalIdsToReceiveFromRank[owningRank].insert(neighborGlobalId);
                    } else if (size > 1 && !foundOwner) { /* Optional Warning */ }
                }
            }
        }
        std::map<int, int> sendCounts; std::map<int, int> recvCounts;
        std::vector<MPI_Request> countSendRequests, countRecvRequests;
        std::map<int, int> recvCountBuffers;
        for (const auto& pair : globalIdsToReceiveFromRank) { int neighborRank = pair.first; if (neighborRank >= 0 && neighborRank < size && neighborRank != rank) { recvCounts[neighborRank] = 0; recvCountBuffers[neighborRank] = 0; MPI_Request req = MPI_REQUEST_NULL; MPI_Irecv(&recvCountBuffers[neighborRank], 1, MPI_INT, neighborRank, MPI_TAG_COUNT + neighborRank, MPI_COMM_WORLD, &req); if (req != MPI_REQUEST_NULL) countRecvRequests.push_back(req); } }
        std::map<int, int> sendCountBuffers;
        for (const auto& pair : localIndicesToSendToRank) { int neighborRank = pair.first; const auto& localIndicesSet = pair.second; if (neighborRank >= 0 && neighborRank < size && neighborRank != rank) { sendCounts[neighborRank] = localIndicesSet.size(); sendCountBuffers[neighborRank] = sendCounts[neighborRank]; MPI_Request req = MPI_REQUEST_NULL; MPI_Isend(&sendCountBuffers[neighborRank], 1, MPI_INT, neighborRank, MPI_TAG_COUNT + rank, MPI_COMM_WORLD, &req); if (req != MPI_REQUEST_NULL) countSendRequests.push_back(req); } }
        if (!countRecvRequests.empty()) { MPI_Waitall(countRecvRequests.size(), countRecvRequests.data(), MPI_STATUSES_IGNORE); for(auto const& [nRank, bufferVal] : recvCountBuffers) { recvCounts[nRank] = bufferVal; } }
        if (!countSendRequests.empty()) { MPI_Waitall(countSendRequests.size(), countSendRequests.data(), MPI_STATUSES_IGNORE); }
        std::vector<MPI_Request> dataSendRequests, dataRecvRequests;
        std::map<int, std::vector<double>> sendDataBuffers; std::map<int, std::vector<double>> recvDataBuffers;
        std::unordered_map<int, SIRCell> ghostCellData;
        for (const auto& pair : recvCounts) { int neighborRank = pair.first; int count = pair.second; if (count > 0) { int expectedDoubles = count * DOUBLES_PER_CELL; recvDataBuffers[neighborRank].resize(expectedDoubles); MPI_Request req = MPI_REQUEST_NULL; MPI_Irecv(recvDataBuffers[neighborRank].data(), expectedDoubles, MPI_DOUBLE, neighborRank, MPI_TAG_DATA + neighborRank, MPI_COMM_WORLD, &req); if (req != MPI_REQUEST_NULL) dataRecvRequests.push_back(req); } }
        for (const auto& pair : localIndicesToSendToRank) { int neighborRank = pair.first; const auto& localIndicesSet = pair.second; if (sendCounts.count(neighborRank) && sendCounts[neighborRank] > 0) { int countToSend = sendCounts[neighborRank]; int expectedDoubles = countToSend * DOUBLES_PER_CELL; sendDataBuffers[neighborRank].reserve(expectedDoubles); for (int localIndex : localIndicesSet) { if (localIndex >= 0 && static_cast<size_t>(localIndex) < grid.size()) { const SIRCell& cell = grid[localIndex]; sendDataBuffers[neighborRank].push_back(cell.getS()); sendDataBuffers[neighborRank].push_back(cell.getI()); sendDataBuffers[neighborRank].push_back(cell.getR()); } else { std::cerr << "FATAL Error: Rank " << rank << " Invalid local index " << localIndex << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); } } if (sendDataBuffers[neighborRank].size() != static_cast<size_t>(expectedDoubles)) { std::cerr << "FATAL Error: Rank " << rank << " Send buffer size mismatch rank " << neighborRank << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); } MPI_Request req = MPI_REQUEST_NULL; MPI_Isend(sendDataBuffers[neighborRank].data(), expectedDoubles, MPI_DOUBLE, neighborRank, MPI_TAG_DATA + rank, MPI_COMM_WORLD, &req); if (req != MPI_REQUEST_NULL) dataSendRequests.push_back(req); } }
        if (!dataRecvRequests.empty()) { MPI_Waitall(dataRecvRequests.size(), dataRecvRequests.data(), MPI_STATUSES_IGNORE); for (auto& pair : recvDataBuffers) { int neighborRank = pair.first; auto& buffer = pair.second; if (!globalIdsToReceiveFromRank.count(neighborRank)) continue; const auto& expectedGlobalIdsSet = globalIdsToReceiveFromRank.at(neighborRank); std::vector<int> expectedGlobalIds(expectedGlobalIdsSet.begin(), expectedGlobalIdsSet.end()); if (buffer.size() != static_cast<size_t>(recvCounts[neighborRank] * DOUBLES_PER_CELL)) { std::cerr << "FATAL Error: Rank " << rank << " Received data size mismatch rank " << neighborRank << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); } if (expectedGlobalIds.size() != static_cast<size_t>(recvCounts[neighborRank])) { std::cerr << "FATAL Error: Rank " << rank << " Mismatch #IDs rank " << neighborRank << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); } for (size_t i = 0; i < expectedGlobalIds.size(); ++i) { int globalId = expectedGlobalIds[i]; size_t bufferIdx = i * DOUBLES_PER_CELL; if (bufferIdx + 2 < buffer.size()) { double s_val = buffer[bufferIdx + 0]; double i_val = buffer[bufferIdx + 1]; double r_val = buffer[bufferIdx + 2]; ghostCellData[globalId] = SIRCell(s_val, i_val, r_val); } else { std::cerr << "FATAL Error: Rank " << rank << " Buffer read index OOB rank " << neighborRank << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); } } } }
        if (!dataSendRequests.empty()) { MPI_Waitall(dataSendRequests.size(), dataSendRequests.data(), MPI_STATUSES_IGNORE); }
        std::vector<SIRCell> newGrid(grid.size());
        for (size_t localIndex = 0; localIndex < grid.size(); ++localIndex) { const SIRCell& currentCell = grid[localIndex]; int globalId = localIndexToGlobalId[localIndex]; std::vector<SIRCell> neighborsForUpdate; if (cellNeighborMap.count(globalId)) { const std::vector<int>& neighborGlobalIds = cellNeighborMap.at(globalId); neighborsForUpdate.reserve(neighborGlobalIds.size()); for (int neighborGlobalId : neighborGlobalIds) { auto it_local = globalToLocalCellIndex.find(neighborGlobalId); if (it_local != globalToLocalCellIndex.end()) { neighborsForUpdate.push_back(grid[it_local->second]); } else { auto it_ghost = ghostCellData.find(neighborGlobalId); if (it_ghost != ghostCellData.end()) { neighborsForUpdate.push_back(it_ghost->second); } } } } newGrid[localIndex] = model.rk4StepWithNeighbors(currentCell, neighborsForUpdate); }
        grid = std::move(newGrid);
        double sumS = 0, sumI = 0, sumR = 0; if (!grid.empty()) { for (const auto& cell : grid) { sumS += cell.getS(); sumI += cell.getI(); sumR += cell.getR(); } sumS /= grid.size(); sumI /= grid.size(); sumR /= grid.size(); } double timeVal = static_cast<double>(step + 1) * model.getDt(); localResults.push_back({timeVal, sumS, sumI, sumR});
        if (size > 1) { MPI_Barrier(MPI_COMM_WORLD); }
    }
    std::cout << "Rank " << rank << " finished simulation loop." << std::endl;
    return localResults; // Make sure return is here
}

void GridSimulation::initialize(const std::vector<SIRCell>& localGrid, int numProcesses) {
    setGrid(localGrid);

    auto cells = createCellsMap();
    auto blocks = divideIntoOptimalBlocks(cells, numProcesses);

    int totalCells = static_cast<int>(localGrid.size()) * numProcesses;
    int numBlocks = static_cast<int>(blocks.size());
    auto [rows, cols] = calculateGridDimensions(totalCells, numBlocks);

    auto neighborMap = build2DGridNeighborMap(rows, cols);
    setNeighborMap(neighborMap);
}

std::unordered_map<int, std::vector<int>> GridSimulation::build2DGridNeighborMap(int rows, int cols) {
    std::unordered_map<int, std::vector<int>> neighbors;
    for (int i = 0; i < rows * cols; ++i) {
        int row = i / cols;
        int col = i % cols;

        std::vector<int> gridNeighbors;

        if (row > 0) gridNeighbors.push_back((row - 1) * cols + col);     // up
        if (row < rows - 1) gridNeighbors.push_back((row + 1) * cols + col); // down
        if (col > 0) gridNeighbors.push_back(i - 1);                       // left
        if (col < cols - 1) gridNeighbors.push_back(i + 1);               // right

        neighbors[i] = gridNeighbors;
    }
    return neighbors;
}

std::pair<int, int> GridSimulation::calculateGridDimensions(int totalCells, int numBlocks) {
    if (totalCells % numBlocks != 0) {
        int extraCells = numBlocks - (totalCells % numBlocks);
        totalCells += extraCells;
    }

    int blockRows = static_cast<int>(std::sqrt(numBlocks));
    while (numBlocks % blockRows != 0) {
        blockRows--;
    }
    int blockCols = numBlocks / blockRows;

    int cellsPerBlock = totalCells / numBlocks;
    int cellsPerBlockRow = static_cast<int>(std::sqrt(cellsPerBlock));
    while (cellsPerBlock % cellsPerBlockRow != 0) {
        cellsPerBlockRow--;
    }
    int cellsPerBlockCol = cellsPerBlock / cellsPerBlockRow;

    int rows = blockRows * cellsPerBlockRow;
    int cols = blockCols * cellsPerBlockCol;

    return {rows, cols};
}
