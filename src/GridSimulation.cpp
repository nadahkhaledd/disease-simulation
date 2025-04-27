#include "../header/GridSimulation.h"
#include <mpi.h>
#include <unordered_map>
#include <map>
#include <list>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>

GridSimulation::GridSimulation(const SIRModel& m, int mpiRank, int mpiSize) 
    : model(m), rank(mpiRank), size(mpiSize) {}

void GridSimulation::setGrid(const std::vector<SIRCell>& initialGrid) {
    grid = initialGrid;
}

std::vector<SIRCell>& GridSimulation::getGrid() {
    return grid;
}

int GridSimulation::getLocalSize() const {
    return grid.size();
}

void GridSimulation::setNeighborMap(const std::unordered_map<int, std::vector<int>>& map) {
    neighborMap = map;
}

void GridSimulation::updateGrid() {
    std::vector<SIRCell> newGrid = grid;
    for (size_t i = 0; i < grid.size(); ++i) {
        newGrid[i] = model.rk4Step(grid[i]);
    }
    grid = newGrid;
}

void GridSimulation::updateGridNew() {
    std::vector<SIRCell> newGrid = grid;

    for (size_t i = 0; i < grid.size(); ++i) {
        std::vector<SIRCell> neighbors;

        // Get neighbors from map (if exists)
        if (neighborMap.find(i) != neighborMap.end()) {
            const std::vector<int>& neighborIDs = neighborMap[i];
            for (int j : neighborIDs) {
                if (j >= 0 && j < static_cast<int>(grid.size())) {
                    neighbors.push_back(grid[j]);
                }
            }
        }

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
    std::ifstream infile("data/sorted_initial_conditions.csv");
    if (!infile) {
        std::cerr << "Error: Could not open sorted_initial_conditions.csv\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string line;
    std::getline(infile, line); // Skip header
    int cellId = 0;

    while (std::getline(infile, line)) {
        std::istringstream ss(line);
        std::string state;
        std::getline(ss, state, ','); // Assuming the first column is the state name
        cells[state] = cellId++;
    }

    return cells;
}

std::map<int, std::list<int>> GridSimulation::divideIntoBlocks(
    const std::map<std::string, int>& cells, int numBlocks) {
    std::map<int, std::list<int>> blocks;

    // Extract and sort cell IDs
    std::vector<int> sortedCellIds;
    for (const auto& [state, cellId] : cells) {
        sortedCellIds.push_back(cellId);
    }
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
    std::vector<std::vector<double>> results; // [time, avg_S, avg_I, avg_R]
    
    for (int step = 0; step < model.getNumSteps(); ++step) {
        // Update grid
        updateGridNew();
        
        // Compute average S, I, R
        double sumS = 0, sumI = 0, sumR = 0;
        for (auto &cell : grid) {
            sumS += cell.getS();
            sumI += cell.getI();
            sumR += cell.getR();
        }
        
        double avgS = sumS / grid.size();
        double avgI = sumI / grid.size();
        double avgR = sumR / grid.size();
        double timeVal = step * model.getDt();
        
        // Ensure values are within valid bounds
        avgS = std::max(0.0, std::min(1.0, avgS));
        avgI = std::max(0.0, std::min(1.0, avgI));
        avgR = std::max(0.0, std::min(1.0, avgR));

        results.push_back({timeVal, avgS, avgI, avgR});
        
        // Synchronize processes
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    return results;
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
